import copy
import random

# typing 
from typing import List, Tuple
import time
import numpy as np
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


_TRAVERSAL_COMPILED_RESIDUAL = None
_TRAVERSAL_COMPILED_RESIDUAL_UNAVAILABLE = False


def _get_traversal_compiled_residual():
    """Lazily build the optional SIMD residual kernel used by cpu_lazy.

    Numba is deliberately optional: environments without it retain the exact
    NumPy implementation. Compilation happens during benchmark warm-up and is
    therefore excluded from measured generation time.
    """
    global _TRAVERSAL_COMPILED_RESIDUAL
    global _TRAVERSAL_COMPILED_RESIDUAL_UNAVAILABLE
    if _TRAVERSAL_COMPILED_RESIDUAL is not None:
        return _TRAVERSAL_COMPILED_RESIDUAL
    if _TRAVERSAL_COMPILED_RESIDUAL_UNAVAILABLE:
        return None
    try:
        from numba import njit
    except (ImportError, ModuleNotFoundError):
        _TRAVERSAL_COMPILED_RESIDUAL_UNAVAILABLE = True
        return None

    @njit(
        cache=False,
        fastmath={"reassoc", "contract"},
        nogil=True,
    )
    def compiled_residual(output, target, draft, scale):
        # Use a widened, reassociable reduction so LLVM can vectorize the
        # vocabulary scan. Individual output probabilities remain float32.
        residual_mass = 0.0
        for token_index in range(target.shape[0]):
            value = target[token_index] * scale - draft[token_index]
            if value < 0.0:
                value = 0.0
            output[token_index] = value
            residual_mass += value
        return residual_mass

    _TRAVERSAL_COMPILED_RESIDUAL = compiled_residual
    return compiled_residual


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def _generate_draft_tree(
        model,
        hidden_states,
        input_ids,
        logits_processor,
        verify_method="default",
):
    """Dispatch draft generation while preserving the legacy return contract."""
    method = verify_method.lower()
    if method == "default":
        model.last_univer_tree_info = None
        draft_tree = model.ea_layer.topK_genrate(
            hidden_states,
            input_ids,
            model.base_model.lm_head,
            logits_processor,
        )
        return (*draft_tree, None)
    if method in {"univer", "greedy"}:
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            tree_info,
        ) = model.ea_layer.univer_generate(
            hidden_states,
            input_ids,
            model.base_model.lm_head,
            logits_processor,
        )
        tree_info["compile_univer"] = bool(
            getattr(model, "compile_univer", False)
        )
        tree_info["collect_diagnostics"] = False
        model.last_univer_tree_info = tree_info
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_info
    if method in {"rrsw", "traversal", "traversal_verification"}:
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            tree_info,
        ) = model.ea_layer.rrsw_generate(
            hidden_states,
            input_ids,
            model.base_model.lm_head,
            logits_processor,
        )
        tree_info["traversal_backend"] = getattr(
            model, "traversal_backend", "gpu"
        )
        tree_info["traversal_compiled_residual"] = bool(getattr(
            model, "traversal_compiled_residual", True
        ))
        if (
                method in {"traversal", "traversal_verification"}
                and tree_info["traversal_backend"] in {"cpu", "cpu_lazy"}
        ):
            distribution = tree_info["draft_distributions"]
            distribution_size = distribution.numel()

            if (
                    getattr(model, "traversal_pinned_buffer", True)
                    and distribution.device.type == "cuda"
            ):
                uniform_count = tree_info["parent_indices"].numel() - 1
                buffer_size = 2 * distribution_size + uniform_count
                cpu_buffer = getattr(model, "_traversal_cpu_buffer", None)
                if (
                        cpu_buffer is None
                        or cpu_buffer.numel() != buffer_size
                        or cpu_buffer.dtype != torch.float32
                ):
                    cpu_buffer = torch.empty(
                        buffer_size,
                        dtype=torch.float32,
                        device="cpu",
                        pin_memory=True,
                    )
                    model._traversal_cpu_buffer = cpu_buffer
                tree_info["_traversal_cpu_buffer"] = cpu_buffer

                # The proposal matrix is complete before target decoding
                # starts. Copy it on a dedicated stream now so its D2H
                # transfer overlaps the much longer target-model forward.
                # Verification then only has to copy target probabilities and
                # random uniforms before entering the CPU traversal loop.
                copy_stream = getattr(model, "_traversal_proposal_copy_stream", None)
                if (
                        copy_stream is None
                        or getattr(model, "_traversal_proposal_copy_device", None)
                        != distribution.device
                ):
                    copy_stream = torch.cuda.Stream(device=distribution.device)
                    model._traversal_proposal_copy_stream = copy_stream
                    model._traversal_proposal_copy_device = distribution.device
                copy_event = getattr(model, "_traversal_proposal_copy_event", None)
                if copy_event is None:
                    copy_event = torch.cuda.Event()
                    model._traversal_proposal_copy_event = copy_event

                current_stream = torch.cuda.current_stream(distribution.device)
                copy_stream.wait_stream(current_stream)
                with torch.cuda.stream(copy_stream):
                    cpu_buffer[
                        distribution_size:2 * distribution_size
                    ].copy_(distribution.reshape(-1), non_blocking=True)
                    copy_event.record(copy_stream)
                tree_info["_traversal_proposal_copy_event"] = copy_event
        model.last_univer_tree_info = tree_info
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_info
    raise ValueError(
        f"Unsupported verify_method={verify_method!r}; expected 'default', "
        "'univer', 'greedy', 'rrsw', or 'traversal_verification'."
    )


def initialize_tree(
        input_ids,
        model,
        past_key_values,
        logits_processor,
        verify_method="default",
        return_tree_info=False,
):
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # Clone the output hidden states
    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states=torch.cat(outputs["hidden_states"],dim=-1)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_info = _generate_draft_tree(
        model,
        hidden_states,
        input_ids,
        logits_processor,
        verify_method,
    )
    result = (
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        orig,
        hidden_states,
        token,
    )
    if return_tree_info:
        return (*result, tree_info)
    return result


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
        compact_logits=False,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    # Probability-aware verifiers already carry a node-to-path topology map
    # and can consume one logit row per physical tree node.  Expanding here to
    # every root-to-leaf path duplicates most rows (63 nodes become 32 x 6
    # rows in a balanced binary depth-5 tree) and is especially expensive for
    # Llama 3's 128K vocabulary.
    logits = tree_logits[0] if compact_logits else tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs





def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


def _univer_allocate_dense_group(
        prefix_probabilities,
        targets,
        residuals,
        child_token_ids,
):
    """Tensor-only UniVer allocation suitable for torch.compile fusion."""
    epsilon = torch.finfo(torch.float32).eps
    sampled_token_ids = child_token_ids[:, -1]
    sampled_masses = residuals.gather(
        1, sampled_token_ids[:, None]
    ).squeeze(1)
    sampled_targets = targets.gather(
        1, sampled_token_ids[:, None]
    ).squeeze(1)
    sampled_acceptance = torch.where(
        sampled_masses > epsilon,
        torch.clamp(
            prefix_probabilities * sampled_targets
            / sampled_masses.clamp_min(epsilon),
            max=1.0,
        ),
        torch.zeros_like(sampled_masses),
    )
    positive_outputs = torch.relu(
        prefix_probabilities[:, None] * targets - residuals
    )
    positive_mass = positive_outputs.sum(dim=1)
    normalization = 1.0 - prefix_probabilities + positive_mass
    remaining_after_sample = 1.0 - sampled_acceptance
    scale = torch.where(
        normalization > epsilon,
        remaining_after_sample / normalization.clamp_min(epsilon),
        torch.zeros_like(normalization),
    )
    unscaled_child_outputs = positive_outputs.gather(1, child_token_ids)
    scaled_child_outputs = unscaled_child_outputs * scale[:, None]
    child_marginals = torch.cat(
        (scaled_child_outputs[:, :-1], sampled_acceptance[:, None]), dim=1
    )
    residual_output_mass = (
        positive_mass - unscaled_child_outputs.sum(dim=1)
    ).clamp_min(0) * scale
    group_rejections = torch.where(
        normalization > epsilon,
        remaining_after_sample * (1.0 - prefix_probabilities)
        / normalization.clamp_min(epsilon),
        torch.zeros_like(normalization),
    )
    remaining_probability = (
        1.0 - child_marginals.sum(dim=1)
    ).clamp_min(0.0)
    group_fallback = torch.where(
        remaining_probability > epsilon,
        torch.clamp(
            residual_output_mass
            / remaining_probability.clamp_min(epsilon),
            0.0,
            1.0,
        ),
        torch.zeros_like(remaining_probability),
    )
    removed_mass_before = torch.cat(
        (
            child_marginals.new_zeros(child_marginals.shape[0], 1),
            child_marginals.cumsum(dim=1)[:, :-1],
        ),
        dim=1,
    )
    remaining_before_child = (1.0 - removed_mass_before).clamp_min(0.0)
    child_effective = torch.where(
        remaining_before_child > epsilon,
        torch.clamp(
            child_marginals / remaining_before_child.clamp_min(epsilon),
            0.0,
            1.0,
        ),
        torch.zeros_like(child_marginals),
    )
    mass_errors = torch.abs(
        child_marginals.sum(dim=1)
        + residual_output_mass + group_rejections - 1.0
    )
    return (
        child_marginals,
        residual_output_mass,
        group_rejections,
        group_fallback,
        child_effective,
        scale,
        mass_errors,
    )


def _univer_allocate_dense_group_fast(
        prefix_probabilities,
        targets,
        residuals,
        child_token_ids,
):
    """Production allocation without benchmark-unused diagnostics."""
    outputs = _univer_allocate_dense_group(
        prefix_probabilities,
        targets,
        residuals,
        child_token_ids,
    )
    # Inductor can eliminate rejection/marginal diagnostics when only the
    # three tensors required by the decision phase are returned.
    return outputs[3], outputs[4], outputs[5]


_compiled_univer_allocate_dense_group = None
_compiled_univer_allocate_dense_group_fast = None


def _get_compiled_univer_allocator():
    global _compiled_univer_allocate_dense_group
    if _compiled_univer_allocate_dense_group is None:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "--compile-univer requires a PyTorch build with torch.compile"
            )
        _compiled_univer_allocate_dense_group = torch.compile(
            _univer_allocate_dense_group,
            fullgraph=True,
            # The balanced tree uses fixed batches of 1, 2, 4, 8 and 16
            # parents. Specialized graphs avoid dynamic reduction guards.
            dynamic=False,
        )
    return _compiled_univer_allocate_dense_group


def _get_compiled_univer_fast_allocator():
    global _compiled_univer_allocate_dense_group_fast
    if _compiled_univer_allocate_dense_group_fast is None:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "--compile-univer requires a PyTorch build with torch.compile"
            )
        _compiled_univer_allocate_dense_group_fast = torch.compile(
            _univer_allocate_dense_group_fast,
            fullgraph=True,
            dynamic=False,
        )
    return _compiled_univer_allocate_dense_group_fast


def evaluate_posterior2(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        retrieve_indices: torch.Tensor,
        logits_processor,
        tree_info,
):
    """Run UniVer allocation followed by one post-order decision pass.

    ``tree_info`` is returned by ``univer_generate`` and carries the full draft
    and residual distributions for every expanded parent.  The return value is
    compatible with ``update_inference_inputs``: a retrieval-path index, the
    accepted node depth, and the distribution used to sample the next token.
    """
    if tree_info is None:
        raise ValueError("evaluate_posterior2 requires UniVer draft-tree metadata")
    collect_diagnostics = tree_info.get("collect_diagnostics", True)

    device = logits.device
    parent_indices = tree_info["parent_indices"].to(device)
    node_token_ids = tree_info["node_token_ids"].to(device)
    expanded_parents = tree_info["expanded_parent_indices"].to(device)
    draft_distributions = tree_info.get("draft_distributions")
    residual_distributions = tree_info["residual_distributions"].to(
        device=device, dtype=torch.float32
    )
    child_groups = [children.to(device) for children in tree_info["child_groups"]]
    node_count = parent_indices.numel()
    vocab_size = logits.shape[-1]

    if node_token_ids.numel() != node_count:
        raise ValueError("node_token_ids and parent_indices must describe the same tree")
    if draft_distributions is not None and draft_distributions.shape != residual_distributions.shape:
        raise ValueError("draft and residual distributions must have matching shapes")
    if draft_distributions is not None and draft_distributions.shape[0] != expanded_parents.numel():
        raise ValueError("one draft distribution is required per expanded parent")
    if draft_distributions is not None and draft_distributions.shape[-1] != vocab_size:
        raise ValueError("draft and target vocabulary sizes do not match")

    # New trees carry these fixed lookup tables.  Keep a compatibility path
    # for hand-built/older metadata, but compute it once on the host rather
    # than scanning the full retrieve tensor separately for every node.
    if "node_rows" in tree_info and "node_depths" in tree_info:
        node_rows = tree_info["node_rows"].to(device)
        node_depths = tree_info["node_depths"].to(device)
    else:
        node_rows_list = [-1] * node_count
        node_depths_list = [-1] * node_count
        for row_index, path in enumerate(retrieve_indices.detach().cpu().tolist()):
            for depth, node_index in enumerate(path):
                if node_index >= 0 and node_rows_list[node_index] < 0:
                    node_rows_list[node_index] = row_index
                    node_depths_list[node_index] = depth
        if any(row < 0 for row in node_rows_list):
            raise ValueError("a tree node is missing from retrieve_indices")
        node_rows = torch.tensor(node_rows_list, dtype=torch.long, device=device)
        node_depths = torch.tensor(node_depths_list, dtype=torch.long, device=device)

    compact_target_logits = logits.ndim == 2
    if compact_target_logits:
        if logits.shape[0] != node_count:
            raise ValueError(
                "Compact target logits must contain one row per tree node: "
                f"got {logits.shape[0]} rows for {node_count} nodes."
            )
        parent_logits = logits[expanded_parents]
    elif logits.ndim == 3 and logits.shape[:2] == candidates.shape:
        parent_logits = logits[
            node_rows[expanded_parents], node_depths[expanded_parents]
        ]
    else:
        raise ValueError(
            "UniVer target logits must be [tree_nodes, vocab] or "
            "[paths, length, vocab]."
        )

    # Allocation only consumes the target distribution at expanded parents.
    # Leaves need a target softmax only when the decision phase actually
    # selects one, so avoid materialising about half of the [node, vocab]
    # probability tensor for a full binary tree.
    def logits_to_probabilities(node_logits):
        if logits_processor is None:
            probabilities = torch.zeros_like(node_logits, dtype=torch.float32)
            probabilities.scatter_(
                1, torch.argmax(node_logits, dim=-1, keepdim=True), 1.0
            )
            return probabilities
        processed_logits = logits_processor(None, node_logits)
        return torch.softmax(
            processed_logits, dim=-1, dtype=torch.float32
        )

    target_parent_probabilities = logits_to_probabilities(parent_logits)

    effective_probabilities = torch.zeros(node_count, dtype=torch.float32, device=device)
    fallback_probabilities = torch.zeros(node_count, dtype=torch.float32, device=device)
    rejection_probabilities = torch.zeros(node_count, dtype=torch.float32, device=device)
    marginal_probabilities = torch.zeros(node_count, dtype=torch.float32, device=device)
    allocation_mass_errors = torch.zeros(
        node_count, dtype=torch.float32, device=device
    )
    # Keep only the scalar needed to reconstruct a fallback distribution.
    # Exactly one fallback can be selected in the decision phase, so eagerly
    # materialising and normalising [expanded_parents, vocab] wasted roughly
    # 15 MiB plus a full-vocabulary division for the 63-node Llama-3 tree.
    allocation_scales = torch.zeros(
        expanded_parents.numel(), dtype=torch.float32, device=device
    )
    effective_probabilities[0] = 1.0
    epsilon = torch.finfo(torch.float32).eps

    if "parent_distribution_rows" in tree_info:
        parent_distribution_rows = tree_info["parent_distribution_rows"].to(device)
    else:
        parent_distribution_rows = torch.full(
            (node_count,), -1, dtype=torch.long, device=device
        )
        parent_distribution_rows[expanded_parents] = torch.arange(
            expanded_parents.numel(), device=device
        )

    if "expanded_depth_groups" in tree_info:
        expanded_depth_groups = tree_info["expanded_depth_groups"]
        expanded_depth_group_tensors = tree_info.get(
            "expanded_depth_group_tensors"
        )
    else:
        # Compatibility for manually constructed metadata.  Generated trees
        # already cache these small Python row lists.
        expanded_list = expanded_parents.detach().cpu().tolist()
        node_depth_list = node_depths.detach().cpu().tolist()
        rows_by_depth = {}
        for row_index, parent_index in enumerate(expanded_list):
            rows_by_depth.setdefault(node_depth_list[parent_index], []).append(row_index)
        expanded_depth_groups = [rows_by_depth[d] for d in sorted(rows_by_depth)]
        expanded_depth_group_tensors = None

    child_index_matrix = tree_info.get("child_index_matrix")
    if child_index_matrix is not None:
        child_index_matrix = child_index_matrix.to(device)

    # A parent's prefix probability depends only on the preceding depth.
    # Therefore all full-vocabulary allocation work at the same depth can be
    # evaluated as one batch.  A depth-5 binary tree now uses five allocation
    # batches rather than 31 independent parent launches.
    for group_index, row_indices_list in enumerate(expanded_depth_groups):
        if not row_indices_list:
            continue
        if expanded_depth_group_tensors is None:
            row_indices = torch.tensor(
                row_indices_list, dtype=torch.long, device=device
            )
        else:
            row_indices = expanded_depth_group_tensors[group_index].to(device)
        parents = expanded_parents[row_indices]
        prefix_probabilities = effective_probabilities[parents]
        targets = target_parent_probabilities[row_indices]
        residuals = residual_distributions[row_indices]
        group_children = (
            child_index_matrix[row_indices]
            if child_index_matrix is not None
            else None
        )
        if group_children is not None:
            child_token_ids = node_token_ids[group_children]
            if collect_diagnostics:
                allocator = (
                    _get_compiled_univer_allocator()
                    if tree_info.get("compile_univer", False)
                    else _univer_allocate_dense_group
                )
                (
                    child_marginals,
                    residual_output_mass,
                    group_rejections,
                    group_fallback,
                    child_effective,
                    scale,
                    group_mass_errors,
                ) = allocator(
                    prefix_probabilities,
                    targets,
                    residuals,
                    child_token_ids,
                )
                marginal_probabilities[
                    group_children.reshape(-1)
                ] = child_marginals.reshape(-1)
                rejection_probabilities[parents] = group_rejections
                allocation_mass_errors[parents] = group_mass_errors
            else:
                allocator = (
                    _get_compiled_univer_fast_allocator()
                    if tree_info.get("compile_univer", False)
                    else _univer_allocate_dense_group_fast
                )
                group_fallback, child_effective, scale = allocator(
                    prefix_probabilities,
                    targets,
                    residuals,
                    child_token_ids,
                )
            allocation_scales[row_indices] = scale
            effective_probabilities[
                group_children.reshape(-1)
            ] = child_effective.reshape(-1)
            fallback_probabilities[parents] = group_fallback
            continue

        sampled_token_ids = (
            node_token_ids[group_children[:, -1]]
            if group_children is not None
            else torch.stack([
                node_token_ids[child_groups[row_index][-1]]
                for row_index in row_indices_list
            ])
        )
        sampled_masses = residuals.gather(1, sampled_token_ids[:, None]).squeeze(1)
        sampled_targets = targets.gather(1, sampled_token_ids[:, None]).squeeze(1)
        sampled_acceptance = torch.where(
            sampled_masses > epsilon,
            torch.clamp(
                prefix_probabilities
                * sampled_targets
                / sampled_masses.clamp_min(epsilon),
                max=1.0,
            ),
            torch.zeros_like(sampled_masses),
        )

        positive_outputs = torch.relu(
            prefix_probabilities[:, None] * targets - residuals
        )
        positive_mass = positive_outputs.sum(dim=1)
        normalization = (
            1.0 - prefix_probabilities + positive_mass
        )
        remaining_after_sample = 1.0 - sampled_acceptance
        scale = torch.where(
            normalization > epsilon,
            remaining_after_sample / normalization.clamp_min(epsilon),
            torch.zeros_like(normalization),
        )
        allocation_scales[row_indices] = scale

        if group_children is not None:
            child_token_ids = node_token_ids[group_children]
            # Keep the full-vocabulary positive part unscaled.  Only its total
            # mass and the two candidate entries are needed by allocation;
            # the fallback distribution is reconstructed lazily only if it is
            # selected in the decision phase.  The previous implementation
            # scaled the entire matrix, scattered candidate zeros into it and
            # then performed a second full-vocabulary reduction.
            unscaled_child_outputs = positive_outputs.gather(
                1, child_token_ids
            )
            child_marginals = unscaled_child_outputs * scale[:, None]
            child_marginals[:, -1] = sampled_acceptance
            marginal_probabilities[group_children.reshape(-1)] = child_marginals.reshape(-1)
            residual_output_mass = (
                positive_mass - unscaled_child_outputs.sum(dim=1)
            ).clamp_min_(0).mul_(scale)
        else:
            group_child_marginals = []
            residual_output_masses = []
            for local_row, row_index in enumerate(row_indices_list):
                children = child_groups[row_index]
                child_token_ids = node_token_ids[children]
                unscaled_child_outputs = positive_outputs[
                    local_row, child_token_ids
                ]
                current_marginals = (
                    unscaled_child_outputs * scale[local_row]
                )
                current_marginals[-1] = sampled_acceptance[local_row]
                group_child_marginals.append(current_marginals)
                marginal_probabilities[children] = current_marginals
                residual_output_masses.append(
                    (
                        positive_mass[local_row]
                        - unscaled_child_outputs.sum()
                    ).clamp_min(0) * scale[local_row]
                )
            residual_output_mass = torch.stack(residual_output_masses)
        group_rejections = torch.where(
            normalization > epsilon,
            remaining_after_sample
            * (1.0 - prefix_probabilities)
            / normalization.clamp_min(epsilon),
            torch.zeros_like(normalization),
        )
        rejection_probabilities[parents] = group_rejections
        child_marginal_sums = (
            child_marginals.sum(dim=1)
            if group_children is not None
            else torch.stack([
                current_marginals.sum()
                for current_marginals in group_child_marginals
            ])
        )
        allocation_mass_errors[parents] = torch.abs(
            child_marginal_sums + residual_output_mass + group_rejections - 1.0
        )

        if group_children is not None:
            removed_mass_before = torch.cat(
                (
                    child_marginals.new_zeros(child_marginals.shape[0], 1),
                    child_marginals.cumsum(dim=1)[:, :-1],
                ),
                dim=1,
            )
            remaining_before_child = (1.0 - removed_mass_before).clamp_min(0.0)
            child_effective = torch.where(
                remaining_before_child > epsilon,
                torch.clamp(
                    child_marginals / remaining_before_child.clamp_min(epsilon),
                    0.0,
                    1.0,
                ),
                torch.zeros_like(child_marginals),
            )
            effective_probabilities[group_children.reshape(-1)] = child_effective.reshape(-1)
            remaining_probability = (1.0 - child_marginals.sum(dim=1)).clamp_min(0.0)
            fallback_probabilities[parents] = torch.where(
                remaining_probability > epsilon,
                torch.clamp(
                    residual_output_mass
                    / remaining_probability.clamp_min(epsilon),
                    0.0,
                    1.0,
                ),
                remaining_probability.new_zeros(()),
            )
        else:
            for local_row, (row_index, current_marginals) in enumerate(
                    zip(row_indices_list, group_child_marginals)
            ):
                children = child_groups[row_index]
                removed_mass_before = torch.cat(
                    (current_marginals.new_zeros(1), current_marginals.cumsum(0)[:-1])
                )
                remaining_before_child = (1.0 - removed_mass_before).clamp_min(0.0)
                effective_probabilities[children] = torch.where(
                    remaining_before_child > epsilon,
                    torch.clamp(
                        current_marginals / remaining_before_child.clamp_min(epsilon),
                        0.0,
                        1.0,
                    ),
                    torch.zeros_like(current_marginals),
                )
                remaining_probability = (1.0 - current_marginals.sum()).clamp_min(0.0)
                fallback_probabilities[parents[local_row]] = torch.where(
                    remaining_probability > epsilon,
                    torch.clamp(
                        residual_output_mass[local_row]
                        / remaining_probability.clamp_min(epsilon),
                        0.0,
                        1.0,
                    ),
                    remaining_probability.new_zeros(()),
                )

    if "postorder" in tree_info:
        postorder = tree_info["postorder"].to(device)
    else:
        parent_list = parent_indices.detach().cpu().tolist()
        children_lists = [[] for _ in range(node_count)]
        for node_index, parent_index in enumerate(parent_list[1:], start=1):
            children_lists[parent_index].append(node_index)
        postorder_list = []

        def visit(node_index):
            for child_index in children_lists[node_index]:
                visit(child_index)
            postorder_list.append(node_index)

        visit(0)
        postorder = torch.tensor(postorder_list, dtype=torch.long, device=device)

    uniforms = torch.rand(node_count, device=device)
    ordered_parent_rows = parent_distribution_rows[postorder]
    thresholds = torch.where(
        ordered_parent_rows >= 0,
        fallback_probabilities[postorder],
        effective_probabilities[postorder],
    )
    accepted_positions = torch.nonzero(
        uniforms[postorder] < thresholds, as_tuple=False
    ).flatten()
    if accepted_positions.numel() == 0:
        raise RuntimeError("UniVer decision phase rejected the root subtree")
    accepted_node = postorder[accepted_positions[0]]
    accepted_parent_row = int(parent_distribution_rows[accepted_node].item())
    if accepted_parent_row >= 0:
        parent_node = expanded_parents[accepted_parent_row]
        sample_p = torch.relu(
            effective_probabilities[parent_node]
            * target_parent_probabilities[accepted_parent_row]
            - residual_distributions[accepted_parent_row]
        )
        sample_p.mul_(allocation_scales[accepted_parent_row])
        selected_children = (
            child_index_matrix[accepted_parent_row]
            if child_index_matrix is not None
            else child_groups[accepted_parent_row]
        )
        sample_p[node_token_ids[selected_children]] = 0
        sample_p = _normalize_probability_distribution(
            sample_p, target_parent_probabilities[accepted_parent_row],
            fallback_is_normalized=True,
        )
    else:
        if compact_target_logits:
            accepted_logits = logits[accepted_node][None]
        else:
            accepted_logits = logits[
                node_rows[accepted_node], node_depths[accepted_node]
            ][None]
        sample_p = logits_to_probabilities(accepted_logits)[0]

    if collect_diagnostics:
        tree_info["effective_acceptance_probabilities"] = effective_probabilities
        tree_info["fallback_acceptance_probabilities"] = fallback_probabilities
        tree_info["marginal_acceptance_probabilities"] = marginal_probabilities
        tree_info["rejection_probabilities"] = rejection_probabilities
        tree_info["allocation_mass_errors"] = allocation_mass_errors
        tree_info["target_parent_distributions"] = target_parent_probabilities
        tree_info["target_distribution_node_indices"] = expanded_parents
        # Compatibility alias. Row i belongs to expanded_parent_indices[i].
        tree_info["target_distributions"] = target_parent_probabilities
        tree_info["postorder"] = postorder

    best_candidate = node_rows[accepted_node].to(candidates.device)
    accept_length = int(node_depths[accepted_node].item())
    return best_candidate, accept_length, sample_p



def _tree_proposal_values(tree_info, source_node_indices, fallback, device):
    """Select per-parent draft distributions from stochastic-tree metadata."""
    expanded = tree_info["expanded_parent_indices"].to(device)
    distributions = tree_info["draft_distributions"].to(
        device=device, dtype=torch.float32
    )
    node_count = tree_info["parent_indices"].numel()
    node_to_row = torch.full((node_count,), -1, dtype=torch.long, device=device)
    node_to_row[expanded] = torch.arange(expanded.numel(), device=device)
    distribution_rows = node_to_row[source_node_indices]
    proposal_values = fallback.clone()
    has_distribution = distribution_rows >= 0
    proposal_values[has_distribution] = distributions[
        distribution_rows[has_distribution]
    ]
    return proposal_values


def _metadata_tree_layout(tree_info, retrieve_indices, candidates, device):
    """Return tensor/host topology without reconstructing token prefixes.

    Generated stochastic trees already contain this information.  The
    compatibility reconstruction is retained for hand-built test metadata and
    older cached trees, but production rounds consume the cached host lists.
    """
    parent_indices = tree_info["parent_indices"]
    parent_list = tree_info.get("parent_indices_list")
    if parent_list is None:
        parent_list = parent_indices.detach().cpu().tolist()
    node_count = len(parent_list)

    children_lists = tree_info.get("children_lists")
    if children_lists is None:
        children_lists = [[] for _ in range(node_count)]
        for node_index, parent_index in enumerate(parent_list[1:], start=1):
            children_lists[parent_index].append(node_index)

    node_rows_list = tree_info.get("node_rows_list")
    node_depths_list = tree_info.get("node_depths_list")
    if node_rows_list is None or node_depths_list is None:
        node_rows_list = [-1] * node_count
        node_depths_list = [-1] * node_count
        for row_index, path in enumerate(retrieve_indices.detach().cpu().tolist()):
            for depth, node_index in enumerate(path):
                if node_index >= 0 and node_rows_list[node_index] < 0:
                    node_rows_list[node_index] = row_index
                    node_depths_list[node_index] = depth
    if any(row < 0 for row in node_rows_list):
        raise ValueError("a tree node is missing from retrieve_indices")

    expanded_list = tree_info.get("expanded_parent_indices_list")
    if expanded_list is None:
        expanded_list = tree_info["expanded_parent_indices"].detach().cpu().tolist()
    parent_rows = tree_info.get("parent_distribution_rows_list")
    if parent_rows is None:
        parent_rows = [-1] * node_count
        for row_index, parent_index in enumerate(expanded_list):
            parent_rows[parent_index] = row_index

    postorder = tree_info.get("postorder_list")
    if postorder is None:
        postorder = []

        def visit(node_index):
            for child_index in children_lists[node_index]:
                visit(child_index)
            postorder.append(node_index)

        visit(0)

    node_path_lists = tree_info.get("node_path_lists")
    descendants_lists = tree_info.get("descendants_lists")
    if node_path_lists is None or descendants_lists is None:
        node_path_lists = [[0]]
        descendants_lists = [[] for _ in range(node_count)]
        for node_index in range(1, node_count):
            node_path = node_path_lists[parent_list[node_index]] + [node_index]
            node_path_lists.append(node_path)
            for ancestor_index in node_path[:-1]:
                descendants_lists[ancestor_index].append(node_index)

    traversal_depth_groups = tree_info.get("traversal_depth_groups")
    if traversal_depth_groups is None:
        traversal_depth_groups = []
        for node_depth in sorted(set(node_depths_list[1:])):
            traversal_depth_groups.append([
                node_index
                for node_index in range(1, node_count)
                if node_depths_list[node_index] == node_depth
            ])
    traversal_depth_group_tensors = tree_info.get(
        "traversal_depth_group_tensors"
    )
    traversal_parent_group_tensors = tree_info.get(
        "traversal_parent_group_tensors"
    )
    traversal_parent_row_group_tensors = tree_info.get(
        "traversal_parent_row_group_tensors"
    )
    if (
            traversal_depth_group_tensors is None
            or traversal_parent_group_tensors is None
            or traversal_parent_row_group_tensors is None
    ):
        traversal_depth_group_tensors = []
        traversal_parent_group_tensors = []
        traversal_parent_row_group_tensors = []
        for nodes_list in traversal_depth_groups:
            parents_list = [parent_list[node] for node in nodes_list]
            traversal_depth_group_tensors.append(torch.tensor(
                nodes_list, dtype=torch.long, device=device
            ))
            traversal_parent_group_tensors.append(torch.tensor(
                parents_list, dtype=torch.long, device=device
            ))
            traversal_parent_row_group_tensors.append(torch.tensor(
                [parent_rows[parent] for parent in parents_list],
                dtype=torch.long, device=device,
            ))
    else:
        traversal_depth_group_tensors = [
            group.to(device) for group in traversal_depth_group_tensors
        ]
        traversal_parent_group_tensors = [
            group.to(device) for group in traversal_parent_group_tensors
        ]
        traversal_parent_row_group_tensors = [
            group.to(device) for group in traversal_parent_row_group_tensors
        ]

    if "node_token_ids" in tree_info:
        node_token_ids = tree_info["node_token_ids"].to(device)
    else:
        rows = torch.tensor(node_rows_list, dtype=torch.long, device=candidates.device)
        depths = torch.tensor(node_depths_list, dtype=torch.long, device=candidates.device)
        node_token_ids = candidates[rows, depths].to(device)

    node_token_ids_list = tree_info.get("node_token_ids_list")
    if node_token_ids_list is None:
        node_token_ids_list = node_token_ids.detach().cpu().tolist()
    traversal_nodes_list = tree_info.get("traversal_nodes_list")
    if traversal_nodes_list is None:
        traversal_nodes_list = [
            node_index
            for group in traversal_depth_groups
            for node_index in group
        ]
    traversal_nodes_tensor = tree_info.get("traversal_nodes_tensor")
    traversal_parent_rows_tensor = tree_info.get("traversal_parent_rows_tensor")
    if traversal_nodes_tensor is None or traversal_parent_rows_tensor is None:
        traversal_nodes_tensor = torch.tensor(
            traversal_nodes_list, dtype=torch.long, device=device
        )
        traversal_parent_rows_tensor = torch.tensor(
            [parent_rows[parent_list[node]] for node in traversal_nodes_list],
            dtype=torch.long, device=device,
        )
    else:
        traversal_nodes_tensor = traversal_nodes_tensor.to(device)
        traversal_parent_rows_tensor = traversal_parent_rows_tensor.to(device)

    return {
        "parent_list": parent_list,
        "children_lists": children_lists,
        "node_rows_list": node_rows_list,
        "node_depths_list": node_depths_list,
        "expanded_list": expanded_list,
        "parent_rows": parent_rows,
        "postorder": postorder,
        "node_path_lists": node_path_lists,
        "descendants_lists": descendants_lists,
        "traversal_depth_group_tensors": traversal_depth_group_tensors,
        "traversal_parent_group_tensors": traversal_parent_group_tensors,
        "traversal_parent_row_group_tensors": traversal_parent_row_group_tensors,
        "traversal_nodes_list": traversal_nodes_list,
        "traversal_nodes_tensor": traversal_nodes_tensor,
        "traversal_parent_rows_tensor": traversal_parent_rows_tensor,
        "node_token_ids": node_token_ids,
        "node_token_ids_list": node_token_ids_list,
    }


def _metadata_parent_probabilities(
        logits, candidates, logits_processor, tree_info, layout, device
):
    expanded = tree_info["expanded_parent_indices"].to(device)
    if logits.ndim == 2:
        # Full breadth-first trees keep all expanded parents in the contiguous
        # prefix 0..N-1. Use a view in that common case instead of launching an
        # advanced-index gather that copies the entire [parents, vocab] block
        # immediately before softmax. Irregular/partial trees retain the
        # general indexed path.
        expanded_list = layout["expanded_list"]
        if expanded_list == list(range(len(expanded_list))):
            parent_logits = logits[:len(expanded_list)]
        else:
            parent_logits = logits[expanded]
    else:
        rows = torch.tensor(
            [layout["node_rows_list"][node] for node in layout["expanded_list"]],
            dtype=torch.long, device=device,
        )
        depths = torch.tensor(
            [layout["node_depths_list"][node] for node in layout["expanded_list"]],
            dtype=torch.long, device=device,
        )
        parent_logits = logits[rows, depths]
    # Both tensors originate from softmax and are already finite normalized
    # distributions.  Re-normalising every [31, 128K] matrix added two full
    # vocabulary scans and temporary tensors per round.
    target = torch.softmax(
        logits_processor(None, parent_logits), dim=-1, dtype=torch.float32
    )
    proposal = tree_info["draft_distributions"].to(
        device=device, dtype=torch.float32
    )
    return target, proposal


def _metadata_node_target_distribution(
        node_index, logits, logits_processor, layout
):
    if logits.ndim == 2:
        selected_logits = logits[node_index][None]
    else:
        selected_logits = logits[
            layout["node_rows_list"][node_index],
            layout["node_depths_list"][node_index],
        ][None]
    return _normalize_probability_distribution(torch.softmax(
        logits_processor(None, selected_logits), dim=-1, dtype=torch.float32
    ))[0]


def _evaluate_posterior3_metadata(
        logits, candidates, logits_processor, tree_info, retrieve_indices
):
    """RRSw using node indices and cached stochastic-tree topology."""
    device = logits.device
    layout = _metadata_tree_layout(tree_info, retrieve_indices, candidates, device)
    target_probs, proposal_probs = _metadata_parent_probabilities(
        logits, candidates, logits_processor, tree_info, layout, device
    )
    target_state = target_probs.clone()
    draft_state = proposal_probs.clone()
    tiny = torch.finfo(target_probs.dtype).tiny
    one = target_probs.new_tensor(1.0)
    zero = target_probs.new_tensor(0.0)

    def verify_from(parent_node):
        row_index = layout["parent_rows"][parent_node]
        current_target = target_state[row_index]
        current_draft = draft_state[row_index]
        for child_node in layout["children_lists"][parent_node]:
            token = layout["node_token_ids"][child_node]
            q = current_draft[token]
            rate = torch.where(
                q > 0,
                torch.minimum(current_target[token] / q.clamp_min(tiny), one),
                zero,
            )
            if torch.rand((), device=device) < rate:
                if layout["parent_rows"][child_node] >= 0:
                    return verify_from(child_node)
                return (
                    torch.tensor(
                        layout["node_rows_list"][child_node],
                        device=candidates.device,
                    ),
                    layout["node_depths_list"][child_node],
                    _metadata_node_target_distribution(
                        child_node, logits, logits_processor, layout
                    ),
                )

            current_target = _normalize_probability_distribution(
                (current_target - current_draft).clamp_min(0),
                target_probs[row_index], fallback_is_normalized=True,
            )
            remaining_draft = current_draft.clone()
            remaining_draft[token] = 0
            current_draft = _normalize_probability_distribution(
                remaining_draft, proposal_probs[row_index],
                fallback_is_normalized=True,
            )
            target_state[row_index] = current_target
            draft_state[row_index] = current_draft

        return (
            torch.tensor(
                layout["node_rows_list"][parent_node], device=candidates.device
            ),
            layout["node_depths_list"][parent_node],
            _normalize_probability_distribution(
                current_target, target_probs[row_index], fallback_is_normalized=True
            ),
        )

    return verify_from(0)


def _evaluate_posterior4_metadata(
        logits, candidates, logits_processor, tree_info, retrieve_indices
):
    """Traversal Verification using compact integer topology and dense rows."""
    device = logits.device
    layout = _metadata_tree_layout(tree_info, retrieve_indices, candidates, device)
    target_probs, proposal_probs = _metadata_parent_probabilities(
        logits, candidates, logits_processor, tree_info, layout, device
    )
    # Keep the dense probability batches read-only and copy only a parent row
    # that is actually rejected. A depth-5 binary tree has 31 parent rows, so
    # eagerly cloning both [parents, vocab] matrices copied roughly 32 MiB per
    # round for Llama 3 even when Traversal accepted immediately.
    target_state = list(target_probs.unbind(0))
    draft_state = list(proposal_probs.unbind(0))
    target_row_owned = [False] * len(target_state)
    draft_row_owned = [False] * len(draft_state)
    node_count = len(layout["parent_list"])
    acceptance_state = [0.0] * node_count
    acceptance_state[0] = 1.0
    postorder = [node for node in layout["postorder"] if node != 0]
    visited_nodes = 0
    rejected_nodes = 0
    refresh_syncs = 0
    refreshed_edges = 0

    def finish(candidate, accept_length, sample_p):
        tree_info["_traversal_stats"] = {
            "visited_nodes": visited_nodes,
            "rejected_nodes": rejected_nodes,
            "refresh_syncs": refresh_syncs,
            "refreshed_edges": refreshed_edges,
            "compiled_residuals": 0,
        }
        return candidate, accept_length, sample_p

    # Gather every edge's p/q values once, then calculate the small tree's
    # prefix probabilities on the CPU.  Pre-draw one independent uniform for
    # every possible post-order visit and fold it into the same host transfer.
    # The previous implementation called ``torch.rand(...).item()`` once per
    # visited node, forcing a CUDA synchronization even when no dirty path had
    # to be refreshed.
    traversal_nodes = layout["traversal_nodes_tensor"]
    traversal_parent_rows = layout["traversal_parent_rows_tensor"]
    traversal_tokens = layout["node_token_ids"][traversal_nodes]
    initial_edge_tensor = torch.stack(
        (
            target_probs[traversal_parent_rows, traversal_tokens],
            proposal_probs[traversal_parent_rows, traversal_tokens],
        ),
        dim=-1,
    )
    initial_edge_value_count = initial_edge_tensor.numel()
    initial_host_values = torch.cat((
        initial_edge_tensor.reshape(-1),
        torch.rand(len(postorder), device=device),
    )).detach().cpu().tolist()
    initial_edge_values = initial_host_values[:initial_edge_value_count]
    uniform_values = initial_host_values[initial_edge_value_count:]
    for offset, node in enumerate(layout["traversal_nodes_list"]):
        p_token = initial_edge_values[2 * offset]
        q_token = initial_edge_values[2 * offset + 1]
        parent = layout["parent_list"][node]
        p_value = float(p_token)
        q_value = float(q_token)
        acceptance_state[node] = (
            min(acceptance_state[parent] * p_value / q_value, 1.0)
            if q_value > 0.0 else 0.0
        )

    # A rejection only marks its statically known subtree dirty.  Rates are
    # refreshed along a root-to-node path immediately before that node is
    # visited.  This avoids eagerly updating every still-active descendant,
    # many of which will never be inspected after an earlier acceptance.
    dirty_nodes = [False] * node_count

    def refresh_path(node, uniform):
        nonlocal refresh_syncs, refreshed_edges
        dirty_path = [
            path_node
            for path_node in layout["node_path_lists"][node][1:]
            if dirty_nodes[path_node]
        ]
        if not dirty_path:
            return uniform
        refresh_syncs += 1
        refreshed_edges += len(dirty_path)

        p_values = []
        q_values = []
        for path_node in dirty_path:
            path_parent = layout["parent_list"][path_node]
            path_parent_row = layout["parent_rows"][path_parent]
            path_token = layout["node_token_ids_list"][path_node]
            p_values.append(target_state[path_parent_row][path_token])
            q_values.append(draft_state[path_parent_row][path_token])
        edge_values = torch.stack(
            (torch.stack(p_values), torch.stack(q_values)), dim=-1
        )
        host_values = edge_values.reshape(-1).detach().cpu().tolist()
        for offset, path_node in enumerate(dirty_path):
            p_value = float(host_values[2 * offset])
            q_value = float(host_values[2 * offset + 1])
            path_parent = layout["parent_list"][path_node]
            acceptance_state[path_node] = (
                min(acceptance_state[path_parent] * p_value / q_value, 1.0)
                if q_value > 0.0 else 0.0
            )
            dirty_nodes[path_node] = False
        return uniform

    def selected_target(node):
        row = layout["parent_rows"][node]
        if row >= 0:
            return target_state[row]
        return _metadata_node_target_distribution(
            node, logits, logits_processor, layout
        )

    for leaf, uniform in zip(postorder, uniform_values):
        visited_nodes += 1
        uniform = refresh_path(leaf, float(uniform))
        parent = layout["parent_list"][leaf]
        parent_row = layout["parent_rows"][parent]
        rejected_token = layout["node_token_ids_list"][leaf]
        rate = acceptance_state[leaf]
        parent_acceptance = acceptance_state[parent]
        if uniform < rate:
            return finish(
                torch.tensor(layout["node_rows_list"][leaf], device=candidates.device),
                layout["node_depths_list"][leaf],
                selected_target(leaf),
            )
        rejected_nodes += 1

        parent_target = target_state[parent_row]
        parent_draft = draft_state[parent_row]
        if target_row_owned[parent_row]:
            positive = parent_target
            positive.mul_(parent_acceptance).sub_(parent_draft).clamp_(min=0)
        else:
            positive = parent_target.mul(parent_acceptance)
            positive.sub_(parent_draft).clamp_(min=0)
            target_row_owned[parent_row] = True
        if not draft_row_owned[parent_row]:
            parent_draft = parent_draft.clone()
            draft_row_owned[parent_row] = True
        parent_draft[rejected_token] = 0
        residual_mass, draft_mass = (
            float(value) for value in torch.stack(
                (positive.sum(), parent_draft.sum())
            ).detach().cpu().tolist()
        )
        if residual_mass > 0.0:
            positive.div_(residual_mass)
            target_state[parent_row] = positive
        else:
            target_state[parent_row] = target_probs[parent_row]
            target_row_owned[parent_row] = False
        if draft_mass > 0.0:
            parent_draft.div_(draft_mass)
            draft_state[parent_row] = parent_draft
        else:
            draft_state[parent_row] = proposal_probs[parent_row]
        if draft_mass <= 0.0:
            draft_row_owned[parent_row] = False
        denominator = residual_mass + 1.0 - parent_acceptance
        acceptance_state[parent] = (
            residual_mass / denominator if denominator > 0.0 else 0.0
        )

        for descendant in layout["descendants_lists"][parent]:
            dirty_nodes[descendant] = True
        # The rejected node is never visited again and should not make a later
        # path appear stale if compatibility metadata contains unusual order.
        dirty_nodes[leaf] = False

    root_row = layout["parent_rows"][0]
    return finish(
        torch.tensor(0, device=candidates.device),
        0,
        target_state[root_row],
    )


def _evaluate_posterior4_metadata_cpu(
        logits, candidates, logits_processor, tree_info, retrieve_indices,
        lazy_state=False,
):
    """Traversal with one bulk D2H copy and CPU-resident sequential state.

    Traversal commonly rejects most of the 63-node tree. Running every
    rejection on CUDA therefore creates dozens of tiny, data-dependent kernel
    launches and synchronizations. This opt-in backend transfers the target
    and draft parent distributions once, executes the sequential control flow
    with NumPy, and copies only the final sampling distribution back.
    """
    device = logits.device
    layout = _metadata_tree_layout(tree_info, retrieve_indices, candidates, device)
    target_probs, proposal_probs = _metadata_parent_probabilities(
        logits, candidates, logits_processor, tree_info, layout, device
    )
    postorder = [node for node in layout["postorder"] if node != 0]
    compiled_residual = (
        _get_traversal_compiled_residual()
        if tree_info.get("traversal_compiled_residual", True)
        else None
    )

    # One contiguous D2H transfer also carries the random uniforms, preserving
    # the CUDA RNG stream used by the GPU backend.
    bulk_transfer_started = time.perf_counter()
    probability_count = target_probs.numel()
    uniforms = torch.rand(len(postorder), device=device)
    packed = tree_info.get("_traversal_cpu_buffer")
    if packed is None:
        packed = torch.cat((
            target_probs.reshape(-1),
            proposal_probs.reshape(-1),
            uniforms,
        )).detach().cpu()
    else:
        packed[:probability_count].copy_(
            target_probs.reshape(-1), non_blocking=True
        )
        proposal_copy_event = tree_info.get(
            "_traversal_proposal_copy_event"
        )
        if proposal_copy_event is None:
            packed[probability_count:2 * probability_count].copy_(
                proposal_probs.reshape(-1), non_blocking=True
            )
        packed[2 * probability_count:].copy_(uniforms, non_blocking=True)
        torch.cuda.current_stream(device).synchronize()
        if proposal_copy_event is not None:
            # This normally returns immediately because the proposal transfer
            # has overlapped target decoding on a separate CUDA stream.
            proposal_copy_event.synchronize()
    target_host = packed[:probability_count].reshape(target_probs.shape).numpy()
    proposal_host = packed[
        probability_count:2 * probability_count
    ].reshape(proposal_probs.shape).numpy()
    uniform_values = packed[2 * probability_count:].tolist()
    bulk_transfer_seconds = time.perf_counter() - bulk_transfer_started
    cpu_algorithm_started = time.perf_counter()

    target_state = list(target_host)
    draft_state = list(proposal_host)
    target_mass = [1.0] * len(target_state)
    draft_mass = [1.0] * len(draft_state)
    target_row_owned = [False] * len(target_state)
    draft_row_owned = [False] * len(draft_state)
    # In lazy mode q only changes by zeroing rejected candidate tokens. Keep
    # those few token ids sparsely instead of copying a full vocabulary row at
    # the first rejection of almost every visited parent.
    rejected_draft_tokens = [[] for _ in draft_state]
    node_count = len(layout["parent_list"])
    acceptance_state = [0.0] * node_count
    acceptance_state[0] = 1.0
    visited_nodes = 0
    rejected_nodes = 0
    refreshed_edges = 0
    compiled_residuals = 0

    for node in layout["traversal_nodes_list"]:
        parent = layout["parent_list"][node]
        parent_row = layout["parent_rows"][parent]
        token = layout["node_token_ids_list"][node]
        p_value = float(target_host[parent_row, token])
        q_value = float(proposal_host[parent_row, token])
        acceptance_state[node] = (
            min(acceptance_state[parent] * p_value / q_value, 1.0)
            if q_value > 0.0 else 0.0
        )

    dirty_nodes = [False] * node_count

    def sample_distribution(node):
        row = layout["parent_rows"][node]
        if row < 0:
            return _metadata_node_target_distribution(
                node, logits, logits_processor, layout
            )
        probability = target_state[row]
        if lazy_state:
            probability = probability / np.float32(target_mass[row])
        probability = np.ascontiguousarray(probability)
        return torch.from_numpy(probability).to(device=device)

    def finish(candidate_node, accept_length, sample_p):
        tree_info["_traversal_stats"] = {
            "visited_nodes": visited_nodes,
            "rejected_nodes": rejected_nodes,
            # All refreshes below are CPU-local and introduce no D2H sync.
            "refresh_syncs": 0,
            "refreshed_edges": refreshed_edges,
            "compiled_residuals": compiled_residuals,
            "cpu_bulk_transfer_seconds": bulk_transfer_seconds,
            "cpu_algorithm_seconds": (
                time.perf_counter() - cpu_algorithm_started
            ),
        }
        return (
            torch.tensor(
                layout["node_rows_list"][candidate_node],
                device=candidates.device,
            ),
            accept_length,
            sample_p,
        )

    for leaf, uniform in zip(postorder, uniform_values):
        visited_nodes += 1
        dirty_path = [
            path_node
            for path_node in layout["node_path_lists"][leaf][1:]
            if dirty_nodes[path_node]
        ]
        refreshed_edges += len(dirty_path)
        for path_node in dirty_path:
            path_parent = layout["parent_list"][path_node]
            path_parent_row = layout["parent_rows"][path_parent]
            path_token = layout["node_token_ids_list"][path_node]
            p_value = float(target_state[path_parent_row][path_token])
            q_value = float(
                proposal_host[path_parent_row, path_token]
                if lazy_state else draft_state[path_parent_row][path_token]
            )
            if lazy_state:
                p_value /= target_mass[path_parent_row]
                if path_token in rejected_draft_tokens[path_parent_row]:
                    q_value = 0.0
                else:
                    q_value /= draft_mass[path_parent_row]
            acceptance_state[path_node] = (
                min(acceptance_state[path_parent] * p_value / q_value, 1.0)
                if q_value > 0.0 else 0.0
            )
            dirty_nodes[path_node] = False

        parent = layout["parent_list"][leaf]
        parent_row = layout["parent_rows"][parent]
        parent_acceptance = acceptance_state[parent]
        if float(uniform) < acceptance_state[leaf]:
            return finish(
                leaf,
                layout["node_depths_list"][leaf],
                sample_distribution(leaf),
            )
        rejected_nodes += 1

        parent_target = target_state[parent_row]
        parent_draft = (
            proposal_host[parent_row]
            if lazy_state else draft_state[parent_row]
        )
        rejected_token = layout["node_token_ids_list"][leaf]
        if lazy_state:
            parent_target_mass = target_mass[parent_row]
            parent_draft_mass = draft_mass[parent_row]
            target_scale = (
                parent_acceptance * parent_draft_mass / parent_target_mass
                if parent_target_mass > 0.0 else 0.0
            )
            removed_tokens = rejected_draft_tokens[parent_row]
            # q is represented as the immutable original row plus a tiny list
            # of zeroed candidate entries. Save the target contribution at
            # those entries before an in-place target update overwrites it.
            restored_target_values = [
                float(parent_target[token]) * target_scale
                for token in removed_tokens
            ]
            positive = (
                parent_target
                if target_row_owned[parent_row]
                else np.empty_like(parent_target)
            )
            if compiled_residual is not None:
                scaled_residual_mass = float(compiled_residual(
                    positive,
                    parent_target,
                    parent_draft,
                    np.float32(target_scale),
                ))
                compiled_residuals += 1
            else:
                np.multiply(
                    parent_target, np.float32(target_scale), out=positive
                )
                np.subtract(positive, parent_draft, out=positive)
                np.maximum(positive, np.float32(0.0), out=positive)
                scaled_residual_mass = float(
                    np.sum(positive, dtype=np.float32)
                )
            target_row_owned[parent_row] = True
            for token, value in zip(removed_tokens, restored_target_values):
                previous_value = float(positive[token])
                positive[token] = np.float32(max(value, 0.0))
                scaled_residual_mass += float(positive[token]) - previous_value
            rejected_draft_mass = float(proposal_host[parent_row, rejected_token])
            removed_tokens.append(rejected_token)
            residual_mass = (
                scaled_residual_mass / parent_draft_mass
                if parent_draft_mass > 0.0 else 0.0
            )
            remaining_draft_mass = max(
                parent_draft_mass - rejected_draft_mass, 0.0
            )
            if residual_mass > 0.0:
                target_state[parent_row] = positive
                target_mass[parent_row] = scaled_residual_mass
            else:
                target_state[parent_row] = target_host[parent_row]
                target_mass[parent_row] = 1.0
                target_row_owned[parent_row] = False
            if remaining_draft_mass > 0.0:
                draft_mass[parent_row] = remaining_draft_mass
            else:
                draft_mass[parent_row] = 1.0
                removed_tokens.clear()
            denominator = residual_mass + 1.0 - parent_acceptance
            acceptance_state[parent] = (
                residual_mass / denominator if denominator > 0.0 else 0.0
            )
            for descendant in layout["descendants_lists"][parent]:
                dirty_nodes[descendant] = True
            dirty_nodes[leaf] = False
            continue

        positive = (
            parent_target
            if target_row_owned[parent_row]
            else np.empty_like(parent_target)
        )
        if compiled_residual is not None:
            residual_mass = float(compiled_residual(
                positive,
                parent_target,
                parent_draft,
                np.float32(parent_acceptance),
            ))
            compiled_residuals += 1
        else:
            np.multiply(
                parent_target, np.float32(parent_acceptance), out=positive
            )
            np.subtract(positive, parent_draft, out=positive)
            np.maximum(positive, np.float32(0.0), out=positive)
            residual_mass = float(np.sum(positive, dtype=np.float32))
        target_row_owned[parent_row] = True
        if not draft_row_owned[parent_row]:
            parent_draft = parent_draft.copy()
            draft_row_owned[parent_row] = True
        parent_draft[rejected_token] = 0.0
        draft_mass = float(np.sum(parent_draft, dtype=np.float32))
        if residual_mass > 0.0:
            positive /= np.float32(residual_mass)
            target_state[parent_row] = positive
        else:
            target_state[parent_row] = target_host[parent_row]
            target_row_owned[parent_row] = False
        if draft_mass > 0.0:
            parent_draft /= np.float32(draft_mass)
            draft_state[parent_row] = parent_draft
        else:
            draft_state[parent_row] = proposal_host[parent_row]
            draft_row_owned[parent_row] = False
        denominator = residual_mass + 1.0 - parent_acceptance
        acceptance_state[parent] = (
            residual_mass / denominator if denominator > 0.0 else 0.0
        )
        for descendant in layout["descendants_lists"][parent]:
            dirty_nodes[descendant] = True
        dir2ty_nodes[leaf] = False

    return finish(0, 0, sample_distribution(0))


@torch.no_grad()
def evaluate_posterior3(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        draft_probs: torch.Tensor,
        candidate_node_indices: torch.Tensor = None,
):
    """Token-level tree verification with RRSw (Algorithm 2 in 2505.12398).

    Unlike :func:`evaluate_posterior2`, this method verifies from the root
    towards a leaf. At each parent it tests siblings in their draft order;
    rejecting a child updates that *parent's* target residual and removes the
    child from its proposal distribution without replacement. An accepted
    child is then verified recursively. ``draft_probs`` uses the same compact
    tree-node representation as Traversal Verification.
    """


    if logits_processor is None:
        raise ValueError("RRSw requires stochastic decoding (temperature > 0).")
    if candidates.ndim != 2:
        raise ValueError("Expected candidates [paths, length].")
    compact_target_logits = logits.ndim == 2
    if not compact_target_logits and (
            logits.ndim != 3 or logits.shape[:2] != candidates.shape
    ):
        raise ValueError(
            "Target logits must be [tree_nodes, vocab] or "
            "[paths, length, vocab]."
        )
    tree_metadata = isinstance(draft_probs, dict)
    compact_draft_probs = tree_metadata or draft_probs.ndim == 2
    if compact_target_logits or compact_draft_probs:
        if candidate_node_indices is None or candidate_node_indices.shape != candidates.shape:
            raise ValueError(
                "Compact target logits or draft_probs require "
                "candidate_node_indices matching candidates."
            )
    elif (
            draft_probs.ndim != 3
            or draft_probs.shape[:2] != candidates.shape
            or draft_probs.shape[-1] != logits.shape[-1]
    ):
        raise ValueError(
            "draft_probs must be [tree_nodes, vocab] or match the candidate "
            "path dimensions and target vocabulary."
        )

    if tree_metadata:
        return _evaluate_posterior3_metadata(
            logits, candidates, logits_processor, draft_probs,
            candidate_node_indices,
        )
    else:
        raise ValueError(
            "draft_probs must be a dict. "

        )




@torch.no_grad()
def evaluate_posterior4(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        draft_probs: torch.Tensor,
        candidate_node_indices: torch.Tensor = None,
):
    """Traversal Verification (Weng et al., arXiv:2505.12398).

    This is a sequence-level, post-order verification replacement for the
    sampling branch of :func:`evaluate_posterior`.  ``candidates`` contains
    root-to-leaf paths (the same layout used by ``retrieve_indices``), with
    ``-1`` used as padding. ``draft_probs`` normally stores the complete draft
    distribution once per tree node, and ``candidate_node_indices`` maps each
    candidate path position back to that tree node.
    Keeping the complete distribution is essential: on rejecting a node,
    Traversal Verification redistributes both the target and draft residuals.

    Args:
        logits: Target logits, shaped ``[tree_nodes, vocab]`` (preferred) or
            ``[num_paths, max_path_len, vocab]`` (legacy).
        candidates: Candidate paths, shaped ``[num_paths, max_path_len]``.
        logits_processor: Sampling processor (temperature, top-p, etc.).  It
            must not be ``None``; greedy decoding should keep using
            :func:`evaluate_posterior`.
        draft_probs: Complete draft distributions, either ``[tree_nodes,
            vocab]`` (preferred) or shaped identically to ``logits`` (legacy).
        candidate_node_indices: ``[num_paths, max_path_len]`` tree-node
            indices used by the preferred compact representation.

    Returns:
        ``(best_candidate, accept_length, sample_p)`` with the same contract
        as :func:`evaluate_posterior`.
    """
    if logits_processor is None:
        raise ValueError("Traversal Verification is defined for stochastic decoding; use evaluate_posterior for greedy decoding.")
    if candidates.ndim != 2:
        raise ValueError("Expected candidates [paths, length].")
    compact_target_logits = logits.ndim == 2
    if not compact_target_logits and (
            logits.ndim != 3 or logits.shape[:2] != candidates.shape
    ):
        raise ValueError(
            "Target logits must be [tree_nodes, vocab] or match candidate "
            "path and sequence dimensions: "
            f"logits={tuple(logits.shape)}, "
            f"candidates={tuple(candidates.shape)}"
        )
    tree_metadata = isinstance(draft_probs, dict)
    compact_draft_probs = tree_metadata or draft_probs.ndim == 2
    if compact_target_logits or compact_draft_probs:
        if candidate_node_indices is None or candidate_node_indices.shape != candidates.shape:
            raise ValueError(
                "Compact target logits or draft_probs require "
                "candidate_node_indices with shape "
                f"{tuple(candidates.shape)}."
            )
    elif (
            draft_probs.ndim != 3
            or draft_probs.shape[:2] != candidates.shape
            or draft_probs.shape[-1] != logits.shape[-1]
    ):
        raise ValueError(
            "draft_probs must be [tree_nodes, vocab] or match the candidate "
            "path dimensions and target vocabulary; "
            f"got draft_probs={tuple(draft_probs.shape)}, logits={tuple(logits.shape)}."
        )

    if tree_metadata:
        traversal_backend = draft_probs.get("traversal_backend", "gpu")
        if traversal_backend in {"cpu", "cpu_lazy"}:
            return _evaluate_posterior4_metadata_cpu(
                logits, candidates, logits_processor, draft_probs,
                candidate_node_indices,
                lazy_state=traversal_backend == "cpu_lazy",
            )
        return _evaluate_posterior4_metadata(
            logits, candidates, logits_processor, draft_probs,
            candidate_node_indices,
        )
    else:
        raise ValueError(
            "draft_probs must be a dict. "

        )



def evaluate_posterior5(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        tree_info,
        retrieve_indices: torch.Tensor,
):
    """Verify a Greedy OT draft tree independently at every reached parent.

    Hu et al.'s Greedy construction keeps Top-(m-1) draft tokens fixed and
    samples the final child ``z`` from the normalized residual draft
    distribution ``r``. Its optimal verifier is the maximal coupling between
    the target distribution ``p`` and ``r``:

    * emit ``z`` with probability ``min(1, p[z] / r[z])``;
    * otherwise sample from ``relu(p - r)``.

    A residual sample that equals one of the deterministic Top-(m-1) children
    is still an accepted draft and verification continues below that child.
    Only residual mass outside the complete sibling group terminates the
    speculative round. Repeating this local coupling from the root down th e
    selected path implements the vertically-myopic Greedy baseline used by
    UniVer, without UniVer's prefix-probability propagation.
    """
    if tree_info is None:
        raise ValueError("evaluate_posterior5 requires Greedy draft-tree metadata")
    if candidates.ndim != 2:
        raise ValueError("Expected candidates [paths, length].")
    if retrieve_indices.shape != candidates.shape:
        raise ValueError("retrieve_indices must match the candidate path layout")
    if tree_info.get("sampling_method", "greedy") != "greedy":
        raise ValueError("Greedy OT verification requires sampling_method='greedy'")

    device = logits.device
    layout = _metadata_tree_layout(
        tree_info, retrieve_indices, candidates, device
    )
    node_count = len(layout["parent_list"])
    if logits.ndim == 2:
        if logits.shape[0] != node_count:
            raise ValueError(
                "Compact target logits must contain one row per tree node: "
                f"got {logits.shape[0]} rows for {node_count} nodes."
            )
    elif logits.ndim != 3 or logits.shape[:2] != candidates.shape:
        raise ValueError(
            "Greedy target logits must be [tree_nodes, vocab] or "
            "[paths, length, vocab]."
        )

    residual_distributions = tree_info["residual_distributions"].to(
        device=device, dtype=torch.float32
    )
    expanded_parents = tree_info["expanded_parent_indices"].to(device)
    if residual_distributions.ndim != 2:
        raise ValueError("residual_distributions must be [expanded_parents, vocab]")
    if residual_distributions.shape[0] != expanded_parents.numel():
        raise ValueError("one residual distribution is required per expanded parent")
    if residual_distributions.shape[-1] != logits.shape[-1]:
        raise ValueError("draft residual and target vocabulary sizes do not match")

    child_groups = tree_info["child_groups"]
    if len(child_groups) != expanded_parents.numel():
        raise ValueError("one child group is required per expanded parent")
    node_is_random = tree_info["node_is_random"].to(device).reshape(-1)
    if node_is_random.numel() != node_count:
        raise ValueError("node_is_random must contain one flag per tree node")

    parent_rows = layout["parent_rows"]
    node_rows = layout["node_rows_list"]
    node_depths = layout["node_depths_list"]
    node_token_ids = layout["node_token_ids"]
    epsilon = torch.finfo(torch.float32).eps

    def target_distribution(node_index):
        if logits.ndim == 2:
            selected_logits = logits[node_index][None]
        else:
            selected_logits = logits[
                node_rows[node_index], node_depths[node_index]
            ][None]
        if logits_processor is None:
            probabilities = torch.zeros_like(selected_logits, dtype=torch.float32)
            probabilities.scatter_(
                1, torch.argmax(selected_logits, dim=-1, keepdim=True), 1.0
            )
            return probabilities[0]
        return torch.softmax(
            logits_processor(None, selected_logits).float(), dim=-1
        )[0]

    current_node = 0
    while True:
        parent_row = parent_rows[current_node]
        if parent_row < 0:
            # The selected draft is a leaf. Its target logits provide the
            # ordinary bonus-token distribution for update_inference_inputs.
            return (
                torch.tensor(node_rows[current_node], device=candidates.device),
                node_depths[current_node],
                target_distribution(current_node),
            )

        children = child_groups[parent_row].to(device)
        if children.numel() == 0:
            raise ValueError("an expanded Greedy parent has no children")
        random_children = children[node_is_random[children]]
        if random_children.numel() != 1:
            raise ValueError(
                "each Greedy sibling group must contain exactly one residual sample"
            )
        random_child = random_children[0]
        random_token = node_token_ids[random_child]
        target = target_distribution(current_node)
        residual = residual_distributions[parent_row]
        residual_mass = residual[random_token]
        if bool((residual_mass <= 0).item()):
            raise ValueError("the sampled Greedy child has zero residual probability")

        acceptance_probability = torch.clamp(
            target[random_token] / residual_mass.clamp_min(epsilon), max=1.0
        )
        if bool((torch.rand((), device=device) < acceptance_probability).item()):
            current_node = int(random_child.item())
            continue

        # This full-vocabulary subtraction is deliberately lazy: the common
        # direct-accept path above needs only p[z] and r[z].
        correction = torch.relu(target - residual)
        correction_mass = correction.sum()
        if bool((correction_mass <= epsilon).item()):
            raise RuntimeError(
                "Greedy maximal coupling rejected with no correction mass"
            )

        child_tokens = node_token_ids[children]
        child_masses = correction[child_tokens]
        fallback = correction.clone()
        fallback[child_tokens] = 0
        fallback_mass = fallback.sum()
        decision_masses = torch.cat((child_masses, fallback_mass[None]))
        if bool((decision_masses.sum() <= epsilon).item()):
            raise RuntimeError("Greedy correction has no selectable output mass")
        decision = int(torch.multinomial(decision_masses, 1).item())

        if decision < children.numel():
            # In exact arithmetic the sampled residual child has zero mass in
            # this branch; accepting any candidate here is primarily how the
            # deterministic Top-(m-1) tokens obtain their target mass.
            current_node = int(children[decision].item())
            continue

        if bool((fallback_mass <= epsilon).item()):
            raise RuntimeError("Greedy selected an empty correction fallback")
        sample_p = fallback / fallback_mass
        return (
            torch.tensor(node_rows[current_node], device=candidates.device),
            node_depths[current_node],
            sample_p,
        )


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p,
        verify_method="default",
        return_tree_info=False,
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Only one retrieval path survives verification.  Expanding hidden states
    # for every root-to-leaf path duplicated shared prefixes (32 x 6 rows for
    # the paper tree) before immediately discarding all but one path.
    accepted_node_indices = retrieve_indices[
        best_candidate, : accept_length + 1
    ].to(hidden_state_new.device)
    accept_hidden_state_new = hidden_state_new[:, accepted_node_indices]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_info = _generate_draft_tree(
        model,
        accept_hidden_state_new,
        input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
        logits_processor=logits_processor,
        verify_method=verify_method,
    )


    new_token += accept_length + 1

    result = (
        input_ids,
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
        new_token,
        None,
        token,
    )
    if return_tree_info:
        return (*result, tree_info)
    return result


def _normalize_probability_distribution(probabilities, fallback=None, fallback_is_normalized=False):
    """Return a finite float32 distribution, with a safe numerical fallback.

    Traversal Verification repeatedly subtracts target and proposal
    distributions.  In fp16, a mathematically positive residual can underflow
    to an all-zero tensor, which later makes ``torch.multinomial`` fail on a
    long benchmark run.  This helper keeps the normal path unchanged while
    making that numerical corner case explicit and recoverable.
    """
    probabilities = torch.nan_to_num(
        probabilities.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0)
    mass = probabilities.sum(dim=-1, keepdim=True)
    normalized = probabilities / mass.clamp_min(torch.finfo(probabilities.dtype).tiny)
    if fallback is None:
        fallback = torch.full_like(probabilities, 1.0 / probabilities.shape[-1])
    elif not fallback_is_normalized:
        fallback = _normalize_probability_distribution(fallback)
    return torch.where(mass > 0, normalized, fallback)



if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
