"""Probability-aware draft-tree construction utilities.

The UniVer paper assumes that every expanded parent has ``m`` children: the
top ``m - 1`` tokens under the draft distribution and one token sampled from
the remaining probability mass.  The stock EAGLE generator instead builds a
large Top-K candidate pool and prunes individual nodes globally.  Pruning an
individual child would invalidate UniVer's per-parent sampling distribution,
so this module expands and retains complete sibling groups.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch


def greedy_residual_sample(
        probabilities: torch.Tensor,
        candidate_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select Top-(m-1) tokens and sample the final token from the residual.

    Args:
        probabilities: One-dimensional normalized draft distribution.
        candidate_count: Requested number of distinct children ``m``.

    Returns:
        candidate_ids: Top-(m-1) token ids followed by the sampled token id.
        residual: The normalized distribution used to sample the last token.
        is_random: Boolean flags aligned with ``candidate_ids``; only the last
            entry is true.

    Hard logits filters (for example top-p) can leave fewer than
    ``candidate_count`` tokens with non-zero mass.  In that case the group is
    reduced to the available support rather than inventing zero-probability
    draft tokens.
    """
    if probabilities.dim() != 1:
        raise ValueError("probabilities must be a one-dimensional tensor")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if torch.any(probabilities < 0):
        raise ValueError("probabilities must be non-negative")

    total = probabilities.sum()
    if total <= 0:
        raise ValueError("probabilities must contain positive mass")
    probabilities = probabilities / total

    support = torch.nonzero(probabilities > 0, as_tuple=True)[0]
    actual_count = min(candidate_count, support.numel())
    if actual_count == 0:
        raise ValueError("the draft distribution has empty support")

    deterministic_count = actual_count - 1
    if deterministic_count:
        top_ids = torch.topk(probabilities, deterministic_count, dim=-1).indices
    else:
        top_ids = torch.empty(0, dtype=torch.long, device=probabilities.device)

    residual = probabilities.clone()
    residual[top_ids] = 0
    residual = residual / residual.sum()
    sampled_id = torch.multinomial(residual, 1)
    candidate_ids = torch.cat((top_ids, sampled_id), dim=0)
    is_random = torch.zeros(actual_count, dtype=torch.bool, device=probabilities.device)
    is_random[-1] = True
    return candidate_ids, residual, is_random


def sample_without_replacement(
        probabilities: torch.Tensor,
        candidate_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-sample the ordered candidates used by RRSw.

    Conceptually, RRSw performs the without-replacement update during
    verification: after candidate ``u_k`` is rejected, its mass is removed
    from the mutable draft distribution and the next candidate is sampled
    from the renormalized residual.  A speculative tree must be materialized
    before the target-model forward pass, so we draw that same ordered list
    here in advance.  This is distributionally equivalent to sampling each
    next sibling on demand after the preceding rejection; it is not a claim
    that Traversal Verification itself changes the tree-construction stage.

    Returns the sampled ids, each selected token's probability under the
    proposal distribution used at its draw, and all-true stochastic flags.
    """
    if probabilities.dim() != 1:
        raise ValueError("probabilities must be a one-dimensional tensor")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    probabilities = torch.nan_to_num(
        probabilities.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0)
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("probabilities must contain positive mass")
    probabilities = probabilities / total
    actual_count = min(
        candidate_count,
        int(torch.count_nonzero(probabilities > 0).item()),
    )

    # PyTorch performs weighted sampling without replacement sequentially
    # internally.  Thus this single call has the same ordered-sample law as
    # repeatedly drawing one token and zeroing its mass ourselves.
    sampled_ids = torch.multinomial(
        probabilities,
        num_samples=actual_count,
        replacement=False,
    )

    # Keep the conditional proposal probability at each draw as additional
    # node metadata.  For the ordered samples (u_1, ..., u_k), it is
    # q(u_k) / (1 - sum_{j<k} q(u_j)).  This does not replace the original
    # full q: generate_univer_tree stores that separately in
    # metadata["draft_distributions"] for verification-time residual updates.
    sampled_mass = probabilities[sampled_ids]
    removed_mass_before = torch.cat(
        (sampled_mass.new_zeros(1), sampled_mass.cumsum(dim=0)[:-1]),
        dim=0,
    )
    remaining_mass = (1 - removed_mass_before).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    selected_probabilities = (sampled_mass / remaining_mass).clamp(0, 1)

    return (
        sampled_ids,
        selected_probabilities,
        torch.ones(actual_count, dtype=torch.bool, device=probabilities.device),
    )


def _batched_binary_samples(
        probabilities: torch.Tensor,
        sampling_method: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw two children for every parent with one set of CUDA launches.

    The benchmark tree is a full binary tree.  The scalar helpers above are
    intentionally defensive and remain the compatibility path for filtered or
    partial sibling groups.  Once probabilities have come from a float32
    softmax, however, validating and sampling every row independently creates
    dozens of device synchronizations and small kernel launches per round.

    Returns candidate ids, the distribution retained for verification, the
    selected proposal probability at each draw, and stochastic-node flags.
    """
    if probabilities.ndim != 2:
        raise ValueError("batched binary sampling expects [parents, vocab]")

    if sampling_method == "greedy":
        top_ids = probabilities.argmax(dim=-1, keepdim=True)
        residual = probabilities.clone()
        residual.scatter_(1, top_ids, 0)
        residual.div_(residual.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(residual.dtype).tiny
        ))
        sampled_ids = torch.multinomial(residual, 1)
        candidate_ids = torch.cat((top_ids, sampled_ids), dim=1)
        selected_probabilities = residual.gather(1, candidate_ids)
        random_flags = torch.zeros_like(candidate_ids, dtype=torch.bool)
        random_flags[:, -1] = True
        return candidate_ids, residual, selected_probabilities, random_flags

    if sampling_method == "without_replacement":
        candidate_ids = torch.multinomial(
            probabilities, num_samples=2, replacement=False
        )
        sampled_mass = probabilities.gather(1, candidate_ids)
        removed_before = torch.cat(
            (sampled_mass.new_zeros(sampled_mass.shape[0], 1), sampled_mass[:, :1]),
            dim=1,
        )
        selected_probabilities = sampled_mass / (1 - removed_before).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        return (
            candidate_ids,
            probabilities,
            selected_probabilities.clamp_(0, 1),
            torch.ones_like(candidate_ids, dtype=torch.bool),
        )

    raise ValueError(f"Unsupported sampling_method={sampling_method!r}")


def _target_distribution(
        distribution: torch.Tensor,
        target_ids: torch.Tensor,
        target_vocab_size: int,
        identity_mapping: bool = False,
) -> torch.Tensor:
    """Map an EAGLE draft-vocabulary distribution to target token ids."""
    if identity_mapping:
        return distribution
    mapped = distribution.new_zeros(target_vocab_size)
    mapped.scatter_add_(0, target_ids, distribution)
    return mapped


def _build_tree_buffers(
        parent_indices: List[int],
        depths: List[int],
        device: torch.device,
        sort_paths: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build EAGLE tree attention and root-to-leaf retrieval buffers."""
    node_count = len(parent_indices) - 1
    # This is a tiny topology problem.  Building it on CUDA launched one
    # logical_or kernel per node (62 launches for the paper's tree).  Construct
    # it on the host and transfer the finished mask once instead.
    tree_mask = torch.eye(node_count + 1, dtype=torch.bool)
    tree_mask[:, 0] = True
    for node_index in range(1, node_count + 1):
        tree_mask[node_index].logical_or_(tree_mask[parent_indices[node_index]])

    nonleaf = set(parent_indices[1:])
    leaf_indices = [i for i in range(node_count + 1) if i not in nonleaf]
    max_depth = max(depths)
    retrieve_indices: List[List[int]] = []
    for leaf in leaf_indices:
        path = [-1] * (max_depth + 1)
        current = leaf
        while current >= 0:
            path[depths[current]] = current
            if current == 0:
                break
            current = parent_indices[current]
        retrieve_indices.append(path)

    if sort_paths:
        sentinel = node_count + 5
        retrieve_indices.sort(
            key=lambda path: [index if index >= 0 else sentinel for index in path]
        )

    return (
        torch.tensor(retrieve_indices, dtype=torch.long),
        tree_mask.to(device=device, dtype=torch.float32)[None, None],
        torch.tensor(depths, dtype=torch.long, device=device),
    )


@torch.no_grad()
def generate_univer_tree(
        model,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        logits_processor,
        project_logits: Callable[[torch.Tensor], torch.Tensor],
        map_to_target: Callable[[torch.Tensor], torch.Tensor],
        target_vocab_size: int,
        sampling_method: str = "greedy",
        store_draft_distributions: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Generate a group-preserving stochastic draft tree.

    ``sampling_method='greedy'`` implements UniVer's Top-(m-1)+one-residual
    law. ``sampling_method='without_replacement'`` pre-materializes the
    ordered sibling samples that RRSw would otherwise draw one by one after
    rejections.  Traversal Verification accepts a valid sampling tree built
    with or without replacement; this option matches the RRSw/EAGLE setup used
    by Weng et al.'s experiments. The returned metadata contains the full
    parent distributions and node-aligned selected probabilities.
    """
    input_ids = input_ids.to(hidden_states.device)
    total_tokens = model.total_tokens
    branching = model.top_k
    sample_token = input_ids[:, -1]

    if total_tokens < 1:
        raise ValueError("UniVer tree generation requires at least one draft token")
    if branching < 1:
        raise ValueError("top_k must be positive for UniVer tree generation")

    prefix_input_ids = input_ids[:, 1:].to(hidden_states.device)
    prefix_length = prefix_input_ids.shape[1]
    model.reset()

    if hasattr(model, "stable_kv") and model.stable_kv is not None:
        kv_len = model.stable_kv[0][0].shape[2]
        out_hidden, past_key_values = model(
            hidden_states,
            input_ids=prefix_input_ids[:, kv_len:],
            past_key_values=model.stable_kv,
            use_cache=True,
        )
    else:
        out_hidden, past_key_values = model(
            hidden_states,
            input_ids=prefix_input_ids,
            use_cache=True,
        )
    model.stable_kv = past_key_values
    root_hidden = out_hidden[:, -1]

    node_target_ids: List[torch.Tensor] = []
    node_draft_probabilities: List[torch.Tensor] = []
    node_residual_probabilities: List[torch.Tensor] = []
    node_is_random: List[torch.Tensor] = []
    parent_indices: List[int] = [-1]
    depths: List[int] = [0]

    expanded_parent_indices: List[int] = []
    parent_draft_distributions: List[torch.Tensor] = []
    parent_residual_distributions: List[torch.Tensor] = []
    child_groups: List[torch.Tensor] = []
    all_target_ids: Optional[torch.Tensor] = None
    identity_vocab_mapping: Optional[bool] = None

    def draft_distribution(hidden: torch.Tensor) -> torch.Tensor:
        logits = project_logits(hidden)
        if logits_processor is not None:
            logits = logits_processor(None, logits)
        return torch.softmax(logits.float(), dim=-1)

    def add_group(
            parent_node_index: int,
            parent_depth: int,
            q: torch.Tensor,
            requested_count: int,
            prepared: Optional[Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prepared is not None:
            candidate_ids, residual, selected_proposal_probabilities, random_flags = prepared
        elif sampling_method == "greedy":
            candidate_ids, residual, random_flags = greedy_residual_sample(
                q, requested_count
            )
            selected_proposal_probabilities = residual[candidate_ids]
        elif sampling_method == "without_replacement":
            (
                candidate_ids,
                selected_proposal_probabilities,
                random_flags,
            ) = sample_without_replacement(q, requested_count)
            residual = q
        else:
            raise ValueError(
                f"Unsupported sampling_method={sampling_method!r}; expected "
                "'greedy' or 'without_replacement'."
            )
        target_ids = map_to_target(candidate_ids)
        first_node_index = len(parent_indices)
        child_indices = torch.arange(
            first_node_index,
            first_node_index + candidate_ids.numel(),
            dtype=torch.long,
            device=candidate_ids.device,
        )

        node_target_ids.extend(target_ids.unbind())
        node_draft_probabilities.extend(q[candidate_ids].unbind())
        node_residual_probabilities.extend(selected_proposal_probabilities.unbind())
        node_is_random.extend(random_flags.unbind())
        parent_indices.extend([parent_node_index] * candidate_ids.numel())
        depths.extend([parent_depth + 1] * candidate_ids.numel())

        # Mapping an entire 128K-token vocabulary used to be repeated for
        # every expanded parent.  The mapping is invariant within a tree, so
        # build it once and reuse it for all parent distributions.
        nonlocal all_target_ids, identity_vocab_mapping
        if all_target_ids is None:
            all_draft_ids = torch.arange(q.numel(), device=q.device)
            all_target_ids = map_to_target(all_draft_ids)
            identity_vocab_mapping = (
                q.numel() == target_vocab_size
                and torch.equal(all_target_ids, all_draft_ids)
            )
        expanded_parent_indices.append(parent_node_index)
        if store_draft_distributions:
            parent_draft_distributions.append(
                _target_distribution(
                    q, all_target_ids, target_vocab_size, bool(identity_vocab_mapping)
                )
            )
        parent_residual_distributions.append(
            _target_distribution(
                residual,
                all_target_ids,
                target_vocab_size,
                bool(identity_vocab_mapping),
            )
        )
        child_groups.append(child_indices)
        return candidate_ids, target_ids, child_indices

    root_q = draft_distribution(root_hidden)[0]
    root_count = min(branching, total_tokens)
    root_prepared = None
    root_unfiltered = (
        logits_processor is None
        or (
            hasattr(logits_processor, "__len__")
            and len(logits_processor) == 0
        )
    )
    if root_count == 2 and branching == 2 and root_unfiltered:
        root_batch = _batched_binary_samples(root_q[None], sampling_method)
        root_prepared = tuple(value[0] for value in root_batch)
    current_draft_ids, current_target_ids, current_node_indices = add_group(
        parent_node_index=0,
        parent_depth=0,
        q=root_q,
        requested_count=root_count,
        prepared=root_prepared,
    )
    generated_tokens = current_draft_ids.numel()
    remaining_tokens = total_tokens - generated_tokens
    current_hidden = root_hidden[:, None].repeat(1, generated_tokens, 1)
    current_input_ids = current_target_ids[None]
    current_scores = torch.log(root_q[current_draft_ids].clamp_min(torch.finfo(root_q.dtype).tiny))
    current_tree_mask = torch.eye(
        generated_tokens,
        dtype=root_hidden.dtype,
        device=root_hidden.device,
    )[None, None]
    generation_depth = 1

    while remaining_tokens > 0 and generation_depth <= model.depth:
        model.tree_mask = current_tree_mask
        position_ids = torch.full(
            (current_input_ids.shape[1],),
            prefix_length + generation_depth - 1,
            dtype=torch.long,
            device=current_input_ids.device,
        )
        out_hidden, past_key_values = model(
            current_hidden,
            input_ids=current_input_ids,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )

        layer_capacity = min(remaining_tokens, current_input_ids.shape[1] * branching)
        full_groups, partial_group = divmod(layer_capacity, branching)
        group_sizes = [branching] * full_groups
        if partial_group:
            group_sizes.append(partial_group)
        if not group_sizes:
            break

        parent_count = len(group_sizes)
        selected_local = torch.topk(current_scores, parent_count, dim=-1).indices
        next_draft_ids: List[torch.Tensor] = []
        next_target_ids: List[torch.Tensor] = []
        next_node_indices: List[torch.Tensor] = []
        next_scores: List[torch.Tensor] = []
        repeated_parent_local: List[torch.Tensor] = []

        layer_q = draft_distribution(out_hidden[0, selected_local])
        prepared_rows = None
        unfiltered_distribution = (
            logits_processor is None
            or (
                hasattr(logits_processor, "__len__")
                and len(logits_processor) == 0
            )
        )
        if (
                branching == 2
                and all(group_size == 2 for group_size in group_sizes)
                and (
                    unfiltered_distribution
                    or torch.count_nonzero(layer_q, dim=1).min().item() >= 2
                )
        ):
            (
                batch_candidate_ids,
                batch_residuals,
                batch_selected_probabilities,
                batch_random_flags,
            ) = _batched_binary_samples(layer_q, sampling_method)
            prepared_rows = [
                (
                    batch_candidate_ids[row],
                    batch_residuals[row],
                    batch_selected_probabilities[row],
                    batch_random_flags[row],
                )
                for row in range(parent_count)
            ]

        # One small transfer per depth replaces one .item() synchronization per
        # expanded parent.
        selected_pairs = torch.stack(
            (selected_local, current_node_indices[selected_local]), dim=1
        ).detach().cpu().tolist()
        for row, ((local_parent, parent_node), group_size) in enumerate(
                zip(selected_pairs, group_sizes)
        ):
            draft_ids, target_ids, child_indices = add_group(
                parent_node_index=parent_node,
                parent_depth=generation_depth,
                q=layer_q[row],
                requested_count=group_size,
                prepared=None if prepared_rows is None else prepared_rows[row],
            )
            actual_count = draft_ids.numel()
            if actual_count == 0:
                continue
            next_draft_ids.append(draft_ids)
            next_target_ids.append(target_ids)
            next_node_indices.append(child_indices)
            next_scores.append(
                current_scores[local_parent]
                + torch.log(layer_q[row, draft_ids].clamp_min(torch.finfo(layer_q.dtype).tiny))
            )
            repeated_parent_local.append(
                torch.full(
                    (actual_count,), local_parent,
                    dtype=torch.long, device=layer_q.device,
                )
            )

        if not next_draft_ids:
            break
        flat_draft_ids = torch.cat(next_draft_ids)
        flat_target_ids = torch.cat(next_target_ids)
        flat_node_indices = torch.cat(next_node_indices)
        flat_scores = torch.cat(next_scores)
        parent_selector = torch.cat(repeated_parent_local)
        new_count = flat_draft_ids.numel()

        selected_hidden = out_hidden[:, parent_selector]
        old_paths = current_tree_mask[:, :, parent_selector]
        new_identity = torch.eye(
            new_count,
            dtype=old_paths.dtype,
            device=old_paths.device,
        )[None, None]
        current_tree_mask = torch.cat((old_paths, new_identity), dim=3)
        current_hidden = selected_hidden
        current_input_ids = flat_target_ids[None]
        current_draft_ids = flat_draft_ids
        current_node_indices = flat_node_indices
        current_scores = flat_scores
        generated_tokens += new_count
        remaining_tokens = total_tokens - generated_tokens
        generation_depth += 1

    draft_token_tensor = torch.stack(node_target_ids)
    draft_tokens = torch.cat((sample_token, draft_token_tensor), dim=0)[None]
    retrieve_indices, tree_mask, tree_position_ids = _build_tree_buffers(
        parent_indices,
        depths,
        hidden_states.device,
        sort_paths=logits_processor is not None,
    )

    # Cache the fixed topology needed by verification.  Computing these from
    # retrieve_indices inside every verification round previously required N
    # full tensor scans plus many device-to-host synchronizations.
    node_rows_list = [-1] * len(parent_indices)
    for row_index, path in enumerate(retrieve_indices.tolist()):
        for node_index in path:
            if node_index >= 0 and node_rows_list[node_index] < 0:
                node_rows_list[node_index] = row_index
    if any(row < 0 for row in node_rows_list):
        raise RuntimeError("generated tree contains a node without a retrieval path")

    children_lists: List[List[int]] = [[] for _ in parent_indices]
    for node_index, parent_index in enumerate(parent_indices[1:], start=1):
        children_lists[parent_index].append(node_index)
    postorder_list: List[int] = []

    def append_postorder(node_index: int) -> None:
        for child_index in children_lists[node_index]:
            append_postorder(child_index)
        postorder_list.append(node_index)

    append_postorder(0)
    # Traversal verification mutates a parent's residual distribution whenever
    # one of its children is rejected.  Cache the affected topology here so the
    # verifier does not rediscover descendants by scanning the active tree on
    # every rejection.  Root-to-node paths support lazy, top-down refreshes of
    # only the nodes that are actually visited later in post-order.
    node_path_lists: List[List[int]] = [[0]]
    descendants_lists: List[List[int]] = [[] for _ in parent_indices]
    for node_index in range(1, len(parent_indices)):
        parent_index = parent_indices[node_index]
        node_path = node_path_lists[parent_index] + [node_index]
        node_path_lists.append(node_path)
        for ancestor_index in node_path[:-1]:
            descendants_lists[ancestor_index].append(node_index)

    parent_distribution_rows = [-1] * len(parent_indices)
    for row_index, parent_index in enumerate(expanded_parent_indices):
        parent_distribution_rows[parent_index] = row_index
    expanded_depth_groups: List[List[int]] = []
    expanded_depth_values = sorted(
        {depths[parent_index] for parent_index in expanded_parent_indices}
    )
    for expanded_depth in expanded_depth_values:
        expanded_depth_groups.append(
            [
                row_index
                for row_index, parent_index in enumerate(expanded_parent_indices)
                if depths[parent_index] == expanded_depth
            ]
        )

    traversal_depth_groups: List[List[int]] = []
    traversal_parent_groups: List[List[int]] = []
    traversal_parent_row_groups: List[List[int]] = []
    for node_depth in sorted(set(depths[1:])):
        nodes_at_depth = [
            node_index
            for node_index in range(1, len(parent_indices))
            if depths[node_index] == node_depth
        ]
        parents_at_depth = [parent_indices[node_index] for node_index in nodes_at_depth]
        traversal_depth_groups.append(nodes_at_depth)
        traversal_parent_groups.append(parents_at_depth)
        traversal_parent_row_groups.append(
            [parent_distribution_rows[parent_index] for parent_index in parents_at_depth]
        )
    traversal_nodes_list = [
        node_index for group in traversal_depth_groups for node_index in group
    ]
    traversal_parent_rows_list = [
        parent_distribution_rows[parent_indices[node_index]]
        for node_index in traversal_nodes_list
    ]

    root_one = root_q.new_ones(1)
    root_zero = root_q.new_zeros(1)
    metadata: Dict[str, object] = {
        "sampling_method": sampling_method,
        "node_token_ids": draft_tokens[0],
        "node_draft_probabilities": torch.cat(
            (root_one, torch.stack(node_draft_probabilities))
        )[None],
        "node_residual_probabilities": torch.cat(
            (root_zero, torch.stack(node_residual_probabilities))
        )[None],
        "node_is_random": torch.cat(
            (
                torch.zeros(1, dtype=torch.bool, device=root_q.device),
                torch.stack(node_is_random),
            )
        )[None],
        "parent_indices": torch.tensor(parent_indices, dtype=torch.long, device=root_q.device),
        "expanded_parent_indices": torch.tensor(
            expanded_parent_indices, dtype=torch.long, device=root_q.device
        ),
        "draft_distributions": (
            torch.stack(parent_draft_distributions)
            if parent_draft_distributions else None
        ),
        "residual_distributions": torch.stack(parent_residual_distributions),
        "child_groups": child_groups,
        "node_rows": torch.tensor(
            node_rows_list, dtype=torch.long, device=root_q.device
        ),
        "node_depths": torch.tensor(depths, dtype=torch.long, device=root_q.device),
        "postorder": torch.tensor(
            postorder_list, dtype=torch.long, device=root_q.device
        ),
        "parent_distribution_rows": torch.tensor(
            parent_distribution_rows, dtype=torch.long, device=root_q.device
        ),
        "expanded_depth_groups": expanded_depth_groups,
        "expanded_depth_group_tensors": [
            torch.tensor(group, dtype=torch.long, device=root_q.device)
            for group in expanded_depth_groups
        ],
        "traversal_depth_groups": traversal_depth_groups,
        "traversal_depth_group_tensors": [
            torch.tensor(group, dtype=torch.long, device=root_q.device)
            for group in traversal_depth_groups
        ],
        "traversal_parent_group_tensors": [
            torch.tensor(group, dtype=torch.long, device=root_q.device)
            for group in traversal_parent_groups
        ],
        "traversal_parent_row_group_tensors": [
            torch.tensor(group, dtype=torch.long, device=root_q.device)
            for group in traversal_parent_row_groups
        ],
        "traversal_nodes_tensor": torch.tensor(
            traversal_nodes_list, dtype=torch.long, device=root_q.device
        ),
        "traversal_parent_rows_tensor": torch.tensor(
            traversal_parent_rows_list, dtype=torch.long, device=root_q.device
        ),
        # Host topology is consumed by the inherently branchy RRSw/Traversal
        # decision phases.  Keeping it here avoids copying candidates back to
        # the CPU and rebuilding prefix dictionaries every verification round.
        "parent_indices_list": parent_indices,
        "children_lists": children_lists,
        "node_rows_list": node_rows_list,
        "node_depths_list": depths,
        "node_token_ids_list": draft_tokens[0].detach().cpu().tolist(),
        "postorder_list": postorder_list,
        "node_path_lists": node_path_lists,
        "descendants_lists": descendants_lists,
        "parent_distribution_rows_list": parent_distribution_rows,
        "expanded_parent_indices_list": expanded_parent_indices,
        "traversal_nodes_list": traversal_nodes_list,
    }
    if child_groups and len({group.numel() for group in child_groups}) == 1:
        metadata["child_index_matrix"] = torch.stack(child_groups)
    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, metadata
