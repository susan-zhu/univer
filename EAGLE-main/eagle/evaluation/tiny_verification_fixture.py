"""Small, reproducible inputs for the four probability-aware verifiers.

The fixture is a complete binary tree with depth 5, 63 total nodes and 32
leaves.  Its compact target logits have a vocabulary of only 10 tokens.

Run a smoke test with::

    python -m eagle.evaluation.tiny_verification_fixture

Or import the inputs with::

    from eagle.evaluation.tiny_verification_fixture import build_fixture
    inputs = build_fixture("greedy", seed=0)
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
import sys
import os

# 将目录添加到 Python 模块搜索路径
eagle_path = r"D:\anconda_workspace\ai_infra_vllm_test\uniVer\EAGLE-main"
sys.path.insert(0, eagle_path)  # 插到最前面，优先级最高
class IdentityLogitsProcessor:
    """Sampling processor equivalent to temperature=1 and no truncation."""

    def __call__(self, input_ids, scores):
        return scores


def _full_binary_topology(depth: int):
    node_count = 2 ** (depth + 1) - 1
    first_leaf = 2 ** depth - 1
    parent_indices = [-1] + [(node - 1) // 2 for node in range(1, node_count)]
    node_depths = [(node + 1).bit_length() - 1 for node in range(node_count)]
    children_lists: List[List[int]] = [[] for _ in range(node_count)]
    for node in range(1, node_count):
        children_lists[parent_indices[node]].append(node)

    paths = []
    for leaf in range(first_leaf, node_count):
        path = []
        node = leaf
        while node >= 0:
            path.append(node)
            node = parent_indices[node]
        paths.append(list(reversed(path)))
    retrieve_indices = torch.tensor(paths, dtype=torch.long)

    node_rows = [-1] * node_count
    for row, path in enumerate(paths):
        for node in path:
            if node_rows[node] < 0:
                node_rows[node] = row

    postorder = []

    def visit(node):
        for child in children_lists[node]:
            visit(child)
        postorder.append(node)

    visit(0)

    node_paths = [[0]]
    descendants = [[] for _ in range(node_count)]
    for node in range(1, node_count):
        node_path = node_paths[parent_indices[node]] + [node]
        node_paths.append(node_path)
        for ancestor in node_path[:-1]:
            descendants[ancestor].append(node)

    return {
        "node_count": node_count,
        "first_leaf": first_leaf,
        "parent_indices_list": parent_indices,
        "node_depths_list": node_depths,
        "children_lists": children_lists,
        "retrieve_indices": retrieve_indices,
        "node_rows_list": node_rows,
        "postorder_list": postorder,
        "node_path_lists": node_paths,
        "descendants_lists": descendants,
    }


def _random_distribution(generator: torch.Generator, vocab_size: int):
    return torch.softmax(torch.randn(vocab_size, generator=generator), dim=-1)


def build_fixture(
        sampling_method: str = "greedy",
        seed: int = 0,
        vocab_size: int = 10,
        depth: int = 5,
) -> Dict[str, object]:
    """Build verifier inputs with internally consistent proposal metadata.

    Args:
        sampling_method: ``"greedy"`` builds the Top-(m-1) plus residual
            sample tree required by Greedy and UniVer. ``"without_replacement"``
            builds the ordered sibling samples required by RRSw and Traversal.
        seed: Local RNG seed. Global PyTorch RNG state is not modified.
        vocab_size: Compact vocabulary size; defaults to 10.
        depth: Root-to-leaf edge depth; 5 gives 32 leaves and 63 nodes.

    Returns:
        A dictionary containing ``logits``, ``candidates``,
        ``logits_processor``, ``tree_info`` and ``retrieve_indices``.
        Logits use the compact ``[tree_nodes, vocab]`` layout.
    """
    if sampling_method not in {"greedy", "without_replacement"}:
        raise ValueError("sampling_method must be 'greedy' or 'without_replacement'")
    if vocab_size < 2:
        raise ValueError("a binary draft tree requires vocab_size >= 2")
    if depth < 1:
        raise ValueError("depth must be positive")

    topology = _full_binary_topology(depth)
    node_count = topology["node_count"]
    expanded_count = topology["first_leaf"]
    generator = torch.Generator().manual_seed(seed)

    node_token_ids = torch.empty(node_count, dtype=torch.long)
    node_token_ids[0] = 0
    draft_distributions = []
    residual_distributions = []
    node_draft_probabilities = torch.ones(node_count, dtype=torch.float32)
    node_residual_probabilities = torch.zeros(node_count, dtype=torch.float32)
    node_is_random = torch.zeros(node_count, dtype=torch.bool)

    for parent in range(expanded_count):
        proposal = _random_distribution(generator, vocab_size)
        children = topology["children_lists"][parent]
        if sampling_method == "greedy":
            top_token = int(torch.argmax(proposal))
            residual = proposal.clone()
            residual[top_token] = 0
            residual /= residual.sum()
            sampled_token = int(torch.multinomial(
                residual, 1, generator=generator
            ))
            child_tokens = [top_token, sampled_token]
            selected_probabilities = [
                float(residual[top_token]),
                float(residual[sampled_token]),
            ]
            node_is_random[children[-1]] = True
        else:
            sampled = torch.multinomial(
                proposal, 2, replacement=False, generator=generator
            )
            child_tokens = sampled.tolist()
            first_mass = proposal[sampled[0]]
            selected_probabilities = [
                float(first_mass),
                float(proposal[sampled[1]] / (1 - first_mass)),
            ]
            residual = proposal
            node_is_random[children] = True

        for child, token, selected_probability in zip(
                children, child_tokens, selected_probabilities
        ):
            node_token_ids[child] = token
            node_draft_probabilities[child] = proposal[token]
            node_residual_probabilities[child] = selected_probability
        draft_distributions.append(proposal)
        residual_distributions.append(residual)

    draft_distributions = torch.stack(draft_distributions)
    residual_distributions = torch.stack(residual_distributions)

    # Make target distributions close to, but not identical to, the draft.
    # This produces a useful mix of acceptance and rejection decisions.
    target_distributions = []
    for node in range(node_count):
        noise = _random_distribution(generator, vocab_size)
        if node < expanded_count:
            target = 0.75 * draft_distributions[node] + 0.25 * noise
        else:
            target = noise
        target_distributions.append(target / target.sum())
    target_distributions = torch.stack(target_distributions)
    logits = target_distributions.clamp_min(1e-12).log()

    retrieve_indices = topology["retrieve_indices"]
    candidates = node_token_ids[retrieve_indices]
    expanded_parents = list(range(expanded_count))
    parent_distribution_rows = expanded_parents + [-1] * (
        node_count - expanded_count
    )
    child_groups = [
        torch.tensor(children, dtype=torch.long)
        for children in topology["children_lists"][:expanded_count]
    ]

    expanded_depth_groups = []
    for parent_depth in range(depth):
        expanded_depth_groups.append([
            parent
            for parent in expanded_parents
            if topology["node_depths_list"][parent] == parent_depth
        ])
    traversal_depth_groups = []
    for node_depth in range(1, depth + 1):
        traversal_depth_groups.append([
            node for node in range(1, node_count)
            if topology["node_depths_list"][node] == node_depth
        ])
    traversal_nodes = [
        node for group in traversal_depth_groups for node in group
    ]

    tree_info = {
        "sampling_method": sampling_method,
        "node_token_ids": node_token_ids,
        "node_draft_probabilities": node_draft_probabilities[None],
        "node_residual_probabilities": node_residual_probabilities[None],
        "node_is_random": node_is_random[None],
        "parent_indices": torch.tensor(
            topology["parent_indices_list"], dtype=torch.long
        ),
        "expanded_parent_indices": torch.arange(expanded_count),
        "draft_distributions": draft_distributions,
        "residual_distributions": residual_distributions,
        "child_groups": child_groups,
        "child_index_matrix": torch.stack(child_groups),
        "node_rows": torch.tensor(topology["node_rows_list"]),
        "node_depths": torch.tensor(topology["node_depths_list"]),
        "postorder": torch.tensor(topology["postorder_list"]),
        "parent_distribution_rows": torch.tensor(parent_distribution_rows),
        "expanded_depth_groups": expanded_depth_groups,
        "expanded_depth_group_tensors": [
            torch.tensor(group) for group in expanded_depth_groups
        ],
        "traversal_depth_groups": traversal_depth_groups,
        "traversal_depth_group_tensors": [
            torch.tensor(group) for group in traversal_depth_groups
        ],
        "traversal_parent_group_tensors": [
            torch.tensor([
                topology["parent_indices_list"][node] for node in group
            ])
            for group in traversal_depth_groups
        ],
        "traversal_parent_row_group_tensors": [
            torch.tensor([
                parent_distribution_rows[
                    topology["parent_indices_list"][node]
                ]
                for node in group
            ])
            for group in traversal_depth_groups
        ],
        "traversal_nodes_tensor": torch.tensor(traversal_nodes),
        "traversal_parent_rows_tensor": torch.tensor([
            parent_distribution_rows[topology["parent_indices_list"][node]]
            for node in traversal_nodes
        ]),
        "parent_indices_list": topology["parent_indices_list"],
        "children_lists": topology["children_lists"],
        "node_rows_list": topology["node_rows_list"],
        "node_depths_list": topology["node_depths_list"],
        "node_token_ids_list": node_token_ids.tolist(),
        "postorder_list": topology["postorder_list"],
        "node_path_lists": topology["node_path_lists"],
        "descendants_lists": topology["descendants_lists"],
        "parent_distribution_rows_list": parent_distribution_rows,
        "expanded_parent_indices_list": expanded_parents,
        "traversal_nodes_list": traversal_nodes,
    }
    return {
        "logits": logits,
        "candidates": candidates,
        "logits_processor": IdentityLogitsProcessor(),
        "tree_info": tree_info,
        "retrieve_indices": retrieve_indices,
    }


def run_smoke_test(seed: int = 0):
    from eagle.model.utils import (
        evaluate_posterior2,
        evaluate_posterior3,
        evaluate_posterior4,
        evaluate_posterior5,
    )

    greedy = build_fixture("greedy", seed=seed)
    without_replacement = build_fixture("without_replacement", seed=seed)
    cases = [
        (
            "UniVer",
            evaluate_posterior2,
            greedy,
            lambda fn, data: fn(
                data["logits"], data["candidates"],
                data["retrieve_indices"], data["logits_processor"],
                data["tree_info"],
            ),
        ),
        (
            "Greedy",
            evaluate_posterior5,
            greedy,
            lambda fn, data: fn(
                data["logits"], data["candidates"],
                data["logits_processor"], data["tree_info"],
                data["retrieve_indices"],
            ),
        ),
        (
            "RRSw",
            evaluate_posterior3,
            without_replacement,
            lambda fn, data: fn(
                data["logits"], data["candidates"],
                data["logits_processor"], data["tree_info"],
                data["retrieve_indices"],
            ),
        ),
        (
            "Traversal",
            evaluate_posterior4,
            without_replacement,
            lambda fn, data: fn(
                data["logits"], data["candidates"],
                data["logits_processor"], data["tree_info"],
                data["retrieve_indices"],
            ),
        ),
    ]
    outputs = {}
    for index, (name, verifier, data, invoke) in enumerate(cases):
        torch.manual_seed(seed + 100 + index)
        best_candidate, accept_length, sample_p = invoke(verifier, data)
        if sample_p.shape != (10,):
            raise AssertionError(f"{name}: expected sample_p [10]")
        torch.testing.assert_close(sample_p.sum(), torch.tensor(1.0))
        outputs[name] = {
            "best_candidate": int(best_candidate),
            "accept_length": int(accept_length),
            "sample_token_distribution": sample_p,
        }
    return greedy, without_replacement, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    greedy, without_replacement, outputs = run_smoke_test(args.seed)
    print("Fixture shapes")
    print("  logits:", tuple(without_replacement["logits"].shape))
    # print(greedy["logits"])
    print("  candidates:", tuple(without_replacement["candidates"].shape))
    print(without_replacement["candidates"])
    print("  retrieve_indices:", tuple(without_replacement["retrieve_indices"].shape))
    print(without_replacement["retrieve_indices"])
    print("  draft_distributions:", tuple(
        without_replacement["tree_info"]["draft_distributions"].shape
    ))
    print("Verifier outputs")
    for method, output in outputs.items():
        print(
            f"  {method}: candidate={output['best_candidate']} "
            f"accept_length={output['accept_length']} "
            f"sample_p_sum={float(output['sample_token_distribution'].sum()):.6f}"
        )


if __name__ == "__main__":
    main()
