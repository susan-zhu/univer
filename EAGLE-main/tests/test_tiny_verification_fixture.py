from unittest.mock import patch

import torch

from eagle.evaluation.tiny_verification_fixture import (
    build_fixture,
    run_smoke_test,
)
from eagle.model.utils import evaluate_posterior4


def test_tiny_fixture_is_a_depth_five_full_binary_tree():
    fixture = build_fixture("greedy", seed=0)
    tree_info = fixture["tree_info"]

    assert fixture["logits"].shape == (63, 10)
    assert fixture["candidates"].shape == (32, 6)
    assert fixture["retrieve_indices"].shape == (32, 6)
    assert tree_info["draft_distributions"].shape == (31, 10)
    assert tree_info["residual_distributions"].shape == (31, 10)
    assert len(tree_info["child_groups"]) == 31
    assert all(group.numel() == 2 for group in tree_info["child_groups"])
    torch.testing.assert_close(
        tree_info["draft_distributions"].sum(dim=1), torch.ones(31)
    )
    torch.testing.assert_close(
        tree_info["residual_distributions"].sum(dim=1), torch.ones(31)
    )


def test_tiny_fixture_runs_all_probability_verifiers():
    _, _, outputs = run_smoke_test(seed=0)

    assert set(outputs) == {"RRSw", "Traversal", "Greedy", "UniVer"}
    for output in outputs.values():
        assert 0 <= output["accept_length"] <= 5
        torch.testing.assert_close(
            output["sample_token_distribution"].sum(), torch.tensor(1.0)
        )


def test_traversal_cpu_backend_matches_gpu_on_rejection_heavy_tree():
    results = []
    for backend in ("gpu", "cpu", "cpu_lazy"):
        fixture = build_fixture("without_replacement", seed=0)
        fixture["tree_info"]["traversal_backend"] = backend
        with patch("torch.rand", return_value=torch.full((62,), 0.999)):
            output = evaluate_posterior4(
                fixture["logits"],
                fixture["candidates"],
                fixture["logits_processor"],
                fixture["tree_info"],
                fixture["retrieve_indices"],
            )
        results.append((output, fixture["tree_info"]["_traversal_stats"]))

    for output, _ in results[1:]:
        assert results[0][0][0].item() == output[0].item()
        assert results[0][0][1] == output[1]
        torch.testing.assert_close(results[0][0][2], output[2])
    assert results[0][1]["refresh_syncs"] > 0
    assert results[1][1]["refresh_syncs"] == 0
    assert results[2][1]["refresh_syncs"] == 0

