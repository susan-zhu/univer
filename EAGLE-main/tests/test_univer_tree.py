import copy
import pathlib
import sys
from unittest.mock import patch

import torch


MODEL_DIR = pathlib.Path(__file__).resolve().parents[1] / "eagle" / "model"
sys.path.insert(0, str(MODEL_DIR))

from univer_tree import (
    generate_univer_tree,
    greedy_residual_sample,
    sample_without_replacement,
)
from cnets1 import Model as LegacyEagleModel
from configs import EConfig
from utils import (
    evaluate_posterior2,
    evaluate_posterior3,
    evaluate_posterior4,
    evaluate_posterior5,
)


def test_legacy_eagle_model_retains_config_for_probability_tree_generation():
    config = EConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        pad_token_id=0,
    )

    model = LegacyEagleModel(
        config,
        total_tokens=3,
        depth=1,
        top_k=2,
    )

    assert model.config is config
    assert model.config.vocab_size == model.vocab_size == 8

    with patch("cnets1.generate_univer_tree", return_value=None) as generator:
        model.univer_generate(
            torch.zeros(1, 1, 4),
            torch.tensor([[1]]),
            lambda states: states,
            None,
        )

    assert generator.call_args.kwargs["store_draft_distributions"] is False


def test_greedy_residual_sample_uses_top_m_minus_one_and_residual():
    torch.manual_seed(0)
    probabilities = torch.tensor([0.6, 0.3, 0.1])

    candidate_ids, residual, is_random = greedy_residual_sample(
        probabilities, candidate_count=2
    )

    assert candidate_ids[0].item() == 0
    assert candidate_ids[1].item() in {1, 2}
    torch.testing.assert_close(residual, torch.tensor([0.0, 0.75, 0.25]))
    assert is_random.tolist() == [False, True]


def test_greedy_residual_sample_reduces_group_to_filtered_support():
    probabilities = torch.tensor([1.0, 0.0, 0.0])

    candidate_ids, residual, is_random = greedy_residual_sample(
        probabilities, candidate_count=3
    )

    assert candidate_ids.tolist() == [0]
    torch.testing.assert_close(residual, probabilities)
    assert is_random.tolist() == [True]


def test_sample_without_replacement_uses_renormalized_proposals():
    torch.manual_seed(1)
    probabilities = torch.tensor([0.6, 0.3, 0.1])

    candidate_ids, selected_probabilities, is_random = sample_without_replacement(
        probabilities, candidate_count=3
    )

    assert sorted(candidate_ids.tolist()) == [0, 1, 2]
    remaining = probabilities.clone()
    expected = []
    for token_id in candidate_ids:
        proposal = remaining / remaining.sum()
        expected.append(proposal[token_id])
        remaining[token_id] = 0
    torch.testing.assert_close(selected_probabilities, torch.stack(expected))
    assert is_random.tolist() == [True, True, True]


def test_generate_univer_tree_keeps_complete_sibling_groups():
    class FakeModel:
        total_tokens = 6
        top_k = 2
        depth = 2
        stable_kv = None
        tree_mask = None

        def reset(self):
            pass

        def __call__(
                self,
                hidden_states,
                input_ids,
                past_key_values=None,
                position_ids=None,
                use_cache=True,
        ):
            batch_size, sequence_length = input_ids.shape
            output = torch.zeros(batch_size, sequence_length, 4)
            cache = [
                (
                    torch.zeros(1, 1, sequence_length, 1),
                    torch.zeros(1, 1, sequence_length, 1),
                )
            ]
            return output, cache

    def project_logits(states):
        return torch.tensor([2.0, 1.0, 0.0]).repeat(states.shape[0], 1)

    model = FakeModel()
    result = generate_univer_tree(
        model=model,
        hidden_states=torch.zeros(1, 3, 4),
        input_ids=torch.tensor([[9, 8, 7]]),
        logits_processor=None,
        project_logits=project_logits,
        map_to_target=lambda token_ids: token_ids,
        target_vocab_size=3,
    )
    draft_tokens, retrieve_indices, tree_mask, position_ids, info = result

    assert draft_tokens.shape == (1, 7)
    assert info["parent_indices"].tolist() == [-1, 0, 0, 1, 1, 2, 2]
    assert info["draft_distributions"].shape == (3, 3)
    assert info["residual_distributions"].shape == (3, 3)
    assert [group.numel() for group in info["child_groups"]] == [2, 2, 2]
    assert tree_mask.shape == (1, 1, 7, 7)
    assert position_ids.tolist() == [0, 1, 1, 2, 2, 2, 2]
    assert retrieve_indices.shape == (4, 3)
    assert info["node_path_lists"] == [
        [0], [0, 1], [0, 2], [0, 1, 3], [0, 1, 4],
        [0, 2, 5], [0, 2, 6],
    ]
    assert info["descendants_lists"] == [
        [1, 2, 3, 4, 5, 6], [3, 4], [5, 6], [], [], [], [],
    ]
    assert [
        group.tolist() for group in info["traversal_depth_group_tensors"]
    ] == [[1, 2], [3, 4, 5, 6]]
    assert [
        group.tolist() for group in info["traversal_parent_row_group_tensors"]
    ] == [[0, 0], [1, 1, 2, 2]]


def test_generate_univer_tree_can_skip_unused_univer_draft_matrix():
    class FakeModel:
        total_tokens = 2
        top_k = 2
        depth = 1
        stable_kv = None
        tree_mask = None

        def reset(self):
            pass

        def __call__(
                self, hidden_states, input_ids, past_key_values=None,
                position_ids=None, use_cache=True,
        ):
            output = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
            cache = [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))]
            return output, cache

    result = generate_univer_tree(
        model=FakeModel(),
        hidden_states=torch.zeros(1, 2, 4),
        input_ids=torch.tensor([[9, 8]]),
        logits_processor=None,
        project_logits=lambda states: torch.tensor([2.0, 1.0, 0.0]).repeat(
            states.shape[0], 1
        ),
        map_to_target=lambda token_ids: token_ids,
        target_vocab_size=3,
        store_draft_distributions=False,
    )
    draft_tokens, retrieve_indices, _, _, info = result
    assert info["draft_distributions"] is None
    assert info["residual_distributions"].shape == (1, 3)

    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    candidates = draft_tokens[0, retrieve_indices]
    _, _, sample_p = evaluate_posterior2(
        torch.zeros(3, 3), candidates, retrieve_indices,
        IdentityProcessor(), info,
    )
    torch.testing.assert_close(sample_p.sum(), torch.tensor(1.0))


def test_evaluate_posterior2_allocates_the_root_example():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    target = torch.tensor([0.3, 0.4, 0.3])
    bonus = torch.tensor([0.2, 0.5, 0.3])
    logits = torch.stack(
        [
            torch.stack([target.log(), bonus.log()]),
            torch.stack([target.log(), bonus.log()]),
        ]
    )
    candidates = torch.tensor([[9, 0], [9, 1]])
    retrieve_indices = torch.tensor([[0, 1], [0, 2]])
    tree_info = {
        "node_token_ids": torch.tensor([9, 0, 1]),
        "parent_indices": torch.tensor([-1, 0, 0]),
        "expanded_parent_indices": torch.tensor([0]),
        "draft_distributions": torch.tensor([[0.6, 0.3, 0.1]]),
        "residual_distributions": torch.tensor([[0.0, 0.75, 0.25]]),
        "child_groups": [torch.tensor([1, 2])],
    }

    torch.manual_seed(3)
    evaluate_posterior2(
        logits,
        candidates,
        retrieve_indices,
        IdentityProcessor(),
        tree_info,
    )

    torch.testing.assert_close(
        tree_info["marginal_acceptance_probabilities"],
        torch.tensor([0.0, 0.4, 8.0 / 15.0]),
    )
    torch.testing.assert_close(
        tree_info["effective_acceptance_probabilities"],
        torch.tensor([1.0, 0.4, 8.0 / 9.0]),
    )
    torch.testing.assert_close(
        tree_info["fallback_acceptance_probabilities"],
        torch.tensor([1.0, 0.0, 0.0]),
    )
    assert tree_info["allocation_mass_errors"].max().item() < 1e-6


def _greedy_one_level_example(target):
    bonus_left = torch.tensor([0.2, 0.5, 0.3])
    bonus_right = torch.tensor([0.4, 0.2, 0.4])
    logits = torch.stack([target.log(), bonus_left.log(), bonus_right.log()])
    candidates = torch.tensor([[9, 0], [9, 1]])
    retrieve_indices = torch.tensor([[0, 1], [0, 2]])
    tree_info = {
        "sampling_method": "greedy",
        "node_token_ids": torch.tensor([9, 0, 1]),
        "node_is_random": torch.tensor([[False, False, True]]),
        "parent_indices": torch.tensor([-1, 0, 0]),
        "expanded_parent_indices": torch.tensor([0]),
        "residual_distributions": torch.tensor([[0.0, 0.75, 0.25]]),
        "child_groups": [torch.tensor([1, 2])],
        "node_rows_list": [0, 0, 1],
        "node_depths_list": [0, 1, 1],
        "parent_distribution_rows_list": [0, -1, -1],
    }
    return logits, candidates, retrieve_indices, tree_info


def test_greedy_ot_correction_can_accept_deterministic_top_child():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    # z=1 is rejected. relu(p-r) then has all its mass on deterministic
    # child 0, which must be accepted instead of terminating the round.
    example = _greedy_one_level_example(torch.tensor([0.7, 0.2, 0.1]))
    with patch("torch.rand", return_value=torch.tensor(0.9)), patch(
            "torch.multinomial", return_value=torch.tensor([0])
    ):
        best_candidate, accept_length, sample_p = evaluate_posterior5(
            example[0], example[1], IdentityProcessor(), example[3], example[2]
        )

    assert best_candidate.item() == 0
    assert accept_length == 1
    torch.testing.assert_close(sample_p, torch.tensor([0.2, 0.5, 0.3]))


def test_greedy_ot_returns_only_non_candidate_correction_mass():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    # After rejecting z=1, relu(p-r) contains candidate token 0 and outside
    # token 2. Choosing the fallback event must return token 2 only.
    example = _greedy_one_level_example(torch.tensor([0.1, 0.2, 0.7]))
    with patch("torch.rand", return_value=torch.tensor(0.9)), patch(
            "torch.multinomial", return_value=torch.tensor([2])
    ):
        best_candidate, accept_length, sample_p = evaluate_posterior5(
            example[0], example[1], IdentityProcessor(), example[3], example[2]
        )

    assert best_candidate.item() == 0
    assert accept_length == 0
    torch.testing.assert_close(sample_p, torch.tensor([0.0, 0.0, 1.0]))


def test_greedy_ot_restarts_local_coupling_down_selected_path():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    root_target = torch.tensor([0.7, 0.2, 0.1])
    left_target = torch.tensor([0.1, 0.1, 0.8])
    bonuses = [
        torch.tensor([0.2, 0.3, 0.5]),
        torch.tensor([0.3, 0.4, 0.3]),
        torch.tensor([0.4, 0.4, 0.2]),
    ]
    logits = torch.stack([
        root_target.log(), left_target.log(), *(bonus.log() for bonus in bonuses)
    ])
    candidates = torch.tensor([
        [9, 0, 0],
        [9, 0, 2],
        [9, 1, -1],
    ])
    retrieve_indices = torch.tensor([
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, -1],
    ])
    tree_info = {
        "sampling_method": "greedy",
        "node_token_ids": torch.tensor([9, 0, 1, 0, 2]),
        "node_is_random": torch.tensor([[False, False, True, False, True]]),
        "parent_indices": torch.tensor([-1, 0, 0, 1, 1]),
        "expanded_parent_indices": torch.tensor([0, 1]),
        "residual_distributions": torch.tensor([
            [0.0, 0.75, 0.25],
            [0.0, 0.4, 0.6],
        ]),
        "child_groups": [torch.tensor([1, 2]), torch.tensor([3, 4])],
        "node_rows_list": [0, 0, 2, 0, 1],
        "node_depths_list": [0, 1, 1, 2, 2],
        "parent_distribution_rows_list": [0, 1, -1, -1, -1],
    }

    # Reject root z=1, select deterministic child 0 from correction, then
    # directly accept the next layer's random z=2 under its fresh local OT.
    with patch("torch.rand", side_effect=[torch.tensor(0.9), torch.tensor(0.0)]), patch(
            "torch.multinomial", return_value=torch.tensor([0])
    ):
        best_candidate, accept_length, sample_p = evaluate_posterior5(
            logits, candidates, IdentityProcessor(), tree_info, retrieve_indices
        )

    assert best_candidate.item() == 1
    assert accept_length == 2
    torch.testing.assert_close(sample_p, bonuses[2])


def test_univer_compact_target_logits_match_expanded_paths():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    target = torch.tensor([0.3, 0.4, 0.3])
    bonus_left = torch.tensor([0.2, 0.5, 0.3])
    bonus_right = torch.tensor([0.4, 0.2, 0.4])
    expanded_logits = torch.stack(
        [
            torch.stack([target.log(), bonus_left.log()]),
            torch.stack([target.log(), bonus_right.log()]),
        ]
    )
    compact_logits = torch.stack(
        [target.log(), bonus_left.log(), bonus_right.log()]
    )
    candidates = torch.tensor([[9, 0], [9, 1]])
    retrieve_indices = torch.tensor([[0, 1], [0, 2]])
    tree_info = {
        "node_token_ids": torch.tensor([9, 0, 1]),
        "parent_indices": torch.tensor([-1, 0, 0]),
        "expanded_parent_indices": torch.tensor([0]),
        "draft_distributions": torch.tensor([[0.6, 0.3, 0.1]]),
        "residual_distributions": torch.tensor([[0.0, 0.75, 0.25]]),
        "child_groups": [torch.tensor([1, 2])],
    }

    outputs = []
    infos = []
    for target_logits in (expanded_logits, compact_logits):
        current_info = copy.deepcopy(tree_info)
        torch.manual_seed(7)
        outputs.append(evaluate_posterior2(
            target_logits,
            candidates,
            retrieve_indices,
            IdentityProcessor(),
            current_info,
        ))
        infos.append(current_info)

    assert outputs[0][0].item() == outputs[1][0].item()
    assert outputs[0][1] == outputs[1][1]
    torch.testing.assert_close(outputs[0][2], outputs[1][2])
    for key in (
            "effective_acceptance_probabilities",
            "fallback_acceptance_probabilities",
            "marginal_acceptance_probabilities",
            "rejection_probabilities",
            "allocation_mass_errors",
            "target_parent_distributions",
    ):
        torch.testing.assert_close(infos[0][key], infos[1][key])


def test_univer_binary_matrix_fast_path_matches_compatibility_path():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    target = torch.tensor([0.3, 0.4, 0.3])
    bonus_left = torch.tensor([0.2, 0.5, 0.3])
    bonus_right = torch.tensor([0.4, 0.2, 0.4])
    logits = torch.stack([target.log(), bonus_left.log(), bonus_right.log()])
    candidates = torch.tensor([[9, 0], [9, 1]])
    retrieve_indices = torch.tensor([[0, 1], [0, 2]])
    base_info = {
        "node_token_ids": torch.tensor([9, 0, 1]),
        "parent_indices": torch.tensor([-1, 0, 0]),
        "expanded_parent_indices": torch.tensor([0]),
        "draft_distributions": torch.tensor([[0.6, 0.3, 0.1]]),
        "residual_distributions": torch.tensor([[0.0, 0.75, 0.25]]),
        "child_groups": [torch.tensor([1, 2])],
    }

    results = []
    infos = []
    for use_matrix in (False, True):
        info = copy.deepcopy(base_info)
        if use_matrix:
            info["child_index_matrix"] = torch.tensor([[1, 2]])
        torch.manual_seed(13)
        results.append(evaluate_posterior2(
            logits, candidates, retrieve_indices, IdentityProcessor(), info
        ))
        infos.append(info)

    assert results[0][0].item() == results[1][0].item()
    assert results[0][1] == results[1][1]
    torch.testing.assert_close(results[0][2], results[1][2])
    for key in (
            "effective_acceptance_probabilities",
            "fallback_acceptance_probabilities",
            "marginal_acceptance_probabilities",
            "rejection_probabilities",
            "allocation_mass_errors",
    ):
        torch.testing.assert_close(infos[0][key], infos[1][key])


def test_univer_no_diagnostics_path_preserves_decision():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    target = torch.tensor([0.3, 0.4, 0.3])
    bonus_left = torch.tensor([0.2, 0.5, 0.3])
    bonus_right = torch.tensor([0.4, 0.2, 0.4])
    logits = torch.stack([target.log(), bonus_left.log(), bonus_right.log()])
    candidates = torch.tensor([[9, 0], [9, 1]])
    retrieve_indices = torch.tensor([[0, 1], [0, 2]])
    base_info = {
        "node_token_ids": torch.tensor([9, 0, 1]),
        "parent_indices": torch.tensor([-1, 0, 0]),
        "expanded_parent_indices": torch.tensor([0]),
        "draft_distributions": torch.tensor([[0.6, 0.3, 0.1]]),
        "residual_distributions": torch.tensor([[0.0, 0.75, 0.25]]),
        "child_groups": [torch.tensor([1, 2])],
        "child_index_matrix": torch.tensor([[1, 2]]),
    }

    results = []
    for collect_diagnostics in (True, False):
        info = copy.deepcopy(base_info)
        info["collect_diagnostics"] = collect_diagnostics
        torch.manual_seed(19)
        results.append(evaluate_posterior2(
            logits, candidates, retrieve_indices, IdentityProcessor(), info
        ))

    assert results[0][0].item() == results[1][0].item()
    assert results[0][1] == results[1][1]
    torch.testing.assert_close(results[0][2], results[1][2])


def test_rrsw_verifiers_accept_stochastic_tree_metadata():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    target = torch.tensor([0.3, 0.4, 0.3])
    bonus = torch.tensor([0.2, 0.5, 0.3])
    logits = torch.stack(
        [
            torch.stack([target.log(), bonus.log()]),
            torch.stack([target.log(), bonus.log()]),
        ]
    )
    candidates = torch.tensor([[9, 0], [9, 1]])
    retrieve_indices = torch.tensor([[0, 1], [0, 2]])
    tree_info = {
        "parent_indices": torch.tensor([-1, 0, 0]),
        "expanded_parent_indices": torch.tensor([0]),
        "draft_distributions": torch.tensor([[0.6, 0.3, 0.1]]),
    }

    for verifier in (evaluate_posterior3, evaluate_posterior4):
        torch.manual_seed(2)
        _, _, sample_p = verifier(
            logits,
            candidates,
            IdentityProcessor(),
            tree_info,
            retrieve_indices,
        )
        torch.testing.assert_close(sample_p.sum(), torch.tensor(1.0))


def test_rrsw_and_traversal_compact_target_logits_match_expanded_paths():
    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    target = torch.tensor([0.3, 0.4, 0.3])
    bonus_left = torch.tensor([0.2, 0.5, 0.3])
    bonus_right = torch.tensor([0.4, 0.2, 0.4])
    expanded_logits = torch.stack(
        [
            torch.stack([target.log(), bonus_left.log()]),
            torch.stack([target.log(), bonus_right.log()]),
        ]
    )
    compact_logits = torch.stack(
        [target.log(), bonus_left.log(), bonus_right.log()]
    )
    candidates = torch.tensor([[9, 0], [9, 1]])
    retrieve_indices = torch.tensor([[0, 1], [0, 2]])
    tree_info = {
        "parent_indices": torch.tensor([-1, 0, 0]),
        "expanded_parent_indices": torch.tensor([0]),
        "draft_distributions": torch.tensor([[0.6, 0.3, 0.1]]),
    }

    for verifier in (evaluate_posterior3, evaluate_posterior4):
        results = []
        for target_logits in (expanded_logits, compact_logits):
            torch.manual_seed(11)
            results.append(verifier(
                target_logits,
                candidates,
                IdentityProcessor(),
                tree_info,
                retrieve_indices,
            ))
        assert results[0][0].item() == results[1][0].item()
        assert results[0][1] == results[1][1]
        torch.testing.assert_close(results[0][2], results[1][2])


def test_traversal_refreshes_later_subtree_rates_after_sibling_rejection():
    """Algorithm 3 line 13 must propagate a changed prefix probability."""

    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    # Tree: root -> A and root -> B -> C.  Rejecting A changes B's rate from
    # 1 to 4/9, which must also reduce C's chain rate from 1 to 4/9.
    root_target = torch.tensor([0.3, 0.4, 0.3])
    b_target = torch.tensor([0.5, 0.25, 0.25])
    bonus = torch.tensor([0.2, 0.3, 0.5])
    logits = torch.stack(
        [
            torch.stack([root_target.log(), bonus.log(), bonus.log()]),
            torch.stack([root_target.log(), b_target.log(), bonus.log()]),
        ]
    )
    candidates = torch.tensor([[9, 0, -1], [9, 1, 0]])
    retrieve_indices = torch.tensor([[0, 1, -1], [0, 2, 3]])
    tree_info = {
        "parent_indices": torch.tensor([-1, 0, 0, 2]),
        "expanded_parent_indices": torch.tensor([0, 2]),
        "draft_distributions": torch.tensor(
            [[0.6, 0.3, 0.1], [0.5, 0.25, 0.25]]
        ),
    }

    # 0.9 rejects A (rate 0.5).  The next 0.7 must reject C after its rate
    # is refreshed to 4/9; the stale implementation accepted it at rate 1.
    with patch(
            "torch.rand",
            return_value=torch.tensor([0.9, 0.7, 0.5]),
    ):
        _, accept_length, sample_p = evaluate_posterior4(
            logits,
            candidates,
            IdentityProcessor(),
            tree_info,
            retrieve_indices,
        )

    assert accept_length == 0
    torch.testing.assert_close(sample_p.sum(), torch.tensor(1.0))


def test_traversal_lazy_refresh_matches_eager_reference():
    """Cached topology and lazy path refresh preserve Algorithm 3 outputs."""

    class IdentityProcessor:
        def __call__(self, input_ids, scores):
            return scores

    root_target = torch.tensor([0.3, 0.4, 0.3])
    b_target = torch.tensor([0.5, 0.25, 0.25])
    bonus = torch.tensor([0.2, 0.3, 0.5])
    root_draft = torch.tensor([0.6, 0.3, 0.1])
    b_draft = torch.tensor([0.5, 0.25, 0.25])
    logits = torch.stack([
        torch.stack([root_target.log(), bonus.log(), bonus.log()]),
        torch.stack([root_target.log(), b_target.log(), bonus.log()]),
    ])
    candidates = torch.tensor([[9, 0, -1], [9, 1, 0]])
    retrieve_indices = torch.tensor([[0, 1, -1], [0, 2, 3]])
    metadata = {
        "parent_indices": torch.tensor([-1, 0, 0, 2]),
        "expanded_parent_indices": torch.tensor([0, 2]),
        "draft_distributions": torch.stack([root_draft, b_draft]),
        "parent_indices_list": [-1, 0, 0, 2],
        "children_lists": [[1, 2], [], [3], []],
        "node_rows_list": [0, 0, 1, 1],
        "node_depths_list": [0, 1, 1, 2],
        "postorder_list": [1, 3, 2, 0],
        "node_path_lists": [[0], [0, 1], [0, 2], [0, 2, 3]],
        "descendants_lists": [[1, 2, 3], [], [3], []],
        "parent_distribution_rows_list": [0, -1, 1, -1],
        "expanded_parent_indices_list": [0, 2],
        "traversal_depth_groups": [[1, 2], [3]],
        "traversal_depth_group_tensors": [
            torch.tensor([1, 2]), torch.tensor([3]),
        ],
        "traversal_parent_group_tensors": [
            torch.tensor([0, 0]), torch.tensor([2]),
        ],
        "traversal_parent_row_group_tensors": [
            torch.tensor([0, 0]), torch.tensor([1]),
        ],
    }
    expanded_draft = torch.stack([
        torch.stack([root_draft, bonus, bonus]),
        torch.stack([root_draft, b_draft, bonus]),
    ])

    for uniforms in (
            [0.1],
            [0.9, 0.1],
            [0.9, 0.7, 0.5],
            [0.9, 0.9, 0.1],
    ):
        lazy_uniforms = torch.tensor(uniforms + [0.5] * (3 - len(uniforms)))
        with patch("torch.rand", return_value=lazy_uniforms):
            lazy_result = evaluate_posterior4(
                logits, candidates, IdentityProcessor(), metadata,
                retrieve_indices,
            )
        random_values = [torch.tensor(value) for value in uniforms]
        with patch("torch.rand", side_effect=random_values):
            eager_result = evaluate_posterior4(
                logits, candidates, IdentityProcessor(), expanded_draft,
            )

        assert lazy_result[0].item() == eager_result[0].item()
        assert lazy_result[1] == eager_result[1]
        torch.testing.assert_close(lazy_result[2], eager_result[2])
