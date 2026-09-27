from types import SimpleNamespace

import torch

from eagle.evaluation.benchmark_verify_methods import (
    build_prompt,
    run_generation,
    summarize,
    uses_llama3_eot,
)


class _FakeTokenizer:
    def __call__(self, prompt, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    def decode(self, token_ids, skip_special_tokens=True):
        return "generated"

    def get_vocab(self):
        return {}


class _FakeModel:
    def __init__(self):
        self.base_model = SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=SimpleNamespace(weight=torch.empty(1))
            )
        )
        self.tokenizer = _FakeTokenizer()

    def get_tokenizer(self):
        return self.tokenizer

    def eagenerate(self, input_ids, **kwargs):
        # Two rounds: zero and one accepted draft token.  Including each
        # round's bonus token gives paper-style acceptance lengths [1, 2].
        self.last_eagenerate_metrics = {
            "verification_rounds": 2,
            "total_accept_length": 3,
            "total_generated_tokens": 3,
            "total_accepted_draft_tokens": 1,
            "average_accept_length": 1.5,
            "average_accepted_draft_tokens": 0.5,
            "acceptance_lengths": [1, 2],
            "accepted_draft_lengths": [0, 1],
        }
        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        return output_ids, 3, 1

    def naivegenerate(self, input_ids, **kwargs):
        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        return output_ids, 3, 2


def test_run_generation_collects_acceptance_lengths():
    args = SimpleNamespace(
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        max_new_tokens=8,
        max_length=32,
    )

    record = run_generation(_FakeModel(), "prompt", args, "RRSw")

    assert record["verification_rounds"] == 2
    assert record["total_accept_length"] == 3
    assert record["average_accept_length"] == 1.5
    assert record["total_accepted_draft_tokens"] == 1
    assert record["average_accepted_draft_tokens"] == 0.5
    assert record["acceptance_lengths"] == [1, 2]
    assert record["accepted_draft_lengths"] == [0, 1]


def test_summary_aggregates_acceptance_length_by_round():
    records = [
        {
            "verify_method": "RRSw",
            "category": "test",
            "output_tokens": 3,
            "verification_rounds": 2,
            "total_accept_length": 3,
            "total_accepted_draft_tokens": 1,
            "seconds": 1.0,
            "tokens_per_round": 1.5,
        },
        {
            "verify_method": "RRSw",
            "category": "test",
            "output_tokens": 5,
            "verification_rounds": 3,
            "total_accept_length": 5,
            "total_accepted_draft_tokens": 2,
            "seconds": 1.0,
            "tokens_per_round": 5 / 3,
        },
    ]

    summary = summarize(records)[0]

    assert summary["total_accept_length"] == 8
    assert summary["average_accept_length"] == 1.6
    assert summary["total_accepted_draft_tokens"] == 3
    assert summary["average_accepted_draft_tokens"] == 0.6
    assert summary["tokens_per_round"] == 1.6
    assert summary["milliseconds_per_round"] == 400.0


def test_baseline_leaves_acceptance_metrics_empty():
    args = SimpleNamespace(
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        max_new_tokens=8,
        max_length=32,
    )

    record = run_generation(_FakeModel(), "prompt", args, "baseline")

    assert record["output_tokens"] == 3
    assert record["verification_rounds"] is None
    assert record["total_accept_length"] is None
    assert record["average_accept_length"] is None
    assert record["acceptance_lengths"] == []
    assert record["tokens_per_round"] is None

    summary = summarize([{**record, "verify_method": "baseline"}])[0]
    assert summary["verification_rounds"] is None
    assert summary["average_accept_length"] is None
    assert summary["tokens_per_round"] is None


def test_build_prompt_prefers_tokenizer_chat_template():
    class ChatTokenizer:
        chat_template = "configured"

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return f"rendered:{messages[0]['content']}"

    prompt = build_prompt(
        ChatTokenizer(),
        [{"role": "user", "content": "Hello."}],
    )

    assert prompt == "rendered:Hello."


def test_uses_llama3_eot_checks_tokenizer_vocabulary():
    tokenizer = SimpleNamespace(get_vocab=lambda: {"<|eot_id|>": 128009})

    assert uses_llama3_eot(tokenizer)
    assert not uses_llama3_eot(_FakeTokenizer())


def test_summary_aggregates_optional_phase_profile_by_round():
    records = []
    for rounds in (2, 3):
        records.append({
            "verify_method": "univer",
            "output_tokens": rounds * 2,
            "verification_rounds": rounds,
            "total_accept_length": rounds * 2,
            "total_accepted_draft_tokens": rounds,
            "seconds": float(rounds),
            "tokens_per_round": 2.0,
            "initial_tree_seconds": 0.1,
            "target_decode_seconds": 0.01 * rounds,
            "verification_seconds": 0.002 * rounds,
            "update_and_draft_seconds": 0.02 * rounds,
            "unattributed_seconds": 0.003 * rounds,
        })

    summary = summarize(records)[0]

    assert summary["initial_tree_seconds"] == 0.2
    assert summary["target_decode_ms_per_round"] == 10.0
    assert summary["verification_ms_per_round"] == 2.0
    assert summary["update_and_draft_ms_per_round"] == 20.0
    assert summary["unattributed_ms_per_round"] == 3.0
