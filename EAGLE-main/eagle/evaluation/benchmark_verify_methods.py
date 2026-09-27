"""Benchmark EAGLE token-level and Traversal Verification on JSONL prompts.

Example:
    python -m eagle.evaluation.benchmark_verify_methods \
      --base-model-path /models/Llama-3.1-8B-Instruct \
      --ea-model-path /models/EAGLE3-LLaMA3.1-Instruct-8B \
      --question-file eagle/data/mt_bench/question.jsonl \
      --output-dir results/verify_benchmark \
      --temperature 0.7 --top-p 0.9 --max-new-tokens 256
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--ea-model-path", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--verify-methods",
        nargs="+",
        default=["default", "RRSw", "traversal_verification", "greedy", "univer"],
        choices=[
            "baseline", "default", "RRSw", "traversal_verification",
            "greedy", "univer",
        ],
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0, help="Sampling top-k (0 disables it).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--draft-top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions.")
    parser.add_argument("--turns", choices=["first", "all"], default="first")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up generations excluded from metrics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help=(
            "Use CUDA Events to report target decoding, verification and "
            "draft/update time separately. Disabled by default."
        ),
    )
    parser.add_argument(
        "--compile-univer",
        action="store_true",
        help=(
            "Experimentally fuse UniVer's dense conditional-OT allocation "
            "with torch.compile. Use --warmup 3 or more."
        ),
    )
    parser.add_argument(
        "--traversal-backend",
        choices=["gpu", "cpu", "cpu_lazy"],
        default="gpu",
        help=(
            "Traversal verifier backend. 'gpu' is the existing fine-grained "
            "implementation; 'cpu' makes one bulk probability transfer and "
            "runs the sequential verifier with NumPy; 'cpu_lazy' additionally "
            "keeps residual rows unnormalised until their values are needed."
        ),
    )
    parser.add_argument(
        "--traversal-pinned-buffer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse pinned host memory for the CPU Traversal backend and copy "
            "target, draft and random values without a temporary GPU concat "
            "(enabled by default; use --no-traversal-pinned-buffer to disable)."
        ),
    )
    parser.add_argument(
        "--traversal-compiled-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use an optional Numba SIMD kernel for Traversal CPU residual "
            "updates when Numba is installed (enabled by default; falls back "
            "to NumPy, or disable with --no-traversal-compiled-residual)."
        ),
    )
    parser.add_argument(
        "--chat-template",
        default="auto",
        help=(
            "Prompt template. 'auto' uses tokenizer.chat_template when present "
            "and otherwise recognizes Vicuna from the tokenizer path. Pass a "
            "FastChat template name such as 'vicuna' to select it explicitly."
        ),
    )
    eagle_version = parser.add_mutually_exclusive_group()
    eagle_version.add_argument(
        "--use-eagle3",
        dest="use_eagle3",
        action="store_true",
        help=(
            "Load EAGLE3 weights (default). Pass --no-use-eagle3 for legacy "
            "EAGLE/EAGLE2 checkpoints such as EAGLE-Vicuna-7B-v1.3."
        ),
    )
    eagle_version.add_argument(
        "--no-use-eagle3",
        dest="use_eagle3",
        action="store_false",
        help="Load a legacy EAGLE/EAGLE2 checkpoint.",
    )
    parser.set_defaults(use_eagle3=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def model_input_device(model):
    # Correct for a normal single-GPU deployment and for the embedding shard of
    # an ``accelerate`` device map.
    return model.base_model.model.embed_tokens.weight.device


def build_prompt(tokenizer, messages, chat_template="auto"):
    """Render chat messages with a tokenizer or FastChat template."""
    tokenizer_template = getattr(tokenizer, "chat_template", None)
    if chat_template in {"auto", "tokenizer"} and tokenizer_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if chat_template == "tokenizer":
        raise ValueError(
            "--chat-template tokenizer was requested, but the tokenizer has "
            "no chat_template. Use --chat-template vicuna for Vicuna models."
        )

    if chat_template == "auto":
        tokenizer_path = str(getattr(tokenizer, "name_or_path", "")).lower()
        if "vicuna" in tokenizer_path:
            chat_template = "vicuna"
        else:
            raise ValueError(
                "The tokenizer has no chat_template and its model family could "
                "not be inferred. Pass --chat-template explicitly (for example, "
                "--chat-template vicuna)."
            )

    # FastChat supplies the canonical Vicuna/Llama-2 prompt formats used by
    # the repository's original evaluation scripts.
    from fastchat.model import get_conversation_template

    conversation = get_conversation_template(chat_template)
    role_map = {
        "user": conversation.roles[0],
        "assistant": conversation.roles[1],
    }
    for message in messages:
        try:
            role = role_map[message["role"]]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported chat role {message.get('role')!r}; expected "
                "'user' or 'assistant'."
            ) from exc
        conversation.append_message(role, message["content"])
    if not messages or messages[-1]["role"] != "assistant":
        conversation.append_message(conversation.roles[1], None)
    return conversation.get_prompt()


def uses_llama3_eot(tokenizer):
    """Whether generation should also stop at Llama-3's end-of-turn token."""
    try:
        return "<|eot_id|>" in tokenizer.get_vocab()
    except (AttributeError, TypeError):
        return False


def load_questions(path, limit):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def run_generation(model, prompt, args, verify_method):
    tokenizer = model.get_tokenizer()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_input_device(model))
    prompt_tokens = input_ids.shape[1]
    is_llama3 = uses_llama3_eot(tokenizer)
    synchronize()
    started = time.perf_counter()
    if verify_method.lower() == "baseline":
        # Use EaModel's autoregressive path rather than Hugging Face
        # ``generate``.  This keeps the custom target model, KV cache,
        # tokenizer, dtype, device placement and sampling processor identical
        # to the speculative methods; only drafting/verification is skipped.
        output_ids, reported_new_tokens, loop_index = model.naivegenerate(
            input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            log=True,
            is_llama3=is_llama3,
        )
    else:
        output_ids, reported_new_tokens, loop_index = model.eagenerate(
            input_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            log=True,
            is_llama3=is_llama3,
            verify_method=verify_method,
            profile=getattr(args, "profile_phases", False),
        )
    synchronize()
    elapsed = time.perf_counter() - started
    phase_seconds = {
        "initial_tree_seconds": None,
        "target_decode_seconds": None,
        "verification_seconds": None,
        "update_and_draft_seconds": None,
        "profiled_seconds": None,
        "unattributed_seconds": None,
    }
    if (
            verify_method.lower() != "baseline"
            and getattr(args, "profile_phases", False)
    ):
        event_groups = getattr(model, "last_eagenerate_profile_events", None)
        if event_groups is not None:
            event_totals = {
                phase: sum(
                    start.elapsed_time(end) for start, end in pairs
                ) / 1000.0
                for phase, pairs in event_groups.items()
            }
            phase_seconds.update({
                "initial_tree_seconds": event_totals.get("initial_tree", 0.0),
                "target_decode_seconds": event_totals.get("target_decode", 0.0),
                "verification_seconds": event_totals.get("verification", 0.0),
                "update_and_draft_seconds": event_totals.get(
                    "update_and_draft", 0.0
                ),
            })
            phase_seconds["profiled_seconds"] = sum(event_totals.values())
            phase_seconds["unattributed_seconds"] = max(
                elapsed - phase_seconds["profiled_seconds"], 0.0
            )
    output_tokens = output_ids.shape[1] - prompt_tokens
    is_baseline = verify_method.lower() == "baseline"
    traversal_stats = {}
    if is_baseline:
        verification_rounds = None
        total_accept_length = None
        depth_accept=None
        total_accepted_draft_tokens = None
        average_accept_length = None
        average_accepted_draft_tokens = None
        acceptance_lengths = []
        accepted_draft_lengths = []
        if output_tokens != int(reported_new_tokens):
            raise RuntimeError(
                "Inconsistent baseline generation metrics: "
                f"output_tokens={output_tokens}, "
                f"reported_new_tokens={reported_new_tokens}."
            )
    else:
        generation_metrics = getattr(model, "last_eagenerate_metrics", {})
        verification_rounds = int(
            generation_metrics.get("verification_rounds", int(loop_index) + 1)
        )
        total_accept_length = int(
            generation_metrics.get("total_accept_length", reported_new_tokens)
        )
        total_accepted_draft_tokens = int(
            generation_metrics.get(
                "total_accepted_draft_tokens",
                max(total_accept_length - verification_rounds, 0),
            )
        )
        acceptance_lengths = [
            int(value) for value in generation_metrics.get("acceptance_lengths", [])
        ]
        accepted_draft_lengths = [
            int(value)
            for value in generation_metrics.get("accepted_draft_lengths", [])
        ]
        if not (
                output_tokens == int(reported_new_tokens) == total_accept_length
        ):
            raise RuntimeError(
                "Inconsistent generation metrics: "
                f"output_tokens={output_tokens}, "
                f"reported_new_tokens={reported_new_tokens}, "
                f"total_accept_length={total_accept_length}."
            )
        if acceptance_lengths and len(acceptance_lengths) != verification_rounds:
            raise RuntimeError(
                "One acceptance length is required per verification round: "
                f"lengths={len(acceptance_lengths)}, rounds={verification_rounds}."
            )
        average_accept_length = (
            total_accept_length / verification_rounds if verification_rounds else 0.0
        )
        average_accepted_draft_tokens = (
            total_accepted_draft_tokens / verification_rounds
            if verification_rounds
            else 0.0
        )
        depth_accept= generation_metrics.get("depth_accept", {})
        if verify_method.lower() in {"traversal", "traversal_verification"}:
            traversal_stats = generation_metrics.get("traversal_stats", {})
    text = tokenizer.decode(output_ids[0, prompt_tokens:], skip_special_tokens=True)
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "reported_new_tokens": int(reported_new_tokens),
        "verification_rounds": verification_rounds,
        "total_accept_length": total_accept_length,
        "depth_accept":depth_accept,
        "total_accepted_draft_tokens": total_accepted_draft_tokens,
        "average_accept_length": average_accept_length,
        "average_accepted_draft_tokens": average_accepted_draft_tokens,
        "acceptance_lengths": acceptance_lengths,
        "accepted_draft_lengths": accepted_draft_lengths,
        "traversal_visited_nodes": traversal_stats.get("visited_nodes"),
        "traversal_rejected_nodes": traversal_stats.get("rejected_nodes"),
        "traversal_refresh_syncs": traversal_stats.get("refresh_syncs"),
        "traversal_refreshed_edges": traversal_stats.get("refreshed_edges"),
        "traversal_compiled_residuals": traversal_stats.get(
            "compiled_residuals"
        ),
        "traversal_cpu_bulk_transfer_seconds": traversal_stats.get(
            "cpu_bulk_transfer_seconds"
        ),
        "traversal_cpu_algorithm_seconds": traversal_stats.get(
            "cpu_algorithm_seconds"
        ),
        "seconds": elapsed,
        "tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
        "tokens_per_round": (
            output_tokens / verification_rounds
            if verification_rounds is not None and verification_rounds > 0
            else None
        ),
        **phase_seconds,
        "output": text,
    }


SUMMARY_COLUMNS = [
    "verify_method",
    "samples",
    "output_tokens",
    "verification_rounds",
    "total_accept_length",
    "total_accepted_draft_tokens",
    "average_accept_length",
    "average_accepted_draft_tokens",
    "seconds",
    "milliseconds_per_round",
    "initial_tree_seconds",
    "target_decode_ms_per_round",
    "verification_ms_per_round",
    "update_and_draft_ms_per_round",
    "unattributed_ms_per_round",
    "traversal_visited_nodes_per_round",
    "traversal_rejected_nodes_per_round",
    "traversal_refresh_syncs_per_round",
    "traversal_refreshed_edges_per_round",
    "traversal_compiled_residuals_per_round",
    "traversal_cpu_bulk_transfer_ms_per_round",
    "traversal_cpu_algorithm_ms_per_round",
    "tokens_per_second",
    "tokens_per_round",
]


def summarize(records, group_field=None):
    """Aggregate the existing benchmark metrics by method and optional field."""
    summaries = []
    group_values = [None] if group_field is None else sorted(
        {record[group_field] for record in records}
    )
    for group_value in group_values:
        for method in sorted({record["verify_method"] for record in records}):
            group = [
                record for record in records
                if record["verify_method"] == method
                and (group_field is None or record[group_field] == group_value)
            ]
            if not group:
                continue
            total_tokens = sum(record["output_tokens"] for record in group)
            total_seconds = sum(record["seconds"] for record in group)
            has_acceptance_metrics = all(
                record["verification_rounds"] is not None for record in group
            )
            if has_acceptance_metrics:
                total_rounds = sum(record["verification_rounds"] for record in group)
                total_accept_length = sum(
                    record["total_accept_length"] for record in group
                )
                total_accepted_draft_tokens = sum(
                    record["total_accepted_draft_tokens"] for record in group
                )
                average_accept_length = (
                    total_accept_length / total_rounds if total_rounds else 0.0
                )
                average_accepted_draft_tokens = (
                    total_accepted_draft_tokens / total_rounds
                    if total_rounds
                    else 0.0
                )
                # Use the same round-weighted definition as acceptance
                # length.  Averaging each sample's tokens/round gives short
                # samples the same weight as long samples and can reverse the
                # apparent method ordering.
                tokens_per_round = (
                    total_tokens / total_rounds if total_rounds else 0.0
                )
                milliseconds_per_round = (
                    total_seconds / total_rounds * 1000
                    if total_rounds else 0.0
                )
            else:
                total_rounds = None
                total_accept_length = None
                total_accepted_draft_tokens = None
                average_accept_length = None
                average_accepted_draft_tokens = None
                tokens_per_round = None
                milliseconds_per_round = None
            row = {
                "verify_method": method,
                "samples": len(group),
                "output_tokens": total_tokens,
                "verification_rounds": total_rounds,
                "total_accept_length": total_accept_length,
                "total_accepted_draft_tokens": total_accepted_draft_tokens,
                "average_accept_length": average_accept_length,
                "average_accepted_draft_tokens": average_accepted_draft_tokens,
                "seconds": total_seconds,
                "milliseconds_per_round": milliseconds_per_round,
                "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
                "tokens_per_round": tokens_per_round,
            }
            has_phase_profile = all(
                record.get("verification_seconds") is not None
                for record in group
            )
            if has_phase_profile and total_rounds:
                initial_tree_seconds = sum(
                    record["initial_tree_seconds"] for record in group
                )
                target_decode_seconds = sum(
                    record["target_decode_seconds"] for record in group
                )
                verification_seconds = sum(
                    record["verification_seconds"] for record in group
                )
                update_and_draft_seconds = sum(
                    record["update_and_draft_seconds"] for record in group
                )
                unattributed_seconds = sum(
                    record["unattributed_seconds"] for record in group
                )
                row.update({
                    "initial_tree_seconds": initial_tree_seconds,
                    "target_decode_ms_per_round": (
                        target_decode_seconds / total_rounds * 1000
                    ),
                    "verification_ms_per_round": (
                        verification_seconds / total_rounds * 1000
                    ),
                    "update_and_draft_ms_per_round": (
                        update_and_draft_seconds / total_rounds * 1000
                    ),
                    "unattributed_ms_per_round": (
                        unattributed_seconds / total_rounds * 1000
                    ),
                })
            else:
                row.update({
                    "initial_tree_seconds": None,
                    "target_decode_ms_per_round": None,
                    "verification_ms_per_round": None,
                    "update_and_draft_ms_per_round": None,
                    "unattributed_ms_per_round": None,
                })
            traversal_stat_names = (
                "visited_nodes", "rejected_nodes",
                "refresh_syncs", "refreshed_edges", "compiled_residuals",
            )
            has_traversal_stats = total_rounds and all(
                record.get(f"traversal_{name}") is not None
                for record in group for name in traversal_stat_names
            )
            for name in traversal_stat_names:
                column = f"traversal_{name}_per_round"
                row[column] = (
                    sum(record[f"traversal_{name}"] for record in group)
                    / total_rounds
                    if has_traversal_stats else None
                )
            cpu_timing_names = ("bulk_transfer", "algorithm")
            has_cpu_timing = total_rounds and all(
                record.get(f"traversal_cpu_{name}_seconds") is not None
                for record in group for name in cpu_timing_names
            )
            for name in cpu_timing_names:
                column = f"traversal_cpu_{name}_ms_per_round"
                row[column] = (
                    sum(
                        record[f"traversal_cpu_{name}_seconds"]
                        for record in group
                    ) / total_rounds * 1000
                    if has_cpu_timing else None
                )
            if group_field is not None:
                row[group_field] = group_value
            summaries.append(row)
    return summaries


def write_csv(rows, path, columns):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(records, path):
    """Write overall and per-category method comparisons in one CSV file."""
    overall = summarize(records)
    by_category = summarize(records, group_field="category")
    for row in overall:
        row["category"] = "summary_result"
    rows = [*overall, *by_category]
    write_csv(rows, path, ["category", *SUMMARY_COLUMNS])
    return rows


def main():
    args = parse_args()
    if args.temperature <= 1e-5 and {"RRSw", "traversal_verification"}.intersection(args.verify_methods):
        raise ValueError("RRSw and Traversal Verification require --temperature > 0.")
    probability_methods = {
        "rrsw", "traversal_verification", "greedy", "univer"
    }.intersection(method.lower() for method in args.verify_methods)
    if probability_methods and args.top_p < 1.0:
        print(
            "Benchmark note: UniVer Table 1/2 uses temperature=1.0 with a "
            "balanced 63-node binary tree and does not report top-p "
            "truncation. The current --top-p setting adds full-vocabulary "
            "sorting at every expanded parent and is not a like-for-like "
            "throughput reproduction. Use --temperature 1.0 --top-p 1.0 "
            "for the paper configuration."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.question_file, args.limit)
    if not questions:
        raise ValueError("No questions found in --question-file.")

    # Delay this import so ``--help`` and argument validation work even in an
    # environment where the EAGLE-compatible Transformers build is not yet set.
    from eagle.model.ea_model import EaModel

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        use_eagle3=args.use_eagle3,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.draft_top_k,
    )
    model.eval()
    model.compile_univer = args.compile_univer
    model.traversal_backend = args.traversal_backend
    model.traversal_pinned_buffer = args.traversal_pinned_buffer
    model.traversal_compiled_residual = args.traversal_compiled_residual
    tokenizer = model.get_tokenizer()

    # Warm up every selected path independently.  Warming only the first
    # method biases short benchmarks because baseline, tree generation and
    # the three verification kernels launch different CUDA operations.
    warmup_prompt = build_prompt(
        tokenizer,
        [{"role": "user", "content": "Hello."}],
        args.chat_template,
    )
    for verify_method in args.verify_methods:
        for warmup_index in range(args.warmup):
            set_seed(args.seed + warmup_index)
            run_generation(model, warmup_prompt, args, verify_method)

    records = []
    for question_index, question in enumerate(questions):
        messages = []
        turns = question["turns"] if args.turns == "all" else question["turns"][:1]
        for turn_index, user_text in enumerate(turns):
            messages.append({"role": "user", "content": user_text})
            prompt = build_prompt(tokenizer, messages, args.chat_template)
            turn_outputs = {}
            for method_index, verify_method in enumerate(args.verify_methods):
                # Same seed makes stochastic comparisons reproducible.  The
                # methods may still diverge after a different accept/reject path.
                set_seed(args.seed + question_index * 1000 + turn_index)
                result = run_generation(model, prompt, args, verify_method)
                result.update({
                    "question_id": question.get("question_id", question_index),
                    "category": question.get("category") or "uncategorized",
                    "turn": turn_index + 1,
                    "verify_method": verify_method,
                })
                records.append(result)
                turn_outputs[verify_method] = result["output"]
                print(
                    f"question={result['question_id']} turn={turn_index + 1} "
                    f"method={verify_method} tokens={result['output_tokens']} "
                    f"time={result['seconds']:.3f}s tok/s={result['tokens_per_second']:.2f}"
                )
            # Use the default response as conversation history for multi-turn
            # MT-Bench prompts, so both methods get the same next-turn prefix.
            if args.turns == "all":
                history_method = (
                    "default"
                    if "default" in turn_outputs
                    else "baseline"
                    if "baseline" in turn_outputs
                    else args.verify_methods[0]
                )
                messages.append(
                    {"role": "assistant", "content": turn_outputs[history_method]}
                )

    records_path = output_dir / f"records.jsonl"
    with open(records_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = write_summary(records, output_dir / "summary.csv")
    print("\nSummary")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
