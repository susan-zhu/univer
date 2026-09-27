"""Benchmark standard autoregressive Hugging Face generation on JSONL prompts.

This intentionally does not import or use EAGLE.  Its JSONL/CSV metrics are
compatible with ``benchmark_verify_methods.py`` for throughput comparison.
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
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables top-k sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions.")
    parser.add_argument("--turns", choices=["first", "all"], default="first")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="Use AutoTokenizer(..., use_fast=False), matching EaModel's tokenizer.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_questions(path, limit):
    questions = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                questions.append(json.loads(line))
            if limit and len(questions) >= limit:
                break
    return questions


def build_prompt(tokenizer, messages):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The tokenizer has no chat template; use the Llama-3-Instruct tokenizer.")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_generation(model, tokenizer, prompt, args):
    device = model.get_input_embeddings().weight.device
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    model_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = model_inputs.input_ids.to(device)
    attention_mask = model_inputs.attention_mask.to(device)
    prompt_tokens = input_ids.shape[1]
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 1e-5:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
        if args.top_k > 0:
            generation_kwargs["top_k"] = args.top_k
    else:
        generation_kwargs["do_sample"] = False

    synchronize()
    started = time.perf_counter()
    output_ids = model.generate(input_ids,attention_mask=attention_mask, **generation_kwargs)
    synchronize()
    elapsed = time.perf_counter() - started
    output_tokens = output_ids.shape[1] - prompt_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "seconds": elapsed,
        "tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
        "output": tokenizer.decode(output_ids[0, prompt_tokens:], skip_special_tokens=True),
    }


def write_summary(records, path):
    """Write overall and per-category baseline metrics in one CSV file."""
    columns = [
        "category",
        "method",
        "samples",
        "output_tokens",
        "seconds",
        "tokens_per_second",
    ]
    rows = []
    categories = ["summary_result", *sorted({record["category"] for record in records})]
    for category in categories:
        group = records if category == "summary_result" else [
            record for record in records if record["category"] == category
        ]
        total_tokens = sum(record["output_tokens"] for record in group)
        total_seconds = sum(record["seconds"] for record in group)
        rows.append({
            "category": category,
            "method": "baseline",
            "samples": len(group),
            "output_tokens": total_tokens,
            "seconds": total_seconds,
            "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
        })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.question_file, args.limit)
    if not questions:
        raise ValueError("No questions found in --question-file.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        use_fast=not args.use_slow_tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    warmup_prompt = build_prompt(tokenizer, [{"role": "user", "content": "Hello."}])
    for index in range(args.warmup):
        set_seed(args.seed + index)
        run_generation(model, tokenizer, warmup_prompt, args)

    records = []
    for question_index, question in enumerate(questions):
        messages = []
        turns = question["turns"] if args.turns == "all" else question["turns"][:1]
        for turn_index, user_text in enumerate(turns):
            messages.append({"role": "user", "content": user_text})
            set_seed(args.seed + question_index * 1000 + turn_index)
            result = run_generation(model, tokenizer, build_prompt(tokenizer, messages), args)
            result.update({
                "question_id": question.get("question_id", question_index),
                "category": question.get("category") or "uncategorized",
                "turn": turn_index + 1,
                "method": "baseline",
            })
            records.append(result)
            print(
                f"question={result['question_id']} turn={result['turn']} "
                f"tokens={result['output_tokens']} time={result['seconds']:.3f}s "
                f"tok/s={result['tokens_per_second']:.2f}"
            )
            if args.turns == "all":
                messages.append({"role": "assistant", "content": result["output"]})

    with open(output_dir / "records_base.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = write_summary(records, output_dir / "summary_base.csv")
    print("\nSummary")
    for row in summary:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
