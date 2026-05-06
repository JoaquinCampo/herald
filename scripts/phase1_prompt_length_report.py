"""Phase 1 prompt-length percentile report (hard rule).

Reports chat-templated prompt-token-length percentiles per task,
**before** truncation. The Phase 1 validity gate is: "if more than a
small fraction of a task's prompts need truncation, that task/config
is not valid for Phase 1." This script is the data point that gate
relies on.

Reports `p50`, `p90`, `p95`, `max`, `n`, `n_over_budget`,
`fraction_over_budget` per task. Also prints the truncation budget
(`LONGBENCH_PROMPT_TOKEN_BUDGET`) and Qwen2.5-7B-Instruct's positional
limit so the operator can read the report against both bounds.

Runs on Orion (needs the HF tokenizer cached + dataset cached). The
model itself is *not* loaded — only the tokenizer.

Usage:

    .venv/bin/python scripts/phase1_prompt_length_report.py \\
        --num-prompts 50 --seed 42 \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --output gold/phase-1-prompt-length-report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from transformers import AutoTokenizer

from herald.prompts import format_chat
from herald.tasks import (
    LONGBENCH_PROMPT_TOKEN_BUDGET,
    PHASE1_TASKS,
    Task,
)


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _measure_task(
    task: Task,
    tokenizer: Any,
    num_prompts: int,
    seed: int,
    budget: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompts = task.load(num_prompts=num_prompts, seed=seed)
    chat_lengths: list[int] = []
    truncation_lengths: list[int] = []  # post-format_prompt
    over_budget = 0
    for p in prompts:
        # Pre-truncation chat-templated length: simulate what the
        # Phase 1 sweep would feed the model if `format_prompt` were
        # a no-op.
        messages = format_chat(
            p["question"], system_prompt=p.get("system_prompt")
        )
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        n_pre = len(
            tokenizer.encode(chat_text, add_special_tokens=False)
        )
        chat_lengths.append(n_pre)
        if n_pre + max_new_tokens > budget:
            over_budget += 1

        # Post-truncation length: what `format_prompt` would actually
        # produce. For non-LongBench tasks this is the same as
        # n_pre (default `format_prompt` is a no-op).
        question_text, _meta = task.format_prompt(
            p,
            tokenizer,
            system_prompt=p.get("system_prompt"),
            max_new_tokens=max_new_tokens,
        )
        messages_post = format_chat(
            question_text, system_prompt=p.get("system_prompt")
        )
        chat_post = tokenizer.apply_chat_template(
            messages_post, tokenize=False, add_generation_prompt=True
        )
        truncation_lengths.append(
            len(tokenizer.encode(chat_post, add_special_tokens=False))
        )

    return {
        "task": task.name,
        "n": len(chat_lengths),
        "pre_truncation_chat_tokens": {
            "p50": _percentile(chat_lengths, 0.50),
            "p90": _percentile(chat_lengths, 0.90),
            "p95": _percentile(chat_lengths, 0.95),
            "max": max(chat_lengths) if chat_lengths else 0,
        },
        "post_truncation_chat_tokens": {
            "p50": _percentile(truncation_lengths, 0.50),
            "p90": _percentile(truncation_lengths, 0.90),
            "p95": _percentile(truncation_lengths, 0.95),
            "max": (
                max(truncation_lengths) if truncation_lengths else 0
            ),
        },
        "n_over_budget": over_budget,
        "fraction_over_budget": (
            over_budget / len(chat_lengths) if chat_lengths else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--budget",
        type=int,
        default=LONGBENCH_PROMPT_TOKEN_BUDGET,
        help=(
            "Chat-templated prompt-token budget; flag a prompt as "
            "over-budget if `prompt_tokens + max_new_tokens > budget`."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gold/phase-1-prompt-length-report.json"),
    )
    parser.add_argument(
        "--max-fraction-over-budget",
        type=float,
        default=0.05,
        help=(
            "Fail (exit 2) if any task's fraction_over_budget exceeds "
            "this. Default 5%."
        ),
    )
    args = parser.parse_args()

    logger.info(f"Loading tokenizer {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)

    rows: list[dict[str, Any]] = []
    for task in PHASE1_TASKS:
        logger.info(f"=== {task.name} ===")
        rows.append(
            _measure_task(
                task,
                tok,
                args.num_prompts,
                args.seed,
                args.budget,
                args.max_new_tokens,
            )
        )

    out_payload = {
        "model": args.model,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "budget": args.budget,
        "qwen_positional_limit": 32768,
        "tasks": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_payload, indent=2))
    logger.info(f"Report -> {args.output}")

    print()
    print(
        f"{'task':<22} {'n':>4} "
        f"{'pre_p50':>8} {'pre_p90':>8} {'pre_p95':>8} {'pre_max':>8} "
        f"{'post_max':>8} {'>budget':>8} {'frac':>6}"
    )
    print("-" * 110)
    bad = []
    for r in rows:
        pre = r["pre_truncation_chat_tokens"]
        post = r["post_truncation_chat_tokens"]
        print(
            f"{r['task']:<22} {r['n']:>4} "
            f"{pre['p50']:>8} {pre['p90']:>8} "
            f"{pre['p95']:>8} {pre['max']:>8} "
            f"{post['max']:>8} {r['n_over_budget']:>8} "
            f"{r['fraction_over_budget']:>6.1%}"
        )
        if r["fraction_over_budget"] > args.max_fraction_over_budget:
            bad.append(r["task"])

    print()
    print(f"Budget: {args.budget} tokens (chat-templated).")
    print("Qwen2.5-7B-Instruct positional limit: 32 768.")
    if bad:
        print()
        print(
            "FAIL: tasks above the truncation-fraction gate "
            f"({args.max_fraction_over_budget:.0%}): {bad}"
        )
        print(
            "These tasks are not valid for Phase 1 with the current "
            "model/budget. Pick a different LongBench subtask, "
            "increase the budget, or switch model."
        )
        return 2
    print("PASS: every task fits the Phase 1 budget gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
