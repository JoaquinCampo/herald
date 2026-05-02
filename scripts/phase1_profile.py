"""Block 2 cost profiler.

Runs a small fixed-ratio slice on Orion to measure per-cell wall-clock
(generation), wall-clock (replay), peak GPU memory, generated-token
length, and per-run storage growth. Emits a per-cell summary JSON and
a `CostBudget` consumable by the Block 3 watchdog.

This script is the gate before any full Phase 1 sweep. The estimate
it writes is what Block 3 holds the sweep to (>25% drift triggers a
watchdog abort).

Usage on Orion (proxy env required for any new dataset; cached
datasets work without it):

    .venv/bin/python scripts/phase1_profile.py \
        --tasks gsm8k \
        --presses streaming_llm,snapkv,knorm,expected_attention,tova,random \
        --ratios 0.5,0.875,0.9375 \
        --num-prompts 50 \
        --max-new-tokens 512 \
        --output-root results/phase1_profile \
        --budget-out gold/phase-1-cost-budget.json

Per-task gates (advisor #6): repeat per task. Token budgets differ
wildly across tasks, so do not extrapolate one task's cell budget to
another.

The profiler reuses Phase 0's per-prompt orchestration and writes
under `output_root/<task>/<press>/<ratio>/`. Re-runs skip cells with
`replay_status == ok`.
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import torch
from loguru import logger

from herald.config import ExperimentConfig, make_run_id
from herald.cost_budget import CellBudget, CostBudget, dump_cost_budget
from herald.experiment import load_model, run_single_with_replay
from herald.metrics.io import PerRunPaths
from herald.tasks import PHASE1_TASKS, Task


def _task_by_name(name: str) -> Task:
    for t in PHASE1_TASKS:
        if t.name == name:
            return t
    raise ValueError(
        f"Unknown task {name!r}. Available: {[t.name for t in PHASE1_TASKS]}"
    )


def _maybe_skip(paths: PerRunPaths) -> bool:
    if not paths.all_exist():
        return False
    df = pl.read_parquet(paths.run)
    return df.height == 1 and df["replay_status"][0] == "ok"


def _cell_root(
    output_root: Path, task: str, press: str, ratio: float
) -> Path:
    return output_root / task / press / f"ratio={ratio:.4f}"


def _profile_cell(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    task: Task,
    prompts: list[dict[str, Any]],
    press_name: str,
    compression_ratio: float,
    output_root: Path,
    model_name: str,
    max_new_tokens: int,
    seed: int,
    prompt_timeout_seconds: float,
    top_k: int,
) -> dict[str, Any]:
    cell_root = _cell_root(
        output_root, task.name, press_name, compression_ratio
    )
    cell_root.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(
        model_name=model_name,
        press_name=press_name,
        compression_ratio=compression_ratio,
        num_prompts=len(prompts),
        seed=seed,
        output_dir=cell_root,
        max_new_tokens=max_new_tokens,
        prompt_timeout_seconds=prompt_timeout_seconds,
    )

    n_runs = 0
    n_failed = 0
    wall_per_token: list[float] = []
    replay_seconds: list[float] = []
    n_tokens_list: list[int] = []
    peak_mem: list[float] = []
    parquet_bytes_total = 0
    cell_t0 = time.perf_counter()

    for p in prompts:
        run_id = make_run_id(p["id"], press_name, compression_ratio, seed)
        baseline_id = (
            None
            if press_name == "none"
            else make_run_id(p["id"], "none", 0.0, seed)
        )
        paths = PerRunPaths(root=cell_root, run_id=run_id)
        if _maybe_skip(paths):
            logger.info(
                f"  skip {task.name}/{press_name}@{compression_ratio} "
                f"{p['id']} (cached)"
            )
        else:
            try:
                run_single_with_replay(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_data=p,
                    config=cfg,
                    baseline_run_id=baseline_id,
                    output_root=cell_root,
                    task=task,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"  FAILED {task.name}/{press_name}@"
                    f"{compression_ratio} {p['id']}: {exc!r}"
                )
                n_failed += 1
                continue
        if not paths.all_exist():
            n_failed += 1
            continue
        rr_df = pl.read_parquet(paths.run)
        n_runs += 1
        wpt = float(rr_df["wall_clock_per_token"][0])
        rwc = float(rr_df["replay_wall_clock_seconds"][0])
        ntok = int(rr_df["num_tokens_generated"][0])
        pmem = float(rr_df["peak_memory_mb"][0])
        if wpt == wpt and wpt > 0:  # NaN-safe finite check
            wall_per_token.append(wpt)
        if rwc == rwc and rwc >= 0:
            replay_seconds.append(rwc)
        n_tokens_list.append(ntok)
        if pmem == pmem and pmem > 0:
            peak_mem.append(pmem)
        for f in (paths.run, paths.tokens, paths.replay):
            parquet_bytes_total += f.stat().st_size

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cell_wall = time.perf_counter() - cell_t0
    summary = {
        "task": task.name,
        "press": press_name,
        "compression_ratio": float(compression_ratio),
        "n_runs": n_runs,
        "n_failed": n_failed,
        "wall_clock_per_token_median": (
            statistics.median(wall_per_token) if wall_per_token else None
        ),
        "wall_clock_per_token_p95": (
            _percentile(wall_per_token, 0.95) if wall_per_token else None
        ),
        "replay_wall_clock_seconds_median": (
            statistics.median(replay_seconds) if replay_seconds else None
        ),
        "n_tokens_generated_median": (
            statistics.median(n_tokens_list) if n_tokens_list else None
        ),
        "peak_memory_mb_median": (
            statistics.median(peak_mem) if peak_mem else None
        ),
        "cell_wall_clock_seconds": cell_wall,
        "parquet_bytes_total": parquet_bytes_total,
        "parquet_bytes_per_run": (
            parquet_bytes_total / n_runs if n_runs else 0
        ),
    }
    summary["replay_fraction"] = _replay_fraction(summary)
    return summary


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def _replay_fraction(summary: dict[str, Any]) -> float | None:
    """Replay time / (generation time + replay time) per run, median."""
    wpt = summary["wall_clock_per_token_median"]
    n = summary["n_tokens_generated_median"]
    rwc = summary["replay_wall_clock_seconds_median"]
    if wpt is None or n is None or rwc is None:
        return None
    gen = wpt * n
    denom = gen + rwc
    if denom <= 0:
        return None
    return rwc / denom


def _print_table(summaries: list[dict[str, Any]]) -> None:
    print()
    print("Per-cell profile (median across runs):")
    header = (
        f"{'task':<18} {'press':<22} {'ratio':>7} {'n':>4} "
        f"{'gen_s/tok':>10} {'replay_s':>10} {'tok':>6} "
        f"{'peak_MiB':>9} {'replay_frac':>11} {'KB/run':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        wpt = s["wall_clock_per_token_median"]
        rwc = s["replay_wall_clock_seconds_median"]
        rf = s["replay_fraction"]
        print(
            f"{s['task']:<18} {s['press']:<22} "
            f"{s['compression_ratio']:>7.4f} {s['n_runs']:>4} "
            f"{(wpt or 0):>10.4f} {(rwc or 0):>10.4f} "
            f"{int(s['n_tokens_generated_median'] or 0):>6} "
            f"{int(s['peak_memory_mb_median'] or 0):>9} "
            f"{(rf or 0):>11.3f} "
            f"{int(s['parquet_bytes_per_run'] / 1024):>8}"
        )


def _to_budget(
    summaries: list[dict[str, Any]],
    model_name: str,
    max_new_tokens: int,
    n_profile_runs: int,
) -> CostBudget:
    cells: dict[tuple[str, str, float], CellBudget] = {}
    for s in summaries:
        if not s["wall_clock_per_token_median"]:
            continue
        wpt = float(s["wall_clock_per_token_median"])
        ntok = float(s["n_tokens_generated_median"] or 0)
        cell = CellBudget(
            task=s["task"],
            press=s["press"],
            compression_ratio=round(s["compression_ratio"], 6),
            predicted_s_per_token=wpt,
            predicted_n_tokens=ntok,
            n_profile_runs=n_profile_runs,
        )
        cells[(cell.task, cell.press, cell.compression_ratio)] = cell
    return CostBudget(
        model=model_name,
        max_new_tokens=max_new_tokens,
        cells=cells,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1 per-cell cost profiler."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k",
        help="Comma-separated task names from herald.tasks.PHASE1_TASKS.",
    )
    parser.add_argument(
        "--presses",
        type=str,
        default="streaming_llm",
        help="Comma-separated press names (or 'none' for the baseline).",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.5,0.875,0.9375",
        help="Comma-separated compression ratios.",
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/phase1_profile")
    )
    parser.add_argument(
        "--budget-out",
        type=Path,
        default=Path("gold/phase-1-cost-budget.json"),
        help="Where to write the CostBudget JSON.",
    )
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--prompt-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help=(
            "Also profile press=none at compression_ratio=0 for each "
            "task. Required if no baseline runs exist yet under "
            "output_root."
        ),
    )
    args = parser.parse_args()

    task_names = [s.strip() for s in args.tasks.split(",") if s.strip()]
    press_names = [s.strip() for s in args.presses.split(",") if s.strip()]
    ratios = [float(s) for s in args.ratios.split(",") if s.strip()]
    tasks = [_task_by_name(n) for n in task_names]

    args.output_root.mkdir(parents=True, exist_ok=True)

    cfg_template = ExperimentConfig(
        model_name=args.model,
        press_name="none",
        compression_ratio=0.0,
        num_prompts=args.num_prompts,
        seed=args.seed,
        output_dir=args.output_root,
        max_new_tokens=args.max_new_tokens,
        prompt_timeout_seconds=args.prompt_timeout_seconds,
    )
    model, tok, device = load_model(cfg_template)

    summaries: list[dict[str, Any]] = []
    overall_t0 = time.perf_counter()
    for task in tasks:
        prompts = task.load(args.num_prompts, args.seed)
        if args.include_baseline:
            logger.info(
                f"=== {task.name} / press=none / ratio=0.0 (baseline) ==="
            )
            summaries.append(
                _profile_cell(
                    model=model,
                    tokenizer=tok,
                    device=device,
                    task=task,
                    prompts=prompts,
                    press_name="none",
                    compression_ratio=0.0,
                    output_root=args.output_root,
                    model_name=args.model,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    prompt_timeout_seconds=args.prompt_timeout_seconds,
                    top_k=args.top_k,
                )
            )
        for press in press_names:
            for ratio in ratios:
                logger.info(f"=== {task.name} / {press} @ {ratio} ===")
                summaries.append(
                    _profile_cell(
                        model=model,
                        tokenizer=tok,
                        device=device,
                        task=task,
                        prompts=prompts,
                        press_name=press,
                        compression_ratio=ratio,
                        output_root=args.output_root,
                        model_name=args.model,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                        prompt_timeout_seconds=args.prompt_timeout_seconds,
                        top_k=args.top_k,
                    )
                )
    total_wall = time.perf_counter() - overall_t0

    _print_table(summaries)

    summary_path = args.output_root / "phase1_profile_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "num_prompts": args.num_prompts,
                "seed": args.seed,
                "total_wall_clock_seconds": total_wall,
                "cells": summaries,
            },
            indent=2,
        )
    )
    logger.info(f"Profile summary -> {summary_path}")

    budget = _to_budget(
        summaries,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        n_profile_runs=args.num_prompts,
    )
    if budget.cells:
        dump_cost_budget(budget, args.budget_out)
        logger.info(f"Cost budget    -> {args.budget_out}")
    else:
        logger.warning(
            "No cells produced finite telemetry; skipping budget write."
        )

    print()
    print(f"Total wall-clock: {total_wall:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
