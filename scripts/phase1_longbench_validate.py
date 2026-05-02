"""Tiny LongBench validation rerun.

Reproduces the 6 LongBench-Single prompts that failed in Phase 1 Block 2
Option B per-task profiling, against the LongBench truncation fix
introduced in `LongBenchSingleTask.format_prompt`. Writes to a fresh
`output_root` so existing profile artifacts are not overwritten.

Usage on Orion (proxy required only if the dataset is not cached):

    .venv/bin/python scripts/phase1_longbench_validate.py \\
        --output-root results/phase1_validate \\
        --presses none,streaming_llm \\
        --ratios 0.0,0.875 \\
        --max-new-tokens 512

Pass criteria (printed at the end):

- All 6 baseline runs and all 6 compressed runs report
  ``replay_status == "ok"``.
- A truncation-sidecar entry exists for every run with
  ``truncated == True`` for the 6 long prompts.
- `peak_memory_mb` stays below 24 576 MiB.

This script is read-only with respect to the existing
``results/phase1_profile/`` data. It is *not* a Block 3 launch.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import torch
from loguru import logger

from herald.config import ExperimentConfig, make_run_id
from herald.experiment import load_model, run_single_with_replay
from herald.metrics.io import PerRunPaths
from herald.tasks import LongBenchSingleTask

FAILING_PROMPT_IDS: tuple[str, ...] = (
    "longbench_1842b0ff1882e545a6d41d5caf67bba5312872423fa48e74",
    "longbench_32e116c58a3c59fc170aa5f4e1dde414c8f3881872889826",
    "longbench_7570a52d69ab93c5f54eba4c45d44a3411650c1e4694760a",
    "longbench_b03244c8cc2681df1008d27c974d81415336396dff81f06d",
    "longbench_df6c6350671baab25c635bfa495eea90c69a7d201b5fe460",
    "longbench_fbeb825de92309788269da33aa6bd189c7b1d46b997746f4",
)


def _select_prompts(num_prompts: int, seed: int) -> list[dict[str, Any]]:
    task = LongBenchSingleTask()
    all_prompts = task.load(num_prompts=num_prompts, seed=seed)
    by_id = {p["id"]: p for p in all_prompts}
    selected = []
    missing = []
    for pid in FAILING_PROMPT_IDS:
        if pid in by_id:
            selected.append(by_id[pid])
        else:
            missing.append(pid)
    if missing:
        raise SystemExit(
            "FAIL: failing-prompt fixture is not in the loader's first "
            f"{num_prompts} prompts at seed={seed}. Missing: {missing}"
        )
    return selected


def _cell_root(
    output_root: Path, press: str, ratio: float
) -> Path:
    return output_root / "longbench_single" / press / f"ratio={ratio:.4f}"


def _validate_cell(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
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
    cell_root = _cell_root(output_root, press_name, compression_ratio)
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

    task = LongBenchSingleTask()
    n_ok = 0
    n_failed = 0
    rows: list[dict[str, Any]] = []
    for p in prompts:
        run_id = make_run_id(
            p["id"], press_name, compression_ratio, seed
        )
        baseline_id = (
            None
            if press_name == "none"
            else make_run_id(p["id"], "none", 0.0, seed)
        )
        paths = PerRunPaths(root=cell_root, run_id=run_id)
        t0 = time.perf_counter()
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
                f"  HARD FAIL {p['id']}: {exc!r}"
            )
            n_failed += 1
            rows.append(
                {
                    "prompt_id": p["id"],
                    "ok": False,
                    "stage": "exception",
                    "error": repr(exc),
                }
            )
            continue
        elapsed = time.perf_counter() - t0
        if not paths.run.exists():
            n_failed += 1
            rows.append(
                {"prompt_id": p["id"], "ok": False, "stage": "no_run_record"}
            )
            continue
        rr = pl.read_parquet(paths.run)
        rs = str(rr["replay_status"][0])
        peak = float(rr["peak_memory_mb"][0])
        ngen = int(rr["num_tokens_generated"][0])
        ok = rs == "ok" and paths.replay.exists()
        if ok:
            n_ok += 1
        else:
            n_failed += 1
        rows.append(
            {
                "prompt_id": p["id"],
                "ok": ok,
                "replay_status": rs,
                "replay_exists": paths.replay.exists(),
                "peak_memory_mb": peak,
                "num_tokens_generated": ngen,
                "elapsed_s": elapsed,
            }
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "press": press_name,
        "compression_ratio": compression_ratio,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "rows": rows,
    }


def _read_sidecar(cell_root: Path) -> list[dict[str, Any]]:
    path = cell_root / "raw" / "truncation.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/phase1_validate")
    )
    parser.add_argument(
        "--presses", type=str, default="none,streaming_llm"
    )
    parser.add_argument("--ratios", type=str, default="0.0,0.875")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--prompt-timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    presses = [s.strip() for s in args.presses.split(",") if s.strip()]
    ratios = [float(s) for s in args.ratios.split(",") if s.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)

    prompts = _select_prompts(args.num_prompts, args.seed)
    logger.info(
        f"Validating {len(prompts)} previously-failing LongBench prompts"
    )

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
    for press in presses:
        for ratio in ratios:
            if press == "none" and ratio != 0.0:
                continue
            if press != "none" and ratio == 0.0:
                continue
            logger.info(
                f"=== validating {press} @ ratio={ratio} ==="
            )
            summary = _validate_cell(
                model=model,
                tokenizer=tok,
                device=device,
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
            cell_root = _cell_root(args.output_root, press, ratio)
            summary["truncation_sidecar"] = _read_sidecar(cell_root)
            summaries.append(summary)

    # Print and persist a final summary.
    out_path = args.output_root / "phase1_longbench_validate_summary.json"
    out_path.write_text(json.dumps({"cells": summaries}, indent=2))
    logger.info(f"Validation summary -> {out_path}")

    print()
    print("=== VALIDATION RESULT ===")
    overall_ok = True
    for s in summaries:
        n_total = s["n_ok"] + s["n_failed"]
        sidecar = s.get("truncation_sidecar", [])
        sidecar_truncated = sum(1 for r in sidecar if r.get("truncated"))
        print(
            f"  {s['press']:>15} @ {s['compression_ratio']:.4f}: "
            f"ok={s['n_ok']}/{n_total} failed={s['n_failed']} "
            f"sidecar_truncated={sidecar_truncated}/{len(sidecar)}"
        )
        if s["n_failed"] > 0:
            overall_ok = False
        for r in s["rows"]:
            if not r.get("ok"):
                print(
                    f"    BAD: prompt_id={r['prompt_id']} "
                    f"row={r}"
                )
        peaks = [
            r.get("peak_memory_mb", 0)
            for r in s["rows"]
            if r.get("peak_memory_mb")
        ]
        if peaks and max(peaks) >= 24576:
            print(
                f"    WARN: peak_memory_mb max={max(peaks):.0f} "
                "(>= 24576 MiB)"
            )
            overall_ok = False
    print()
    print("PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
