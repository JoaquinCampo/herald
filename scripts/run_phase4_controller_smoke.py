"""Phase 4 controller smoke runner (calibration-aware).

Supports two modes:
  default mode (single-threshold)  : matches the original 3-prompt smoke
  calibration mode (--thresholds)  : runs HERALD at each candidate
                                     threshold + a fresh matched-random
                                     baseline per threshold

Layout per prompt:
  fixed_{b} for b in budgets                                  (once)
  herald_t<T> for T in thresholds                             (per T)
  random_t<T> for T in thresholds (replays herald_t<T>)       (per T)

Outputs under --output-dir:
  controller_runs.parquet      one row per run (with threshold col)
  controller_segments.parquet  one row per (run, segment)
  controller_events.parquet    one row per compress fire
  controller_summary.json      aggregate summary
  online_distribution.json     fixed_64 segment-score quantiles
  drift_table.json             expected (offline) vs observed (online)
                               relax-rate per threshold

Run with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 (no HF Hub access).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from herald.phase4_controller import (
    DEFAULT_BUDGETS,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
    RUN_COLUMNS,
    SEGMENT_COLUMNS,
    ControllerRun,
    FixedBudgetPolicy,
    RandomMatchedBudgetPolicy,
    RiskBudgetStepPolicy,
    attach_cache_size_per_segment,
    run_controller_once,
)
from herald.phase4_feasibility import (
    EVENTS_COLUMNS,
    write_parquet,
)
from herald.phase4_predictor import ExportedPredictor

# Extends RUN_COLUMNS with a `threshold` slot (where applicable).
RUN_COLUMNS_EXT: tuple[str, ...] = ("threshold",) + RUN_COLUMNS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictor",
        type=Path,
        default=Path("models/phase4_lr_all_cheap.json"),
    )
    ap.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct"
    )
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS)
    )
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Calibration sweep over multiple thresholds. If omitted, "
            "uses the single --threshold value (legacy behavior)."
        ),
    )
    ap.add_argument(
        "--offline-threshold-table",
        type=Path,
        default=None,
        help=(
            "Path to threshold_table.json from the offline calibration "
            "(scripts/phase4_calibration/offline_segment_distribution.py). "
            "If provided, drift_table.json compares expected (offline) "
            "vs observed (online) relax rates per threshold."
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-seed", type=int, default=7)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4/controller_smoke"),
    )
    return ap.parse_args(argv)


def _build_press_fn(model: Any) -> Any:
    """Returns a closure (target_size, interval) -> DecodingPress."""
    from kvpress import (  # noqa: PLC0415
        DecodingPress,
        KnormPress,
    )

    def build(target_size: int, interval: int) -> Any:
        return DecodingPress(
            base_press=KnormPress(),
            target_size=int(target_size),
            compression_interval=int(interval),
        )

    return build


def _segment_rows(
    run: ControllerRun, k: int
) -> list[dict[str, Any]]:
    """Project ControllerRun.segments + cache attribution to rows."""
    base = attach_cache_size_per_segment(run, k=k)
    out = []
    for r in base:
        out.append(
            {
                "run_id": run.run_id,
                "policy": run.policy_name,
                **r,
            }
        )
    return out


def _run_row(run: ControllerRun, threshold: float) -> dict[str, Any]:
    n_relax = sum(1 for s in run.segments if s.decision == "relax")
    n_tighten = sum(1 for s in run.segments if s.decision == "tighten")
    n_keep = sum(1 for s in run.segments if s.decision == "keep")
    return {
        "run_id": run.run_id,
        "prompt_id": run.prompt_id,
        "policy": run.policy_name,
        "threshold": float(threshold),
        "initial_budget": run.initial_budget,
        "num_tokens_generated": run.num_tokens_generated,
        "stop_reason": run.stop_reason,
        "wall_clock_seconds": run.wall_clock_seconds,
        "wall_clock_per_token": run.wall_clock_per_token,
        "peak_memory_mb": run.peak_memory_mb,
        "compression_event_count": run.compression_event_count,
        "decode_event_count": run.decode_event_count,
        "total_evicted_tokens": run.total_evicted_tokens,
        "n_segments": len(run.segments),
        "n_relax": n_relax,
        "n_tighten": n_tighten,
        "n_keep": n_keep,
        "task_score": (
            float(run.correct) if run.correct is not None else None
        ),
        "correct": run.correct,
        "predicted_answer": run.predicted_answer,
        "ground_truth": run.ground_truth,
        "generated_text": run.generated_text,
    }


def _event_rows(run: ControllerRun) -> list[dict[str, Any]]:
    out = []
    for ev in run.events:
        out.append(
            {
                "run_id": run.run_id,
                "step_idx": ev.step_idx,
                "layer_idx": ev.layer_idx,
                "event_type": ev.event_type,
                "retained_cache_len_before": ev.retained_cache_len_before,
                "retained_cache_len_after": ev.retained_cache_len_after,
                "target_size": ev.target_size,
                "threshold": ev.threshold,
                "wall_clock_seconds": ev.wall_clock_seconds,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.predictor.exists():
        print(f"predictor not found: {args.predictor}", file=sys.stderr)
        return 2

    print(f"loading predictor {args.predictor}...")
    predictor = ExportedPredictor.from_path(args.predictor)

    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from herald.prompts import load_gsm8k  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("controller smoke requires CUDA", file=sys.stderr)
        return 2
    device = "cuda"

    prompts = load_gsm8k(args.num_prompts, seed=args.seed)
    print(f"loaded {len(prompts)} prompts")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device)  # type: ignore[arg-type]
    model.eval()

    build_press = _build_press_fn(model)
    budgets = tuple(int(b) for b in sorted(args.budgets))
    thresholds = (
        tuple(float(t) for t in args.thresholds)
        if args.thresholds
        else (float(args.threshold),)
    )
    print(f"thresholds: {list(thresholds)}")

    def _count(run: ControllerRun, d: str) -> int:
        return sum(1 for s in run.segments if s.decision == d)

    def _t_tag(t: float) -> str:
        return f"{t:.4f}".rstrip("0").rstrip(".") or "0"

    runs: list[tuple[ControllerRun, float]] = []
    started_at = time.time()

    for prompt in prompts:
        # Fixed baselines, run once per prompt regardless of #thresholds.
        for b in budgets:
            policy = FixedBudgetPolicy(budget=b, name=f"fixed_{b}")
            run = run_controller_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                predictor=predictor,
                policy=policy,
                build_press_fn=build_press,
                initial_budget=b,
                compression_interval=args.k,
                max_new_tokens=args.max_new_tokens,
                device=device,
                k=args.k,
                threshold=float("nan"),
            )
            runs.append((run, float("nan")))
            print(
                f"  fixed_{b} {prompt['id']}: "
                f"tokens={run.num_tokens_generated} "
                f"wct/tok={run.wall_clock_per_token:.4f}s "
                f"correct={run.correct}"
            )

        # Per-threshold HERALD + matched-random replays.
        for t in thresholds:
            tag = _t_tag(t)
            herald_policy = RiskBudgetStepPolicy(
                budgets=budgets,
                threshold=t,
                start_index=0,
                name=f"herald_t{tag}",
            )
            herald_run = run_controller_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                predictor=predictor,
                policy=herald_policy,
                build_press_fn=build_press,
                initial_budget=herald_policy.initial_budget(),
                compression_interval=args.k,
                max_new_tokens=args.max_new_tokens,
                device=device,
                k=args.k,
                threshold=t,
            )
            runs.append((herald_run, t))
            print(
                f"  herald_t{tag} {prompt['id']}: "
                f"tokens={herald_run.num_tokens_generated} "
                f"wct/tok={herald_run.wall_clock_per_token:.4f}s "
                f"correct={herald_run.correct} "
                f"relax/tighten/keep="
                f"{_count(herald_run, 'relax')}/"
                f"{_count(herald_run, 'tighten')}/"
                f"{_count(herald_run, 'keep')}"
            )

            herald_actions = [herald_run.initial_budget] + [
                s.next_budget for s in herald_run.segments
            ]
            random_policy = RandomMatchedBudgetPolicy(
                actions=tuple(herald_actions),
                seed=args.random_seed,
                name=f"random_t{tag}",
            )
            random_run = run_controller_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                predictor=predictor,
                policy=random_policy,
                build_press_fn=build_press,
                initial_budget=random_policy.initial_budget(),
                compression_interval=args.k,
                max_new_tokens=args.max_new_tokens,
                device=device,
                k=args.k,
                threshold=t,
            )
            runs.append((random_run, t))
            print(
                f"  random_t{tag} {prompt['id']}: "
                f"tokens={random_run.num_tokens_generated} "
                f"wct/tok={random_run.wall_clock_per_token:.4f}s "
                f"correct={random_run.correct}"
            )

    # Emit artifacts.
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_rows = [_run_row(r, t) for (r, t) in runs]
    seg_rows: list[dict[str, Any]] = []
    ev_rows: list[dict[str, Any]] = []
    for r, _t in runs:
        seg_rows.extend(_segment_rows(r, k=args.k))
        ev_rows.extend(_event_rows(r))

    write_parquet(
        out_dir, "controller_runs", runs_rows, RUN_COLUMNS_EXT
    )
    write_parquet(
        out_dir, "controller_segments", seg_rows, SEGMENT_COLUMNS
    )
    write_parquet(
        out_dir, "controller_events", ev_rows, EVENTS_COLUMNS
    )

    # Online distribution from fixed_64 segment scores.
    online_dist = _online_distribution(runs, target_policy="fixed_64")
    (out_dir / "online_distribution.json").write_text(
        json.dumps(online_dist, indent=2)
    )

    # Drift table: expected (offline) vs observed (online) relax rate.
    drift = _drift_table(
        runs=runs,
        thresholds=thresholds,
        offline_table_path=args.offline_threshold_table,
    )
    (out_dir / "drift_table.json").write_text(
        json.dumps(drift, indent=2)
    )

    summary = {
        "predictor_path": str(args.predictor),
        "model": args.model,
        "num_prompts": len(prompts),
        "budgets": list(budgets),
        "k": args.k,
        "threshold_default": args.threshold,
        "thresholds": list(thresholds),
        "max_new_tokens": args.max_new_tokens,
        "n_runs": len(runs),
        "n_segments": len(seg_rows),
        "n_events": len(ev_rows),
        "wall_clock_seconds_total": time.time() - started_at,
        "policies": sorted({r.policy_name for (r, _t) in runs}),
    }
    (out_dir / "controller_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        f"wrote artifacts to {out_dir} "
        f"(runs={len(runs)} segments={len(seg_rows)} events={len(ev_rows)})"
    )
    return 0


def _online_distribution(
    runs: list[tuple[ControllerRun, float]],
    target_policy: str,
) -> dict[str, Any]:
    """Quantile snapshot of segment scores from fixed_64 runs."""
    import numpy as np  # noqa: PLC0415

    scores: list[float] = []
    for run, _t in runs:
        if run.policy_name != target_policy:
            continue
        for s in run.segments:
            scores.append(float(s.segment_score_mean))
    if not scores:
        return {
            "source_policy": target_policy,
            "n_segments": 0,
            "note": "no segments observed",
        }
    arr = np.asarray(scores)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "source_policy": target_policy,
        "n_segments": int(arr.shape[0]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "percentiles": {
            f"p{p}": float(np.percentile(arr, p)) for p in pcts
        },
    }


def _drift_table(
    runs: list[tuple[ControllerRun, float]],
    thresholds: tuple[float, ...],
    offline_table_path: Path | None,
) -> list[dict[str, Any]]:
    """Per-threshold expected (offline) vs observed (online) relax rate."""
    expected_by_threshold: dict[float, float] = {}
    if offline_table_path is not None and offline_table_path.exists():
        offline = json.loads(offline_table_path.read_text())
        for row in offline:
            expected_by_threshold[float(row["threshold"])] = float(
                row["relax_rate"]
            )
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        herald_segs = [
            s
            for run, rt in runs
            for s in run.segments
            if rt == t and run.policy_name.startswith("herald_t")
        ]
        n_total = len(herald_segs)
        n_relax = sum(1 for s in herald_segs if s.decision == "relax")
        n_tighten = sum(
            1 for s in herald_segs if s.decision == "tighten"
        )
        n_keep = sum(1 for s in herald_segs if s.decision == "keep")
        observed_relax = (n_relax / n_total) if n_total else None
        expected_relax = _nearest_expected(
            t, expected_by_threshold
        )
        rows.append(
            {
                "threshold": float(t),
                "n_segments": int(n_total),
                "observed_relax_rate": observed_relax,
                "observed_tighten_rate": (
                    n_tighten / n_total if n_total else None
                ),
                "observed_keep_rate": (
                    n_keep / n_total if n_total else None
                ),
                "expected_offline_relax_rate": expected_relax,
                "drift": (
                    None
                    if (observed_relax is None or expected_relax is None)
                    else observed_relax - expected_relax
                ),
            }
        )
    return rows


def _nearest_expected(
    t: float, expected: dict[float, float]
) -> float | None:
    """Lookup expected relax rate for threshold t (exact or nearest)."""
    if not expected:
        return None
    if t in expected:
        return expected[t]
    keys = sorted(expected.keys())
    nearest = min(keys, key=lambda k: abs(k - t))
    if abs(nearest - t) <= 1e-6:
        return expected[nearest]
    return None


if __name__ == "__main__":
    sys.exit(main())
