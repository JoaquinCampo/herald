"""Apply the Phase 1 success criterion from gold/research-plan.md.

Phase 1 success criterion (verbatim from research-plan.md):

  "Intrinsic metrics significantly predict extrinsic damage under
   held-out prompts and remain directionally stable across at least 3
   of 4 tasks. Reported via Spearman correlation AND AUROC/AUPRC for
   intrinsic-to-extrinsic classification (e.g., does future_max_JS_H
   classify baseline_correct AND compressed_wrong; does trajectory NLL
   ratio classify diagnostic failures)."

Operationalization (pre-registered in this script, not tunable
post-hoc):

  - "Intrinsic metric" = the trajectory aggregates that survived Phase
    0 saturation (sum_kl, sum_js, nll_ratio).
  - "Extrinsic damage" = outcome harm (baseline_correct AND NOT
    compressed_correct).
  - "Significantly predict" = AUROC > 0.65 with bootstrap 95% CI lower
    bound > 0.5, on the per-task pooled sample (all presses + ratios in
    that task).
  - "Directionally stable" = Spearman ρ has the same sign in all
    qualifying tasks (we expect positive ρ between intrinsic damage and
    outcome harm).
  - The criterion passes if at least 3 of 4 tasks meet both bars for
    at least one intrinsic metric.

Inputs:
  - <root>/metrics/intrinsic_to_outcome_phase1.parquet (from
    scripts/build_phase1_alignment.py).

Outputs:
  - <root>/metrics/phase1_success_criterion.json.
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl

INTRINSIC_METRICS: tuple[str, ...] = ("sum_kl", "sum_js", "nll_ratio")
TASKS_EXPECTED: tuple[str, ...] = (
    "gsm8k",
    "humaneval",
    "ifeval",
    "longbench_single",
)
AUROC_BAR: float = 0.65


def _per_task_pooled(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Apply the criterion to the task-pooled rows.

    The alignment script writes one task-pooled row per (task, metric)
    where `press == "__pooled__"` and `compression_ratio == -1.0`.
    These rows aggregate every press and ratio for the task. We pick
    the best intrinsic metric per task and decide pass/fail.
    """
    pooled = df.filter(
        (pl.col("press") == "__pooled__")
        & (pl.col("compression_ratio") == -1.0)
        & (pl.col("task") != "__pooled__")
    )
    out_rows: list[dict[str, object]] = []
    for task in TASKS_EXPECTED:
        sub = pooled.filter(pl.col("task") == task)
        if sub.is_empty():
            out_rows.append(
                {
                    "task": task,
                    "metric": None,
                    "best_auroc": None,
                    "best_auroc_lo": None,
                    "passes": False,
                    "reason": "task missing from results",
                }
            )
            continue
        # Drop NaN AUROCs (cells with degenerate label distributions).
        sub = sub.filter(~pl.col("auroc_outcome_harm").is_nan())
        if sub.is_empty():
            out_rows.append(
                {
                    "task": task,
                    "metric": None,
                    "best_auroc": None,
                    "best_auroc_lo": None,
                    "passes": False,
                    "reason": (
                        "task-pooled AUROC undefined (no outcome harm "
                        "in any compressed cell with positives)"
                    ),
                }
            )
            continue
        best: dict[str, object] | None = None
        for metric in INTRINSIC_METRICS:
            mrows = sub.filter(pl.col("metric") == metric)
            if mrows.is_empty():
                continue
            row = mrows.row(0, named=True)
            cand = {
                "task": task,
                "metric": metric,
                "press": "task_pool",
                "compression_ratio": -1.0,
                "n": row["n"],
                "n_pos": row["n_pos"],
                "best_auroc": row["auroc_outcome_harm"],
                "best_auroc_lo": row["auroc_lo"],
                "best_auroc_hi": row["auroc_hi"],
                "best_spearman": row["spearman_vs_outcome_harm"],
                "passes": (
                    row["auroc_outcome_harm"] >= AUROC_BAR
                    and row["auroc_lo"] > 0.5
                ),
            }
            if (
                best is None
                or (cand["passes"] and not best["passes"])
                or (
                    cand["passes"]
                    and best["passes"]
                    and float(cand["best_auroc"]) > float(best["best_auroc"])
                )
            ):
                best = cand
        if best is None:
            out_rows.append(
                {
                    "task": task,
                    "metric": None,
                    "best_auroc": None,
                    "best_auroc_lo": None,
                    "passes": False,
                    "reason": "no intrinsic metric had a finite AUROC",
                }
            )
        else:
            out_rows.append(best)
    return pl.DataFrame(out_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    args = ap.parse_args()

    src = args.root / "metrics" / "intrinsic_to_outcome_phase1.parquet"
    if not src.exists():
        sys.exit(f"missing {src}; run scripts/build_phase1_alignment.py.")
    df = pl.read_parquet(src)

    summary = _per_task_pooled(df)
    n_pass = int(summary["passes"].sum())
    direction_ok = True
    if n_pass >= 3:
        signs = [
            float(r["best_spearman"])
            for r in summary.iter_rows(named=True)
            if r.get("passes") and r.get("best_spearman") is not None
        ]
        direction_ok = len({1 if s > 0 else -1 for s in signs}) <= 1

    verdict = {
        "n_tasks_passing": n_pass,
        "direction_ok": direction_ok,
        "criterion_pass": (n_pass >= 3 and direction_ok),
        "auroc_bar": AUROC_BAR,
        "per_task": summary.to_dicts(),
    }
    out_path = args.root / "metrics" / "phase1_success_criterion.json"
    out_path.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
