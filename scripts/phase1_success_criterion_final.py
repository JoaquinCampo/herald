"""Apply the Phase 1 success criterion against the corrected labels.

Companion to ``scripts/phase1_success_criterion.py``: the operational
bar (AUROC ≥ 0.65, bootstrap CI lower bound > 0.5, ≥ 3 of 4 tasks,
direction stable) is unchanged. The only substitution is the extrinsic
label: this script reads
``metrics/intrinsic_to_outcome_phase1.final.parquet`` produced by
``scripts/build_phase1_alignment_final.py`` and uses the
``gross_harm_final`` rows.

Outputs ``metrics/phase1_success_criterion.final.json``. The original
``phase1_success_criterion.json`` (placeholder grader) is preserved.
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


def _per_task_pooled(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the criterion to the per-task pooled rows."""
    pooled = df.filter(
        (pl.col("press") == "__pooled__")
        & (pl.col("compression_ratio") == -1.0)
        & (pl.col("task") != "__pooled__")
        & (pl.col("label") == "gross_harm_final")
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
        sub = sub.filter(~pl.col("auroc").is_nan())
        if sub.is_empty():
            out_rows.append(
                {
                    "task": task,
                    "metric": None,
                    "best_auroc": None,
                    "best_auroc_lo": None,
                    "passes": False,
                    "reason": (
                        "task-pooled AUROC undefined (no gross_harm_final "
                        "positives in this task; label degenerate)"
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
                "best_auroc": row["auroc"],
                "best_auroc_lo": row["auroc_lo"],
                "best_auroc_hi": row["auroc_hi"],
                "best_spearman": row["spearman"],
                "passes": (
                    row["auroc"] >= AUROC_BAR and row["auroc_lo"] > 0.5
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

    src = args.root / "metrics" / "intrinsic_to_outcome_phase1.final.parquet"
    if not src.exists():
        sys.exit(
            f"missing {src}; run scripts/build_phase1_alignment_final.py."
        )
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
        "label": "gross_harm_final",
        "n_tasks_passing": n_pass,
        "direction_ok": direction_ok,
        "criterion_pass": (n_pass >= 3 and direction_ok),
        "auroc_bar": AUROC_BAR,
        "per_task": summary.to_dicts(),
        "note": (
            "Same pre-registered bar as phase1_success_criterion.json; only "
            "the extrinsic label substitutes (gross_harm_final from the "
            "post-hoc deterministic rescore). The original placeholder-"
            "grader verdict is preserved at phase1_success_criterion.json."
        ),
    }
    out_path = args.root / "metrics" / "phase1_success_criterion.final.json"
    out_path.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
