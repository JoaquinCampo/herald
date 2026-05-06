"""Phase 2b Task 1: paired-bootstrap CIs on predictor-vs-baseline deltas.

Upgrades the Phase 2 point-estimate "lr_all_cheap beats entropy_mean_8 by
>= 0.05 AUROC on every (split, horizon)" claim into a CI-backed claim.

Method
------
For each (split_kind, horizon, fold):
  1. Threshold derived from training-fold rows of the continuous label.
  2. Refit `lr_all_cheap` and score `feature::entropy_mean_8` on the
     test fold using the existing `collect_fold_predictions` helper.
  3. Test rows are clustered by `run_id` (tokens within a run share
     within-run rolling features and many features depend on run-local
     temporal context — row-level bootstrap underestimates variance).

Bootstrap procedure (paired, two-stage):
  - Per replicate, for each fold: resample unique run_ids in the test
    set with replacement, gather all rows belonging to the resample
    (with multiplicity), and recompute AUROC for both scores. Take the
    fold delta. Cross-fold mean delta = mean over folds.
  - The CI is the percentile of cross-fold mean delta across B
    replicates.

Outputs
-------
- results/phase2/baselines/phase2_delta_ci.parquet
- results/phase2/baselines/phase2_delta_ci_summary.json

Each row in the parquet captures:
  split_kind, horizon, label, fold_id, n_test, n_groups,
  point_delta_fold, fold_lr_auroc, fold_entropy_auroc.

The summary JSON captures the cross-fold mean delta and CI per
(split_kind, horizon) cell, plus the pre-registered "passes >=0.05
with CI excluding zero" decision.
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl
from loguru import logger

from herald.predictor_baselines import (
    DEFAULT_QUANTILE,
    collect_fold_predictions,
    cross_fold_clustered_bootstrap,
    safe_auroc,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2/baselines"),
    )
    ap.add_argument(
        "--horizons", type=int, nargs="+", default=[5, 10, 25, 50]
    )
    ap.add_argument("--label-base", type=str, default="future_sum_js")
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["prompts", "ratios", "presses", "tasks"],
    )
    ap.add_argument("--n-prompt-folds", type=int, default=5)
    ap.add_argument("--threshold-q", type=float, default=DEFAULT_QUANTILE)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=200_000,
        help="Subsample for tractable LR fits (matches Phase 2 runner).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--target",
        type=str,
        default="lr_all_cheap",
        help="Baseline whose delta is being tested.",
    )
    ap.add_argument(
        "--reference",
        type=str,
        default="feature::entropy_mean_8",
        help="Baseline being compared against.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading {}", args.dataset)
    df = pl.read_parquet(args.dataset)
    logger.info("dataset rows={} cols={}", df.height, len(df.columns))
    if args.max_train_rows > 0 and df.height > args.max_train_rows:
        df = df.sample(n=args.max_train_rows, seed=args.seed, shuffle=True)
        logger.info("subsampled to {} rows", df.height)

    score_a_key = f"score::{args.target}"
    score_b_key = f"score::{args.reference}"

    fold_rows: list[dict[str, object]] = []
    summary: dict[str, dict[str, object]] = {}
    t0 = time.time()
    for h in args.horizons:
        label = f"{args.label_base}_{h}"
        if label not in df.columns:
            logger.warning("label {} not in dataset, skipping", label)
            continue
        logger.info("collecting predictions for label {}", label)
        preds = collect_fold_predictions(
            df,
            label_col=label,
            splits=tuple(args.splits),
            n_prompt_folds=args.n_prompt_folds,
            threshold_q=args.threshold_q,
            seed=args.seed,
            baselines_to_keep=(args.target, args.reference),
            group_col="run_id",
        )
        # Group by split_kind and compute cross-fold CI.
        by_split: dict[str, list[dict[str, object]]] = {}
        for fp in preds:
            kind = str(fp["split_kind"])
            if score_a_key not in fp or score_b_key not in fp:
                logger.warning(
                    "fold {} missing scores; skipping",
                    fp.get("fold_id"),
                )
                continue
            by_split.setdefault(kind, []).append(fp)
            fold_lr = safe_auroc(list(fp["y_true"]), list(fp[score_a_key]))
            fold_en = safe_auroc(list(fp["y_true"]), list(fp[score_b_key]))
            fold_rows.append(
                {
                    "split_kind": kind,
                    "horizon": h,
                    "label": label,
                    "fold_id": fp["fold_id"],
                    "n_test": int(fp["n_test"]),
                    "n_groups": int(len(set(fp["groups"].tolist()))),
                    "auroc_target": fold_lr,
                    "auroc_reference": fold_en,
                    "fold_delta": (
                        None
                        if fold_lr is None or fold_en is None
                        else round(fold_lr - fold_en, 4)
                    ),
                }
            )

        for kind, fps in by_split.items():
            inputs = [
                {
                    "y_true": fp["y_true"],
                    "score_a": fp[score_a_key],
                    "score_b": fp[score_b_key],
                    "groups": fp["groups"],
                }
                for fp in fps
            ]
            ci = cross_fold_clustered_bootstrap(
                inputs,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            cell_key = f"{kind}::H={h}"
            decision: str
            dlo = ci.get("delta_lo")
            dmean = ci.get("delta_mean")
            if dmean is None or dlo is None:
                decision = "ci_failed"
            elif dmean >= 0.05 and dlo > 0:
                decision = "passes_005_ci"
            elif dmean >= 0.05 and dlo <= 0:
                decision = "point_passes_ci_crosses_zero"
            elif dmean < 0.05 and dlo > 0:
                decision = "ci_excludes_zero_below_005"
            else:
                decision = "fails"
            summary[cell_key] = {
                **ci,
                "split_kind": kind,
                "horizon": h,
                "label": label,
                "target": args.target,
                "reference": args.reference,
                "decision": decision,
            }
            logger.info(
                "{}: delta={} CI=[{},{}] decision={}",
                cell_key,
                ci.get("delta_mean"),
                ci.get("delta_lo"),
                ci.get("delta_hi"),
                decision,
            )

    fold_path = args.output_dir / "phase2_delta_ci.parquet"
    pl.DataFrame(fold_rows).write_parquet(fold_path)
    summary_obj: dict[str, object] = {
        "target": args.target,
        "reference": args.reference,
        "label_base": args.label_base,
        "horizons": args.horizons,
        "splits": args.splits,
        "n_boot": args.n_boot,
        "n_prompt_folds": args.n_prompt_folds,
        "threshold_q": args.threshold_q,
        "max_train_rows": args.max_train_rows,
        "wall_seconds": round(time.time() - t0, 2),
        "cells": summary,
    }
    summary_path = args.output_dir / "phase2_delta_ci_summary.json"
    summary_path.write_text(json.dumps(summary_obj, indent=2, default=str))

    print()
    print("=== Phase 2b paired-bootstrap CI summary ===")
    print(f"target = {args.target}, reference = {args.reference}")
    print(f"label  = {args.label_base}_(H)")
    print()
    for k, v in sorted(summary.items()):
        print(
            f"  {k:30s}  delta={v['delta_mean']}  "
            f"CI=[{v['delta_lo']},{v['delta_hi']}]  "
            f"n_folds={v['n_folds']}  decision={v['decision']}"
        )
    print()
    logger.info("results -> {}", fold_path)
    logger.info("summary -> {}", summary_path)


if __name__ == "__main__":
    main()
