"""Phase 2 baseline-first evaluation.

Reads `phase2_tokens.parquet` and runs the spec-required baselines
(random, position, ratio, metadata, single-feature thresholds,
logistic regression on Tier 0 + all cheap features) across four
held-out splits (prompts, ratios, presses, tasks) for the chosen
horizons.

NO XGBoost. The spec is explicit: do not train XGBoost until the
baseline table exists and is summarized.

Outputs:
- `phase2_baseline_results.parquet` — one row per
  (split_kind, fold_id, label, baseline)
- `phase2_baseline_summary.json` — per (split_kind, label): best
  baseline overall and best cheap baseline, plus paired delta of
  lr_tier0 / lr_all_cheap vs the best cheap baseline and a
  ≥ 0.05 AUROC pass/fail flag
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl
from loguru import logger

from herald.predictor_baselines import (
    DEFAULT_QUANTILE,
    evaluate_split,
    summarize_baselines,
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
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50],
    )
    ap.add_argument(
        "--label-bases",
        type=str,
        nargs="+",
        default=["future_sum_js"],
        help="Continuous label families. Each is paired with each horizon.",
    )
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
        default=300_000,
        help="Subsample size for tractable LR training.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading {}", args.dataset)
    df = pl.read_parquet(args.dataset)
    logger.info("dataset rows={} cols={}", df.height, len(df.columns))

    if args.max_train_rows > 0 and df.height > args.max_train_rows:
        sampled = df.sample(
            n=min(args.max_train_rows, df.height),
            seed=args.seed,
            shuffle=True,
        )
        logger.info(
            "subsampled to {} rows for tractable evaluation",
            sampled.height,
        )
        df = sampled

    all_rows: list[dict[str, object]] = []
    t0 = time.time()
    for base in args.label_bases:
        for h in args.horizons:
            label = f"{base}_{h}"
            if label not in df.columns:
                logger.warning("label {} not in dataset, skipping", label)
                continue
            logger.info("evaluating label={}", label)
            t_label = time.time()
            rows = evaluate_split(
                df,
                label_col=label,
                threshold_q=args.threshold_q,
                splits=tuple(args.splits),
                n_prompt_folds=args.n_prompt_folds,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            all_rows.extend(rows)
            logger.info(
                "  {} done in {:.1f}s ({} result rows)",
                label,
                time.time() - t_label,
                len(rows),
            )

    results_path = args.output_dir / "phase2_baseline_results.parquet"
    pl.DataFrame(all_rows).write_parquet(results_path)
    summary = summarize_baselines(all_rows)
    summary["wall_seconds"] = round(time.time() - t0, 2)
    summary["n_result_rows"] = len(all_rows)
    summary["dataset_path"] = str(args.dataset)
    summary["horizons"] = args.horizons
    summary["label_bases"] = args.label_bases
    summary["splits"] = args.splits
    summary["threshold_q"] = args.threshold_q
    summary["n_prompt_folds"] = args.n_prompt_folds
    summary["max_train_rows"] = args.max_train_rows
    summary_path = args.output_dir / "phase2_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info(
        "done: {} result rows in {:.1f}s",
        len(all_rows),
        time.time() - t0,
    )
    logger.info("results -> {}", results_path)
    logger.info("summary -> {}", summary_path)


if __name__ == "__main__":
    main()
