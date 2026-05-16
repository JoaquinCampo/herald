"""Phase 2c: predictor comparison with and without EWS features.

CPU-only.  Reuses the Phase 2 baseline runner with `extra_features`
set to the canonical EWS column list.  Outputs:

- results/phase2c_early_warning/ews_baselines.parquet  — per
  (split_kind, fold_id, label, baseline) AUROC + AUPRC
- results/phase2c_early_warning/ews_baselines_summary.json — per
  (split, label) best baseline and `lr_all_cheap_plus_extras` delta
- results/phase2c_early_warning/ews_delta_ci.parquet — cluster-
  bootstrap CI on the `lr_all_cheap_plus_extras minus lr_all_cheap`
  delta, per (split, horizon)
- results/phase2c_early_warning/ews_delta_ci_summary.json — same
  data as JSON

The cluster bootstrap follows Phase 2b: resample run_ids per fold,
recompute AUROC for both scores, take per-fold delta, average over
folds, take percentile CI across replicates.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from herald.early_warning_features import (
    DEFAULT_BASE_SIGNALS,
    DEFAULT_WINDOWS,
    NON_DEPLOYABLE_COLUMNS,
    ews_feature_names,
)
from herald.predictor_baselines import (
    DEFAULT_QUANTILE,
    collect_fold_predictions,
    cross_fold_clustered_bootstrap,
    evaluate_split,
    summarize_baselines,
)


def _ci_one_pair(
    folds: list[dict[str, np.ndarray]],
    score_a_key: str,
    score_b_key: str,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Cluster-bootstrap CI on AUROC(a) - AUROC(b) across folds."""
    pairs = []
    for fp in folds:
        if score_a_key not in fp or score_b_key not in fp:
            continue
        pairs.append(
            {
                "y_true": fp["y_true"],
                "score_a": fp[score_a_key],
                "score_b": fp[score_b_key],
                "groups": fp["groups"],
            }
        )
    if not pairs:
        return {
            "delta_mean": None,
            "delta_lo": None,
            "delta_hi": None,
            "n_folds": 0,
        }
    return cross_fold_clustered_bootstrap(pairs, n_boot=n_boot, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "results/phase2c_early_warning/phase2_dataset_ews.parquet"
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2c_early_warning"),
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
        default=200_000,
        help="Subsample size for tractable LR training.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ci-horizons",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50],
        help="Horizons to compute cluster-bootstrap CIs for.",
    )
    ap.add_argument(
        "--ci-n-boot",
        type=int,
        default=200,
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading {}", args.dataset)
    df = pl.read_parquet(args.dataset)
    logger.info("rows={} cols={}", df.height, len(df.columns))

    ews_cols = tuple(
        ews_feature_names(
            signals=DEFAULT_BASE_SIGNALS, windows=DEFAULT_WINDOWS
        )
    )
    present_ews = tuple(c for c in ews_cols if c in df.columns)
    missing_ews = [c for c in ews_cols if c not in df.columns]
    if missing_ews:
        logger.warning(
            "EWS missing from dataset: {} (proceeding with {} present)",
            missing_ews[:5],
            len(present_ews),
        )

    # Schema hygiene: refuse to run if any non-deployable column accidentally
    # appears in the configured extras.
    leak = set(present_ews) & NON_DEPLOYABLE_COLUMNS
    if leak:
        raise SystemExit(f"non-deployable columns in extras: {sorted(leak)}")

    if args.max_train_rows > 0 and df.height > args.max_train_rows:
        df = df.sample(n=args.max_train_rows, seed=args.seed, shuffle=True)
        logger.info("subsampled to {} rows", df.height)

    # ------------------- Baselines table -------------------
    all_rows: list[dict[str, object]] = []
    t0 = time.time()
    for base in args.label_bases:
        for h in args.horizons:
            label = f"{base}_{h}"
            if label not in df.columns:
                logger.warning("label {} missing, skipping", label)
                continue
            t_label = time.time()
            rows = evaluate_split(
                df,
                label_col=label,
                threshold_q=args.threshold_q,
                splits=tuple(args.splits),
                n_prompt_folds=args.n_prompt_folds,
                n_boot=args.n_boot,
                seed=args.seed,
                extra_features=present_ews,
            )
            all_rows.extend(rows)
            logger.info(
                "label={} folds_done in {:.1f}s ({} rows)",
                label,
                time.time() - t_label,
                len(rows),
            )

    results_path = args.output_dir / "ews_baselines.parquet"
    pl.DataFrame(all_rows).write_parquet(results_path)
    summary = summarize_baselines(all_rows)
    summary["wall_seconds"] = round(time.time() - t0, 2)
    summary["n_result_rows"] = len(all_rows)
    summary["dataset_path"] = str(args.dataset)
    summary["horizons"] = args.horizons
    summary["label_bases"] = args.label_bases
    summary["splits"] = args.splits
    summary["n_prompt_folds"] = args.n_prompt_folds
    summary["max_train_rows"] = args.max_train_rows
    summary["ews_features_used"] = list(present_ews)
    summary["ews_features_missing"] = missing_ews
    summary_path = args.output_dir / "ews_baselines_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("baselines -> {}", results_path)
    logger.info("summary   -> {}", summary_path)

    # ------------------- Cluster-bootstrap CIs -------------------
    logger.info("computing cluster-bootstrap CIs (run_id clusters)")
    ci_rows: list[dict[str, object]] = []
    for base in args.label_bases:
        for h in args.ci_horizons:
            label = f"{base}_{h}"
            if label not in df.columns:
                continue
            folds = collect_fold_predictions(
                df,
                label_col=label,
                splits=tuple(args.splits),
                n_prompt_folds=args.n_prompt_folds,
                threshold_q=args.threshold_q,
                seed=args.seed,
                baselines_to_keep=(
                    "lr_all_cheap",
                    "lr_all_cheap_plus_extras",
                    "feature::entropy_mean_8",
                ),
                extra_features=present_ews,
            )
            # Per split: collect folds for that split kind only.
            split_kinds = sorted({fp["split_kind"] for fp in folds})
            for sk in split_kinds:
                sk_folds = [fp for fp in folds if fp["split_kind"] == sk]
                # Pack into the format cross_fold_clustered_bootstrap expects.
                pairs_ews_vs_lr = []
                pairs_lr_vs_ent = []
                pairs_ews_vs_ent = []
                for fp in sk_folds:
                    if (
                        "score::lr_all_cheap" in fp
                        and "score::lr_all_cheap_plus_extras" in fp
                    ):
                        pairs_ews_vs_lr.append(
                            {
                                "y_true": fp["y_true"],
                                "score_a": fp[
                                    "score::lr_all_cheap_plus_extras"
                                ],
                                "score_b": fp["score::lr_all_cheap"],
                                "groups": fp["groups"],
                            }
                        )
                    if (
                        "score::lr_all_cheap" in fp
                        and "score::feature::entropy_mean_8" in fp
                    ):
                        pairs_lr_vs_ent.append(
                            {
                                "y_true": fp["y_true"],
                                "score_a": fp["score::lr_all_cheap"],
                                "score_b": fp[
                                    "score::feature::entropy_mean_8"
                                ],
                                "groups": fp["groups"],
                            }
                        )
                    if (
                        "score::lr_all_cheap_plus_extras" in fp
                        and "score::feature::entropy_mean_8" in fp
                    ):
                        pairs_ews_vs_ent.append(
                            {
                                "y_true": fp["y_true"],
                                "score_a": fp[
                                    "score::lr_all_cheap_plus_extras"
                                ],
                                "score_b": fp[
                                    "score::feature::entropy_mean_8"
                                ],
                                "groups": fp["groups"],
                            }
                        )
                ews_vs_lr = cross_fold_clustered_bootstrap(
                    pairs_ews_vs_lr,
                    n_boot=args.ci_n_boot,
                    seed=args.seed,
                )
                lr_vs_ent = cross_fold_clustered_bootstrap(
                    pairs_lr_vs_ent,
                    n_boot=args.ci_n_boot,
                    seed=args.seed,
                )
                ews_vs_ent = cross_fold_clustered_bootstrap(
                    pairs_ews_vs_ent,
                    n_boot=args.ci_n_boot,
                    seed=args.seed,
                )
                ci_rows.append(
                    {
                        "split_kind": sk,
                        "label": label,
                        "horizon": h,
                        "ews_vs_lr_delta_mean": ews_vs_lr.get("delta_mean"),
                        "ews_vs_lr_delta_lo": ews_vs_lr.get("delta_lo"),
                        "ews_vs_lr_delta_hi": ews_vs_lr.get("delta_hi"),
                        "ews_vs_lr_n_folds": ews_vs_lr.get("n_folds"),
                        "lr_vs_ent_delta_mean": lr_vs_ent.get("delta_mean"),
                        "lr_vs_ent_delta_lo": lr_vs_ent.get("delta_lo"),
                        "lr_vs_ent_delta_hi": lr_vs_ent.get("delta_hi"),
                        "ews_vs_ent_delta_mean": ews_vs_ent.get("delta_mean"),
                        "ews_vs_ent_delta_lo": ews_vs_ent.get("delta_lo"),
                        "ews_vs_ent_delta_hi": ews_vs_ent.get("delta_hi"),
                    }
                )
                logger.info(
                    "{} h={}: EWS-vs-LR Δ={} CI=[{},{}]",
                    sk,
                    h,
                    ews_vs_lr.get("delta_mean"),
                    ews_vs_lr.get("delta_lo"),
                    ews_vs_lr.get("delta_hi"),
                )

    ci_df = pl.DataFrame(ci_rows)
    ci_path = args.output_dir / "ews_delta_ci.parquet"
    ci_df.write_parquet(ci_path)
    ci_summary_path = args.output_dir / "ews_delta_ci_summary.json"
    ci_summary_path.write_text(
        json.dumps({"rows": ci_rows}, indent=2, default=str)
    )
    logger.info("CIs -> {}", ci_path)
    logger.info("done in {:.1f}s", time.time() - t0)


if __name__ == "__main__":
    main()
