"""Run-level wrapper for HERALD v1: second-stage regressor on
`rouge_l_drop` (and other run-level targets).

Motivation: the per-token regressor is bounded by the substrate
ceiling (Spearman ≈ 0.78 vs `rouge_l_drop` from any aggregation of
`future_sum_js_H`). A run-level wrapper sees additional run-scope
context (press, ratio, sequence length, full distribution of per-
token predictions) and can push the ceiling — at the cost of
becoming a per-run damage estimator rather than a per-token signal.

This wrapper consumes OOF token-level predictions from a HERALD v1
prediction file and trains an HGB regressor on:

  - 8 aggregates of `pred_raw` per run (max, p95, p75, p50, mean,
    std, top3_mean, last5_mean, n_tokens).
  - 2 aggregates of `y_raw` (would be cheating; excluded here).
  - press (one-hot), compression_ratio, task (one-hot).

Targets: `rouge_l_drop`, `sum_js`, `sum_kl`, `char_edit_ratio`.

No-leakage protocol: when input is a `prompt_group` OOF file, runs
are grouped by `prompt_id`, GroupKFold(5) is reused at the wrapper
stage, and the wrapper for fold k is trained on aggregates from
folds {0..4}\\{k} only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from herald.regression_metrics import (
    clustered_spearman_ci,
    ece_quantile,
    safe_spearman,
)

RUN_TARGETS = ("rouge_l_drop", "sum_js", "sum_kl", "char_edit_ratio")


def _build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-run aggregate features from token-level OOF."""
    return (
        df.sort(["run_id", "token_pos"])
        .group_by("run_id")
        .agg(
            pl.col("pred_raw").max().alias("pred_max"),
            pl.col("pred_raw").quantile(0.95).alias("pred_p95"),
            pl.col("pred_raw").quantile(0.75).alias("pred_p75"),
            pl.col("pred_raw").quantile(0.50).alias("pred_p50"),
            pl.col("pred_raw").mean().alias("pred_mean"),
            pl.col("pred_raw").std().alias("pred_std"),
            pl.col("pred_raw").tail(5).mean().alias("pred_last5"),
            pl.col("pred_raw").sort(descending=True)
                .head(3).mean().alias("pred_top3"),
            pl.col("pred_raw").count().alias("n_tokens"),
            pl.col("press").first().alias("press"),
            pl.col("compression_ratio").first().alias("ratio"),
            pl.col("task").first().alias("task"),
            pl.col("prompt_id").first().alias("prompt_id"),
            pl.col("fold").first().alias("fold"),
        )
    )


def _one_hot(df: pl.DataFrame, col: str) -> pl.DataFrame:
    vals = sorted(df[col].unique().to_list())
    for v in vals:
        df = df.with_columns(
            (pl.col(col) == v).cast(pl.Float32).alias(f"{col}__{v}")
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds-path", type=Path,
        default=Path("results/phase3/preds/prompt_group__h25.parquet"))
    parser.add_argument(
        "--run-damage-path", type=Path,
        default=Path("results/phase1/metrics/run_damage.parquet"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/phase3/run_level_wrapper.json"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    print(f"[load] preds: {args.preds_path}")
    df = pl.read_parquet(args.preds_path)

    feats_df = _build_features(df)
    print(f"[agg] {feats_df.height:,} runs")

    rd = pl.read_parquet(args.run_damage_path).select(
        ["run_id"] + list(RUN_TARGETS))
    joined = feats_df.join(rd, on="run_id", how="inner")
    print(f"[join] {joined.height:,} runs with run_damage")

    joined = _one_hot(joined, "press")
    joined = _one_hot(joined, "task")

    feat_cols = [
        "pred_max", "pred_p95", "pred_p75", "pred_p50",
        "pred_mean", "pred_std", "pred_last5", "pred_top3",
        "n_tokens", "ratio",
    ] + [c for c in joined.columns
         if c.startswith("press__") or c.startswith("task__")]

    print(f"[features] {len(feat_cols)} feature columns")

    results: dict[str, dict] = {}
    for target in RUN_TARGETS:
        sub = joined.drop_nulls(subset=[target])
        # Realign indices after drop_nulls.
        groups_t = sub["prompt_id"].to_numpy()
        gkf_t = GroupKFold(n_splits=args.n_folds)
        splits_t = list(gkf_t.split(
            np.arange(sub.height), np.arange(sub.height),
            groups=groups_t))

        y = sub[target].to_numpy().astype(np.float64)
        y_log1p = np.log1p(np.maximum(y, 0.0))
        X = sub.select(feat_cols).to_numpy().astype(np.float32)

        oof = np.full(X.shape[0], np.nan, dtype=np.float64)
        for k, (tr, te) in enumerate(splits_t):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=args.max_iter,
                max_depth=4,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=args.seed + k,
            )
            model.fit(X[tr], y_log1p[tr])
            oof[te] = np.expm1(model.predict(X[te]))

        rho_ci = clustered_spearman_ci(
            oof, y, sub["prompt_id"].to_numpy(),
            n_boot=500, seed=args.seed,
        )
        rho_baseline = safe_spearman(
            sub["pred_max"].to_numpy(), y)
        ece = ece_quantile(oof, y, n_bins=10)
        results[target] = {
            "n_runs": int(sub.height),
            "rho_wrapper": rho_ci,
            "rho_baseline_pred_max": rho_baseline,
            "ece_wrapper_vs_target": ece,
        }
        print(f"[{target:<18s}] wrapper ρ={rho_ci['rho']:.4f} "
              f"[{rho_ci['lo']:.4f}, {rho_ci['hi']:.4f}]  "
              f"baseline pred_max ρ={rho_baseline:.4f}  "
              f"ECE={ece:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
