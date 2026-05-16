"""Press-agnostic run-level wrapper for HERALD v1.

Variant of `scripts/run_run_level_wrapper.py` that respects the
press-agnostic constraint from `gold/research-plan.md`:

  - No `press__*` one-hot features.
  - Cross-validation: leave-one-press-out (6 folds), not
    GroupKFold(prompt_id). The press appearing in the held-out
    fold has never been seen during training.

The task one-hot is optional; default keeps it because
cross-task transfer is a separate retention slice rather than a
hard agnosticism requirement, but `--drop-task-onehot` removes it.

Compare wrapper ρ here vs the in-distribution wrapper from
`run_run_level_wrapper.py` to quantify the metadata leakage in
the headline 0.94 sum_js number flagged by the user on
2026-05-15.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

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
        default=Path("results/phase3/run_level_wrapper_agnostic.json"))
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument(
        "--drop-task-onehot", action="store_true",
        help="Also drop task one-hot features (stricter).")
    parser.add_argument(
        "--strict", action="store_true",
        help="Strictest: keep only the 8 per-token-pred aggregates. "
             "Drop press one-hot (already dropped), task one-hot, "
             "compression_ratio, and n_tokens. This isolates how "
             "much of the wrapper's lift comes from per-token "
             "prediction quality vs run-level metadata.")
    args = parser.parse_args()

    print(f"[load] preds: {args.preds_path}")
    df = pl.read_parquet(args.preds_path)

    feats_df = _build_features(df)
    print(f"[agg] {feats_df.height:,} runs")

    rd = pl.read_parquet(args.run_damage_path).select(
        ["run_id"] + list(RUN_TARGETS))
    joined = feats_df.join(rd, on="run_id", how="inner")
    print(f"[join] {joined.height:,} runs with run_damage")

    # No press one-hot. Optionally drop task one-hot.
    drop_task = args.drop_task_onehot or args.strict
    if not drop_task:
        joined = _one_hot(joined, "task")
        task_cols = [
            c for c in joined.columns if c.startswith("task__")]
    else:
        task_cols = []

    pred_aggs = [
        "pred_max", "pred_p95", "pred_p75", "pred_p50",
        "pred_mean", "pred_std", "pred_last5", "pred_top3",
    ]
    if args.strict:
        feat_cols = pred_aggs
    else:
        feat_cols = pred_aggs + ["n_tokens", "ratio"] + task_cols

    print(f"[features] {len(feat_cols)} feature columns "
          f"(press one-hot dropped"
          + (", task one-hot dropped" if drop_task else "")
          + (", ratio + n_tokens dropped (STRICT)"
             if args.strict else "")
          + ")")

    presses = sorted(joined["press"].unique().to_list())
    print(f"[splits] leave-one-press-out over {len(presses)} presses")

    results: dict[str, dict] = {}
    for target in RUN_TARGETS:
        sub = joined.drop_nulls(subset=[target])
        y = sub[target].to_numpy().astype(np.float64)
        y_log1p = np.log1p(np.maximum(y, 0.0))
        X = sub.select(feat_cols).to_numpy().astype(np.float32)
        press_arr = sub["press"].to_numpy()
        prompt_arr = sub["prompt_id"].to_numpy()

        oof = np.full(X.shape[0], np.nan, dtype=np.float64)
        per_press: dict[str, dict] = {}
        for k, held in enumerate(presses):
            te = np.where(press_arr == held)[0]
            tr = np.where(press_arr != held)[0]
            if te.size < 10 or tr.size < 100:
                continue
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
            rho_held = safe_spearman(oof[te], y[te])
            per_press[held] = {
                "n_runs": int(te.size),
                "rho": float(rho_held),
            }

        mask = np.isfinite(oof)
        rho_ci = clustered_spearman_ci(
            oof[mask], y[mask], prompt_arr[mask],
            n_boot=500, seed=args.seed,
        )
        rho_baseline = safe_spearman(
            sub["pred_max"].to_numpy()[mask], y[mask])
        ece = ece_quantile(oof[mask], y[mask], n_bins=10)
        results[target] = {
            "n_runs": int(mask.sum()),
            "rho_wrapper": rho_ci,
            "rho_baseline_pred_max": rho_baseline,
            "ece_wrapper_vs_target": ece,
            "per_press": per_press,
        }
        print(f"[{target:<18s}] wrapper ρ={rho_ci['rho']:.4f} "
              f"[{rho_ci['lo']:.4f}, {rho_ci['hi']:.4f}]  "
              f"baseline pred_max ρ={rho_baseline:.4f}  "
              f"ECE={ece:.4f}")
        for press, p in per_press.items():
            print(f"    held={press:<20s} ρ={p['rho']:.4f}  "
                  f"n={p['n_runs']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
