"""Meta-only baseline for HERALD v1 wrapper.

Trains a per-run regressor on (compression_ratio, task one-hot,
n_tokens) ONLY, with leave-one-press-out CV. No per-token-pred
aggregates. Measures how much of the wrapper's headline ρ comes
purely from run-level metadata that a deployer would have at
inference, vs the per-token regressor's lift.

If meta-only ρ ≈ wrapper ρ, the per-token regressor is not
adding signal at the run level; the wrapper number is metadata.
If meta-only ρ << wrapper ρ, the wrapper's lift is real and the
per-token regressor is informative beyond metadata.
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
        default=Path("results/phase3/run_level_wrapper_meta_only.json"))
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    print(f"[load] preds: {args.preds_path}")
    df = pl.read_parquet(args.preds_path)

    feats_df = (
        df.group_by("run_id")
        .agg(
            pl.col("pred_raw").count().alias("n_tokens"),
            pl.col("press").first().alias("press"),
            pl.col("compression_ratio").first().alias("ratio"),
            pl.col("task").first().alias("task"),
            pl.col("prompt_id").first().alias("prompt_id"),
        )
    )
    print(f"[agg] {feats_df.height:,} runs")

    rd = pl.read_parquet(args.run_damage_path).select(
        ["run_id"] + list(RUN_TARGETS))
    joined = feats_df.join(rd, on="run_id", how="inner")
    print(f"[join] {joined.height:,} runs with run_damage")

    joined = _one_hot(joined, "task")
    task_cols = [
        c for c in joined.columns if c.startswith("task__")]

    feat_cols = ["n_tokens", "ratio"] + task_cols
    print(f"[features] {len(feat_cols)} META-ONLY feature columns: "
          "n_tokens, ratio, task one-hot")

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
        ece = ece_quantile(oof[mask], y[mask], n_bins=10)
        results[target] = {
            "n_runs": int(mask.sum()),
            "rho_meta_only": rho_ci,
            "ece_meta_only_vs_target": ece,
            "per_press": per_press,
        }
        print(f"[{target:<18s}] meta-only ρ={rho_ci['rho']:.4f} "
              f"[{rho_ci['lo']:.4f}, {rho_ci['hi']:.4f}]  "
              f"ECE={ece:.4f}")
        for press, p in per_press.items():
            print(f"    held={press:<20s} ρ={p['rho']:.4f}  "
                  f"n={p['n_runs']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
