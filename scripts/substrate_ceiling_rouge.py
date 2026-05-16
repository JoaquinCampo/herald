"""Multivariate substrate ceiling for `rouge_l_drop` (bar 3).

The univariate ceiling on `rouge_l_drop` from the per-token
substrate features (`future_sum_js_25` aggregated by max/p95/
mean/sum/auc) sits at ~0.77-0.85 depending on aggregation.

Bar 3 (rouge_l_drop ρ >= 0.85) failing at 0.7565 raises the
question: is that headroom in the wrapper or substrate cap?

This script computes the **multivariate** substrate ceiling:
the best ρ achievable from a joint HGB on all per-token-label
aggregations (max, p95, mean, sum, auc) with prompt_group OOF
validation. If multivariate ceiling is at the bar (~0.85), bar 3
is reachable. If well below, bar 3 is structurally capped.

Output: results/phase3/substrate_ceiling_rouge.json
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from herald.regression_metrics import (
    clustered_spearman_ci,
    safe_spearman,
)

LABEL = "future_sum_js_25"
TARGET = "rouge_l_drop"
AGGS = ("y_max", "y_p95", "y_mean", "y_sum", "y_auc")


def main() -> None:
    out_path = Path(
        "results/phase3/substrate_ceiling_rouge.json")

    tokens_path = Path(
        "results/phase2/dataset/phase2_tokens.parquet")
    run_damage_path = Path(
        "results/phase1/metrics/run_damage.parquet")

    print(f"[load] {tokens_path}")
    df = (
        pl.scan_parquet(tokens_path)
        .select(
            ["run_id", "prompt_id", "token_pos",
             "compression_ratio", LABEL])
        .drop_nulls(subset=[LABEL])
        .collect()
    )
    print(f"[load] {df.height:,} rows")

    agg = (
        df.sort(["run_id", "token_pos"])
        .group_by("run_id", maintain_order=True)
        .agg([
            pl.col(LABEL).max().alias("y_max"),
            pl.col(LABEL).quantile(0.95).alias("y_p95"),
            pl.col(LABEL).mean().alias("y_mean"),
            pl.col(LABEL).sum().alias("y_sum"),
            pl.col(LABEL).count().alias("n_tok"),
            pl.col("compression_ratio").first().alias("ratio"),
            pl.col("prompt_id").first().alias("prompt_id"),
        ])
        .with_columns(
            (pl.col("y_mean") * pl.col("n_tok")).alias("y_auc"),
        )
    )

    rd = pl.read_parquet(run_damage_path).select(
        ["run_id", TARGET])
    joined = agg.join(rd, on="run_id", how="inner").drop_nulls(
        subset=[TARGET])
    print(f"[join] {joined.height:,} runs with {TARGET}")

    y = joined[TARGET].to_numpy().astype(np.float64)
    prompt_arr = joined["prompt_id"].to_numpy()
    X = joined.select(list(AGGS)).to_numpy().astype(np.float32)

    print("[univariate ceilings]")
    uni = {}
    for a in AGGS:
        v = joined[a].to_numpy().astype(np.float64)
        rho = safe_spearman(v, y)
        uni[a] = float(rho)
        print(f"  {a}: ρ={rho:.4f}")

    print("\n[multivariate ceiling] GroupKFold(prompt_id), 5 folds")
    gkf = GroupKFold(n_splits=5)
    oof = np.full(X.shape[0], np.nan, dtype=np.float64)
    for k, (tr, te) in enumerate(
        gkf.split(X, y, groups=prompt_arr)
    ):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=400,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=20260515 + k,
        )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
        rho_te = safe_spearman(oof[te], y[te])
        print(f"  [fold {k}] n_tr={tr.size:,} n_te={te.size:,} "
              f"ρ_te={rho_te:.4f}")

    overall_ci = clustered_spearman_ci(
        oof, y, prompt_arr, n_boot=500, seed=20260515,
    )
    print(f"\n[multivariate] ρ={overall_ci['rho']:.4f} "
          f"[{overall_ci['lo']:.4f}, {overall_ci['hi']:.4f}]")

    out = {
        "target": TARGET,
        "label_source": LABEL,
        "univariate": uni,
        "multivariate_oracle": overall_ci,
        "n_runs": int(joined.height),
        "bar_3": 0.85,
        "passes_bar_3_under_oracle": bool(
            overall_ci["rho"] >= 0.85),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
