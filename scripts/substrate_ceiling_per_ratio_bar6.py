"""Per-ratio substrate ceiling for the bar 6 (`sum_js`) decision.

The oracle reachability test (`oracle_reachability_bar6.json`)
shows that with both per-token and wrapper layers in-distribution
for ratio, the heavy-compression slice ρ saturates at ~0.70
(retention 0.795 vs 0.95 bar). The remaining question: is the
oracle saturated at the substrate ceiling, or is there headroom?

This script computes, *per ratio*, the best Spearman achievable
by max/p95/mean/sum/auc aggregation of the TRUE label
`future_sum_js_25` against `run_damage.sum_js`. If at heavy
ratios the ceiling is itself ~0.70, the per-token regressor and
wrapper are already at substrate ceiling and bar 6 is provably
unreachable as currently defined. If the ceiling is ~0.90+, the
oracle has headroom and slice-specific work could close part of
the gap.

Output: `results/phase3/substrate_ceiling_per_ratio_bar6.json`
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

LABEL = "future_sum_js_25"
TARGET = "sum_js"
AGGS = ("y_max", "y_p95", "y_mean", "y_sum", "y_auc")


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    r = spearmanr(a[mask], b[mask]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def main() -> None:
    out_path = Path(
        "results/phase3/substrate_ceiling_per_ratio_bar6.json")

    tokens_path = Path(
        "results/phase2/dataset/phase2_tokens.parquet")
    run_damage_path = Path(
        "results/phase1/metrics/run_damage.parquet")

    print(f"[load] {tokens_path}")
    df = (
        pl.scan_parquet(tokens_path)
        .select(
            ["run_id", "token_pos", "compression_ratio", LABEL])
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
        ])
        .with_columns(
            (pl.col("y_mean") * pl.col("n_tok")).alias("y_auc"),
        )
    )
    print(f"[agg] {agg.height:,} runs")

    rd = pl.read_parquet(run_damage_path).select(
        ["run_id", TARGET])
    joined = agg.join(rd, on="run_id", how="inner").drop_nulls(
        subset=[TARGET])
    print(f"[join] {joined.height:,} runs with run_damage")

    ratios = sorted(joined["ratio"].unique().to_list())
    result: dict = {
        "label": LABEL,
        "target": TARGET,
        "aggs": list(AGGS),
        "per_ratio": {},
        "pooled": {},
    }

    for r in ratios:
        sub = joined.filter(pl.col("ratio") == r)
        y_true = sub[TARGET].to_numpy().astype(np.float64)
        per_agg = {}
        for a in AGGS:
            v = sub[a].to_numpy().astype(np.float64)
            per_agg[a] = _rho(v, y_true)
        best_agg = max(per_agg, key=lambda k: per_agg[k])
        result["per_ratio"][str(r)] = {
            "n_runs": int(sub.height),
            "by_agg": per_agg,
            "best_agg": best_agg,
            "best_rho": per_agg[best_agg],
        }
        print(f"[ratio={r:<7.4f}] best={best_agg} "
              f"ρ={per_agg[best_agg]:.4f}  "
              + "  ".join(
                  f"{a}={per_agg[a]:.3f}" for a in AGGS))

    y_pool = joined[TARGET].to_numpy().astype(np.float64)
    pooled_per_agg = {}
    for a in AGGS:
        v = joined[a].to_numpy().astype(np.float64)
        pooled_per_agg[a] = _rho(v, y_pool)
    best_pooled = max(pooled_per_agg, key=lambda k: pooled_per_agg[k])
    result["pooled"] = {
        "n_runs": int(joined.height),
        "by_agg": pooled_per_agg,
        "best_agg": best_pooled,
        "best_rho": pooled_per_agg[best_pooled],
    }
    print(f"\n[pooled] best={best_pooled} "
          f"ρ={pooled_per_agg[best_pooled]:.4f}")

    worst_ceiling = min(
        d["best_rho"] for d in result["per_ratio"].values())
    worst_ratio_for_ceiling = min(
        result["per_ratio"].items(),
        key=lambda kv: kv[1]["best_rho"])[0]
    result["worst_ratio_ceiling"] = float(worst_ceiling)
    result["worst_ratio"] = worst_ratio_for_ceiling
    print(f"[ceiling] worst per-ratio substrate ceiling: "
          f"ρ={worst_ceiling:.4f} at ratio={worst_ratio_for_ceiling}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
