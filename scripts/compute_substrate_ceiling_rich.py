"""Extended per-run substrate ceiling with richer aggregations.

The original ceiling (`compute_substrate_ceiling.py`) used only
max/p95/mean of `future_*_H` per run. This script adds five more
aggregations whose information content over a sequence is
qualitatively different from quantiles:

  - sum: total accumulated damage (telescoping caveat: for the
    `future_sum_js` family at large H this is correlated with
    `run_damage.sum_js` by construction). Reported for awareness.
  - auc: mean × n_tokens, an "area-under-curve" of the labelled
    trajectory.
  - dwell_p90: fraction of tokens with label above the 90th
    percentile of the same run's labels (always ~0.1, mostly
    constant; reported for sanity).
  - dwell_abs_q90: fraction of tokens above the 90th percentile
    computed from the *global* label distribution. Captures
    "how many high-damage moments per run".
  - peak_count: number of local maxima (centre exceeds both
    immediate neighbours) in the labelled trajectory.

The point is to check whether any of these lifts the ceiling for
`rouge_l_drop` beyond the 0.78 max ceiling reported in
`gold/phase-3-substrate-ceiling.md`. If yes, the per-token
regressor has more headroom than we thought; if no, the 0.85 bar
on rouge_l_drop is structurally unreachable, full stop.

Output:
  results/phase3_smoke/substrate_ceiling_rich.json
  gold/phase-3-substrate-ceiling-rich.md
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

HORIZONS = [5, 10, 25, 50]
LABEL_FAMILIES = ["future_sum_js", "future_sum_kl", "future_max_js"]
TARGETS = [
    "rouge_l_drop", "sum_js", "sum_kl", "char_edit_ratio",
]


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    r = spearmanr(a[mask], b[mask]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _aggregate_per_run(
    df: pl.DataFrame, col: str, global_q90: float
) -> pl.DataFrame:
    """Compute per-run aggregates of `col` from a token-level df."""
    return (
        df.sort(["run_id"])
        .group_by("run_id", maintain_order=True)
        .agg([
            pl.col(col).max().alias("y_max"),
            pl.col(col).quantile(0.95).alias("y_p95"),
            pl.col(col).quantile(0.90).alias("y_p90"),
            pl.col(col).mean().alias("y_mean"),
            pl.col(col).sum().alias("y_sum"),
            pl.col(col).count().alias("n_tok"),
            (pl.col(col) > global_q90).cast(pl.Float64)
                .mean().alias("y_dwell_abs_q90"),
        ])
        .with_columns(
            (pl.col("y_mean") * pl.col("n_tok")).alias("y_auc"),
        )
    )


def _count_local_maxima(series: list[float]) -> int:
    a = np.asarray(series, dtype=np.float64)
    if a.size < 3:
        return 0
    finite = np.isfinite(a)
    if not finite.all():
        a = np.where(finite, a, -np.inf)
    return int(np.sum((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])))


def main() -> None:
    out_path = Path("results/phase3_smoke/substrate_ceiling_rich.json")
    md_path = Path("gold/phase-3-substrate-ceiling-rich.md")

    label_cols = [f"{fam}_{h}"
                  for fam in LABEL_FAMILIES for h in HORIZONS]
    print(f"[load] labels: {label_cols}")
    t0 = time.time()
    df = (
        pl.scan_parquet(
            "results/phase2/dataset/phase2_tokens.parquet")
        .select(["run_id", "token_pos"] + label_cols)
        .collect()
    )
    print(f"[load] {df.height:,} rows in {time.time()-t0:.1f}s")

    rd = pl.read_parquet(
        "results/phase1/metrics/run_damage.parquet"
    ).select(["run_id"] + TARGETS)

    AGGS = (
        "y_max", "y_p95", "y_mean", "y_sum", "y_auc",
        "y_dwell_abs_q90", "y_peak_count",
    )

    rows: list[dict] = []
    for fam in LABEL_FAMILIES:
        for h in HORIZONS:
            col = f"{fam}_{h}"
            sub = df.select(["run_id", "token_pos", col]).drop_nulls(
                subset=[col])
            if sub.height < 10:
                continue
            # Global q90 of `col` for the dwell-absolute feature
            global_q90 = float(np.nanquantile(
                sub[col].to_numpy(), 0.90))
            agg_df = _aggregate_per_run(sub, col, global_q90)

            # Local-maxima count requires sorted-by-token_pos.
            sorted_df = sub.sort(["run_id", "token_pos"])
            peak_df = (
                sorted_df.group_by("run_id", maintain_order=True)
                .agg(pl.col(col).alias("vals"))
                .with_columns(
                    pl.col("vals").map_elements(
                        _count_local_maxima, return_dtype=pl.Int64
                    ).alias("y_peak_count")
                )
                .select(["run_id", "y_peak_count"])
            )
            agg_df = agg_df.join(peak_df, on="run_id", how="inner")

            joined = agg_df.join(rd, on="run_id", how="inner")
            for agg in AGGS:
                for tgt in TARGETS:
                    s = joined.drop_nulls(subset=[agg, tgt])
                    if s.height < 10:
                        rho = float("nan")
                    else:
                        rho = _rho(
                            s[agg].to_numpy(),
                            s[tgt].to_numpy(),
                        )
                    rows.append({
                        "label": col,
                        "family": fam,
                        "horizon": h,
                        "agg": agg,
                        "target": tgt,
                        "n_runs": int(s.height),
                        "rho": rho,
                    })
            print(f"  done {col} (global_q90={global_q90:.4f})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\n[done] wrote {out_path}")

    rdf = pl.DataFrame(rows)

    md = [
        "# HERALD v1 Rich Substrate Ceiling",
        "",
        "Extends `gold/phase-3-substrate-ceiling.md` with five "
        "aggregations beyond max/p95/mean: sum, AUC, dwell above "
        "the global q90, dwell above the per-run q90 (constant, "
        "informational only and excluded here), and local-peak "
        "count. The question this answers: does a richer "
        "aggregator of the TRUE label raise the per-run ceiling "
        "vs `rouge_l_drop`, `sum_js`, and friends?",
        "",
        "## Best ceiling per target across all (family, H, agg)",
        "",
    ]
    best = (
        rdf
        .group_by("target")
        .agg([
            pl.col("rho").max().alias("rho_best"),
            pl.struct(["family", "horizon", "agg"]).filter(
                pl.col("rho") == pl.col("rho").max()
            ).first().alias("best_combo"),
        ])
        .sort("target")
    )
    md.append("| target | best ρ | family | horizon | agg |")
    md.append("|---|---|---|---|---|")
    for r in best.iter_rows(named=True):
        bc = r["best_combo"] or {}
        md.append(
            f"| `{r['target']}` | {r['rho_best']:.4f} | "
            f"`{bc.get('family', '?')}` | {bc.get('horizon', '?')} "
            f"| `{bc.get('agg', '?')}` |"
        )

    md += ["", "## Best ceiling per (target, agg)",
           "", "| target | agg | best ρ | family | horizon |",
           "|---|---|---|---|---|"]
    best_ta = (
        rdf
        .group_by(["target", "agg"])
        .agg([
            pl.col("rho").max().alias("rho_best"),
            pl.struct(["family", "horizon"]).filter(
                pl.col("rho") == pl.col("rho").max()
            ).first().alias("best_combo"),
        ])
        .sort(["target", "agg"])
    )
    for r in best_ta.iter_rows(named=True):
        bc = r["best_combo"] or {}
        md.append(
            f"| `{r['target']}` | `{r['agg']}` | "
            f"{r['rho_best']:.4f} | `{bc.get('family', '?')}` | "
            f"{bc.get('horizon', '?')} |"
        )

    md += [
        "",
        "## Reachability of the /goal bars",
        "",
        "| target | bar | best ceiling | reachable? |",
        "|---|---|---|---|",
    ]
    bars = [
        ("sum_js", 0.85),
        ("sum_kl", 0.85),
        ("rouge_l_drop", 0.85),
        ("char_edit_ratio", 0.75),
    ]
    for tgt, bar in bars:
        best_for_tgt = rdf.filter(
            pl.col("target") == tgt)["rho"].max()
        ok = "✓" if (best_for_tgt is not None
                     and best_for_tgt >= bar) else "✗"
        md.append(
            f"| `{tgt}` | {bar:.2f} | {best_for_tgt:.4f} | {ok} |"
        )

    md += [
        "",
        "## Interpretation",
        "",
        "If the rich-aggregator ceiling for `rouge_l_drop` is "
        "still below 0.85, then any per-token regressor (no "
        "matter how good, no matter how rich the run-level "
        "aggregator) is structurally bounded below the bar. "
        "That justifies the option-(c) framing in "
        "`gold/phase-3-herald-v1-results.md`.",
        "",
        "If the ceiling lifts above 0.85, the bar is reachable "
        "in principle and we should explore the corresponding "
        "aggregator family in HERALD's run-level wrapper.",
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md))
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
