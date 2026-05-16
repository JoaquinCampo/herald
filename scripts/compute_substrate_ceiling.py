"""Compute the per-run substrate ceiling for HERALD v1.

For each (H, agg, target), compute Spearman between
  aggregate(future_*_H over all tokens in a run)
and
  run_damage[target].

This bounds what ANY per-token regressor + aggregator can hit
on the chosen run-level validator. The model can only close the
gap between its OOF aggregation and this ceiling.

y_sum is excluded: telescoping makes it ~H * sum_js, which is
tautological for sum_js / sum_kl targets.
"""

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

HORIZONS = [5, 10, 25, 50]
LABEL_FAMILIES = ["future_sum_js", "future_sum_kl", "future_max_js"]
AGGS = ["max", "p95", "mean"]
TARGETS = [
    "rouge_l_drop", "sum_js", "sum_kl", "char_edit_ratio",
]


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def main() -> None:
    out_path = Path("results/phase3_smoke/substrate_ceiling.json")
    md_path = Path("gold/phase-3-substrate-ceiling.md")

    label_cols = [f"{fam}_{h}" for fam in LABEL_FAMILIES for h in HORIZONS]
    print(f"[load] labels: {label_cols}")
    t0 = time.time()
    df = (
        pl.scan_parquet("results/phase2/dataset/phase2_tokens.parquet")
        .select(["run_id"] + label_cols)
        .collect()
    )
    print(f"[load] {df.height:,} rows in {time.time()-t0:.1f}s")

    rd = pl.read_parquet(
        "results/phase1/metrics/run_damage.parquet"
    ).select(["run_id"] + TARGETS)

    rows = []
    for fam in LABEL_FAMILIES:
        for h in HORIZONS:
            col = f"{fam}_{h}"
            sub = df.select(["run_id", col]).drop_nulls()
            agg_df = (
                sub.group_by("run_id")
                .agg(
                    pl.col(col).max().alias("y_max"),
                    pl.col(col).quantile(0.95).alias("y_p95"),
                    pl.col(col).mean().alias("y_mean"),
                )
            )
            joined = agg_df.join(rd, on="run_id", how="inner")
            for agg in AGGS:
                a_col = f"y_{agg}"
                for tgt in TARGETS:
                    s = joined.drop_nulls(subset=[a_col, tgt])
                    if s.height < 10:
                        rho = float("nan")
                    else:
                        rho = _rho(
                            s[a_col].to_numpy(),
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
                    print(f"  ρ({col} {agg:>4s} → {tgt:<22s}) "
                          f"= {rho:.4f}  n={s.height}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\n[done] wrote {out_path}")

    rdf = pl.DataFrame(rows)
    md = [
        "# HERALD v1 Substrate Ceiling",
        "",
        "Upper bound for per-run Spearman achievable by aggregating "
        "any of the `future_*_H` labels per run. The HERALD v1 model "
        "can never exceed these numbers, because they are computed "
        "from the TRUE label, not predictions.",
        "",
        "All correlations are Spearman ρ between "
        "`aggregate_over_tokens_in_run(future_*_H)` and "
        "`run_damage[target]`. `y_sum` is excluded because it "
        "telescopes to ~H × `sum_js`, which is tautological.",
        "",
        "## Best ceiling per (target, family)",
        "",
    ]
    best = (
        rdf
        .group_by(["family", "target"])
        .agg(
            pl.col("rho").max().alias("rho_max"),
            pl.col("horizon").filter(
                pl.col("rho") == pl.col("rho").max()
            ).first().alias("best_h"),
            pl.col("agg").filter(
                pl.col("rho") == pl.col("rho").max()
            ).first().alias("best_agg"),
        )
        .sort(["target", "family"])
    )
    md.append("| family | target | best H | best agg | ρ_ceiling |")
    md.append("|---|---|---|---|---|")
    for r in best.iter_rows(named=True):
        md.append(
            f"| `{r['family']}` | `{r['target']}` | "
            f"{r['best_h']} | `{r['best_agg']}` | "
            f"{r['rho_max']:.4f} |"
        )

    md += ["", "## Full matrix",
           "", "Columns: family / target. Rows: H × agg.", ""]
    fams_targets = [(f, t) for f in LABEL_FAMILIES for t in TARGETS]
    md.append("| H | agg | " + " | ".join(
        f"{f}<br>{t}" for f, t in fams_targets) + " |")
    md.append("|" + "---|" * (2 + len(fams_targets)))
    for h in HORIZONS:
        for agg in AGGS:
            row = [str(h), agg]
            for f, t in fams_targets:
                m = rdf.filter(
                    (pl.col("family") == f)
                    & (pl.col("horizon") == h)
                    & (pl.col("agg") == agg)
                    & (pl.col("target") == t)
                )
                v = m["rho"][0] if m.height > 0 else float("nan")
                row.append(f"{v:.3f}" if np.isfinite(v) else "—")
            md.append("| " + " | ".join(row) + " |")

    md += [
        "",
        "## Implications for HERALD v1 headline bars",
        "",
        "The /goal spec sets per-run Spearman bars at ≥ 0.85 vs "
        "`rouge_l_drop` and `sum_js`. These ceilings tell us which "
        "are reachable:",
        "",
    ]
    bars = [
        ("sum_js", 0.85),
        ("sum_kl", 0.85),
        ("rouge_l_drop", 0.85),
        ("char_edit_ratio", 0.75),
    ]
    md.append("| target | bar | best ceiling | reachable? |")
    md.append("|---|---|---|---|")
    for tgt, bar in bars:
        best_for_tgt = rdf.filter(pl.col("target") == tgt)["rho"].max()
        ok = "✓" if best_for_tgt is not None and best_for_tgt >= bar \
            else "✗"
        md.append(
            f"| `{tgt}` | {bar:.2f} | "
            f"{best_for_tgt:.4f} | {ok} |"
        )

    md += [
        "",
        "## Strategic note",
        "",
        "The honest framing: `future_sum_js_H` and `future_sum_kl_H` "
        "are process metrics (trajectory divergence accumulated over "
        "the next H tokens). `rouge_l_drop` is an end-of-sequence "
        "quality delta. A per-token regressor over the JS/KL labels "
        "cannot exceed the substrate-ceiling Spearman against "
        "`rouge_l_drop`, regardless of model capacity. Closing that "
        "gap requires either a different label, a run-level wrapper "
        "stage, or accepting that `rouge_l_drop` is reported as an "
        "extrinsic validator below the 0.85 bar.",
        "",
    ]

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md))
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
