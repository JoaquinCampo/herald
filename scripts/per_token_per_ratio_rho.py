"""Per-ratio per-token Spearman on the in-distribution preds.

Slices `prompt_group__h25.parquet` (per-token model trained
5-fold by prompt_id, so every ratio is in-distribution) by
`compression_ratio` and computes Spearman of `pred_raw` vs
`future_sum_js_25` within each slice. This localises the bar-6
bottleneck: if per-token ρ at heavy compression is at per-token
substrate ceiling, the wrapper inherits a hard cap and slice-
specific work on per-token features is needed. If per-token ρ
has headroom, the wrapper itself is leaving signal on the table.

Output: `results/phase3/per_token_per_ratio_rho.json`
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr


LABEL = "y_raw"


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    r = spearmanr(a[mask], b[mask]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def main() -> None:
    preds_path = Path(
        "results/phase3/preds/prompt_group__h25.parquet")
    out_path = Path(
        "results/phase3/per_token_per_ratio_rho.json")

    print(f"[load] {preds_path}")
    df = pl.read_parquet(preds_path).select(
        ["pred_raw", LABEL, "compression_ratio"]).drop_nulls(
            subset=[LABEL])
    print(f"[load] {df.height:,} rows")

    ratios = sorted(df["compression_ratio"].unique().to_list())
    per_ratio: dict[str, dict] = {}
    for r in ratios:
        sub = df.filter(pl.col("compression_ratio") == r)
        p = sub["pred_raw"].to_numpy().astype(np.float64)
        y = sub[LABEL].to_numpy().astype(np.float64)
        rho = _rho(p, y)
        per_ratio[str(r)] = {
            "n_tokens": int(sub.height),
            "rho": rho,
        }
        print(f"[ratio={r:<7.4f}] n={sub.height:>9,}  ρ={rho:.4f}")

    p_all = df["pred_raw"].to_numpy().astype(np.float64)
    y_all = df[LABEL].to_numpy().astype(np.float64)
    pooled = _rho(p_all, y_all)
    print(f"\n[pooled] ρ={pooled:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "label": LABEL,
        "preds_path": str(preds_path),
        "regime": "in_distribution_prompt_group",
        "pooled_rho": pooled,
        "per_ratio": per_ratio,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
