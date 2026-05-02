"""Phase 0 analysis #7: do trajectory aggregates discriminate ratios?

For each press (streaming_llm, snapkv) and each metric in
{sum_kl, sum_js, nll_ratio}:
  - report median + IQR by compression ratio;
  - run pairwise Mann-Whitney U tests between ratios.

If aggregates also saturate (no separation between 0.875 and 0.9375),
that contradicts the Phase 2 methodology story and should be flagged.

Output: results/phase0/analysis/ratio_discrimination.json
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu


def _iqr(arr: np.ndarray) -> tuple[float, float]:
    return (
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
    )


def main() -> None:
    root = Path("results/phase0")
    runs = pl.read_parquet(root / "final" / "runs.parquet").select(
        ["run_id", "press", "compression_ratio"]
    )
    traj = pl.read_parquet(root / "metrics" / "trajectory_metrics.parquet")
    df = runs.join(traj, on="run_id", how="inner")

    metrics = ["sum_kl", "sum_js", "nll_ratio"]
    presses = ["streaming_llm", "snapkv"]
    ratio_pairs = [(0.5, 0.875), (0.5, 0.9375), (0.875, 0.9375)]

    out: dict = {"by_press": {}}
    for press in presses:
        sub = df.filter(pl.col("press") == press)
        ratios = sorted(set(sub["compression_ratio"].to_list()))
        cell: dict = {"ratios": ratios, "metrics": {}}
        for metric in metrics:
            mblock: dict = {"distribution": {}, "pairwise": {}}
            for r in ratios:
                arr = sub.filter(pl.col("compression_ratio") == r)[
                    metric
                ].to_numpy()
                arr = arr[~np.isnan(arr)]
                lo, hi = _iqr(arr)
                mblock["distribution"][f"{r:.4f}"] = {
                    "n": int(arr.size),
                    "median": float(np.median(arr)),
                    "iqr_lo": lo,
                    "iqr_hi": hi,
                    "min": float(arr.min()) if arr.size else None,
                    "max": float(arr.max()) if arr.size else None,
                }
            for r1, r2 in ratio_pairs:
                a = sub.filter(pl.col("compression_ratio") == r1)[
                    metric
                ].to_numpy()
                b = sub.filter(pl.col("compression_ratio") == r2)[
                    metric
                ].to_numpy()
                a = a[~np.isnan(a)]
                b = b[~np.isnan(b)]
                if a.size and b.size:
                    u = mannwhitneyu(a, b, alternative="two-sided")
                    mblock["pairwise"][f"{r1:.4f}_vs_{r2:.4f}"] = {
                        "n_a": int(a.size),
                        "n_b": int(b.size),
                        "u": float(u.statistic),
                        "p_value": float(u.pvalue),
                        "median_a": float(np.median(a)),
                        "median_b": float(np.median(b)),
                    }
            cell["metrics"][metric] = mblock
        out["by_press"][press] = cell

    out_path = root / "analysis" / "ratio_discrimination.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    for press in presses:
        print(f"\n=== {press} ===")
        for metric in metrics:
            print(f"\n{metric}:")
            dist = out["by_press"][press]["metrics"][metric]["distribution"]
            for r, st in dist.items():
                print(
                    f"  ratio={r}: n={st['n']} median={st['median']:.4f} "
                    f"IQR=[{st['iqr_lo']:.4f}, {st['iqr_hi']:.4f}]"
                )
            pw = out["by_press"][press]["metrics"][metric]["pairwise"]
            for k, st in pw.items():
                print(f"  {k}: U={st['u']:.0f} p={st['p_value']:.2e}")


if __name__ == "__main__":
    main()
