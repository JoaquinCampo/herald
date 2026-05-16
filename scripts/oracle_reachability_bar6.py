"""Oracle reachability test for bar 6 (cross-ratio retention).

The bar-6 failure (held-out 0.375 ρ = 0.640) could be either:
  (a) a generalisation problem: per-token model trained without
      0.375 cannot transfer there, OR
  (b) a structural problem: even with 0.375 in-distribution, the
      run-level signal at heavy compression is qualitatively
      different and the wrapper cannot rank within that slice.

This script distinguishes the two. It uses
`prompt_group__h25.parquet` (per-token model trained 5-fold by
prompt_id, so every ratio is in-distribution). Then it trains the
trajectory wrapper with GroupKFold(prompt_id) (same in-distribution
regime) and breaks out per-ratio ρ on the OOF predictions.

If oracle per-ratio retention >= 0.95, bar 6 is a generalisation
problem solvable by feature/regime engineering. If < 0.95, the
heavy-compression slices are structurally unrankable from the
trajectory signal and no amount of retraining will close the gap.
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
    safe_spearman,
)

RUN_TARGET = "sum_js"


def _run_features(pred: np.ndarray) -> dict[str, float]:
    n = pred.size
    if n == 0:
        return {}
    pmax = float(np.nanmax(pred))
    pmean = float(np.nanmean(pred))
    pstd = float(np.nanstd(pred))
    p50 = float(np.nanquantile(pred, 0.50))
    p75 = float(np.nanquantile(pred, 0.75))
    p90 = float(np.nanquantile(pred, 0.90))
    p95 = float(np.nanquantile(pred, 0.95))
    half = n // 2
    early = float(np.nanmean(pred[:half])) if half > 0 else pmean
    late = float(np.nanmean(pred[half:])) if half < n else pmean
    if n >= 2:
        diffs = np.diff(pred)
        max_deriv = float(np.nanmax(np.abs(diffs)))
    else:
        max_deriv = 0.0
    max_pos = int(np.nanargmax(pred))
    max_rel = float(max_pos) / float(max(n - 1, 1))
    above_p75 = pred > p75
    longest = 0
    cur = 0
    for v in above_p75:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    above_p90_count = int(np.sum(pred > p90))
    if n >= 3:
        ll = pred[:-2]
        cc = pred[1:-1]
        rr = pred[2:]
        local_max_count = int(np.sum((cc > ll) & (cc > rr)))
    else:
        local_max_count = 0
    auc = float(np.nansum(pred)) / float(max(n, 1))
    inv_n = 1.0 / float(max(n, 1))
    return {
        "pred_max_pos": max_rel,
        "pred_late_minus_early": late - early,
        "pred_max_minus_p50": pmax - p50,
        "pred_p95_minus_p50": p95 - p50,
        "pred_auc": auc,
        "pred_max_derivative": max_deriv,
        "pred_above_p90_rate": float(above_p90_count) * inv_n,
        "pred_longest_run_above_p75_rate":
            float(longest) * inv_n,
        "pred_count_local_maxima_rate":
            float(local_max_count) * inv_n,
        "pred_max": pmax,
        "pred_p95": p95,
        "pred_p75": p75,
        "pred_p50": p50,
        "pred_mean": pmean,
        "pred_std": pstd,
        "pred_last5": float(np.nanmean(pred[-5:])) if n >= 5
            else pmean,
        "pred_top3": float(np.nanmean(
            np.sort(pred)[-3:])) if n >= 3 else pmax,
    }


PRED_AGG_COLS = (
    "pred_max", "pred_p95", "pred_p75", "pred_p50",
    "pred_mean", "pred_std", "pred_last5", "pred_top3",
)
TRAJECTORY_COLS = (
    "pred_max_pos", "pred_late_minus_early",
    "pred_max_minus_p50", "pred_p95_minus_p50",
    "pred_auc", "pred_max_derivative",
    "pred_above_p90_rate", "pred_longest_run_above_p75_rate",
    "pred_count_local_maxima_rate",
)


def _build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["run_id", "token_pos"])
    grouped = (
        df.group_by("run_id", maintain_order=True)
        .agg(
            pl.col("pred_raw").alias("preds"),
            pl.col("press").first().alias("press"),
            pl.col("compression_ratio").first().alias("ratio"),
            pl.col("task").first().alias("task"),
            pl.col("prompt_id").first().alias("prompt_id"),
            pl.col("pred_raw").count().alias("n_tokens"),
        )
    )
    rows = []
    for r in grouped.iter_rows(named=True):
        preds = np.asarray(r["preds"], dtype=np.float64)
        feats = _run_features(preds)
        feats.update({
            "run_id": r["run_id"],
            "press": r["press"],
            "ratio": float(r["ratio"]),
            "task": r["task"],
            "prompt_id": r["prompt_id"],
            "n_tokens": int(r["n_tokens"]),
        })
        rows.append(feats)
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds-path", type=Path,
        default=Path(
            "results/phase3/preds/prompt_group__h25.parquet"))
    parser.add_argument(
        "--run-damage-path", type=Path,
        default=Path("results/phase1/metrics/run_damage.parquet"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument(
        "--out", type=Path,
        default=Path(
            "results/phase3/oracle_reachability_bar6.json"))
    args = parser.parse_args()

    print(f"[load] preds: {args.preds_path}")
    df = pl.read_parquet(args.preds_path)
    print(f"[load] {df.height:,} rows")

    feats_df = _build_features(df)
    print(f"[agg] {feats_df.height:,} runs")

    rd = pl.read_parquet(args.run_damage_path).select(
        ["run_id", RUN_TARGET])
    joined = feats_df.join(rd, on="run_id", how="inner")
    print(f"[join] {joined.height:,} runs with run_damage")

    feat_cols = list(PRED_AGG_COLS) + list(TRAJECTORY_COLS)

    sub = joined.drop_nulls(subset=[RUN_TARGET])
    y = sub[RUN_TARGET].to_numpy().astype(np.float64)
    y_log1p = np.log1p(np.maximum(y, 0.0))
    X = sub.select(feat_cols).to_numpy().astype(np.float32)
    ratio_arr = sub["ratio"].to_numpy()
    prompt_arr = sub["prompt_id"].to_numpy()

    print(f"[oracle] GroupKFold(prompt_id), {args.n_folds} folds, "
          "all ratios in-distribution for per-token AND wrapper")
    gkf = GroupKFold(n_splits=args.n_folds)
    oof = np.full(X.shape[0], np.nan, dtype=np.float64)
    for k, (tr, te) in enumerate(
        gkf.split(X, y_log1p, groups=prompt_arr)
    ):
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
        rho_te = safe_spearman(oof[te], y[te])
        print(f"[fold {k}] n_tr={tr.size:,} n_te={te.size:,} "
              f"ρ_te={rho_te:.4f}")

    mask = np.isfinite(oof)
    overall_ci = clustered_spearman_ci(
        oof[mask], y[mask], prompt_arr[mask],
        n_boot=args.n_boot, seed=args.seed,
    )
    print(f"\n[overall] oracle ρ={overall_ci['rho']:.4f} "
          f"[{overall_ci['lo']:.4f}, {overall_ci['hi']:.4f}]")

    ratios = sorted(np.unique(ratio_arr).tolist())
    per_ratio: dict[str, dict] = {}
    for r in ratios:
        sel = (ratio_arr == r) & mask
        if sel.sum() < 50:
            continue
        rho_ci = clustered_spearman_ci(
            oof[sel], y[sel], prompt_arr[sel],
            n_boot=args.n_boot, seed=args.seed,
        )
        per_ratio[str(r)] = {
            "n_runs": int(sel.sum()),
            **rho_ci,
        }
        print(f"[ratio={r:<7.4f}] oracle ρ={rho_ci['rho']:.4f} "
              f"[{rho_ci['lo']:.4f}, {rho_ci['hi']:.4f}] "
              f"n={int(sel.sum())}")

    bar = 0.95
    worst = min(d["rho"] for d in per_ratio.values())
    retention = worst / overall_ci["rho"]
    threshold = bar * overall_ci["rho"]
    print(f"\n[retention] worst={worst:.4f}, "
          f"threshold={threshold:.4f} "
          f"(bar {bar:.2f} × overall {overall_ci['rho']:.4f})")
    print(f"[retention] retention ratio = "
          f"{retention:.4f} {'PASSES' if retention >= bar else 'FAILS'}"
          f" bar 6")

    out = {
        "target": RUN_TARGET,
        "regime": "oracle_in_distribution_both_layers",
        "n_folds": args.n_folds,
        "overall": overall_ci,
        "per_ratio": per_ratio,
        "retention_bar": bar,
        "retention_threshold": threshold,
        "worst_ratio_rho": float(worst),
        "retention_ratio": float(retention),
        "passes_bar_6": bool(retention >= bar),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
