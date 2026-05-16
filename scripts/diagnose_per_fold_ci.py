"""Per-fold bootstrap CIs for LOO-task and LOO-ratio wrappers.

Re-runs the trajectory wrapper (same 17 features, same HGB params,
same V2 normalisation) for sum_js, but adds a cluster-bootstrap-by-
prompt_id CI to every held-out fold's per-fold ρ. The headline
JSON only reports point estimates; this script adds the noise band
to decide whether bars 6 and 7 are within statistical noise of
their thresholds (0.95 × overall for retention).

Run with --split-col task for bar 7, --split-col ratio for bar 6.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

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
    parser.add_argument(
        "--split-col", type=str, default="task",
        choices=("task", "ratio", "press"))
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or Path(
        f"results/phase3/per_fold_ci_{args.split_col}.json")

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
    split_arr = sub[args.split_col].to_numpy()
    prompt_arr = sub["prompt_id"].to_numpy()

    holds = sorted(np.unique(split_arr).tolist())
    print(f"[splits] leave-one-{args.split_col}-out over "
          f"{len(holds)} {args.split_col}s: {holds}")

    oof = np.full(X.shape[0], np.nan, dtype=np.float64)
    per_fold: dict[str, dict] = {}
    for k, held in enumerate(holds):
        te = np.where(split_arr == held)[0]
        tr = np.where(split_arr != held)[0]
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
        rho_ci = clustered_spearman_ci(
            oof[te], y[te], prompt_arr[te],
            n_boot=args.n_boot, seed=args.seed + k,
        )
        held_key = (
            str(held) if not isinstance(held, str) else held)
        per_fold[held_key] = {
            "n_runs": int(te.size),
            **rho_ci,
        }
        print(f"[fold] {args.split_col}={held_key:<20s} "
              f"ρ={rho_ci['rho']:.4f} "
              f"[{rho_ci['lo']:.4f}, {rho_ci['hi']:.4f}] "
              f"n={te.size}")

    mask = np.isfinite(oof)
    overall_ci = clustered_spearman_ci(
        oof[mask], y[mask], prompt_arr[mask],
        n_boot=args.n_boot, seed=args.seed,
    )
    bar = {"task": 0.90, "ratio": 0.95, "press": 0.95}[
        args.split_col]
    print(f"\n[overall] ρ={overall_ci['rho']:.4f} "
          f"[{overall_ci['lo']:.4f}, {overall_ci['hi']:.4f}]")
    threshold = bar * overall_ci["rho"]
    print(f"[retention] worst-fold ρ must clear "
          f"{bar:.2f} × {overall_ci['rho']:.4f} = "
          f"{threshold:.4f} for bar to pass")

    out = {
        "target": RUN_TARGET,
        "split_col": args.split_col,
        "retention_bar": bar,
        "overall": overall_ci,
        "retention_threshold": threshold,
        "per_fold": per_fold,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
