"""Trajectory-shape pred-aggregates wrapper for HERALD v1.

Extends the strict press-agnostic wrapper from
`run_run_level_wrapper_agnostic.py --strict` with richer
per-run aggregates of `pred_raw`. All features remain
pred-derived (no metadata leakage). Leave-one-press-out CV.

The strict baseline (8 quantile/mean aggregates) gave
ρ_sum_js = 0.828 vs meta-only = 0.811 (+0.017 lift). The bar
is 0.85 and the no-leakage protocol requires strict-vs-meta
margin ≥ 0.05. This script tests whether trajectory-shape
features close both gaps simultaneously.

Trajectory features added on top of the 8 strict aggregates:
  - pred_max_pos: position (token_pos / n_tokens) of pred_max
  - pred_late_minus_early: mean(last_half) - mean(first_half)
  - pred_max_minus_p50: peak prominence over median
  - pred_p95_minus_p50: tail prominence over median
  - pred_auc: sum of pred over the run, normalised by n_tokens
  - pred_max_derivative: max |pred[t+1] - pred[t]|
  - pred_above_p90_count: count of tokens with pred above the
    run's own p90 (always ~10% by construction; informative
    only when paired with other features)
  - pred_longest_run_above_p75: longest consecutive run of
    tokens with pred above the run's own p75
  - pred_count_local_maxima: count of local maxima in pred
    (peak in a window of 3)
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


def _run_features(pred: np.ndarray) -> dict[str, float]:
    """Per-run features from a sorted (by token_pos) pred array."""
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
    # Dwell features
    above_p75 = pred > p75
    longest = 0
    cur = 0
    for v in above_p75:
        if v:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    # Normalize counts by n to avoid implicit n_tokens leakage
    # via unnormalized run-length-dependent counts. Three of the
    # trajectory features are counts; without dividing by n they
    # smuggle run length back in as a proxy for the dropped
    # n_tokens metadata feature.
    above_p90_count = int(np.sum(pred > p90))
    # Local maxima: pred[i] > pred[i-1] and pred[i] > pred[i+1]
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
        # Strict-aggregates (recomputed here to keep features
        # in one place for clarity)
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
    """Build per-run features from the token-level pred parquet."""
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
        default=Path("results/phase3/preds/prompt_group__h25.parquet"))
    parser.add_argument(
        "--run-damage-path", type=Path,
        default=Path("results/phase1/metrics/run_damage.parquet"))
    parser.add_argument(
        "--out", type=Path,
        default=Path(
            "results/phase3/run_level_wrapper_trajectory.json"))
    parser.add_argument(
        "--split-col", type=str, default="press",
        help="Column to hold out per fold: 'press' "
             "(default), 'ratio', or 'task'.")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    print(f"[load] preds: {args.preds_path}")
    df = pl.read_parquet(args.preds_path)
    print(f"[load] {df.height:,} rows")

    feats_df = _build_features(df)
    print(f"[agg] {feats_df.height:,} runs")

    rd = pl.read_parquet(args.run_damage_path).select(
        ["run_id"] + list(RUN_TARGETS))
    joined = feats_df.join(rd, on="run_id", how="inner")
    print(f"[join] {joined.height:,} runs with run_damage")

    feat_cols = list(PRED_AGG_COLS) + list(TRAJECTORY_COLS)
    print(f"[features] {len(feat_cols)} pred-derived feature cols "
          f"(strict 8 + trajectory 9). No metadata.")

    split_col = args.split_col
    if split_col not in {"press", "ratio", "task"}:
        raise SystemExit(
            f"--split-col must be press|ratio|task, got {split_col}")
    presses = sorted(joined[split_col].unique().to_list())
    print(f"[splits] leave-one-{split_col}-out over {len(presses)} "
          f"{split_col}s: {presses}")

    results: dict[str, dict] = {}
    for target in RUN_TARGETS:
        sub = joined.drop_nulls(subset=[target])
        y = sub[target].to_numpy().astype(np.float64)
        y_log1p = np.log1p(np.maximum(y, 0.0))
        X = sub.select(feat_cols).to_numpy().astype(np.float32)
        press_arr = sub[split_col].to_numpy()
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
            held_key = (
                str(held) if not isinstance(held, str) else held)
            per_press[held_key] = {
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
            print(f"    held={str(press):<20s} ρ={p['rho']:.4f}  "
                  f"n={p['n_runs']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
