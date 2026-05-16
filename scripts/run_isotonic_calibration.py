"""Isotonic post-hoc calibration of HERALD v1 per-run predictions.

The raw OOF gives ECE(pred_max → y_max) = 0.058, marginally over
the /goal bar of 0.05. Isotonic regression is the standard fix
for monotone calibration drift; it preserves Spearman ρ exactly
(it is rank-preserving) and only changes the mapping pred → y.

Per fold k, we fit isotonic on the OOF predictions of the other
folds and apply to fold k. This is the textbook nested-OOF
calibration that avoids using a row's own prediction to
calibrate itself.

Output:
  results/phase3/eval/isotonic_calibration.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from herald.regression_metrics import (
    clustered_spearman_ci,
    ece_quantile,
    safe_spearman,
)

RUN_TARGETS = ("rouge_l_drop", "sum_js", "sum_kl", "char_edit_ratio")


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
        default=Path("results/phase3/eval/isotonic_calibration.json"))
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    df = pl.read_parquet(args.preds_path)
    print(f"[load] {df.height:,} rows from {args.preds_path}")
    print(f"[load] folds = {sorted(df['fold'].unique().to_list())}")

    per_run = (
        df.sort(["run_id", "token_pos"])
        .group_by("run_id")
        .agg(
            pl.col("pred_raw").max().alias("pred_max"),
            pl.col("y_raw").max().alias("y_max"),
            pl.col("fold").first().alias("fold"),
            pl.col("prompt_id").first().alias("prompt_id"),
        )
    )
    rd = pl.read_parquet(args.run_damage_path).select(
        ["run_id"] + list(RUN_TARGETS))
    joined = per_run.join(rd, on="run_id", how="inner")
    print(f"[agg] {joined.height:,} runs after join")

    folds = sorted(joined["fold"].unique().to_list())
    pred = joined["pred_max"].to_numpy()
    y_max = joined["y_max"].to_numpy()
    fold = joined["fold"].to_numpy()
    prompt = joined["prompt_id"].to_numpy()

    cal = np.full_like(pred, np.nan, dtype=np.float64)
    for f in folds:
        te = fold == f
        tr = ~te
        iso = IsotonicRegression(
            out_of_bounds="clip", increasing=True)
        iso.fit(pred[tr], y_max[tr])
        cal[te] = iso.transform(pred[te])

    mask = np.isfinite(cal) & np.isfinite(y_max)
    ece_raw = ece_quantile(pred[mask], y_max[mask], n_bins=10)
    ece_cal = ece_quantile(cal[mask], y_max[mask], n_bins=10)
    rho_raw = safe_spearman(pred[mask], y_max[mask])
    rho_cal = safe_spearman(cal[mask], y_max[mask])
    rho_cal_ci = clustered_spearman_ci(
        cal[mask], y_max[mask], prompt[mask],
        n_boot=500, seed=args.seed,
    )

    # Apply same calibration mapping to each run_target.
    target_results: dict[str, dict] = {}
    for tgt in RUN_TARGETS:
        sub = joined.drop_nulls(subset=[tgt])
        sub_pred = sub["pred_max"].to_numpy()
        sub_y = sub[tgt].to_numpy()
        sub_fold = sub["fold"].to_numpy()
        sub_prompt = sub["prompt_id"].to_numpy()
        sub_cal = np.full_like(sub_pred, np.nan, dtype=np.float64)
        for f in folds:
            te = sub_fold == f
            tr = ~te
            iso = IsotonicRegression(
                out_of_bounds="clip", increasing=True)
            iso.fit(sub_pred[tr], sub_y[tr])
            sub_cal[te] = iso.transform(sub_pred[te])
        m = np.isfinite(sub_cal) & np.isfinite(sub_y)
        ece_raw_t = ece_quantile(
            sub_pred[m], sub_y[m], n_bins=10)
        ece_cal_t = ece_quantile(
            sub_cal[m], sub_y[m], n_bins=10)
        rho_cal_t = clustered_spearman_ci(
            sub_cal[m], sub_y[m], sub_prompt[m],
            n_boot=500, seed=args.seed,
        )
        target_results[tgt] = {
            "n_runs": int(m.sum()),
            "ece_raw": ece_raw_t,
            "ece_calibrated": ece_cal_t,
            "rho_calibrated": rho_cal_t,
        }
        print(f"[{tgt:<18s}] ECE raw={ece_raw_t:.4f} → "
              f"cal={ece_cal_t:.4f}  ρ_cal={rho_cal_t['rho']:.4f}")

    results = {
        "n_runs": int(mask.sum()),
        "ece_raw_pred_max_y_max": ece_raw,
        "ece_calibrated_pred_max_y_max": ece_cal,
        "rho_raw_pred_max_y_max": rho_raw,
        "rho_calibrated_pred_max_y_max": rho_cal,
        "rho_calibrated_pred_max_y_max_ci": rho_cal_ci,
        "per_target": target_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] ECE pred_max → y_max: "
          f"{ece_raw:.4f} → {ece_cal:.4f}")
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
