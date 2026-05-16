"""Phase 2c: lead-time analysis with EWS-augmented predictor.

Mirrors `scripts/run_phase2_lead_time.py` but trains three predictors
on the *EWS-augmented* dataset and reports their pre-onset AUROC
profile against catastrophic (looping or non_termination) runs:

  - entropy (raw, no model)
  - lr_all_cheap (Phase 2 baseline; existing rolling/EWMA only)
  - lr_all_cheap + EWS

The hypothesis is that EWS captures the rising variance / skew /
flicker dynamics that precede collapse, so the EWS predictor's
pre-onset AUROC should be > 0.5 (and ideally > 0.65) where the
Phase 2 baseline was inverted (~0.27-0.33).

Outputs:
- results/phase2c_early_warning/ews_lead_time.parquet
- results/phase2c_early_warning/ews_lead_time_summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from herald.early_warning_features import (
    DEFAULT_BASE_SIGNALS,
    DEFAULT_WINDOWS,
    ews_feature_names,
)
from herald.predictor_baselines import (
    CHEAP_ALL_FEATURES,
    binarize_with_threshold,
    compute_train_quantile_threshold,
)


def _clean_X(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.size < 2 or len(set(y.tolist())) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def _train_lr(
    train_df: pl.DataFrame,
    label_col: str,
    feats: list[str],
    threshold_q: float,
) -> tuple[StandardScaler, LogisticRegression, float] | None:
    thr = compute_train_quantile_threshold(train_df[label_col], q=threshold_q)
    if thr is None:
        return None
    y = binarize_with_threshold(train_df[label_col], thr)
    mask = ~y.is_null()
    train = train_df.filter(mask)
    yv = y.filter(mask).cast(pl.Int64).to_numpy()
    if len(set(yv.tolist())) < 2:
        return None
    X = _clean_X(train.select(feats).fill_null(0.0).to_numpy().astype(float))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=200, class_weight="balanced", solver="lbfgs"
    )
    clf.fit(Xs, yv)
    return scaler, clf, thr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("results/phase1"),
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "results/phase2c_early_warning/phase2_dataset_ews.parquet"
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2c_early_warning"),
    )
    ap.add_argument("--label", type=str, default="future_sum_js_25")
    ap.add_argument("--threshold-q", type=float, default=0.9)
    ap.add_argument("--max-train-rows", type=int, default=400_000)
    ap.add_argument("--window-before", type=int, default=200)
    ap.add_argument("--window-after", type=int, default=50)
    ap.add_argument("--position-stride", type=int, default=2)
    ap.add_argument("--max-controls-per-stratum", type=int, default=20)
    ap.add_argument("--max-cats-per-stratum", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rd_path = args.root / "metrics" / "run_damage.parquet"
    runs_path = args.root / "final" / "runs.parquet"
    if not runs_path.exists() or not rd_path.exists():
        raise SystemExit("missing runs.parquet or run_damage.parquet")

    logger.info("loading run_damage")
    rd = pl.read_parquet(rd_path)
    runs = pl.read_parquet(runs_path).filter(pl.col("replay_status") == "ok")
    rd_meta = rd.join(
        runs.select(
            [
                "run_id",
                "task",
                "press",
                "compression_ratio",
                "num_tokens_generated",
                "max_new_tokens",
            ]
        ),
        on="run_id",
        how="left",
    )

    cats = rd_meta.filter(
        pl.col("has_looping") | pl.col("has_non_termination")
    )
    healthy = rd_meta.filter(
        ~pl.col("has_looping")
        & ~pl.col("has_non_termination")
        & ~pl.col("has_format_break")
        & ~pl.col("has_drift")
    )
    logger.info("n_cat={} n_healthy={}", cats.height, healthy.height)

    selected_cat: list[dict[str, Any]] = []
    selected_ctrl: list[dict[str, Any]] = []
    for (task, press, ratio), cat_sub in cats.group_by(
        ["task", "press", "compression_ratio"]
    ):
        n_cat = min(args.max_cats_per_stratum, cat_sub.height)
        cat_pick = cat_sub.sample(n=n_cat, seed=args.seed)
        ctrl_sub = healthy.filter(
            (pl.col("task") == task)
            & (pl.col("press") == press)
            & (pl.col("compression_ratio") == ratio)
        )
        if ctrl_sub.is_empty():
            continue
        n_ctrl = min(args.max_controls_per_stratum, ctrl_sub.height)
        ctrl_pick = ctrl_sub.sample(n=n_ctrl, seed=args.seed)
        cat_pick = cat_pick.with_columns(
            pl.when(pl.col("has_non_termination"))
            .then((pl.col("max_new_tokens") * 0.75).cast(pl.Int64))
            .otherwise((pl.col("num_tokens_generated") * 0.5).cast(pl.Int64))
            .alias("onset_token")
        )
        median_onset = int(cat_pick["onset_token"].median() or 0)
        ctrl_pick = ctrl_pick.with_columns(
            pl.lit(median_onset).alias("stratum_median_onset"),
            pl.min_horizontal(
                pl.lit(median_onset),
                pl.col("num_tokens_generated") - 1,
            ).alias("onset_token"),
        )
        for r in cat_pick.iter_rows(named=True):
            selected_cat.append({**r, "label": 1})
        for r in ctrl_pick.iter_rows(named=True):
            selected_ctrl.append({**r, "label": 0})

    if not selected_cat or not selected_ctrl:
        raise SystemExit("no matched cats/ctrls")
    cat_df = pl.DataFrame(selected_cat, infer_schema_length=None)
    ctrl_df = pl.DataFrame(selected_ctrl, infer_schema_length=None)
    logger.info("matched: cats={} ctrls={}", cat_df.height, ctrl_df.height)

    logger.info("loading EWS dataset")
    ds = pl.read_parquet(args.dataset)

    selected_run_ids = set(
        cat_df["run_id"].to_list() + ctrl_df["run_id"].to_list()
    )
    train_pool = ds.filter(~pl.col("run_id").is_in(list(selected_run_ids)))
    if args.max_train_rows > 0 and train_pool.height > args.max_train_rows:
        train_pool = train_pool.sample(
            n=args.max_train_rows, seed=args.seed, shuffle=True
        )
    logger.info("train pool: {} rows", train_pool.height)

    feat_lr = [c for c in CHEAP_ALL_FEATURES if c in ds.columns]
    ews_cols = list(
        ews_feature_names(
            signals=DEFAULT_BASE_SIGNALS, windows=DEFAULT_WINDOWS
        )
    )
    feat_ews = feat_lr + [c for c in ews_cols if c in ds.columns]
    logger.info(
        "n_feat lr_all_cheap={}  lr_all_cheap+EWS={}",
        len(feat_lr),
        len(feat_ews),
    )

    lr_pack = _train_lr(train_pool, args.label, feat_lr, args.threshold_q)
    ews_pack = _train_lr(train_pool, args.label, feat_ews, args.threshold_q)
    if lr_pack is None or ews_pack is None:
        raise SystemExit("training failed")

    selected_full = ds.filter(pl.col("run_id").is_in(list(selected_run_ids)))
    X_lr = _clean_X(
        selected_full.select(feat_lr).fill_null(0.0).to_numpy().astype(float)
    )
    X_ews = _clean_X(
        selected_full.select(feat_ews).fill_null(0.0).to_numpy().astype(float)
    )
    sc_lr, clf_lr, _ = lr_pack
    sc_ews, clf_ews, _ = ews_pack
    s_lr = clf_lr.predict_proba(sc_lr.transform(X_lr))[:, 1]
    s_ews = clf_ews.predict_proba(sc_ews.transform(X_ews))[:, 1]
    selected_full = selected_full.with_columns(
        pl.Series("predictor_lr", s_lr),
        pl.Series("predictor_ews", s_ews),
    )

    onset_map = pl.concat(
        [
            cat_df.select(["run_id", "onset_token"]).with_columns(
                pl.lit(1).alias("label")
            ),
            ctrl_df.select(["run_id", "onset_token"]).with_columns(
                pl.lit(0).alias("label")
            ),
        ]
    )
    long = selected_full.join(onset_map, on="run_id", how="inner")
    long = long.with_columns(
        (pl.col("token_pos") - pl.col("onset_token")).alias("relative_pos")
    )
    long = long.filter(
        pl.col("relative_pos").is_between(
            -args.window_before, args.window_after
        )
    )
    if args.position_stride > 1:
        long = long.filter(
            (pl.col("relative_pos") % args.position_stride) == 0
        )

    rows: list[dict[str, Any]] = []
    feature_signals = ("entropy", "predictor_lr", "predictor_ews")
    for fname in feature_signals:
        if fname not in long.columns:
            continue
        for rp, sub in long.group_by("relative_pos", maintain_order=True):
            y = sub["label"].to_numpy().astype(np.int8)
            s = sub[fname].to_numpy().astype(np.float64)
            auc = _safe_auroc(y, s)
            rows.append(
                {
                    "feature": fname,
                    "relative_pos": int(rp[0]),
                    "n_pos": int((y == 1).sum()),
                    "n_neg": int((y == 0).sum()),
                    "auroc": auc,
                }
            )
    by_feat = pl.DataFrame(rows)
    by_feat.write_parquet(args.output_dir / "ews_lead_time.parquet")

    # Aggregate diagnostics: mean AUROC in [-200, 0] and [-50, 0].
    diag: dict[str, Any] = {}
    for fname in feature_signals:
        sub = by_feat.filter(pl.col("feature") == fname)
        if sub.is_empty():
            continue
        for window_lo, window_hi, key in (
            (-args.window_before, 0, "pre_onset_full"),
            (-50, 0, "pre_onset_50"),
            (0, args.window_after, "post_onset"),
        ):
            sw = sub.filter(
                pl.col("relative_pos").is_between(window_lo, window_hi)
            ).filter(pl.col("auroc").is_not_null())
            if sw.is_empty():
                continue
            arr = sw["auroc"].to_numpy().astype(float)
            diag.setdefault(fname, {})[key] = {
                "mean_auroc": float(arr.mean()),
                "max_auroc": float(arr.max()),
                "min_auroc": float(arr.min()),
                "n_positions": int(sw.height),
            }

    summary = {
        "label": args.label,
        "threshold_q": args.threshold_q,
        "n_catastrophic_runs": int(cat_df.height),
        "n_control_runs": int(ctrl_df.height),
        "window_before": args.window_before,
        "window_after": args.window_after,
        "position_stride": args.position_stride,
        "feature_signals": list(feature_signals),
        "diagnostics_by_feature": diag,
        "by_feature_path": str(args.output_dir / "ews_lead_time.parquet"),
        "feat_lr_count": len(feat_lr),
        "feat_ews_count": len(feat_ews),
    }
    summary_path = args.output_dir / "ews_lead_time_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("summary -> {}", summary_path)

    print()
    print("=== EWS lead-time headline ===")
    for fname in feature_signals:
        d = diag.get(fname, {})
        pre = d.get("pre_onset_full", {})
        post = d.get("post_onset", {})
        print(
            f"  {fname:18s} pre[-200,0] mean={pre.get('mean_auroc')} "
            f"max={pre.get('max_auroc')}  "
            f"post[0,+50] mean={post.get('mean_auroc')}"
        )


if __name__ == "__main__":
    main()
