"""Extended-feature per-token model targeting bar 6 heavy slices.

The per-ratio per-token retraining showed +0.006 at the worst
slice — confirming that the bottleneck is the per-token *feature
set*, not the training regime. This script adds 17 causal features
that the canonical `CHEAP_ALL_FEATURES` set omits, especially
regime-shift signals (delta_h and its rollings, std-rollings for
all base signals, missing _32 rollings).

Excluded as oracle / leaky: js_full, kl_unc_comp_full
(per-token paired divergences vs uncompressed — these are the
label substrate).

Trains GroupKFold(prompt_id, 5) on log1p(future_sum_js_25),
writes OOF preds to a parallel parquet so per-ratio per-token
ρ and oracle wrapper retention can be re-evaluated.

Output: results/phase3/preds/prompt_group_extfeat__h25.parquet
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from herald.predictor_baselines import CHEAP_ALL_FEATURES

HORIZON = 25
LABEL = f"future_sum_js_{HORIZON}"

# Causal features available in phase2_tokens.parquet but not in
# CHEAP_ALL. All are O(1) per-token streaming signals.
EXTRA_FEATURES: tuple[str, ...] = (
    "avg_logp",
    "delta_h",
    "delta_h_mean_8", "delta_h_std_8",
    "delta_h_mean_32", "delta_h_ewma_hl8", "delta_h_ewma_hl32",
    "entropy_std_32",
    "top1_prob_std_8", "top1_prob_std_32",
    "top1_prob_ewma_hl32",
    "h_alts_std_8", "h_alts_mean_32", "h_alts_ewma_hl8",
    "kl_div_std_8", "kl_div_mean_32", "kl_div_ewma_hl8",
    "top10_jaccard_std_8", "top10_jaccard_ewma_hl8",
    "output_length_so_far",
)

EXT_FEATURES: tuple[str, ...] = tuple(CHEAP_ALL_FEATURES) + EXTRA_FEATURES


def _fit_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
    seed: int, max_iter: int, learning_rate: float, max_depth: int,
) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens-path", type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"))
    parser.add_argument(
        "--out-path", type=Path,
        default=Path(
            "results/phase3/preds/prompt_group_extfeat__h25.parquet"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    feats = list(EXT_FEATURES)
    keep = (
        ["run_id", "prompt_id", "task", "press", "token_pos",
         "compression_ratio", LABEL]
        + [c for c in feats
           if c not in {"token_pos", "compression_ratio"}]
    )
    keep = list(dict.fromkeys(keep))

    print(f"[features] {len(feats)} ({len(CHEAP_ALL_FEATURES)} canonical "
          f"+ {len(EXTRA_FEATURES)} extras)")
    print(f"[load] {args.tokens_path}")
    t0 = time.time()
    df = (
        pl.scan_parquet(args.tokens_path)
        .select(keep)
        .filter(pl.col(LABEL).is_not_null())
        .collect()
    )
    df = df.drop_nulls(subset=feats + [LABEL])
    print(f"[load] {df.height:,} rows in {time.time()-t0:.1f}s")

    y_raw = df[LABEL].to_numpy().astype(np.float64)
    y = np.log1p(y_raw)
    X = df.select(feats).to_numpy().astype(np.float32)
    groups = df["prompt_id"].to_numpy()

    gkf = GroupKFold(n_splits=args.n_folds)
    pred = np.full(X.shape[0], np.nan, dtype=np.float64)
    fold_id = np.full(X.shape[0], "", dtype=object)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        t0 = time.time()
        pred[te] = _fit_predict(
            X[tr], y[tr], X[te],
            seed=args.seed + k,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )
        fold_id[te] = f"fold{k}"
        print(f"[fold{k}] n_tr={len(tr):>9,} n_te={len(te):>9,} "
              f"{time.time()-t0:.1f}s")

    out = pl.DataFrame({
        "run_id": df["run_id"],
        "prompt_id": df["prompt_id"],
        "task": df["task"],
        "press": df["press"],
        "compression_ratio": df["compression_ratio"],
        "token_pos": df["token_pos"],
        "y_raw": y_raw,
        "y_log1p": y,
        "pred_log1p": pred,
        "pred_raw": np.expm1(pred),
        "fold": fold_id.astype(str),
    })
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out_path)
    meta = {
        "regime": "extfeat_prompt_group",
        "horizon": HORIZON,
        "label": LABEL,
        "feats": feats,
        "extra_feats": list(EXTRA_FEATURES),
        "n_rows": int(out.height),
        "hparams": {
            "n_folds": args.n_folds,
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "min_samples_leaf": 200,
            "l2_regularization": 1.0,
        },
        "seed": args.seed,
    }
    (args.out_path.with_suffix(".meta.json")).write_text(
        json.dumps(meta, indent=2))
    print(f"[done] wrote {args.out_path}  ({out.height:,} rows)")


if __name__ == "__main__":
    main()
