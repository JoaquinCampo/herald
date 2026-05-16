"""Per-ratio per-token model for bar 6 lift.

Hypothesis: a single joint HGB across all ratios averages over
heterogeneous regimes and produces weak per-token ρ at heavy
compression (0.25, 0.375, 0.5: ρ ~ 0.53-0.56 vs y_max ceiling
0.85). A ratio-stratified per-token model that fits each ratio
independently should lift per-token ρ at heavy slices.

For each ratio in the in-distribution prompt_group regime:
  1. Filter `phase2_tokens.parquet` to that ratio.
  2. 5-fold GroupKFold by prompt_id.
  3. Train HGB on log1p(future_sum_js_25), OOF predict.
  4. Save OOF preds.

Then concatenate all per-ratio OOF preds into a single parquet
that mirrors `prompt_group__h25.parquet` schema so the oracle
wrapper script can be re-run unchanged.

Output: results/phase3/preds/prompt_group_per_ratio__h25.parquet
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


def _fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    seed: int,
    max_iter: int,
    learning_rate: float,
    max_depth: int,
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
            "results/phase3/preds/"
            "prompt_group_per_ratio__h25.parquet"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    feats = list(CHEAP_ALL_FEATURES)
    keep = (
        ["run_id", "prompt_id", "task", "press", "token_pos",
         "compression_ratio", LABEL]
        + [c for c in feats
           if c not in {"token_pos", "compression_ratio"}]
    )
    keep = list(dict.fromkeys(keep))

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

    ratios = sorted(df["compression_ratio"].unique().to_list())
    print(f"[ratios] {ratios}")

    out_chunks: list[pl.DataFrame] = []
    per_ratio_meta: dict[str, dict] = {}
    for r in ratios:
        sub = df.filter(pl.col("compression_ratio") == r)
        y_raw = sub[LABEL].to_numpy().astype(np.float64)
        y = np.log1p(y_raw)
        X = sub.select(feats).to_numpy().astype(np.float32)
        groups = sub["prompt_id"].to_numpy()

        gkf = GroupKFold(n_splits=args.n_folds)
        pred = np.full(X.shape[0], np.nan, dtype=np.float64)
        fold_id = np.full(X.shape[0], "", dtype=object)
        print(f"\n=== ratio={r:.4f}  n={sub.height:,} ===")
        for k, (tr, te) in enumerate(
            gkf.split(X, y, groups=groups)
        ):
            tk = time.time()
            pred[te] = _fit_predict(
                X[tr], y[tr], X[te],
                seed=args.seed + k,
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
            )
            fold_id[te] = f"fold{k}"
            print(f"  [fold{k}] n_tr={len(tr):>9,} "
                  f"n_te={len(te):>9,} {time.time()-tk:.1f}s")

        chunk = pl.DataFrame({
            "run_id": sub["run_id"],
            "prompt_id": sub["prompt_id"],
            "task": sub["task"],
            "press": sub["press"],
            "compression_ratio": sub["compression_ratio"],
            "token_pos": sub["token_pos"],
            "y_raw": y_raw,
            "y_log1p": y,
            "pred_log1p": pred,
            "pred_raw": np.expm1(pred),
            "fold": fold_id.astype(str),
        })
        out_chunks.append(chunk)
        per_ratio_meta[str(r)] = {
            "n_rows": int(sub.height),
            "n_folds": args.n_folds,
        }
        del sub, X, y, groups, pred, fold_id
        gc.collect()

    out = pl.concat(out_chunks, how="vertical")
    out.write_parquet(args.out_path)
    meta = {
        "regime": "per_ratio_per_token_prompt_group",
        "horizon": HORIZON,
        "label": LABEL,
        "feats": feats,
        "n_rows": int(out.height),
        "per_ratio": per_ratio_meta,
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
    print(f"\n[done] wrote {args.out_path}  ({out.height:,} rows)")


if __name__ == "__main__":
    main()
