"""Train a single HERALD v1 regressor on all data and pickle it.

Used by `test_herald_v1_streaming_parity.py` to verify that the
`StreamingHeraldRegressor` produces predictions matching batched
inference within tolerance.

This is NOT the OOF model used for headline numbers; it is a
saved artifact for the streaming demo only.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from herald.predictor_baselines import CHEAP_ALL_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens-path", type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("models/herald_v1_h25.pkl"))
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    feats = list(CHEAP_ALL_FEATURES)
    label = f"future_sum_js_{args.horizon}"
    keep = (
        ["run_id", "prompt_id", "task", "press", "token_pos",
         "compression_ratio", label]
        + [c for c in feats
           if c not in {"token_pos", "compression_ratio"}]
    )
    keep = list(dict.fromkeys(keep))

    print(f"[load] {args.tokens_path}")
    t0 = time.time()
    df = (
        pl.scan_parquet(args.tokens_path)
        .select(keep)
        .filter(pl.col(label).is_not_null())
        .collect()
    )
    df = df.drop_nulls(subset=feats + [label])
    print(f"[load] {df.height:,} rows in {time.time()-t0:.1f}s")

    y_raw = df[label].to_numpy().astype(np.float64)
    y = np.log1p(y_raw)
    X = df.select(feats).to_numpy().astype(np.float32)

    print(f"[fit] HGB max_iter={args.max_iter} "
          f"max_depth={args.max_depth} lr={args.learning_rate}")
    t0 = time.time()
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=args.seed,
    )
    model.fit(X, y)
    print(f"[fit] done in {time.time()-t0:.1f}s; "
          f"n_iter={model.n_iter_}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fp:
        pickle.dump(model, fp)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
