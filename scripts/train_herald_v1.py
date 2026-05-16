"""HERALD v1 full training: per-token regressor + transfer slices.

For each horizon H in {5, 10, 25, 50}, train HistGradientBoosting
regressor on log1p(future_sum_js_H) with feature set
`CHEAP_ALL_FEATURES`. Run four split strategies:

  1. GroupKFold(5) by prompt_id (in-distribution headline).
  2. Held-out press (6 leave-one-out folds).
  3. Held-out compression_ratio (LOO folds).
  4. Held-out task (LOO folds).

For each (split_kind, fold, horizon) we save OOF predictions to
`results/phase3/preds/{split_kind}__h{H}.parquet`. Eval is a
separate stage so re-scoring is cheap.

Designed to run on CPU; uses Polars streaming where possible.
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

SPLIT_KINDS = ("prompt_group", "loo_press", "loo_ratio", "loo_task")


def _load(
    path: Path, horizon: int, feats: list[str]
) -> pl.DataFrame:
    label = f"future_sum_js_{horizon}"
    keep = (
        ["run_id", "prompt_id", "task", "press", "token_pos",
         "compression_ratio", label]
        + [c for c in feats
           if c not in {"token_pos", "compression_ratio"}]
    )
    keep = list(dict.fromkeys(keep))
    t0 = time.time()
    df = (
        pl.scan_parquet(path)
        .select(keep)
        .filter(pl.col(label).is_not_null())
        .collect()
    )
    df = df.drop_nulls(subset=feats + [label])
    print(f"[load h={horizon}] {df.height:,} rows in "
          f"{time.time()-t0:.1f}s")
    return df


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


def _splits_prompt(
    df: pl.DataFrame, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    groups = df["prompt_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_folds)
    X_idx = np.arange(df.height)
    out = []
    for k, (tr, te) in enumerate(
        gkf.split(X_idx, X_idx, groups=groups)
    ):
        out.append((tr, te, f"fold{k}"))
    return out


def _splits_loo(
    df: pl.DataFrame, col: str
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    vals = sorted(df[col].unique().to_list())
    levels = df[col].to_numpy()
    X_idx = np.arange(df.height)
    out = []
    for v in vals:
        te = X_idx[levels == v]
        tr = X_idx[levels != v]
        out.append((tr, te, f"hold_{v}"))
    return out


def _train_split(
    df: pl.DataFrame,
    feats: list[str],
    label: str,
    splits: list[tuple[np.ndarray, np.ndarray, str]],
    seed: int,
    max_iter: int,
    learning_rate: float,
    max_depth: int,
) -> pl.DataFrame:
    y_raw = df[label].to_numpy().astype(np.float64)
    y = np.log1p(y_raw)
    X = df.select(feats).to_numpy().astype(np.float32)

    pred = np.full(X.shape[0], np.nan, dtype=np.float64)
    fold_id = np.full(X.shape[0], "", dtype=object)
    for tr, te, name in splits:
        t0 = time.time()
        pred[te] = _fit_predict(
            X[tr], y[tr], X[te],
            seed=seed,
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        fold_id[te] = name
        print(f"  [{name}] n_tr={len(tr):>9,} n_te={len(te):>9,} "
              f"{time.time()-t0:.1f}s")

    return pl.DataFrame({
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens-path", type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"))
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("results/phase3/preds"))
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[5, 10, 25, 50])
    parser.add_argument("--splits", type=str, nargs="+",
                        default=list(SPLIT_KINDS),
                        choices=SPLIT_KINDS)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--row-cap", type=int, default=None,
                        help="If set, uniformly sample this many rows.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feats = list(CHEAP_ALL_FEATURES)

    for h in args.horizons:
        label = f"future_sum_js_{h}"
        df = _load(args.tokens_path, horizon=h, feats=feats)
        if args.row_cap is not None and df.height > args.row_cap:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(
                df.height, size=args.row_cap, replace=False)
            df = df[idx]
            print(f"[h={h}] row_cap → {df.height:,}")

        for kind in args.splits:
            print(f"\n=== h={h} split={kind} ===")
            t0 = time.time()
            if kind == "prompt_group":
                splits = _splits_prompt(df, args.n_folds, args.seed)
            elif kind == "loo_press":
                splits = _splits_loo(df, "press")
            elif kind == "loo_ratio":
                splits = _splits_loo(df, "compression_ratio")
            elif kind == "loo_task":
                splits = _splits_loo(df, "task")
            else:
                raise ValueError(kind)

            preds = _train_split(
                df=df, feats=feats, label=label,
                splits=splits, seed=args.seed,
                max_iter=args.max_iter,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
            )
            out_path = args.out_dir / f"{kind}__h{h}.parquet"
            preds.write_parquet(out_path)
            meta = {
                "split_kind": kind,
                "horizon": h,
                "n_rows": int(preds.height),
                "n_folds": len(splits),
                "elapsed_seconds": time.time() - t0,
                "feats": feats,
                "label": label,
                "model": "HistGradientBoostingRegressor",
                "hparams": {
                    "max_iter": args.max_iter,
                    "learning_rate": args.learning_rate,
                    "max_depth": args.max_depth,
                    "min_samples_leaf": 200,
                    "l2_regularization": 1.0,
                },
                "seed": args.seed,
            }
            (out_path.with_suffix(".meta.json")).write_text(
                json.dumps(meta, indent=2)
            )
            print(f"[h={h} {kind}] wrote {out_path} "
                  f"({time.time()-t0:.0f}s)")

        del df
        gc.collect()


if __name__ == "__main__":
    main()
