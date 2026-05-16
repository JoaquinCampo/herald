"""Slope/trend per-token features targeting bar 6 worst slice.

Direction-of-change is the single per-token signal class our
canonical CHEAP_ALL_FEATURES set never carries. Rolling means
and EWMAs encode level; std-rollings encode volatility; neither
captures whether entropy / kl / delta_h are *trending up* over a
short window. Heavy compression is hypothesised to manifest as
sustained drift rather than as a level shift the rolling-mean
features have already absorbed.

We add four extras on top of `CHEAP_ALL_FEATURES`:

1. Slope features (8): window OLS slope over windows 8 and 32 on
   entropy, kl_div, delta_h, top1_prob. Closed-form weighted sum,
   O(1)-streamable per token. Computed per-run via convolution
   over `token_pos`-sorted arrays.
2. Press one-hot (6): never used as a per-token input. Free.
3. Ratio x signal interactions (4): compression_ratio multiplied
   by entropy, kl_div, delta_h, top1_prob. Lets the model bend
   the response curve per regime without a per-ratio retrain.
4. relative_progress: position-in-run normalised, mentioned in the
   parquet schema, not in CHEAP_ALL.

Excluded as oracle / leaky: js_full, kl_unc_comp_full
(per-token paired divergences vs uncompressed).

Pre-committed kill criterion (per advisor consult, 2026-05-15):
worst-slice per-token rho at ratio=0.375 must clear 0.65 (current
0.530, extfeat 0.536). If not, the per-token substrate is
empirically exhausted (third independent triangulation) and we
revert to 6/9 framing.

Output: results/phase3/preds/prompt_group_slopefeat__h25.parquet
"""

import argparse
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

SLOPE_BASES: tuple[str, ...] = (
    "entropy", "kl_div", "delta_h", "top1_prob",
)
SLOPE_WINDOWS: tuple[int, ...] = (8, 32)
INTERACTION_BASES: tuple[str, ...] = (
    "entropy", "kl_div", "delta_h", "top1_prob",
)
PRESSES: tuple[str, ...] = (
    "expected_attention", "knorm", "random", "snapkv",
    "streaming_llm", "tova",
)


def _slope_weights(window: int) -> np.ndarray:
    """Closed-form OLS slope weights for x[t-W+1..t] vs k=0..W-1.

    slope = (12 / (W * (W^2 - 1))) * sum_k (k - (W-1)/2) * x[k].
    """
    k = np.arange(window, dtype=np.float64)
    centred = k - (window - 1) / 2.0
    return centred * (12.0 / (window * (window * window - 1)))


def _slope_for_run(
    values: np.ndarray, window: int, weights: np.ndarray,
) -> np.ndarray:
    """Causal rolling slope. Pads first W-1 tokens with NaN."""
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    flipped = weights[::-1]
    conv = np.convolve(values, flipped, mode="valid")
    out[window - 1:] = conv
    return out


def _add_slope_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Compute slope columns per run via numpy groupby."""
    df = df.sort(["run_id", "token_pos"])
    run_ids = df["run_id"].to_numpy()
    boundaries = np.where(run_ids[1:] != run_ids[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(run_ids)]])

    cols: dict[str, np.ndarray] = {}
    for base in SLOPE_BASES:
        base_arr = df[base].to_numpy().astype(np.float64)
        for win in SLOPE_WINDOWS:
            weights = _slope_weights(win)
            out = np.full(len(base_arr), np.nan, dtype=np.float64)
            for s, e in zip(starts, ends, strict=True):
                vals = base_arr[s:e]
                if np.isnan(vals).any():
                    vals = np.nan_to_num(vals, nan=0.0)
                out[s:e] = _slope_for_run(vals, win, weights)
            cols[f"{base}_slope_{win}"] = out
    return df.with_columns([
        pl.Series(name, arr) for name, arr in cols.items()
    ])


def _add_interaction_and_onehot(df: pl.DataFrame) -> pl.DataFrame:
    ratio = df["compression_ratio"].to_numpy().astype(np.float64)
    new_cols: list[pl.Series] = []
    for base in INTERACTION_BASES:
        base_arr = df[base].to_numpy().astype(np.float64)
        new_cols.append(pl.Series(f"ratio_x_{base}", ratio * base_arr))
    press_arr = df["press"].to_numpy()
    for press in PRESSES:
        new_cols.append(
            pl.Series(f"press_is_{press}",
                      (press_arr == press).astype(np.float32))
        )
    return df.with_columns(new_cols)


def _bootstrap_ci_spearman(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray,
    n_boot: int = 500, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    unique_groups, group_codes = np.unique(groups, return_inverse=True)
    order = np.argsort(group_codes, kind="stable")
    sorted_codes = group_codes[order]
    edges = np.concatenate(
        [[0], np.flatnonzero(np.diff(sorted_codes)) + 1,
         [len(sorted_codes)]])
    group_idx: list[np.ndarray] = [
        order[edges[i]:edges[i + 1]] for i in range(len(unique_groups))
    ]
    overall, _ = spearmanr(y, p)
    boots = []
    n_groups = len(unique_groups)
    for _ in range(n_boot):
        chosen = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([group_idx[int(g)] for g in chosen])
        rho, _ = spearmanr(y[idx], p[idx])
        boots.append(rho)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(overall), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens-path", type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"))
    parser.add_argument(
        "--out-path", type=Path,
        default=Path(
            "results/phase3/preds/prompt_group_slopefeat__h25.parquet"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument(
        "--diag-out", type=Path,
        default=Path("results/phase3/per_token_per_ratio_slopefeat.json"))
    args = parser.parse_args()

    base_cols = list(CHEAP_ALL_FEATURES)
    keep_core = [
        "run_id", "prompt_id", "task", "press", "token_pos",
        "compression_ratio", LABEL, "relative_progress",
    ]
    signals_needed = set(SLOPE_BASES) | set(INTERACTION_BASES)
    keep = list(dict.fromkeys(
        keep_core + base_cols + list(signals_needed)
    ))

    print(f"[load] {args.tokens_path}")
    t0 = time.time()
    df = (
        pl.scan_parquet(args.tokens_path)
        .select(keep)
        .filter(pl.col(LABEL).is_not_null())
        .collect()
    )
    print(f"[load] {df.height:,} rows in {time.time()-t0:.1f}s")

    print("[features] building slope cols...")
    t0 = time.time()
    df = _add_slope_columns(df)
    print(f"[features] slope cols done in {time.time()-t0:.1f}s")

    print("[features] adding interaction + press one-hot...")
    df = _add_interaction_and_onehot(df)

    feats = list(base_cols)
    if "relative_progress" not in feats:
        feats.append("relative_progress")
    for base in SLOPE_BASES:
        for win in SLOPE_WINDOWS:
            feats.append(f"{base}_slope_{win}")
    for base in INTERACTION_BASES:
        feats.append(f"ratio_x_{base}")
    for press in PRESSES:
        feats.append(f"press_is_{press}")

    feats = list(dict.fromkeys(feats))
    feats = [c for c in feats if c in df.columns]
    print(f"[features] {len(feats)} total "
          f"(base {len(base_cols)} + slope "
          f"{len(SLOPE_BASES) * len(SLOPE_WINDOWS)} + "
          f"interaction {len(INTERACTION_BASES)} + press "
          f"{len(PRESSES)} + 1 progress)")

    df = df.drop_nulls(subset=[LABEL])

    y_raw = df[LABEL].to_numpy().astype(np.float64)
    y = np.log1p(y_raw)
    feat_mat = df.select(feats).to_numpy().astype(np.float64)
    feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)
    X = feat_mat.astype(np.float32)
    groups = df["prompt_id"].to_numpy()

    gkf = GroupKFold(n_splits=args.n_folds)
    pred = np.full(X.shape[0], np.nan, dtype=np.float64)
    fold_id = np.full(X.shape[0], "", dtype=object)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
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
            random_state=args.seed + k,
        )
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
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
        "regime": "slopefeat_prompt_group",
        "horizon": HORIZON,
        "label": LABEL,
        "feats": feats,
        "n_feats": len(feats),
        "n_rows": int(out.height),
        "slope_bases": list(SLOPE_BASES),
        "slope_windows": list(SLOPE_WINDOWS),
        "interaction_bases": list(INTERACTION_BASES),
        "presses": list(PRESSES),
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

    print()
    print("[diag] per-ratio per-token rho")
    diag: dict[str, dict[str, float]] = {}
    rho_all, lo_all, hi_all = _bootstrap_ci_spearman(
        y_raw, pred, groups, n_boot=200, seed=args.seed,
    )
    print(f"  overall   rho={rho_all:.4f} [{lo_all:.4f}, {hi_all:.4f}]")
    diag["overall"] = {"rho": rho_all, "lo": lo_all, "hi": hi_all}
    ratios = sorted(df["compression_ratio"].unique().to_list())
    ratio_arr = df["compression_ratio"].to_numpy()
    for r in ratios:
        mask = ratio_arr == r
        rho, lo, hi = _bootstrap_ci_spearman(
            y_raw[mask], pred[mask], groups[mask],
            n_boot=200, seed=args.seed,
        )
        n = int(mask.sum())
        print(f"  ratio={r:.4f}  rho={rho:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"n={n}")
        diag[f"ratio_{r}"] = {
            "rho": rho, "lo": lo, "hi": hi, "n": n,
        }
    args.diag_out.parent.mkdir(parents=True, exist_ok=True)
    args.diag_out.write_text(json.dumps(diag, indent=2))
    print(f"[diag] wrote {args.diag_out}")

    worst = min(
        diag[k]["rho"] for k in diag if k.startswith("ratio_")
    )
    kill = 0.65
    print()
    print(f"[kill-criterion] worst-slice rho = {worst:.4f}, "
          f"threshold = {kill:.2f}")
    if worst >= kill:
        print("[kill-criterion] PASS — proceed to oracle wrapper "
              "retention re-check.")
    else:
        print("[kill-criterion] FAIL — substrate empirically "
              "exhausted. Revert to 6/9 framing.")


if __name__ == "__main__":
    main()
