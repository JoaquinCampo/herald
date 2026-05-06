"""Phase 2 baseline-first evaluation (CPU-only).

The headline rule: no XGBoost until cheap baselines are summarized.
This module provides the leakage-safe primitives the Phase 2
baseline runner relies on, plus the runner itself.

Spec: gold/phase-2-dataset.md.

Leakage rules (do not regress):
- Binary thresholds for `future_*` labels are computed from training-
  fold rows only; test rows never enter `compute_train_quantile_threshold`.
- Held-out-prompts splits use `prompt_id` (not `run_id`) as the group:
  the same prompt under different (press, ratio) is one group.
- Held-out-press splits drop `press` from the feature set on both
  sides — leaving it in lets the model fit a constant per side.
"""

import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DEFAULT_QUANTILE = 0.9
DEFAULT_N_BOOT = 200
DEFAULT_BOOT_SEED = 0
DEFAULT_BOOT_ALPHA = 0.05
DEFAULT_BOOT_MAX_N = 200_000


# ---------------------------------------------------------------
# Threshold + binarization
# ---------------------------------------------------------------


def compute_train_quantile_threshold(
    train_values: pl.Series,
    q: float = DEFAULT_QUANTILE,
) -> float | None:
    """Quantile of `train_values`, dropping nulls.

    Returns None if the train set has no usable rows. The baseline
    runner records the None case so it never silently coerces to 0.
    """
    s = train_values.drop_nulls()
    if s.len() == 0:
        return None
    q_val = s.quantile(q, interpolation="linear")
    return None if q_val is None else float(q_val)


def binarize_with_threshold(
    values: pl.Series,
    threshold: float | None,
) -> pl.Series:
    """Binary `values >= threshold`, propagating nulls.

    If `threshold` is None (e.g. the train fold was all-null on this
    label), every output is null. Never coerce a missing threshold to
    a default, that would silently produce all-zero labels.
    """
    if threshold is None:
        return pl.Series(values.name, [None] * values.len(), dtype=pl.Int8)
    out = values.map_elements(
        lambda v: None if v is None else int(v >= threshold),
        return_dtype=pl.Int8,
    )
    return out.alias(values.name)


# ---------------------------------------------------------------
# Splits
# ---------------------------------------------------------------


def iter_splits(
    df: pl.DataFrame,
    kind: str,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Yield (train_idx, test_idx, fold_id) for the requested split.

    kind:
      - "prompts": GroupKFold over `prompt_id` (n_splits folds)
      - "ratios":  leave-one-ratio-out
      - "presses": leave-one-press-out
      - "tasks":   leave-one-task-out
    """
    if kind == "prompts":
        groups = df["prompt_id"].to_numpy()
        n = len(groups)
        gkf = GroupKFold(n_splits=n_splits)
        for fold_i, (tr, te) in enumerate(
            gkf.split(np.arange(n), groups=groups)
        ):
            yield tr.astype(np.int64), te.astype(np.int64), f"fold{fold_i}"
        return

    col_map = {
        "ratios": "compression_ratio",
        "presses": "press",
        "tasks": "task",
    }
    if kind not in col_map:
        raise ValueError(f"unknown split kind: {kind}")
    col = col_map[kind]
    values = df[col].to_numpy()
    unique = sorted(set(values.tolist()))
    for v in unique:
        test_mask = values == v
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx, f"{col}={v}"


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------


def safe_auroc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def safe_auprc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def bootstrap_auroc_ci(
    y_true: list[int],
    y_score: list[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
    alpha: float = DEFAULT_BOOT_ALPHA,
    max_n: int = DEFAULT_BOOT_MAX_N,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for AUROC. Returns (lo, hi) or (None, None)."""
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None, None
    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    if yt.size > max_n:
        idx = rng.choice(yt.size, size=max_n, replace=False)
        yt = yt[idx]
        ys = ys[idx]
    boots: list[float] = []
    n = yt.size
    for _ in range(n_boot):
        sample = rng.integers(0, n, size=n)
        st = yt[sample]
        if np.unique(st).size < 2:
            continue
        try:
            boots.append(float(roc_auc_score(st, ys[sample])))
        except ValueError:
            continue
    if len(boots) < max(10, n_boot // 10):
        return None, None
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def clustered_paired_bootstrap_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
    alpha: float = DEFAULT_BOOT_ALPHA,
) -> dict[str, float | None]:
    """Cluster-bootstrap CI on AUROC(A) - AUROC(B), resampling groups.

    Tokens within a run are not independent (rolling features carry
    within-run state), so a row-level bootstrap underestimates the
    variance of the AUROC. The cluster unit must be the run (or
    prompt). This helper resamples unique values of `groups` with
    replacement, gathers all rows belonging to the resampled groups
    (with multiplicity), and recomputes AUROC for both scores on the
    same resample. Returns the point estimate (no resampling) plus
    the percentile CI.
    """
    yt = np.asarray(y_true)
    sa = np.asarray(score_a, dtype=float)
    sb = np.asarray(score_b, dtype=float)
    gr = np.asarray(groups)
    if yt.size < 2 or len(set(yt.tolist())) < 2:
        return {"delta": None, "delta_lo": None, "delta_hi": None}

    point = (safe_auroc(list(yt), list(sa)) or 0.0) - (
        safe_auroc(list(yt), list(sb)) or 0.0
    )

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(gr)
    if unique_groups.size < 2:
        return {
            "delta": round(point, 4),
            "delta_lo": None,
            "delta_hi": None,
        }
    # Pre-bucket row indices per group to avoid repeated np.where.
    group_to_rows: dict[Any, np.ndarray] = {
        g: np.where(gr == g)[0] for g in unique_groups
    }

    boots: list[float] = []
    n_groups = unique_groups.size
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=n_groups, replace=True)
        idx_lists = [group_to_rows[g] for g in sampled]
        idx = np.concatenate(idx_lists)
        st = yt[idx]
        if np.unique(st).size < 2:
            continue
        try:
            a = float(roc_auc_score(st, sa[idx]))
            b = float(roc_auc_score(st, sb[idx]))
            boots.append(a - b)
        except ValueError:
            continue
    if len(boots) < max(10, n_boot // 10):
        return {
            "delta": round(point, 4),
            "delta_lo": None,
            "delta_hi": None,
        }
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "delta": round(point, 4),
        "delta_lo": round(lo, 4),
        "delta_hi": round(hi, 4),
        "n_boot_used": len(boots),
    }


def cross_fold_clustered_bootstrap(
    fold_predictions: list[dict[str, np.ndarray]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
    alpha: float = DEFAULT_BOOT_ALPHA,
) -> dict[str, Any]:
    """Cluster-bootstrap CI on the cross-fold mean of AUROC(A)-AUROC(B).

    Each entry in `fold_predictions` must have keys
    {"y_true", "score_a", "score_b", "groups"}. The procedure:
    1. Per replicate, for each fold, resample unique groups with
       replacement and recompute AUROC(A) and AUROC(B) on the
       resampled rows.
    2. Per-fold delta = AUROC(A) - AUROC(B).
    3. Cross-fold mean delta = mean over folds of step 2.
    4. CI = percentile of step 3 across replicates.

    Folds where the resampled labels are degenerate are dropped from
    that replicate's mean (NaN); a replicate is discarded if all folds
    are degenerate.
    """
    if not fold_predictions:
        return {
            "delta_mean": None,
            "delta_lo": None,
            "delta_hi": None,
            "n_folds": 0,
        }
    point_per_fold: list[float] = []
    for fp in fold_predictions:
        a = safe_auroc(list(fp["y_true"]), list(fp["score_a"]))
        b = safe_auroc(list(fp["y_true"]), list(fp["score_b"]))
        if a is None or b is None:
            continue
        point_per_fold.append(a - b)
    if not point_per_fold:
        return {
            "delta_mean": None,
            "delta_lo": None,
            "delta_hi": None,
            "n_folds": 0,
        }
    point_mean = float(np.mean(point_per_fold))

    rng = np.random.default_rng(seed)
    FoldCache = tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[Any, np.ndarray],
    ]
    fold_caches: list[FoldCache] = []
    for fp in fold_predictions:
        gr = np.asarray(fp["groups"])
        unique = np.unique(gr)
        bucket = {g: np.where(gr == g)[0] for g in unique}
        fold_caches.append(
            (
                np.asarray(fp["y_true"]),
                np.asarray(fp["score_a"], dtype=float),
                np.asarray(fp["score_b"], dtype=float),
                unique,
                bucket,
            )
        )

    boots: list[float] = []
    for _ in range(n_boot):
        per_fold_d: list[float] = []
        for yt, sa, sb, unique, bucket in fold_caches:
            if unique.size < 2:
                continue
            sampled = rng.choice(unique, size=unique.size, replace=True)
            idx = np.concatenate([bucket[g] for g in sampled])
            st = yt[idx]
            if np.unique(st).size < 2:
                continue
            try:
                a = float(roc_auc_score(st, sa[idx]))
                b = float(roc_auc_score(st, sb[idx]))
            except ValueError:
                continue
            per_fold_d.append(a - b)
        if per_fold_d:
            boots.append(float(np.mean(per_fold_d)))

    if len(boots) < max(10, n_boot // 10):
        return {
            "delta_mean": round(point_mean, 4),
            "delta_lo": None,
            "delta_hi": None,
            "n_folds": len(point_per_fold),
            "n_boot_used": len(boots),
        }
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "delta_mean": round(point_mean, 4),
        "delta_lo": round(lo, 4),
        "delta_hi": round(hi, 4),
        "n_folds": len(point_per_fold),
        "n_boot_used": len(boots),
    }


def paired_bootstrap_delta(
    y_true: list[int],
    score_a: list[float],
    score_b: list[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
    alpha: float = DEFAULT_BOOT_ALPHA,
    max_n: int = DEFAULT_BOOT_MAX_N,
) -> dict[str, float | None]:
    """Paired-bootstrap CI on AUROC(A) - AUROC(B)."""
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return {"delta": None, "delta_lo": None, "delta_hi": None}
    yt = np.asarray(y_true)
    sa = np.asarray(score_a, dtype=float)
    sb = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    if yt.size > max_n:
        idx = rng.choice(yt.size, size=max_n, replace=False)
        yt = yt[idx]
        sa = sa[idx]
        sb = sb[idx]
    boots: list[float] = []
    n = yt.size
    for _ in range(n_boot):
        sample = rng.integers(0, n, size=n)
        st = yt[sample]
        if np.unique(st).size < 2:
            continue
        try:
            a = float(roc_auc_score(st, sa[sample]))
            b = float(roc_auc_score(st, sb[sample]))
            boots.append(a - b)
        except ValueError:
            continue
    if len(boots) < max(10, n_boot // 10):
        return {"delta": None, "delta_lo": None, "delta_hi": None}
    point = (safe_auroc(y_true, list(sa)) or 0.0) - (
        safe_auroc(y_true, list(sb)) or 0.0
    )
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {
        "delta": round(point, 4),
        "delta_lo": round(lo, 4),
        "delta_hi": round(hi, 4),
    }


# ---------------------------------------------------------------
# Single-feature scoring
# ---------------------------------------------------------------


# Features where lower value = more dangerous: negate them so AUROC
# always interprets larger score as more likely positive.
INVERSE_FEATURES: frozenset[str] = frozenset(
    {
        "top1_prob",
        "top5_prob",
        "top10_jaccard",
        "top1_prob_mean_8",
        "top1_prob_mean_32",
        "top1_prob_ewma_hl8",
        "top1_prob_ewma_hl32",
        "top10_jaccard_mean_8",
        "top10_jaccard_mean_32",
        "top10_jaccard_ewma_hl8",
        "top10_jaccard_ewma_hl32",
    }
)


def directional_score(
    df: pl.DataFrame,
    feature: str,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Return the per-row score for a single-feature baseline.

    Negates inverse features so larger always means more dangerous.
    Nulls are filled with `fill_value` because AUROC needs a numeric
    score; the rows that are dropped from y at evaluation time are
    chosen by null label, not by null score.
    """
    raw = df[feature].fill_null(fill_value).to_numpy().astype(float)
    raw = np.nan_to_num(
        raw, nan=fill_value, posinf=fill_value, neginf=fill_value
    )
    if feature in INVERSE_FEATURES:
        return np.asarray(-raw, dtype=float)
    return np.asarray(raw, dtype=float)


# ---------------------------------------------------------------
# Baseline definitions
# ---------------------------------------------------------------


CHEAP_TIER0_FEATURES: tuple[str, ...] = (
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "kl_div",
    "top10_jaccard",
    "tail_mass",
    "logit_range",
    "eff_vocab_size",
)
CHEAP_ALL_FEATURES: tuple[str, ...] = CHEAP_TIER0_FEATURES + (
    "entropy_mean_8",
    "entropy_std_8",
    "entropy_mean_32",
    "entropy_ewma_hl8",
    "entropy_ewma_hl32",
    "top1_prob_mean_8",
    "top1_prob_mean_32",
    "top1_prob_ewma_hl8",
    "kl_div_mean_8",
    "h_alts_mean_8",
    "top10_jaccard_mean_8",
    "top10_jaccard_mean_32",
    "token_pos",
    "relative_progress",
    "compression_ratio",
)

# The "metadata" baseline must one-hot task and press; ratio is
# already numeric.
METADATA_CATEGORICAL: tuple[str, ...] = ("task", "press")
METADATA_NUMERIC: tuple[str, ...] = ("compression_ratio",)


def _onehot_columns(
    df: pl.DataFrame,
    cols: tuple[str, ...],
    level_overrides: dict[str, list[Any]] | None = None,
) -> np.ndarray:
    """One-hot encode `cols` of `df`.

    `level_overrides`: per-column list of levels to use. If supplied,
    that fixed level list is used; otherwise levels are taken from
    the unique values in `df` itself. The override is required when
    train and test halves can have disjoint level sets (e.g. held-
    out-task split: train has 3 tasks, test has 1) — otherwise the
    train-fit scaler and the test transform see different shapes.
    """
    parts: list[np.ndarray] = []
    for c in cols:
        if c not in df.columns:
            continue
        if level_overrides and c in level_overrides:
            levels = level_overrides[c]
        else:
            levels = sorted(df[c].drop_nulls().unique().to_list())
        for lvl in levels:
            parts.append(
                (df[c] == lvl).fill_null(False).to_numpy().astype(float)
            )
    if not parts:
        return np.zeros((df.height, 0), dtype=float)
    return np.column_stack(parts)


def _clean_X(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0 for sklearn compatibility.

    Several Tier 0 features carry NaN at token_pos=0 (delta_h, kl_div,
    top10_jaccard) by construction, and rolling_std with min_samples=2
    is null at the first row of each run — fill_null doesn't catch
    NaN, so handle it explicitly here.
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _logreg_score(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
) -> np.ndarray | None:
    """Standardize, fit LR, return predicted positive-class probability."""
    if train_X.shape[0] < 4 or len(set(train_y.tolist())) < 2:
        return None
    train_X = _clean_X(train_X)
    test_X = _clean_X(test_X)
    pipe_scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = pipe_scaler.fit_transform(train_X)
    Xt = pipe_scaler.transform(test_X)
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    # solver="lbfgs" uses BLAS for the gradient step; on Mac the
    # Accelerate framework parallelises matmul implicitly.
    # liblinear is single-threaded coordinate descent (~3-10x slower).
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(Xs, train_y)
    return np.asarray(clf.predict_proba(Xt)[:, 1], dtype=float)


# ---------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------


def _build_baseline_scores(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_y: np.ndarray,
    test_y: np.ndarray,
    drop_press_feature: bool,
) -> dict[str, np.ndarray]:
    """Compute per-baseline scores on the test set."""
    n_test = test_df.height
    out: dict[str, np.ndarray] = {}

    # Random.
    rng = np.random.default_rng(0)
    out["random"] = rng.uniform(size=n_test)

    # Position-only.
    out["position"] = test_df["token_pos"].to_numpy().astype(float)
    if "relative_progress" in test_df.columns:
        out["relative_progress"] = (
            test_df["relative_progress"].to_numpy().astype(float)
        )

    # Ratio-only.
    if "compression_ratio" in test_df.columns:
        out["ratio"] = test_df["compression_ratio"].to_numpy().astype(float)

    # Single-feature thresholds.
    for feat in CHEAP_TIER0_FEATURES + (
        "entropy_mean_8",
        "entropy_ewma_hl8",
        "kl_div_mean_8",
        "top1_prob_mean_8",
    ):
        if feat in test_df.columns:
            out[f"feature::{feat}"] = directional_score(test_df, feat)

    # Metadata-only LR (task, press, ratio).
    meta_cat = METADATA_CATEGORICAL
    if drop_press_feature:
        meta_cat = tuple(c for c in meta_cat if c != "press")
    # Build the union of levels across train+test so train_X and test_X
    # have matching shape (held-out-task / held-out-press splits put
    # disjoint level sets on each side).
    level_overrides: dict[str, list[Any]] = {}
    for c in meta_cat:
        if c not in train_df.columns:
            continue
        union = sorted(
            set(train_df[c].drop_nulls().to_list())
            | set(test_df[c].drop_nulls().to_list())
        )
        level_overrides[c] = union
    train_meta_X = np.column_stack(
        [
            _onehot_columns(train_df, meta_cat, level_overrides),
            train_df.select(METADATA_NUMERIC).to_numpy().astype(float),
        ]
    )
    test_meta_X = np.column_stack(
        [
            _onehot_columns(test_df, meta_cat, level_overrides),
            test_df.select(METADATA_NUMERIC).to_numpy().astype(float),
        ]
    )
    s = _logreg_score(train_meta_X, train_y, test_meta_X)
    if s is not None:
        out["metadata_lr"] = s

    # LR on Tier 0 features.
    tier0 = [c for c in CHEAP_TIER0_FEATURES if c in train_df.columns]
    if tier0:
        s = _logreg_score(
            train_df.select(tier0).fill_null(0.0).to_numpy().astype(float),
            train_y,
            test_df.select(tier0).fill_null(0.0).to_numpy().astype(float),
        )
        if s is not None:
            out["lr_tier0"] = s

    # LR on all cheap features.
    all_cheap = [c for c in CHEAP_ALL_FEATURES if c in train_df.columns]
    if drop_press_feature:
        all_cheap = [c for c in all_cheap if c != "compression_ratio"]
        # Note: ratio kept generally; only dropped when ratio is the
        # split itself.  Press is never a numeric feature; skip.
    if all_cheap:
        s = _logreg_score(
            train_df.select(all_cheap)
            .fill_null(0.0)
            .to_numpy()
            .astype(float),
            train_y,
            test_df.select(all_cheap).fill_null(0.0).to_numpy().astype(float),
        )
        if s is not None:
            out["lr_all_cheap"] = s

    return out


def collect_fold_predictions(
    df: pl.DataFrame,
    label_col: str,
    splits: tuple[str, ...] = ("prompts", "ratios", "presses", "tasks"),
    n_prompt_folds: int = 5,
    threshold_q: float = DEFAULT_QUANTILE,
    seed: int = DEFAULT_BOOT_SEED,
    baselines_to_keep: tuple[str, ...] = (
        "lr_all_cheap",
        "feature::entropy_mean_8",
    ),
    group_col: str = "run_id",
) -> list[dict[str, Any]]:
    """Return per-fold predictions for paired CI computation.

    For each (split_kind, fold_id), retrains the requested baselines
    on the training half and returns the test-half predictions plus
    the cluster identifier (default `run_id`). Output is one dict per
    fold containing y_true, scores per baseline, and groups. Folds
    where the test labels are degenerate (single class) are skipped.
    """
    out: list[dict[str, Any]] = []
    if label_col not in df.columns:
        raise ValueError(f"label_col {label_col!r} missing from df")
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} missing from df")

    for kind in splits:
        drop_press = kind == "presses"
        for tr_idx, te_idx, fold_id in iter_splits(
            df, kind=kind, n_splits=n_prompt_folds, seed=seed
        ):
            train_df = df[tr_idx]
            test_df = df[te_idx]
            train_y_cont = train_df[label_col]
            test_y_cont = test_df[label_col]
            thr = compute_train_quantile_threshold(
                train_y_cont, q=threshold_q
            )
            train_y_bin = binarize_with_threshold(train_y_cont, thr)
            test_y_bin = binarize_with_threshold(test_y_cont, thr)

            test_mask = ~test_y_bin.is_null()
            keep_test_y = (
                test_y_bin.filter(test_mask).cast(pl.Int64).to_numpy()
            )
            keep_test_df = test_df.filter(test_mask)
            train_mask = ~train_y_bin.is_null()
            keep_train_df = train_df.filter(train_mask)
            keep_train_y = (
                train_y_bin.filter(train_mask).cast(pl.Int64).to_numpy()
            )

            if (
                keep_test_y.size < 2
                or len(set(keep_test_y.tolist())) < 2
                or keep_train_y.size < 4
                or len(set(keep_train_y.tolist())) < 2
            ):
                continue

            scores = _build_baseline_scores(
                keep_train_df,
                keep_test_df,
                keep_train_y,
                keep_test_y,
                drop_press_feature=drop_press,
            )
            entry: dict[str, Any] = {
                "split_kind": kind,
                "fold_id": fold_id,
                "label": label_col,
                "threshold": thr,
                "n_test": int(keep_test_df.height),
                "y_true": keep_test_y,
                "groups": keep_test_df[group_col].to_numpy(),
            }
            for b in baselines_to_keep:
                if b in scores and scores[b].shape[0] == keep_test_y.shape[0]:
                    entry[f"score::{b}"] = scores[b]
            out.append(entry)
    return out


def evaluate_split(
    df: pl.DataFrame,
    label_col: str,
    threshold_q: float = DEFAULT_QUANTILE,
    splits: tuple[str, ...] = ("prompts", "ratios", "presses", "tasks"),
    n_prompt_folds: int = 5,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
) -> list[dict[str, Any]]:
    """For each split kind / fold / baseline, compute AUROC + AUPRC.

    Threshold is computed on training rows only and applied to the
    test fold to derive the binary label. Returns one dict per
    (split_kind, fold_id, baseline).
    """
    rows: list[dict[str, Any]] = []
    if label_col not in df.columns:
        raise ValueError(f"label_col {label_col!r} missing from df")

    for kind in splits:
        drop_press = kind == "presses"
        for tr_idx, te_idx, fold_id in iter_splits(
            df, kind=kind, n_splits=n_prompt_folds, seed=seed
        ):
            train_df = df[tr_idx]
            test_df = df[te_idx]
            train_y_cont = train_df[label_col]
            test_y_cont = test_df[label_col]
            thr = compute_train_quantile_threshold(
                train_y_cont, q=threshold_q
            )
            train_y_bin = binarize_with_threshold(train_y_cont, thr)
            test_y_bin = binarize_with_threshold(test_y_cont, thr)

            test_mask = ~test_y_bin.is_null()
            keep_test = test_y_bin.filter(test_mask).cast(pl.Int64).to_numpy()
            keep_test_df = test_df.filter(test_mask)

            train_mask = ~train_y_bin.is_null()
            keep_train_df = train_df.filter(train_mask)
            keep_train_y = (
                train_y_bin.filter(train_mask).cast(pl.Int64).to_numpy()
            )

            if keep_test.size < 2 or len(set(keep_test.tolist())) < 2:
                rows.append(
                    {
                        "split_kind": kind,
                        "fold_id": fold_id,
                        "label": label_col,
                        "threshold": thr,
                        "n_train": int(keep_train_df.height),
                        "n_test": int(keep_test_df.height),
                        "train_pos_rate": (
                            float((keep_train_y == 1).mean())
                            if keep_train_y.size
                            else None
                        ),
                        "test_pos_rate": (
                            float((keep_test == 1).mean())
                            if keep_test.size
                            else None
                        ),
                        "baseline": "_skipped",
                        "auroc": None,
                        "auprc": None,
                        "auroc_lo": None,
                        "auroc_hi": None,
                        "skipped_reason": "degenerate test labels",
                    }
                )
                continue

            scores = _build_baseline_scores(
                keep_train_df,
                keep_test_df,
                keep_train_y,
                keep_test,
                drop_press_feature=drop_press,
            )
            for bname, sarr in scores.items():
                if sarr.shape[0] != keep_test.shape[0]:
                    continue
                a = safe_auroc(list(keep_test), list(sarr))
                p = safe_auprc(list(keep_test), list(sarr))
                lo, hi = bootstrap_auroc_ci(
                    list(keep_test), list(sarr), n_boot=n_boot, seed=seed
                )
                rows.append(
                    {
                        "split_kind": kind,
                        "fold_id": fold_id,
                        "label": label_col,
                        "threshold": thr,
                        "n_train": int(keep_train_df.height),
                        "n_test": int(keep_test_df.height),
                        "train_pos_rate": float((keep_train_y == 1).mean()),
                        "test_pos_rate": float((keep_test == 1).mean()),
                        "baseline": bname,
                        "auroc": a,
                        "auprc": p,
                        "auroc_lo": lo,
                        "auroc_hi": hi,
                        "skipped_reason": None,
                    }
                )
            logger.info(
                "{} fold {}: n_train={} n_test={} pos_rate={:.3f} thr={:.4f}",
                kind,
                fold_id,
                keep_train_df.height,
                keep_test_df.height,
                (keep_test == 1).mean(),
                thr if thr is not None else math.nan,
            )

    return rows


def summarize_baselines(
    results: list[dict[str, Any]],
    cheap_baselines: tuple[str, ...] = (
        "feature::entropy",
        "feature::entropy_mean_8",
        "feature::entropy_ewma_hl8",
    ),
) -> dict[str, Any]:
    """Per (split_kind, label): identify the best baseline and the
    paired delta of `lr_tier0` / `lr_all_cheap` versus the best cheap
    baseline. Decision rule: ≥ 0.05 AUROC and CI not crossing zero."""
    df = pl.DataFrame(results)
    summary: dict[str, Any] = {}
    if df.is_empty():
        return summary

    for (split_kind, label), sub in df.group_by(["split_kind", "label"]):
        sub_clean = sub.filter(pl.col("auroc").is_not_null())
        if sub_clean.is_empty():
            continue
        per_baseline = (
            sub_clean.group_by("baseline")
            .agg(
                pl.col("auroc").mean().alias("mean_auroc"),
                pl.col("auprc").mean().alias("mean_auprc"),
                pl.col("auroc_lo").mean().alias("mean_auroc_lo"),
                pl.col("auroc_hi").mean().alias("mean_auroc_hi"),
                pl.len().alias("n_folds"),
            )
            .sort("mean_auroc", descending=True)
        )
        cheap_present = per_baseline.filter(
            pl.col("baseline").is_in(list(cheap_baselines))
        )
        best_cheap = (
            cheap_present.head(1).to_dicts()[0]
            if not cheap_present.is_empty()
            else None
        )
        best_overall = per_baseline.head(1).to_dicts()[0]
        target_keys = ("lr_tier0", "lr_all_cheap")
        target_rows = {}
        for tk in target_keys:
            t = per_baseline.filter(pl.col("baseline") == tk).to_dicts()
            target_rows[tk] = t[0] if t else None
        decision: dict[str, Any] = {}
        if best_cheap is not None:
            for tk, trow in target_rows.items():
                if trow is None:
                    decision[tk] = "missing"
                    continue
                delta = trow["mean_auroc"] - best_cheap["mean_auroc"]
                decision[tk] = {
                    "delta_vs_best_cheap": round(delta, 4),
                    "best_cheap_baseline": best_cheap["baseline"],
                    "best_cheap_auroc": round(best_cheap["mean_auroc"], 4),
                    "passes_005_bar": (
                        delta >= 0.05 if delta is not None else None
                    ),
                }
        summary[f"{split_kind}::{label}"] = {
            "best_overall": best_overall,
            "best_cheap": best_cheap,
            "decision": decision,
            "n_baselines": int(per_baseline.height),
        }
    return summary


__all__ = [
    "CHEAP_ALL_FEATURES",
    "CHEAP_TIER0_FEATURES",
    "DEFAULT_BOOT_ALPHA",
    "DEFAULT_BOOT_SEED",
    "DEFAULT_N_BOOT",
    "DEFAULT_QUANTILE",
    "INVERSE_FEATURES",
    "binarize_with_threshold",
    "bootstrap_auroc_ci",
    "clustered_paired_bootstrap_delta",
    "collect_fold_predictions",
    "compute_train_quantile_threshold",
    "cross_fold_clustered_bootstrap",
    "directional_score",
    "evaluate_split",
    "iter_splits",
    "paired_bootstrap_delta",
    "safe_auprc",
    "safe_auroc",
    "summarize_baselines",
]
