"""Feature-information / information-ceiling analysis.

Quantifies whether cheap online features carry information about
future compression damage *beyond* trivial position and metadata
proxies (press, ratio, task). This is the HERALD rebuttal to the
LimitsLearned-style concern that the predictor's signal might be
indistinguishable from `position + ratio` alone.

Method
------
1. Build a per-token table for compressed runs
   (press != 'none'), joining run-level (task, press, ratio) onto the
   token-level features. Future-damage labels per horizon H are
   `1` iff the run's catastrophic onset falls in `(t, t+H]` and the
   run is not yet at/past onset. Tokens at or past onset are
   excluded from the analysis (they are trivially separable post-hoc
   and would inflate any ceiling).
2. Estimate per-feature mutual information with each future-damage
   label using `sklearn.feature_selection.mutual_info_classif`.
3. Train a balanced regularized logistic regression per feature group
   (`position`, `metadata`, `online`, `position_metadata`,
   `position_metadata_online`) using `GroupKFold(groups=run_id)` to
   prevent within-run label leakage. Score = mean cross-validated
   AUROC, with run-level bootstrap CIs.
4. Headline number: incremental AUROC gain of
   `position+metadata+online` over `position+metadata`.

Phase 0 caveat: 1 task, 2 presses, 3 ratios. Per-press stratification
runs only when each press has at least `min_runs_per_press` runs. The
summary JSON exposes `phase0_smoke=True` whenever the cell coverage
falls below the threshold.
"""

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from herald.analysis.event_study import (
    _add_derived_features,
    derive_onset,
    load_tokens_for_runs,
)
from herald.labeling import DEFAULT_NT_ONSET_FRAC

DEFAULT_HORIZONS: tuple[int, ...] = (5, 10, 25, 50)
DEFAULT_ONLINE_FEATURES: tuple[str, ...] = (
    "entropy",
    "top1_prob",
    "top1_top2_margin",
    "top5_prob",
    "h_alts",
    "delta_h",
    "kl_div",
    "top10_jaccard",
    "tail_mass",
    "eff_vocab_size",
)
POSITION_FEATURES: tuple[str, ...] = (
    "token_pos",
    "relative_progress",
    "output_length_so_far",
)
METADATA_FEATURES: tuple[str, ...] = (
    "compression_ratio",
    "press_code",
    "task_code",
)
DEFAULT_MAX_SAMPLES = 50_000
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_N_SPLITS = 5
DEFAULT_MIN_RUNS_PER_PRESS = 10
DEFAULT_PLOT_DPI = 120


@dataclass(frozen=True)
class InformationCeilingConfig:
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    online_features: tuple[str, ...] = DEFAULT_ONLINE_FEATURES
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC
    max_samples: int = DEFAULT_MAX_SAMPLES
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    n_splits: int = DEFAULT_N_SPLITS
    min_runs_per_press: int = DEFAULT_MIN_RUNS_PER_PRESS
    seed: int = 0


@dataclass
class InformationCeilingResult:
    summary: dict[str, Any]
    by_feature: pl.DataFrame = field(default_factory=pl.DataFrame)
    by_group: pl.DataFrame = field(default_factory=pl.DataFrame)


# ----- dataset construction ------------------------------------------


def collect_run_onsets(
    runs: pl.DataFrame,
    cfg: InformationCeilingConfig,
) -> tuple[dict[str, int | None], dict[str, str]]:
    """Per-run onset and onset_source.

    Iterates compressed runs and returns
    `(run_id -> onset_or_None, run_id -> source)`. Censored runs map
    to `None`.
    """
    onsets: dict[str, int | None] = {}
    sources: dict[str, str] = {}
    for row in runs.iter_rows(named=True):
        if str(row.get("press")) == "none":
            continue
        token_ids = list(row.get("generated_token_ids") or [])
        if not token_ids:
            onsets[row["run_id"]] = None
            sources[row["run_id"]] = "none"
            continue
        onset, src = derive_onset(
            list(row.get("catastrophes") or []),
            token_ids,
            int(row.get("max_new_tokens") or 512),
            cfg.nt_onset_frac,
        )
        onsets[row["run_id"]] = onset
        sources[row["run_id"]] = src
    return onsets, sources


def build_token_dataset(
    runs: pl.DataFrame,
    tokens: pl.DataFrame,
    cfg: InformationCeilingConfig,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Per-token feature + label table, restricted to compressed runs.

    Adds:
    - `relative_progress = token_pos / num_tokens_generated`
    - `output_length_so_far = token_pos`
    - `press_code`, `task_code`: integer-encoded categoricals
    - `future_damage_h{H}` for each H in cfg.horizons
    - drops tokens at/past derived onset for catastrophic runs

    Returns (df, used_online_features, missing_online_features).
    """
    if tokens.is_empty():
        return pl.DataFrame(), [], list(cfg.online_features)

    tokens = _add_derived_features(tokens)
    used = [f for f in cfg.online_features if f in tokens.columns]
    missing = [f for f in cfg.online_features if f not in tokens.columns]

    compressed_runs = runs.filter(pl.col("press") != "none")
    if compressed_runs.is_empty():
        return pl.DataFrame(), used, missing

    run_meta = compressed_runs.select(
        [
            "run_id",
            "task",
            "press",
            "compression_ratio",
            "num_tokens_generated",
        ]
    )

    onsets, sources = collect_run_onsets(compressed_runs, cfg)
    onset_df = pl.DataFrame(
        {
            "run_id": list(onsets.keys()),
            "onset_token": [
                (-1 if v is None else int(v)) for v in onsets.values()
            ],
            "onset_source": [sources[k] for k in onsets.keys()],
        }
    )

    df = tokens.join(run_meta, on="run_id", how="inner")
    df = df.join(onset_df, on="run_id", how="left")

    # Position features.
    df = df.with_columns(
        pl.col("token_pos").cast(pl.Float32).alias("output_length_so_far"),
        (
            pl.col("token_pos").cast(pl.Float32)
            / pl.max_horizontal(
                pl.col("num_tokens_generated").cast(pl.Float32),
                pl.lit(1.0).cast(pl.Float32),
            )
        ).alias("relative_progress"),
    )

    # Drop tokens at/past onset for catastrophic runs.
    df = df.filter(
        (pl.col("onset_token") < 0)
        | (pl.col("token_pos") < pl.col("onset_token"))
    )

    # Future-damage labels.
    for h in cfg.horizons:
        df = df.with_columns(
            (
                (pl.col("onset_token") >= 0)
                & (pl.col("onset_token") - pl.col("token_pos") <= h)
                & (pl.col("onset_token") - pl.col("token_pos") > 0)
            )
            .cast(pl.Int8)
            .alias(f"future_damage_h{h}")
        )

    # Categorical encodings.
    press_codes = {
        v: i for i, v in enumerate(sorted(df["press"].unique().to_list()))
    }
    task_codes = {
        v: i for i, v in enumerate(sorted(df["task"].unique().to_list()))
    }
    df = df.with_columns(
        pl.col("press")
        .replace_strict(press_codes, default=-1)
        .cast(pl.Int32)
        .alias("press_code"),
        pl.col("task")
        .replace_strict(task_codes, default=-1)
        .cast(pl.Int32)
        .alias("task_code"),
    )

    # Cast position columns explicitly to keep schema stable.
    df = df.with_columns(
        pl.col("token_pos").cast(pl.Float32),
        pl.col("compression_ratio").cast(pl.Float32),
    )

    return df, used, missing


# ----- mutual information --------------------------------------------


def _subsample_indices(
    n: int, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n)
    return rng.choice(n, size=max_samples, replace=False)


def compute_per_feature_mi(
    df: pl.DataFrame,
    feature_names: list[str],
    discrete_mask: list[bool],
    horizons: Iterable[int],
    max_samples: int,
    seed: int,
) -> pl.DataFrame:
    """Per (feature, horizon): mutual information with the future-
    damage label.

    Drops rows with non-finite values for the feature. Subsamples
    when n > max_samples to keep runtime CPU-friendly.
    """
    rows: list[dict[str, Any]] = []
    if df.is_empty() or not feature_names:
        return pl.DataFrame()

    rng = np.random.default_rng(seed)
    for h in horizons:
        label_col = f"future_damage_h{h}"
        if label_col not in df.columns:
            continue
        # Build dense matrix once per horizon, dropping NaN-rich rows.
        cols = feature_names + [label_col]
        sub = df.select(cols).drop_nulls()
        if sub.is_empty():
            for f in feature_names:
                rows.append(
                    {
                        "feature": f,
                        "horizon": h,
                        "mi": float("nan"),
                        "n": 0,
                        "n_pos": 0,
                    }
                )
            continue
        X = sub.select(feature_names).to_numpy()
        finite_mask = np.all(np.isfinite(X), axis=1)
        X = X[finite_mask]
        y = sub[label_col].to_numpy()[finite_mask].astype(np.int8)
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            for f in feature_names:
                rows.append(
                    {
                        "feature": f,
                        "horizon": h,
                        "mi": float("nan"),
                        "n": int(X.shape[0]),
                        "n_pos": int(np.sum(y == 1)),
                    }
                )
            continue
        idx = _subsample_indices(X.shape[0], max_samples, rng)
        Xs, ys = X[idx], y[idx]
        try:
            mi = mutual_info_classif(
                Xs,
                ys,
                discrete_features=discrete_mask,
                random_state=seed,
            )
        except ValueError:
            mi = np.full(len(feature_names), np.nan)
        for f, m in zip(feature_names, mi):
            rows.append(
                {
                    "feature": f,
                    "horizon": h,
                    "mi": float(m),
                    "n": int(X.shape[0]),
                    "n_pos": int(np.sum(y == 1)),
                }
            )
    return pl.DataFrame(rows)


# ----- group-level cross-validated AUROC ------------------------------


def _fit_and_score(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> float | None:
    """Mean group-CV AUROC of a balanced LR. None if degenerate.

    Skips folds where the validation set is single-class. Returns
    None if no fold yielded a usable AUROC.
    """
    if X.shape[0] == 0 or len(np.unique(y)) < 2:
        return None
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        return None
    gkf = GroupKFold(n_splits=actual_splits)
    aurocs: list[float] = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        y_tr, y_va = y[train_idx], y[val_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr = scaler.fit_transform(X[train_idx])
        Xva = scaler.transform(X[val_idx])
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            solver="lbfgs",
            random_state=seed,
        )
        try:
            clf.fit(Xtr, y_tr)
        except ValueError:
            continue
        proba = clf.predict_proba(Xva)[:, 1]
        try:
            aurocs.append(float(roc_auc_score(y_va, proba)))
        except ValueError:
            continue
    if not aurocs:
        return None
    return float(np.mean(aurocs))


def _bootstrap_auroc_runs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    n_splits: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Run-level bootstrap percentile CI for the group-CV AUROC.

    Each bootstrap iteration resamples *runs* (groups) with
    replacement and re-fits a fresh GroupKFold-CV LR on the
    resampled token rows. Token-level resampling would collapse the
    CI; this preserves within-run dependence.

    Returns (lo, hi) or (None, None) if degenerate.
    """
    if n_boot <= 0:
        return None, None
    rng = np.random.default_rng(seed + 1)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return None, None
    group_to_idx: dict[Any, np.ndarray] = {
        g: np.where(groups == g)[0] for g in unique_groups
    }
    boot: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(
            unique_groups, size=unique_groups.size, replace=True
        )
        # Build fresh "groups" labels distinguishing duplicate
        # samples so GroupKFold sees them as separate folds.
        idx_pieces: list[np.ndarray] = []
        group_pieces: list[np.ndarray] = []
        for k, g in enumerate(sampled):
            ix = group_to_idx[g]
            idx_pieces.append(ix)
            group_pieces.append(np.full(ix.size, k, dtype=np.int64))
        idx = np.concatenate(idx_pieces)
        gboot = np.concatenate(group_pieces)
        score = _fit_and_score(
            X[idx], y[idx], gboot, n_splits=n_splits, seed=seed
        )
        if score is not None:
            boot.append(score)
    if not boot:
        return None, None
    return (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def compute_group_auroc(
    df: pl.DataFrame,
    group_features: dict[str, list[str]],
    horizons: Iterable[int],
    cfg: InformationCeilingConfig,
) -> pl.DataFrame:
    """Per (group, horizon): cross-validated AUROC + bootstrap CI."""
    if df.is_empty() or not group_features:
        return pl.DataFrame()

    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []
    for h in horizons:
        label_col = f"future_damage_h{h}"
        if label_col not in df.columns:
            continue
        for gname, feats in group_features.items():
            present = [f for f in feats if f in df.columns]
            if not present:
                rows.append(
                    {
                        "group": gname,
                        "horizon": h,
                        "n_features": 0,
                        "n_samples": 0,
                        "n_pos": 0,
                        "n_runs": 0,
                        "auroc": None,
                        "ci_lo": None,
                        "ci_hi": None,
                        "features": [],
                    }
                )
                continue
            sub_cols = present + [label_col, "run_id"]
            sub = df.select(sub_cols).drop_nulls()
            if sub.is_empty():
                rows.append(
                    {
                        "group": gname,
                        "horizon": h,
                        "n_features": len(present),
                        "n_samples": 0,
                        "n_pos": 0,
                        "n_runs": 0,
                        "auroc": None,
                        "ci_lo": None,
                        "ci_hi": None,
                        "features": present,
                    }
                )
                continue
            X = sub.select(present).to_numpy()
            finite_mask = np.all(np.isfinite(X), axis=1)
            X = X[finite_mask]
            y = sub[label_col].to_numpy()[finite_mask].astype(np.int8)
            groups = sub["run_id"].to_numpy()[finite_mask]
            # Subsample large datasets to keep CV runtime manageable.
            if X.shape[0] > cfg.max_samples:
                idx = _subsample_indices(X.shape[0], cfg.max_samples, rng)
                X, y, groups = X[idx], y[idx], groups[idx]
            score = _fit_and_score(X, y, groups, cfg.n_splits, cfg.seed)
            ci_lo, ci_hi = _bootstrap_auroc_runs(
                X, y, groups, cfg.n_bootstrap, cfg.n_splits, cfg.seed
            )
            rows.append(
                {
                    "group": gname,
                    "horizon": h,
                    "n_features": len(present),
                    "n_samples": int(X.shape[0]),
                    "n_pos": int(np.sum(y == 1)),
                    "n_runs": int(np.unique(groups).size),
                    "auroc": score,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "features": present,
                }
            )
    return pl.DataFrame(rows)


# ----- permutation importance ----------------------------------------


def compute_permutation_importance(
    df: pl.DataFrame,
    feature_names: list[str],
    horizon: int,
    cfg: InformationCeilingConfig,
) -> pl.DataFrame:
    """Permutation importance from a single balanced LR.

    Uses a held-out group split (one GroupKFold fold) to fit and
    evaluate; permutes each feature in the held-out set and reports
    the AUROC drop. Returns rows of (feature, importance,
    base_auroc).
    """
    label_col = f"future_damage_h{horizon}"
    if df.is_empty() or label_col not in df.columns or not feature_names:
        return pl.DataFrame()

    sub_cols = feature_names + [label_col, "run_id"]
    sub = df.select(sub_cols).drop_nulls()
    if sub.is_empty():
        return pl.DataFrame()

    X = sub.select(feature_names).to_numpy()
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    y = sub[label_col].to_numpy()[finite_mask].astype(np.int8)
    groups = sub["run_id"].to_numpy()[finite_mask]

    rng = np.random.default_rng(cfg.seed + 2)
    if X.shape[0] > cfg.max_samples:
        idx = _subsample_indices(X.shape[0], cfg.max_samples, rng)
        X, y, groups = X[idx], y[idx], groups[idx]
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return pl.DataFrame()

    n_splits = min(cfg.n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return pl.DataFrame()
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = next(iter(gkf.split(X, y, groups)))
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
        return pl.DataFrame()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train_idx])
    Xva = scaler.transform(X[val_idx])
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        random_state=cfg.seed,
    )
    try:
        clf.fit(Xtr, y[train_idx])
    except ValueError:
        return pl.DataFrame()
    proba = clf.predict_proba(Xva)[:, 1]
    try:
        base = float(roc_auc_score(y[val_idx], proba))
    except ValueError:
        return pl.DataFrame()

    rows: list[dict[str, Any]] = []
    for j, fname in enumerate(feature_names):
        Xperm = Xva.copy()
        rng.shuffle(Xperm[:, j])
        proba_p = clf.predict_proba(Xperm)[:, 1]
        try:
            score = float(roc_auc_score(y[val_idx], proba_p))
        except ValueError:
            score = float("nan")
        rows.append(
            {
                "feature": fname,
                "horizon": horizon,
                "base_auroc": base,
                "perm_auroc": score,
                "importance": base - score,
            }
        )
    return pl.DataFrame(rows)


# ----- plotting ------------------------------------------------------


def plot_information_by_horizon(
    by_group: pl.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Bar chart of group-level AUROC vs horizon.

    Headline: shows position-only, metadata-only,
    position+metadata, all (online layered on top).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if by_group.is_empty():
        output_path.write_bytes(b"")
        return

    horizons = sorted(by_group["horizon"].unique().to_list())
    groups = sorted(by_group["group"].unique().to_list())
    if not horizons or not groups:
        output_path.write_bytes(b"")
        return

    fig, ax = plt.subplots(figsize=(max(6.0, 2.0 * len(horizons)), 4.5))
    width = 0.8 / max(1, len(groups))
    xs = np.arange(len(horizons), dtype=float)
    for i, g in enumerate(groups):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for h in horizons:
            row = by_group.filter(
                (pl.col("group") == g) & (pl.col("horizon") == h)
            )
            if row.is_empty() or row["auroc"][0] is None:
                means.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
                continue
            m = float(row["auroc"][0])
            lo = row["ci_lo"][0]
            hi = row["ci_hi"][0]
            means.append(m)
            lows.append(m if lo is None else float(lo))
            highs.append(m if hi is None else float(hi))
        offsets = xs + (i - (len(groups) - 1) / 2.0) * width
        means_arr = np.asarray(means)
        lows_arr = np.asarray(lows)
        highs_arr = np.asarray(highs)
        err_lo = np.maximum(0.0, means_arr - lows_arr)
        err_hi = np.maximum(0.0, highs_arr - means_arr)
        ax.bar(
            offsets,
            means_arr,
            width=width,
            label=g,
            yerr=[err_lo, err_hi],
            capsize=3,
        )
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"H={h}" for h in horizons])
    ax.set_ylabel("Group-CV AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_feature_group_comparison(
    by_feature: pl.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Per-feature MI heatmap-style bar plot, one panel per horizon."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if by_feature.is_empty():
        output_path.write_bytes(b"")
        return

    horizons = sorted(by_feature["horizon"].unique().to_list())
    n = len(horizons)
    cols = min(2, n) if n > 0 else 1
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.5 * cols, 4.0 * rows), squeeze=False
    )
    for i, h in enumerate(horizons):
        ax = axes[i // cols][i % cols]
        sub = by_feature.filter(pl.col("horizon") == h).sort("mi")
        feats = sub["feature"].to_list()
        mis = sub["mi"].to_numpy()
        ys = np.arange(len(feats))
        ax.barh(ys, mis, color="C0")
        ax.set_yticks(ys)
        ax.set_yticklabels(feats, fontsize=8)
        ax.set_xlabel("Mutual information")
        ax.set_title(f"H={h}", fontsize=10)
        ax.grid(True, axis="x", alpha=0.2)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


# ----- top-level entry point -----------------------------------------


def _feature_groups(
    used_online: list[str],
) -> dict[str, list[str]]:
    """Define the groups used in the head-to-head comparison.

    Critical: `compression_ratio` lives in `metadata`, never in
    `online`. The whole point of the analysis is to show online
    features add information *beyond* ratio + position.
    """
    pos = list(POSITION_FEATURES)
    meta = list(METADATA_FEATURES)
    online = list(used_online)
    return {
        "position": pos,
        "metadata": meta,
        "online": online,
        "position_metadata": pos + meta,
        "position_metadata_online": pos + meta + online,
    }


def _scalar_score(
    by_group: pl.DataFrame, group: str, horizon: int
) -> float | None:
    if by_group.is_empty():
        return None
    row = by_group.filter(
        (pl.col("group") == group) & (pl.col("horizon") == horizon)
    )
    if row.is_empty() or row["auroc"][0] is None:
        return None
    return float(row["auroc"][0])


def run_information_ceiling(
    input_root: Path,
    output_dir: Path,
    cfg: InformationCeilingConfig,
) -> InformationCeilingResult:
    """Top-level CPU entry point.

    Reads runs.parquet + token partitions from `input_root` and
    writes:
      - `information_by_feature.parquet`
      - `information_by_group.parquet`
      - `information_summary.json`
      - `information_by_horizon.png`
      - `information_by_feature.png` (optional, MI per feature)
    """
    runs_path = input_root / "final" / "runs.parquet"
    tokens_root = input_root / "final" / "tokens"

    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "runs_path": str(runs_path),
        "tokens_root": str(tokens_root),
        "horizons": list(cfg.horizons),
        "online_features_requested": list(cfg.online_features),
        "position_features": list(POSITION_FEATURES),
        "metadata_features": list(METADATA_FEATURES),
        "max_samples": cfg.max_samples,
        "n_bootstrap": cfg.n_bootstrap,
        "n_splits": cfg.n_splits,
        "min_runs_per_press": cfg.min_runs_per_press,
        "nt_onset_frac": cfg.nt_onset_frac,
        "method": (
            "Per-feature MI (sklearn.mutual_info_classif) and "
            "group-level CV AUROC (LogisticRegression with "
            "class_weight=balanced and GroupKFold(groups=run_id)). "
            "Bootstrap CIs resample run_ids, never tokens. "
            "Tokens at or past derived onset are excluded so the "
            "ceiling is a *future-damage* ceiling, not a post-hoc "
            "discrimination."
        ),
        "label_definition": (
            "future_damage_h{H} = 1 iff onset is in (token_pos, "
            "token_pos+H] for the run; censored runs and tokens "
            "past onset get label 0; tokens at/past onset are "
            "excluded from the analysis frame."
        ),
    }

    blockers: list[str] = []
    if not runs_path.exists():
        blockers.append(f"missing runs parquet: {runs_path}")
    if not tokens_root.exists():
        blockers.append(f"missing tokens directory: {tokens_root}")
    if blockers:
        summary["blockers"] = blockers
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "information_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return InformationCeilingResult(summary=summary)

    runs = pl.read_parquet(runs_path)
    summary["n_total_runs"] = runs.height
    compressed = runs.filter(pl.col("press") != "none")
    summary["n_compressed_runs"] = compressed.height

    if compressed.is_empty():
        summary["status"] = "no_compressed_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "information_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return InformationCeilingResult(summary=summary)

    tokens = load_tokens_for_runs(tokens_root, compressed["run_id"].to_list())
    summary["n_token_rows_loaded"] = tokens.height

    df, used_online, missing = build_token_dataset(runs, tokens, cfg)
    summary["features_online_used"] = used_online
    summary["features_online_missing"] = missing
    summary["n_token_rows_after_filter"] = df.height

    if df.is_empty() or not used_online:
        summary["status"] = "no_token_rows"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "information_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return InformationCeilingResult(summary=summary)

    pos_counts: dict[str, int] = {}
    for h in cfg.horizons:
        pos_counts[str(h)] = int(df[f"future_damage_h{h}"].sum())
    summary["positive_label_counts_by_horizon"] = pos_counts

    # Phase 0 smoke flag: any press with < min_runs_per_press runs.
    press_run_counts = (
        compressed.group_by("press")
        .agg(pl.col("run_id").n_unique().alias("n_runs"))
        .to_dicts()
    )
    summary["runs_per_press"] = {
        d["press"]: int(d["n_runs"]) for d in press_run_counts
    }
    phase0_smoke = any(
        d["n_runs"] < cfg.min_runs_per_press for d in press_run_counts
    )
    if len(set(df["task"].unique().to_list())) <= 1:
        phase0_smoke = True
    summary["phase0_smoke"] = phase0_smoke
    if phase0_smoke:
        summary["phase0_caveat"] = (
            "Phase 0 smoke output: at least one press has fewer than "
            f"{cfg.min_runs_per_press} runs and/or only one task is "
            "present. Per-press stratification is reported but treat "
            "the headline incremental AUROC gain as smoke-only until "
            "Phase 1 broadens coverage."
        )

    # Per-feature MI: position + metadata + online together.
    all_feature_names = (
        list(POSITION_FEATURES) + list(METADATA_FEATURES) + used_online
    )
    discrete_mask = [
        f in {"press_code", "task_code"} for f in all_feature_names
    ]
    by_feature = compute_per_feature_mi(
        df,
        all_feature_names,
        discrete_mask,
        cfg.horizons,
        cfg.max_samples,
        cfg.seed,
    )

    # Group-level CV AUROC.
    feat_groups = _feature_groups(used_online)
    by_group = compute_group_auroc(df, feat_groups, cfg.horizons, cfg)

    # Permutation importance: smallest horizon (most actionable).
    perm = compute_permutation_importance(
        df,
        all_feature_names,
        horizon=cfg.horizons[0],
        cfg=cfg,
    )

    # Per-press stratification.
    by_group_press: list[pl.DataFrame] = []
    for press_val in sorted(df["press"].unique().to_list()):
        sub = df.filter(pl.col("press") == press_val)
        if sub.is_empty():
            continue
        # Drop the press feature inside the per-press cell since it
        # is constant. Keep ratio + task_code.
        per_press_groups = {
            "position": list(POSITION_FEATURES),
            "metadata": [f for f in METADATA_FEATURES if f != "press_code"],
            "online": used_online,
            "position_metadata": (
                list(POSITION_FEATURES)
                + [f for f in METADATA_FEATURES if f != "press_code"]
            ),
            "position_metadata_online": (
                list(POSITION_FEATURES)
                + [f for f in METADATA_FEATURES if f != "press_code"]
                + used_online
            ),
        }
        bg = compute_group_auroc(sub, per_press_groups, cfg.horizons, cfg)
        if not bg.is_empty():
            bg = bg.with_columns(pl.lit(press_val).alias("press"))
            by_group_press.append(bg)
    by_group_press_df = (
        pl.concat(by_group_press, how="vertical_relaxed")
        if by_group_press
        else pl.DataFrame()
    )

    # Headline numbers per horizon.
    headline: dict[str, dict[str, float | None]] = {}
    for h in cfg.horizons:
        pos_only = _scalar_score(by_group, "position", h)
        meta_only = _scalar_score(by_group, "metadata", h)
        online_only = _scalar_score(by_group, "online", h)
        pm = _scalar_score(by_group, "position_metadata", h)
        all_groups = _scalar_score(by_group, "position_metadata_online", h)
        gain = (
            None
            if (pm is None or all_groups is None)
            else float(all_groups - pm)
        )
        # MI gains: sum of MI for online features at this horizon.
        if not by_feature.is_empty():
            mi_pos = float(
                by_feature.filter(
                    (pl.col("horizon") == h)
                    & pl.col("feature").is_in(list(POSITION_FEATURES))
                )["mi"]
                .fill_nan(0.0)
                .sum()
            )
            mi_meta = float(
                by_feature.filter(
                    (pl.col("horizon") == h)
                    & pl.col("feature").is_in(list(METADATA_FEATURES))
                )["mi"]
                .fill_nan(0.0)
                .sum()
            )
            mi_online = float(
                by_feature.filter(
                    (pl.col("horizon") == h)
                    & pl.col("feature").is_in(list(used_online))
                )["mi"]
                .fill_nan(0.0)
                .sum()
            )
        else:
            mi_pos = mi_meta = mi_online = 0.0
        headline[str(h)] = {
            "auroc_position_only": pos_only,
            "auroc_metadata_only": meta_only,
            "auroc_online_only": online_only,
            "auroc_position_plus_metadata": pm,
            "auroc_all": all_groups,
            "incremental_auroc_gain": gain,
            "mi_sum_position": mi_pos,
            "mi_sum_metadata": mi_meta,
            "mi_sum_online": mi_online,
            "incremental_mi_gain": mi_online,
        }
    summary["headline_by_horizon"] = headline
    online_signal_present = False
    for hrec in headline.values():
        gain_val = hrec.get("incremental_auroc_gain")
        if gain_val is not None and gain_val > 0.0:
            online_signal_present = True
            break
    summary["online_features_add_signal"] = online_signal_present

    # Write artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    if not by_feature.is_empty():
        by_feature.write_parquet(
            output_dir / "information_by_feature.parquet"
        )
    if not by_group.is_empty():
        # `features` is a list-of-strings column; polars writes lists
        # of strings to parquet natively, but cast to keep schema
        # stable across versions.
        by_group.write_parquet(output_dir / "information_by_group.parquet")
    if not by_group_press_df.is_empty():
        by_group_press_df.write_parquet(
            output_dir / "information_by_group_press.parquet"
        )
    if not perm.is_empty():
        perm.write_parquet(output_dir / "permutation_importance.parquet")

    plot_information_by_horizon(
        by_group,
        output_dir / "information_by_horizon.png",
        title=(
            "Group-CV AUROC for future-damage prediction "
            f"(horizons={list(cfg.horizons)})"
        ),
    )
    plot_feature_group_comparison(
        by_feature,
        output_dir / "information_by_feature.png",
        title="Per-feature mutual information with future-damage label",
    )

    summary["artifacts"] = {
        "by_feature_parquet": str(
            output_dir / "information_by_feature.parquet"
        ),
        "by_group_parquet": str(output_dir / "information_by_group.parquet"),
        "by_group_press_parquet": str(
            output_dir / "information_by_group_press.parquet"
        ),
        "permutation_parquet": str(
            output_dir / "permutation_importance.parquet"
        ),
        "headline_png": str(output_dir / "information_by_horizon.png"),
        "feature_mi_png": str(output_dir / "information_by_feature.png"),
    }
    summary_path = output_dir / "information_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    return InformationCeilingResult(
        summary=summary,
        by_feature=by_feature,
        by_group=by_group,
    )


__all__ = [
    "DEFAULT_HORIZONS",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_MIN_RUNS_PER_PRESS",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_N_SPLITS",
    "DEFAULT_ONLINE_FEATURES",
    "InformationCeilingConfig",
    "InformationCeilingResult",
    "METADATA_FEATURES",
    "POSITION_FEATURES",
    "build_token_dataset",
    "collect_run_onsets",
    "compute_group_auroc",
    "compute_per_feature_mi",
    "compute_permutation_importance",
    "plot_feature_group_comparison",
    "plot_information_by_horizon",
    "run_information_ceiling",
]
