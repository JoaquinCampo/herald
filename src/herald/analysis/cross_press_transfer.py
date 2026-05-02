"""Cross-press transfer matrix.

Trains a simple future-damage predictor on tokens from one press and
evaluates it on tokens from every other press, building an (m x m)
matrix of AUROC / AUPRC scores. Directly probes the HERALD black-box
"compressor-agnostic" claim. If the diagonal is high but the
off-diagonal collapses, HERALD is a per-compressor calibration
framework rather than a universal predictor.

Method
------
1. Reuse `information_ceiling.build_token_dataset` to assemble the
   per-token feature + label table over compressed runs (press !=
   'none'), with the same onset derivation, the same drop-tokens-
   past-onset rule, and the same `future_damage_h{H}` labels.
2. Cross-press feature sets deliberately exclude `press_code`. Under
   train-on-A / test-on-B, `press_code` is constant on each side and
   takes a literally unseen value at test, which would silently
   degrade the model and confound the matrix. The metadata baseline
   here is `(token_pos, relative_progress, output_length_so_far,
   compression_ratio, task_code)`.
3. Diagonal cells (train_press == test_press) use `GroupKFold(
   groups=run_id)` over the runs of that press; the cell value is
   the mean fold AUROC. Off-diagonal cells (train_press !=
   test_press) train a single `StandardScaler -> LogisticRegression`
   pipeline on every train-press row and score every test-press row
   in one shot. Run-level disjointness is automatic off-diagonal
   (a run lives in exactly one press).
4. A cell is marked `insufficient` if either side has fewer than
   `min_pos` positives or `min_neg` negatives at the chosen horizon.
   We never force a score on a degenerate cell.
5. Off-diagonal cells additionally produce a percentile bootstrap CI
   over test runs: resample test `run_id`s with replacement, recompute
   AUROC, take the 2.5/97.5 percentiles. Diagonal CIs come from the
   spread of GroupKFold folds (min/max), not from a bootstrap.

Phase 0 caveat
--------------
Phase 0 has only two compressed presses, so the matrix is 2x2:
two diagonal cells, two off-diagonal cells. The transfer-gap point
estimate is therefore directional only and has no real CI at this
scale. Treat the Phase 0 output as a smoke read of the plumbing,
not as a publishable transfer claim.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from herald.analysis.event_study import load_tokens_for_runs
from herald.analysis.information_ceiling import (
    DEFAULT_HORIZONS,
    DEFAULT_ONLINE_FEATURES,
    POSITION_FEATURES,
    InformationCeilingConfig,
    build_token_dataset,
)
from herald.labeling import DEFAULT_NT_ONSET_FRAC

# Cross-press metadata: drop press_code (constant per side, unseen at
# test). Keep ratio + task_code.
CROSS_PRESS_METADATA_FEATURES: tuple[str, ...] = (
    "compression_ratio",
    "task_code",
)
DEFAULT_FEATURE_SETS: tuple[str, ...] = (
    "online",
    "position_metadata",
    "all",
    "entropy_only",
)
DEFAULT_MIN_POS = 5
DEFAULT_MIN_NEG = 5
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_N_SPLITS = 5
DEFAULT_MAX_TRAIN_SAMPLES = 50_000
DEFAULT_MAX_TEST_SAMPLES = 50_000
DEFAULT_PLOT_DPI = 120


@dataclass(frozen=True)
class CrossPressTransferConfig:
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    online_features: tuple[str, ...] = DEFAULT_ONLINE_FEATURES
    feature_sets: tuple[str, ...] = DEFAULT_FEATURE_SETS
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    n_splits: int = DEFAULT_N_SPLITS
    min_pos: int = DEFAULT_MIN_POS
    min_neg: int = DEFAULT_MIN_NEG
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES
    max_test_samples: int = DEFAULT_MAX_TEST_SAMPLES
    seed: int = 0


@dataclass
class CrossPressTransferResult:
    summary: dict[str, Any]
    matrix: pl.DataFrame = field(default_factory=pl.DataFrame)


# ----- feature-set resolution ----------------------------------------


def resolve_feature_sets(
    used_online: list[str],
    requested: tuple[str, ...],
) -> dict[str, list[str]]:
    """Resolve named feature sets to concrete column lists.

    `press_code` is intentionally excluded from every cross-press
    set: it is constant within each side of a train/test split and
    takes a literally unseen value at test time, which would just
    confound the matrix.
    """
    pos = list(POSITION_FEATURES)
    meta = list(CROSS_PRESS_METADATA_FEATURES)
    online = list(used_online)
    catalog = {
        "position": pos,
        "metadata": meta,
        "position_metadata": pos + meta,
        "online": online,
        "all": pos + meta + online,
        "entropy_only": (["entropy"] if "entropy" in online else []),
    }
    out: dict[str, list[str]] = {}
    for name in requested:
        if name not in catalog:
            raise ValueError(f"Unknown feature set: {name}")
        out[name] = catalog[name]
    return out


# ----- per-cell evaluation -------------------------------------------


def _prepare_side(
    df: pl.DataFrame,
    press: str,
    features: list[str],
    label_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter to one press, drop nulls + non-finite rows, return
    (X, y, groups) where groups are run_ids.
    """
    sub = (
        df.filter(pl.col("press") == press)
        .select(features + [label_col, "run_id"])
        .drop_nulls()
    )
    if sub.is_empty():
        return (
            np.zeros((0, len(features)), dtype=np.float64),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=object),
        )
    X = sub.select(features).to_numpy().astype(np.float64)
    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    y = sub[label_col].to_numpy()[finite_mask].astype(np.int8)
    groups = sub["run_id"].to_numpy()[finite_mask]
    return X, y, groups


def _classification_counts(y: np.ndarray) -> tuple[int, int]:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return n_pos, n_neg


def _insufficient_reason(
    train_n_pos: int,
    train_n_neg: int,
    test_n_pos: int,
    test_n_neg: int,
    cfg: CrossPressTransferConfig,
) -> str | None:
    """Return None if the cell is usable, otherwise a short reason."""
    msgs: list[str] = []
    if train_n_pos < cfg.min_pos:
        msgs.append(f"train_pos<{cfg.min_pos}")
    if train_n_neg < cfg.min_neg:
        msgs.append(f"train_neg<{cfg.min_neg}")
    if test_n_pos < cfg.min_pos:
        msgs.append(f"test_pos<{cfg.min_pos}")
    if test_n_neg < cfg.min_neg:
        msgs.append(f"test_neg<{cfg.min_neg}")
    if not msgs:
        return None
    return ",".join(msgs)


def _subsample(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] <= cap:
        return X, y, groups
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    return X[idx], y[idx], groups[idx]


def _fit_lr(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression] | None:
    """Fit StandardScaler + balanced LR. Returns None if degenerate."""
    if X_tr.shape[0] == 0 or len(np.unique(y_tr)) < 2:
        return None
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_tr)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        random_state=seed,
    )
    try:
        clf.fit(Xs, y_tr)
    except ValueError:
        return None
    return scaler, clf


def _score(
    scaler: StandardScaler,
    clf: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float | None, float | None, np.ndarray | None]:
    """Return (auroc, auprc, proba). None if y is single-class."""
    if X.shape[0] == 0 or len(np.unique(y)) < 2:
        return None, None, None
    proba = clf.predict_proba(scaler.transform(X))[:, 1]
    try:
        auroc = float(roc_auc_score(y, proba))
    except ValueError:
        auroc = None
    try:
        auprc = float(average_precision_score(y, proba))
    except ValueError:
        auprc = None
    return auroc, auprc, proba


def _bootstrap_test_runs(
    proba: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Percentile CI for AUROC, resampling test run_ids."""
    if n_boot <= 0 or proba is None:
        return None, None
    rng = np.random.default_rng(seed + 7)
    unique = np.unique(groups)
    if unique.size < 2:
        return None, None
    group_to_idx = {g: np.where(groups == g)[0] for g in unique}
    boot: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        try:
            boot.append(float(roc_auc_score(ys, proba[idx])))
        except ValueError:
            continue
    if not boot:
        return None, None
    return (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def evaluate_diagonal_cell(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: CrossPressTransferConfig,
) -> dict[str, Any]:
    """GroupKFold CV within a press. Mean fold AUROC + min/max as CI."""
    unique_groups = np.unique(groups)
    n_splits = min(cfg.n_splits, unique_groups.size)
    if n_splits < 2 or X.shape[0] == 0 or len(np.unique(y)) < 2:
        return {
            "auroc": None,
            "auprc": None,
            "ci_lo": None,
            "ci_hi": None,
            "fold_aurocs": [],
        }
    gkf = GroupKFold(n_splits=n_splits)
    fold_aurocs: list[float] = []
    fold_auprcs: list[float] = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        y_tr, y_va = y[train_idx], y[val_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
        fit = _fit_lr(X[train_idx], y_tr, cfg.seed)
        if fit is None:
            continue
        scaler, clf = fit
        au, ap, _ = _score(scaler, clf, X[val_idx], y_va)
        if au is not None:
            fold_aurocs.append(au)
        if ap is not None:
            fold_auprcs.append(ap)
    if not fold_aurocs:
        return {
            "auroc": None,
            "auprc": None,
            "ci_lo": None,
            "ci_hi": None,
            "fold_aurocs": [],
        }
    return {
        "auroc": float(np.mean(fold_aurocs)),
        "auprc": (float(np.mean(fold_auprcs)) if fold_auprcs else None),
        "ci_lo": float(np.min(fold_aurocs)),
        "ci_hi": float(np.max(fold_aurocs)),
        "fold_aurocs": fold_aurocs,
    }


def evaluate_offdiagonal_cell(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    groups_te: np.ndarray,
    cfg: CrossPressTransferConfig,
) -> dict[str, Any]:
    """Train on one press, score on another press. Bootstrap test
    runs for the AUROC CI.
    """
    fit = _fit_lr(X_tr, y_tr, cfg.seed)
    if fit is None:
        return {
            "auroc": None,
            "auprc": None,
            "ci_lo": None,
            "ci_hi": None,
        }
    scaler, clf = fit
    auroc, auprc, proba = _score(scaler, clf, X_te, y_te)
    if proba is None:
        ci_lo, ci_hi = None, None
    else:
        ci_lo, ci_hi = _bootstrap_test_runs(
            proba, y_te, groups_te, cfg.n_bootstrap, cfg.seed
        )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def compute_cell(
    df: pl.DataFrame,
    train_press: str,
    test_press: str,
    features: list[str],
    horizon: int,
    cfg: CrossPressTransferConfig,
) -> dict[str, Any]:
    """Score one (train_press, test_press) cell at one horizon."""
    label_col = f"future_damage_h{horizon}"
    if label_col not in df.columns or not features:
        return {
            "train_press": train_press,
            "test_press": test_press,
            "horizon": horizon,
            "auroc": None,
            "auprc": None,
            "ci_lo": None,
            "ci_hi": None,
            "n_train_rows": 0,
            "n_train_pos": 0,
            "n_train_neg": 0,
            "n_train_runs": 0,
            "n_test_rows": 0,
            "n_test_pos": 0,
            "n_test_neg": 0,
            "n_test_runs": 0,
            "n_features": len(features),
            "insufficient": True,
            "insufficient_reason": "no_features_or_label",
        }

    rng = np.random.default_rng(cfg.seed + horizon)

    X_tr, y_tr, g_tr = _prepare_side(df, train_press, features, label_col)
    X_te, y_te, g_te = _prepare_side(df, test_press, features, label_col)

    train_n_pos, train_n_neg = _classification_counts(y_tr)
    test_n_pos, test_n_neg = _classification_counts(y_te)
    n_train_runs = int(np.unique(g_tr).size) if g_tr.size else 0
    n_test_runs = int(np.unique(g_te).size) if g_te.size else 0

    base = {
        "train_press": train_press,
        "test_press": test_press,
        "horizon": horizon,
        "n_train_rows": int(X_tr.shape[0]),
        "n_train_pos": train_n_pos,
        "n_train_neg": train_n_neg,
        "n_train_runs": n_train_runs,
        "n_test_rows": int(X_te.shape[0]),
        "n_test_pos": test_n_pos,
        "n_test_neg": test_n_neg,
        "n_test_runs": n_test_runs,
        "n_features": len(features),
    }

    reason = _insufficient_reason(
        train_n_pos, train_n_neg, test_n_pos, test_n_neg, cfg
    )
    if reason is not None:
        return {
            **base,
            "auroc": None,
            "auprc": None,
            "ci_lo": None,
            "ci_hi": None,
            "insufficient": True,
            "insufficient_reason": reason,
        }

    X_tr, y_tr, g_tr = _subsample(
        X_tr, y_tr, g_tr, cfg.max_train_samples, rng
    )
    X_te, y_te, g_te = _subsample(X_te, y_te, g_te, cfg.max_test_samples, rng)

    if train_press == test_press:
        # Diagonal: GroupKFold over runs of this press.
        result = evaluate_diagonal_cell(X_tr, y_tr, g_tr, cfg)
        return {
            **base,
            "auroc": result["auroc"],
            "auprc": result["auprc"],
            "ci_lo": result["ci_lo"],
            "ci_hi": result["ci_hi"],
            "insufficient": result["auroc"] is None,
            "insufficient_reason": (
                None if result["auroc"] is not None else "cv_degenerate"
            ),
        }

    # Off-diagonal: train on full train side, score on full test side.
    result = evaluate_offdiagonal_cell(X_tr, y_tr, X_te, y_te, g_te, cfg)
    return {
        **base,
        "auroc": result["auroc"],
        "auprc": result["auprc"],
        "ci_lo": result["ci_lo"],
        "ci_hi": result["ci_hi"],
        "insufficient": result["auroc"] is None,
        "insufficient_reason": (
            None if result["auroc"] is not None else "score_degenerate"
        ),
    }


# ----- matrix construction -------------------------------------------


def compute_transfer_matrix(
    df: pl.DataFrame,
    presses: list[str],
    feature_sets: dict[str, list[str]],
    horizons: tuple[int, ...],
    cfg: CrossPressTransferConfig,
) -> pl.DataFrame:
    """Long-format matrix of every (horizon, feature_set, train, test)
    cell.
    """
    rows: list[dict[str, Any]] = []
    for h in horizons:
        for fs_name, feats in feature_sets.items():
            for tr in presses:
                for te in presses:
                    cell = compute_cell(df, tr, te, feats, h, cfg)
                    cell["feature_set"] = fs_name
                    rows.append(cell)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


# ----- summary -------------------------------------------------------


def _diag_offdiag_means(
    matrix: pl.DataFrame,
    horizon: int,
    feature_set: str,
) -> dict[str, Any]:
    """Mean AUROC on the diagonal vs off-diagonal for one cell of the
    (horizon, feature_set) grid. Skips insufficient cells.
    """
    sub = matrix.filter(
        (pl.col("horizon") == horizon)
        & (pl.col("feature_set") == feature_set)
        & (~pl.col("insufficient"))
        & pl.col("auroc").is_not_null()
    )
    if sub.is_empty():
        return {
            "diagonal_mean": None,
            "offdiagonal_mean": None,
            "transfer_gap": None,
            "n_diagonal_cells": 0,
            "n_offdiagonal_cells": 0,
        }
    diag = sub.filter(pl.col("train_press") == pl.col("test_press"))
    off = sub.filter(pl.col("train_press") != pl.col("test_press"))

    def _mean_or_none(series_df: pl.DataFrame) -> float | None:
        if series_df.is_empty():
            return None
        vals = [v for v in series_df["auroc"].to_list() if v is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    diag_mean = _mean_or_none(diag)
    off_mean = _mean_or_none(off)
    gap = (
        None
        if (diag_mean is None or off_mean is None)
        else float(diag_mean - off_mean)
    )
    return {
        "diagonal_mean": diag_mean,
        "offdiagonal_mean": off_mean,
        "transfer_gap": gap,
        "n_diagonal_cells": diag.height,
        "n_offdiagonal_cells": off.height,
    }


# ----- plotting ------------------------------------------------------


def plot_transfer_heatmap(
    matrix: pl.DataFrame,
    horizon: int,
    feature_set: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    """Plot one heatmap (rows = train_press, cols = test_press)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub = matrix.filter(
        (pl.col("horizon") == horizon)
        & (pl.col("feature_set") == feature_set)
    )
    if sub.is_empty():
        output_path.write_bytes(b"")
        return
    presses = sorted(
        set(sub["train_press"].to_list()) | set(sub["test_press"].to_list())
    )
    grid = np.full((len(presses), len(presses)), np.nan, dtype=np.float64)
    for r, tr in enumerate(presses):
        for c, te in enumerate(presses):
            row = sub.filter(
                (pl.col("train_press") == tr) & (pl.col("test_press") == te)
            )
            if row.is_empty() or row["auroc"][0] is None:
                continue
            grid[r, c] = float(row["auroc"][0])

    fig, ax = plt.subplots(
        figsize=(0.9 * len(presses) + 2.5, 0.9 * len(presses) + 2.0)
    )
    im = ax.imshow(grid, vmin=0.4, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(presses)))
    ax.set_yticks(range(len(presses)))
    ax.set_xticklabels(presses, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(presses, fontsize=9)
    ax.set_xlabel("test press")
    ax.set_ylabel("train press")
    for r in range(len(presses)):
        for c in range(len(presses)):
            v = grid[r, c]
            label = "n/a" if not np.isfinite(v) else f"{v:.3f}"
            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                color="white" if (np.isfinite(v) and v < 0.7) else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUROC")
    ax.set_title(
        title
        or (f"Cross-press transfer (H={horizon}, features={feature_set})"),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


# ----- top-level entry point -----------------------------------------


def _build_dataset_via_information_ceiling(
    runs: pl.DataFrame,
    tokens: pl.DataFrame,
    cfg: CrossPressTransferConfig,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Reuse the information-ceiling table builder so onset logic,
    feature engineering, and label construction stay identical.
    """
    ic_cfg = InformationCeilingConfig(
        horizons=cfg.horizons,
        online_features=cfg.online_features,
        nt_onset_frac=cfg.nt_onset_frac,
        seed=cfg.seed,
    )
    return build_token_dataset(runs, tokens, ic_cfg)


def run_cross_press_transfer(
    input_root: Path,
    output_dir: Path,
    cfg: CrossPressTransferConfig,
) -> CrossPressTransferResult:
    """Top-level CPU entry point.

    Reads runs.parquet + token partitions from `input_root` and writes
    the long-format matrix parquet, the summary JSON, and one heatmap
    per (horizon, feature_set).
    """
    runs_path = input_root / "final" / "runs.parquet"
    tokens_root = input_root / "final" / "tokens"

    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "runs_path": str(runs_path),
        "tokens_root": str(tokens_root),
        "horizons": list(cfg.horizons),
        "feature_sets_requested": list(cfg.feature_sets),
        "online_features_requested": list(cfg.online_features),
        "cross_press_metadata_features": list(CROSS_PRESS_METADATA_FEATURES),
        "min_pos": cfg.min_pos,
        "min_neg": cfg.min_neg,
        "n_bootstrap": cfg.n_bootstrap,
        "n_splits": cfg.n_splits,
        "max_train_samples": cfg.max_train_samples,
        "max_test_samples": cfg.max_test_samples,
        "nt_onset_frac": cfg.nt_onset_frac,
        "method": (
            "Train logistic regression on all tokens of one press, "
            "score on all tokens of another press. Diagonal cells "
            "use GroupKFold(run_id) within the press; off-diagonal "
            "cells train on the full train-press table and score on "
            "the full test-press table. StandardScaler is fit on "
            "train rows only. Bootstrap CI for off-diagonal cells "
            "resamples test run_ids; diagonal CI is the spread of "
            "GroupKFold folds (min/max). press_code is dropped from "
            "every cross-press feature set because it is constant "
            "per side and unseen at test."
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
        (output_dir / "cross_press_transfer_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return CrossPressTransferResult(summary=summary)

    runs = pl.read_parquet(runs_path)
    summary["n_total_runs"] = runs.height
    compressed = runs.filter(pl.col("press") != "none")
    summary["n_compressed_runs"] = compressed.height
    presses = sorted(compressed["press"].unique().to_list())
    summary["presses_found"] = presses
    summary["n_presses"] = len(presses)

    if compressed.is_empty():
        summary["status"] = "no_compressed_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cross_press_transfer_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return CrossPressTransferResult(summary=summary)

    if len(presses) < 2:
        summary["status"] = "insufficient_presses"
        summary["phase0_caveat"] = (
            "Fewer than 2 compressed presses present. The cross-press "
            "matrix degenerates to a single diagonal cell; off-"
            "diagonal transfer cannot be measured at all."
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cross_press_transfer_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return CrossPressTransferResult(summary=summary)

    tokens = load_tokens_for_runs(tokens_root, compressed["run_id"].to_list())
    summary["n_token_rows_loaded"] = tokens.height

    df, used_online, missing_online = _build_dataset_via_information_ceiling(
        runs, tokens, cfg
    )
    summary["features_online_used"] = used_online
    summary["features_online_missing"] = missing_online
    summary["n_token_rows_after_filter"] = df.height

    if df.is_empty():
        summary["status"] = "no_token_rows"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cross_press_transfer_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return CrossPressTransferResult(summary=summary)

    # Per-press / per-horizon label counts.
    label_counts: dict[str, dict[str, dict[str, int]]] = {}
    for press in presses:
        sub = df.filter(pl.col("press") == press)
        per_h: dict[str, dict[str, int]] = {}
        for h in cfg.horizons:
            col = f"future_damage_h{h}"
            if col not in sub.columns or sub.is_empty():
                per_h[str(h)] = {"n_pos": 0, "n_neg": 0, "n_rows": 0}
                continue
            y = sub[col].to_numpy().astype(np.int8)
            per_h[str(h)] = {
                "n_pos": int(np.sum(y == 1)),
                "n_neg": int(np.sum(y == 0)),
                "n_rows": int(y.size),
            }
        label_counts[press] = per_h
    summary["label_counts_by_press_horizon"] = label_counts

    feature_sets = resolve_feature_sets(used_online, cfg.feature_sets)
    # Drop empty feature sets (e.g., entropy_only when no entropy).
    feature_sets = {k: v for k, v in feature_sets.items() if v}
    summary["feature_sets_resolved"] = {k: v for k, v in feature_sets.items()}
    if not feature_sets:
        summary["status"] = "no_feature_sets"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "cross_press_transfer_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return CrossPressTransferResult(summary=summary)

    matrix = compute_transfer_matrix(
        df, presses, feature_sets, cfg.horizons, cfg
    )

    # Aggregate per (horizon, feature_set).
    agg: dict[str, dict[str, dict[str, Any]]] = {}
    for h in cfg.horizons:
        per_fs: dict[str, dict[str, Any]] = {}
        for fs_name in feature_sets:
            per_fs[fs_name] = _diag_offdiag_means(matrix, h, fs_name)
        agg[str(h)] = per_fs
    summary["aggregate_by_horizon_and_feature_set"] = agg

    # Insufficient-cell counts.
    insufficient_rows = matrix.filter(pl.col("insufficient"))
    summary["n_cells_total"] = matrix.height
    summary["n_cells_insufficient"] = insufficient_rows.height
    if not insufficient_rows.is_empty():
        summary["insufficient_cells"] = insufficient_rows.select(
            [
                "horizon",
                "feature_set",
                "train_press",
                "test_press",
                "insufficient_reason",
                "n_train_pos",
                "n_test_pos",
            ]
        ).to_dicts()
    else:
        summary["insufficient_cells"] = []

    summary["phase0_caveat"] = (
        "Phase 0 has only two compressed presses, so the matrix is "
        "2x2 (two diagonal cells, two off-diagonal cells). The "
        "diagonal/off-diagonal means and the transfer gap are "
        "directional only at this scale; do not interpret a small "
        "off-diagonal collapse as evidence against compressor-"
        "agnostic transfer until the Phase 1 grid lands. Cells "
        "marked insufficient are excluded from the means."
    )

    # Write artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)
    if not matrix.is_empty():
        # Schema-stable: drop list/dict columns if any.
        matrix.write_parquet(output_dir / "cross_press_transfer.parquet")

    headline_horizon = cfg.horizons[0]
    headline_feature_set = (
        "all" if "all" in feature_sets else next(iter(feature_sets))
    )
    plot_transfer_heatmap(
        matrix,
        horizon=headline_horizon,
        feature_set=headline_feature_set,
        output_path=(
            output_dir / f"cross_press_transfer_h{headline_horizon}.png"
        ),
        title=(
            f"Cross-press transfer AUROC (H={headline_horizon}, "
            f"{headline_feature_set})"
        ),
    )

    # Optional: one heatmap per (horizon, feature_set) pair.
    for h in cfg.horizons:
        for fs_name in feature_sets:
            if h == headline_horizon and fs_name == headline_feature_set:
                continue
            plot_transfer_heatmap(
                matrix,
                horizon=h,
                feature_set=fs_name,
                output_path=(
                    output_dir / f"cross_press_transfer_h{h}_{fs_name}.png"
                ),
            )

    summary["artifacts"] = {
        "matrix_parquet": str(output_dir / "cross_press_transfer.parquet"),
        "headline_png": str(
            output_dir / f"cross_press_transfer_h{headline_horizon}.png"
        ),
    }
    summary_path = output_dir / "cross_press_transfer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    return CrossPressTransferResult(summary=summary, matrix=matrix)


__all__ = [
    "CROSS_PRESS_METADATA_FEATURES",
    "CrossPressTransferConfig",
    "CrossPressTransferResult",
    "DEFAULT_FEATURE_SETS",
    "DEFAULT_MAX_TEST_SAMPLES",
    "DEFAULT_MAX_TRAIN_SAMPLES",
    "DEFAULT_MIN_NEG",
    "DEFAULT_MIN_POS",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_N_SPLITS",
    "compute_cell",
    "compute_transfer_matrix",
    "evaluate_diagonal_cell",
    "evaluate_offdiagonal_cell",
    "plot_transfer_heatmap",
    "resolve_feature_sets",
    "run_cross_press_transfer",
]
