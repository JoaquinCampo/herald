"""Median-mean mix switch predictor with train-only alpha selection.

The candidate predicts ``median + alpha * (mean - median)`` from grouped
train statistics keyed by (task, ratio, position_bucket), the same key
structure as the locked ``task_ratio_position_bucket`` baseline. The
median anchor is MAE-aligned for the zero-inflated dq target, while the
mean tilt restores graded ranking for the controller top decile. Alpha
is selected per outer split by an internal leave-one-TRAIN-compressor-
out grid, so the held-out compressor never influences model selection.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from herald.switch_baselines import leave_one_compressor_splits

MODEL_INPUT_FIELDS: tuple[str, ...] = ("task", "ratio", "position_bucket")
GROUP_KEYS: tuple[str, ...] = ("task", "ratio", "position_bucket")
DEFAULT_ALPHAS: tuple[float, ...] = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
)
DUPLICATE_KEY_FIELDS: tuple[str, ...] = (
    "model",
    "task",
    "prompt_id",
    "compressor",
    "ratio",
    "s",
)


@dataclass(frozen=True)
class GroupedStat:
    """Backoff tables for one grouped train statistic."""

    stat: str
    bucket_size: int
    tables: tuple[dict[tuple[object, ...], float], ...]


class GridEntry(NamedTuple):
    """Internal CV outcome for one alpha value."""

    mean_relative_improvement: float
    min_top_decile_lift: float


@dataclass(frozen=True)
class MixResult:
    """End-to-end outputs of the median-mean mix predictor."""

    predicted_rows: list[dict[str, Any]]
    alpha_by_heldout: dict[str, float]
    internal_grids: dict[str, dict[float, GridEntry]]
    model_input_fields: tuple[str, ...]


def fit_grouped_stat(
    rows: Sequence[dict[str, Any]],
    *,
    stat: str,
    bucket_size: int = 16,
) -> GroupedStat:
    """Fit grouped-statistic backoff tables on train rows only."""
    if stat not in ("median", "mean"):
        raise ValueError(f"unsupported stat: {stat!r}")
    tables: list[dict[tuple[object, ...], float]] = []
    for width in range(len(GROUP_KEYS), -1, -1):
        groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
        for row in rows:
            groups[_group_key(row, width, bucket_size)].append(
                float(row["dq"])
            )
        aggregate = np.median if stat == "median" else np.mean
        tables.append(
            {key: float(aggregate(values)) for key, values in groups.items()}
        )
    return GroupedStat(
        stat=stat, bucket_size=bucket_size, tables=tuple(tables)
    )


def predict_grouped_stat(model: GroupedStat, row: dict[str, Any]) -> float:
    """Predict with backoff from narrow to broader group keys."""
    for offset, table in enumerate(model.tables):
        width = len(GROUP_KEYS) - offset
        key = _group_key(row, width, model.bucket_size)
        if key in table:
            return table[key]
    raise ValueError("grouped statistic has no backoff level for row")


def mix_predictions(
    train: Sequence[dict[str, Any]],
    test: Sequence[dict[str, Any]],
    *,
    alpha: float,
    bucket_size: int = 16,
) -> list[float]:
    """Predict ``median + alpha * (mean - median)`` from train tables."""
    median = fit_grouped_stat(train, stat="median", bucket_size=bucket_size)
    mean = fit_grouped_stat(train, stat="mean", bucket_size=bucket_size)
    out: list[float] = []
    for row in test:
        low = predict_grouped_stat(median, row)
        high = predict_grouped_stat(mean, row)
        out.append(low + alpha * (high - low))
    return out


def internal_alpha_grid(
    train_rows: Sequence[dict[str, Any]],
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    bucket_size: int = 16,
) -> dict[float, GridEntry]:
    """Score alphas by leave-one-TRAIN-compressor-out internal CV."""
    compressors = sorted({str(row["compressor"]) for row in train_rows})
    if len(compressors) < 2:
        raise ValueError(
            "internal alpha selection needs at least two train compressors"
        )
    folds: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for inner_heldout in compressors:
        fit = [
            row
            for row in train_rows
            if str(row["compressor"]) != inner_heldout
        ]
        evaluate = [
            row
            for row in train_rows
            if str(row["compressor"]) == inner_heldout
        ]
        folds.append((fit, evaluate))

    grid: dict[float, GridEntry] = {}
    for alpha in alphas:
        improvements: list[float] = []
        lifts: list[float] = []
        for fit, evaluate in folds:
            y = np.asarray(
                [float(row["dq"]) for row in evaluate],
                dtype=np.float64,
            )
            pred = np.asarray(
                mix_predictions(
                    fit, evaluate, alpha=alpha, bucket_size=bucket_size
                ),
                dtype=np.float64,
            )
            locked = fit_grouped_stat(
                fit, stat="mean", bucket_size=bucket_size
            )
            locked_pred = np.asarray(
                [predict_grouped_stat(locked, row) for row in evaluate],
                dtype=np.float64,
            )
            locked_mae = float(np.mean(np.abs(locked_pred - y)))
            mae = float(np.mean(np.abs(pred - y)))
            if locked_mae > 0.0:
                improvements.append((locked_mae - mae) / locked_mae)
            lifts.append(_best_top_decile_lift(y, pred))
        grid[alpha] = GridEntry(
            mean_relative_improvement=float(np.mean(improvements)),
            min_top_decile_lift=min(lifts),
        )
    return grid


def select_alpha(
    grid: Mapping[float, tuple[float, float]],
    *,
    min_lift: float = 2.0,
) -> float:
    """Pick alpha train-only: lift-feasible first, then improvement.

    Among alphas whose internal minimum top-decile lift clears
    ``min_lift``, choose the best internal relative improvement. If no
    alpha is feasible, choose the highest internal lift; ties break by
    improvement and then by smaller alpha, deterministically.
    """
    if not grid:
        raise ValueError("alpha grid is empty")
    feasible = {
        alpha: entry for alpha, entry in grid.items() if entry[1] >= min_lift
    }
    pool = feasible if feasible else dict(grid)
    if feasible:
        return min(pool, key=lambda alpha: (-pool[alpha][0], alpha))
    return min(
        pool,
        key=lambda alpha: (-pool[alpha][1], -pool[alpha][0], alpha),
    )


def run_median_mean_mix(
    rows: Sequence[dict[str, Any]],
    *,
    compressors: Sequence[str],
    seed: int = 0,
    test_group_fraction: float = 0.25,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    bucket_size: int = 16,
    min_lift: float = 2.0,
) -> MixResult:
    """Fit and predict the mix on canonical leave-one-compressor splits."""
    selected = set(compressors)
    usable: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("compressor")) not in selected:
            continue
        dq = row.get("dq")
        if dq is None:
            continue
        value = float(dq)
        if not math.isfinite(value):
            continue
        usable.append(row)

    splits = leave_one_compressor_splits(
        usable,
        compressors=compressors,
        seed=seed,
        test_group_fraction=test_group_fraction,
    )
    predictions: dict[tuple[object, ...], float] = {}
    alpha_by_heldout: dict[str, float] = {}
    internal_grids: dict[str, dict[float, GridEntry]] = {}
    for split in splits:
        grid = internal_alpha_grid(
            split.train, alphas=alphas, bucket_size=bucket_size
        )
        alpha = select_alpha(grid, min_lift=min_lift)
        alpha_by_heldout[split.heldout_compressor] = alpha
        internal_grids[split.heldout_compressor] = grid
        predicted = mix_predictions(
            split.train, split.test, alpha=alpha, bucket_size=bucket_size
        )
        for row, value in zip(split.test, predicted, strict=True):
            predictions[_duplicate_key(row)] = float(value)

    predicted_rows: list[dict[str, Any]] = []
    for row in usable:
        out = dict(row)
        key = _duplicate_key(row)
        out["predicted_dq"] = predictions.get(key, 0.0)
        out["scored_by_predictor"] = key in predictions
        predicted_rows.append(out)
    return MixResult(
        predicted_rows=predicted_rows,
        alpha_by_heldout=alpha_by_heldout,
        internal_grids=internal_grids,
        model_input_fields=MODEL_INPUT_FIELDS,
    )


def _best_top_decile_lift(
    y: np.ndarray[Any, np.dtype[np.float64]],
    scores: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """Best top-decile lift across the two controller thresholds."""
    lifts = [
        _top_decile_lift(y, scores, threshold=0.0, strict=True),
        _top_decile_lift(y, scores, threshold=0.5, strict=False),
    ]
    finite = [value for value in lifts if value is not None]
    if not finite:
        return 0.0
    return max(finite)


def _top_decile_lift(
    y: np.ndarray[Any, np.dtype[np.float64]],
    scores: np.ndarray[Any, np.dtype[np.float64]],
    *,
    threshold: float,
    strict: bool,
) -> float | None:
    """Top-decile enrichment matching the canonical evaluator."""
    n = int(y.shape[0])
    if n == 0:
        return None
    labels = (y > threshold) if strict else (y >= threshold)
    prevalence = float(labels.mean())
    if prevalence == 0.0:
        return None
    top_n = max(1, math.ceil(0.10 * n))
    order = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
    top_rate = float(labels[order[:top_n]].mean())
    return top_rate / prevalence


def _group_key(
    row: dict[str, Any], width: int, bucket_size: int
) -> tuple[object, ...]:
    """Build a grouped-stat key of the requested backoff width."""
    values: list[object] = []
    for field in GROUP_KEYS[:width]:
        if field == "position_bucket":
            values.append(_position_bucket(row, bucket_size))
        else:
            values.append(row.get(field))
    return tuple(values)


def _position_bucket(row: dict[str, Any], bucket_size: int) -> int:
    """Bucket the switch position exactly like the locked baseline."""
    if "position_bucket" in row:
        return int(row["position_bucket"])
    s = float(row["s"])
    return math.floor(s / bucket_size) * bucket_size


def _duplicate_key(row: dict[str, Any]) -> tuple[object, ...]:
    """Canonical switch identity used to attach predictions."""
    return tuple(row[field] for field in DUPLICATE_KEY_FIELDS)
