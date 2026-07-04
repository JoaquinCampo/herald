"""Train-only baselines for switch-level damage prediction."""

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BaselineSpec:
    """A mean-prediction baseline keyed by train-row fields."""

    name: str
    keys: tuple[str, ...]


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec("global_mean", ()),
    BaselineSpec("ratio", ("ratio",)),
    BaselineSpec("task_ratio", ("task", "ratio")),
    BaselineSpec(
        "task_ratio_position_bucket",
        ("task", "ratio", "position_bucket"),
    ),
)


@dataclass(frozen=True)
class SplitRows:
    """Rows for one held-out-compressor baseline evaluation."""

    heldout_compressor: str
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]


@dataclass(frozen=True)
class MeanBaseline:
    """A fitted train-only grouped-mean baseline."""

    spec: BaselineSpec
    global_mean: float
    means: dict[tuple[object, ...], float]


def leave_one_compressor_splits(
    rows: Sequence[dict[str, Any]],
    *,
    compressors: Sequence[str] | None = None,
    seed: int = 0,
    test_group_fraction: float = 0.25,
) -> list[SplitRows]:
    """Build compressor-held-out splits with prompt groups disjoint."""
    if not 0.0 < test_group_fraction < 1.0:
        raise ValueError("test_group_fraction must be in (0, 1)")

    selected = set(compressors) if compressors is not None else None
    all_compressors = sorted(
        {
            str(row["compressor"])
            for row in rows
            if selected is None or str(row.get("compressor")) in selected
        }
    )
    splits: list[SplitRows] = []
    for heldout in all_compressors:
        train: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []
        for row in rows:
            compressor = str(row.get("compressor"))
            if selected is not None and compressor not in selected:
                continue
            is_test_group = _is_test_group(row, seed, test_group_fraction)
            if compressor == heldout and is_test_group:
                test.append(row)
            elif compressor != heldout and not is_test_group:
                train.append(row)
        if train and test:
            splits.append(SplitRows(heldout, train, test))
    return splits


def evaluate_baselines(
    rows: Sequence[dict[str, Any]],
    *,
    compressors: Sequence[str] | None = None,
    seed: int = 0,
    test_group_fraction: float = 0.25,
    position_bucket_size: int = 16,
) -> dict[str, Any]:
    """Evaluate train-only baselines on compressor-held-out splits."""
    selected = set(compressors) if compressors is not None else None
    prepared = [
        _with_position_bucket(row, position_bucket_size)
        for row in rows
        if _as_float(row.get("dq")) is not None
        and (selected is None or str(row.get("compressor")) in selected)
    ]
    splits = leave_one_compressor_splits(
        prepared,
        compressors=compressors,
        seed=seed,
        test_group_fraction=test_group_fraction,
    )

    split_summaries: list[dict[str, Any]] = []
    for split in splits:
        split_summaries.append(_evaluate_split(split))

    return {
        "config": {
            "seed": seed,
            "test_group_fraction": test_group_fraction,
            "position_bucket_size": position_bucket_size,
            "compressors": list(compressors)
            if compressors is not None
            else None,
        },
        "n_rows": len(prepared),
        "splits": split_summaries,
    }


def baseline_report(summary: dict[str, Any]) -> str:
    """Render a Markdown report for ``evaluate_baselines`` output."""
    lines = [
        "# Switch Predictor Baselines",
        "",
        "Train-only baselines for the pre-compression controller target.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(summary["config"], indent=2, sort_keys=True),
        "```",
        "",
        f"Rows: {summary['n_rows']}",
        "",
        "## Leave-One-Compressor-Out Results",
        "",
        (
            "| Held-out compressor | Baseline | Train rows | Test rows | "
            "MAE | RMSE | Mean dq | Mean pred |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in summary["splits"]:
        heldout = split["heldout_compressor"]
        for metric in split["baselines"]:
            lines.append(
                "| "
                f"{heldout} | {metric['baseline']} | "
                f"{split['n_train']} | {split['n_test']} | "
                f"{_fmt(metric['mae'])} | {_fmt(metric['rmse'])} | "
                f"{_fmt(metric['mean_y'])} | {_fmt(metric['mean_pred'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def _evaluate_split(split: SplitRows) -> dict[str, Any]:
    """Evaluate all baseline specs for one split."""
    baselines = [fit_mean_baseline(split.train, spec) for spec in BASELINES]
    metrics = [
        _metrics_for_predictions(
            spec=baseline.spec,
            y_pred=[
                predict_mean_baseline(baseline, row) for row in split.test
            ],
            y_true=[_required_float(row["dq"]) for row in split.test],
        )
        for baseline in baselines
    ]
    return {
        "heldout_compressor": split.heldout_compressor,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "baselines": metrics,
    }


def fit_mean_baseline(
    rows: Sequence[dict[str, Any]], spec: BaselineSpec
) -> MeanBaseline:
    """Fit a grouped train-mean baseline."""
    values = [_required_float(row["dq"]) for row in rows]
    global_mean = _mean(values)
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[_key(row, spec.keys)].append(_required_float(row["dq"]))
    means = {key: _mean(vals) for key, vals in groups.items()}
    return MeanBaseline(spec, global_mean, means)


def predict_mean_baseline(model: MeanBaseline, row: dict[str, Any]) -> float:
    """Predict with grouped means, backing off to broader keys."""
    keys = model.spec.keys
    for width in range(len(keys), -1, -1):
        key = _key(row, keys[:width])
        if key in model.means:
            return model.means[key]
    return model.global_mean


def _metrics_for_predictions(
    *, spec: BaselineSpec, y_pred: Sequence[float], y_true: Sequence[float]
) -> dict[str, Any]:
    """Compute basic regression metrics for one prediction vector."""
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    err = pred - y
    return {
        "baseline": spec.name,
        "mae": _mean_abs(err),
        "rmse": _root_mean_square(err),
        "mean_y": _mean_array(y),
        "mean_pred": _mean_array(pred),
    }


def _with_position_bucket(
    row: dict[str, Any], bucket_size: int
) -> dict[str, Any]:
    """Return a copy with integer position bucket added."""
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    out = dict(row)
    s = _required_float(row["s"])
    out["position_bucket"] = _position_bucket(s, bucket_size)
    return out


def _is_test_group(
    row: dict[str, Any], seed: int, test_group_fraction: float
) -> bool:
    """Return whether a prompt group belongs to the deterministic test set."""
    key = "\0".join(
        [
            str(seed),
            str(row.get("model")),
            str(row.get("task")),
            str(row.get("prompt_id")),
        ]
    )
    digest = hashlib.sha256(key.encode()).digest()
    value = _unit_interval_from_digest(digest)
    return value < test_group_fraction


def _mean(values: Sequence[float]) -> float:
    """Return a finite mean for a non-empty float sequence."""
    if not values:
        raise ValueError("cannot average an empty sequence")
    return _mean_array(np.asarray(values, dtype=np.float64))


def _mean_array(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return a finite mean for a non-empty NumPy vector."""
    if values.size == 0:
        raise ValueError("cannot average an empty array")
    try:
        result = values.mean().item()
    except (TypeError, ValueError) as exc:
        raise ValueError("could not compute mean") from exc
    return _required_float(result)


def _mean_abs(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return the mean absolute value for a vector."""
    try:
        abs_values = np.abs(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("could not compute absolute errors") from exc
    return _mean_array(abs_values)


def _root_mean_square(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return root mean square for a vector."""
    try:
        squared = values * values
        mean_square = _mean_array(squared)
        result = np.sqrt(mean_square).item()
    except (TypeError, ValueError) as exc:
        raise ValueError("could not compute root mean square") from exc
    return _required_float(result)


def _position_bucket(s: float, bucket_size: int) -> int:
    """Return the lower edge of the position bucket containing ``s``."""
    try:
        return math.floor(s / bucket_size) * bucket_size
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"invalid position {s!r}") from exc


def _unit_interval_from_digest(digest: bytes) -> float:
    """Map the first eight digest bytes into [0, 1)."""
    try:
        numerator = int.from_bytes(digest[:8], "big")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid digest") from exc
    return numerator / 18_446_744_073_709_551_616.0


def _key(row: dict[str, Any], keys: Sequence[str]) -> tuple[object, ...]:
    """Return a hashable row key for grouped means."""
    return tuple(row.get(key) for key in keys)


def _required_float(value: object) -> float:
    """Return a finite float or raise a clear error."""
    out = _as_float(value)
    if out is None:
        raise ValueError(f"expected finite float, got {value!r}")
    return out


def _as_float(value: object) -> float | None:
    """Return a finite float when conversion is possible."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _fmt(value: object) -> str:
    """Format report values compactly."""
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"
