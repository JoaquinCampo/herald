"""Free causal feature engineering (protocol section 5).

Every transform uses only rows of the same run with token positions at or
before the current row. Frames must be sorted by (run_id, token_pos).
"""

import numpy as np
import pandas as pd

BASES = (
    "entropy",
    "top1_prob",
    "h_alts",
    "delta_h",
    "kl_div",
    "top10_jaccard",
)
EXTRA_BASES = ("tail_mass", "eff_vocab_size", "focus_ratio")
START_ROWS = 8
SLOPE_WINDOWS = (8, 32)
LONG_WINDOW = 64
RATIO_EPS = 1e-6


def engineered_names() -> tuple[str, ...]:
    """Names produced by add_engineered_features, in order."""
    names = ["focus_ratio"]
    for base in BASES:
        names.append(f"{base}_startdiff_{START_ROWS}")
        names.append(f"{base}_startratio_{START_ROWS}")
    for base in BASES:
        for window in SLOPE_WINDOWS:
            names.append(f"{base}_slope_{window}")
            names.append(f"{base}_posfrac_{window}")
    for base in (*EXTRA_BASES,):
        for window in SLOPE_WINDOWS:
            names.append(f"{base}_causal_mean_{window}")
    for base in EXTRA_BASES:
        for window in SLOPE_WINDOWS:
            names.append(f"{base}_causal_min_{window}")
    for base in BASES:
        names.append(f"{base}_causal_mean_{LONG_WINDOW}")
        names.append(f"{base}_causal_std_{LONG_WINDOW}")
    return tuple(names)


ENGINEERED_NAMES = engineered_names()


def run_boundaries(run_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(run_ids) == 0:
        return []
    changes = np.flatnonzero(run_ids[1:] != run_ids[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(run_ids)]))
    return [
        (int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def trailing_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Least-squares slope over the trailing window (finite points only)."""
    output = np.full(len(values), np.nan, dtype=np.float32)
    for end in range(1, len(values) + 1):
        segment = values[max(0, end - window) : end]
        finite = segment[np.isfinite(segment)]
        if len(finite) < 2:
            continue
        positions = np.arange(len(finite), dtype=np.float64)
        centered = positions - positions.mean()
        output[end - 1] = np.float32(
            np.dot(centered, finite - finite.mean())
            / np.dot(centered, centered)
        )
    return output


def trailing_posfrac(values: np.ndarray, window: int) -> np.ndarray:
    """Fraction of positive steps in the trailing window."""
    output = np.full(len(values), np.nan, dtype=np.float32)
    for end in range(1, len(values) + 1):
        segment = values[max(0, end - window) : end]
        diffs = np.diff(segment)
        finite = diffs[np.isfinite(diffs)]
        if len(finite) == 0:
            continue
        output[end - 1] = np.float32(np.mean(finite > 0))
    return output


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing finite-only mean (NaN where no finite point)."""
    output = np.full(len(values), np.nan, dtype=np.float32)
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
    counts = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    right = np.arange(1, len(values) + 1)
    left = np.maximum(right - window, 0)
    count = counts[right] - counts[left]
    total = cumulative[right] - cumulative[left]
    output[count > 0] = (total[count > 0] / count[count > 0]).astype(
        np.float32
    )
    return output


def trailing_min(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing finite-only minimum (NaN where no finite point)."""
    output = np.full(len(values), np.nan, dtype=np.float32)
    masked = np.where(np.isfinite(values), values, np.inf)
    for end in range(1, len(values) + 1):
        segment = masked[max(0, end - window) : end]
        best = segment.min()
        if np.isfinite(best):
            output[end - 1] = np.float32(best)
    return output


def trailing_std(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing finite-only sample std (NaN with fewer than 2 points)."""
    output = np.full(len(values), np.nan, dtype=np.float32)
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
    squared = np.concatenate(
        ([0.0], np.cumsum(clean * clean, dtype=np.float64))
    )
    counts = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    right = np.arange(1, len(values) + 1)
    left = np.maximum(right - window, 0)
    count = counts[right] - counts[left]
    total = cumulative[right] - cumulative[left]
    total_squared = squared[right] - squared[left]
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = (
            total_squared - total * total / np.maximum(count, 1)
        ) / np.maximum(count - 1, 1)
    use = count > 1
    output[use] = np.sqrt(np.maximum(variance[use], 0.0)).astype(np.float32)
    return output


def start_anchor(values: np.ndarray, rows: int = START_ROWS) -> float:
    """Finite-only mean of the first rows (NaN when none finite)."""
    segment = values[:rows]
    finite = segment[np.isfinite(segment)]
    if len(finite) == 0:
        return float("nan")
    return float(finite.mean())


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns in place; returns the frame."""
    top1 = frame["top1_prob"].to_numpy(dtype=np.float64)
    top5 = frame["top5_prob"].to_numpy(dtype=np.float64)
    frame["focus_ratio"] = (top1 / (top5 + RATIO_EPS)).astype(np.float32)
    boundaries = run_boundaries(frame["run_id"].to_numpy())
    columns: dict[str, np.ndarray] = {
        name: np.empty(len(frame), dtype=np.float32)
        for name in ENGINEERED_NAMES
        if name != "focus_ratio"
    }
    for base in BASES:
        values = frame[base].to_numpy(dtype=np.float64)
        diff = np.empty(len(frame), dtype=np.float32)
        ratio = np.empty(len(frame), dtype=np.float32)
        for start, end in boundaries:
            anchor = start_anchor(values[start:end])
            diff[start:end] = (values[start:end] - anchor).astype(np.float32)
            ratio[start:end] = (
                values[start:end] / (anchor + RATIO_EPS)
            ).astype(np.float32)
        columns[f"{base}_startdiff_{START_ROWS}"] = diff
        columns[f"{base}_startratio_{START_ROWS}"] = ratio
        for window in SLOPE_WINDOWS:
            slope = np.empty(len(frame), dtype=np.float32)
            posfrac = np.empty(len(frame), dtype=np.float32)
            for start, end in boundaries:
                slope[start:end] = trailing_slope(values[start:end], window)
                posfrac[start:end] = trailing_posfrac(
                    values[start:end], window
                )
            columns[f"{base}_slope_{window}"] = slope
            columns[f"{base}_posfrac_{window}"] = posfrac
        mean64 = np.empty(len(frame), dtype=np.float32)
        std64 = np.empty(len(frame), dtype=np.float32)
        for start, end in boundaries:
            mean64[start:end] = trailing_mean(values[start:end], LONG_WINDOW)
            std64[start:end] = trailing_std(values[start:end], LONG_WINDOW)
        columns[f"{base}_causal_mean_{LONG_WINDOW}"] = mean64
        columns[f"{base}_causal_std_{LONG_WINDOW}"] = std64
    for base in EXTRA_BASES:
        values = frame[base].to_numpy(dtype=np.float64)
        for window in SLOPE_WINDOWS:
            mean = np.empty(len(frame), dtype=np.float32)
            minimum = np.empty(len(frame), dtype=np.float32)
            for start, end in boundaries:
                mean[start:end] = trailing_mean(values[start:end], window)
                minimum[start:end] = trailing_min(values[start:end], window)
            columns[f"{base}_causal_mean_{window}"] = mean
            columns[f"{base}_causal_min_{window}"] = minimum
    for name, array in columns.items():
        frame[name] = array
    return frame
