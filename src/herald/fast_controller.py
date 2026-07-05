"""Vectorized replica of the locked tau selection, for internal CV.

``herald.controller_metrics.select_tau`` is pure Python over a
201-point grid and dominates internal cross-validation wall-clock.
This module reproduces it exactly (verified by tests) with padded
NumPy matrices. Final numbers always come from the locked evaluator;
this replica is used only inside train-only knob selection loops.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from herald.controller_metrics import (
    EPSILON,
    GROUP_FIELDS,
    TAU_GRID_SIZE,
)


def group_matrices(
    rows: Sequence[dict[str, Any]],
    *,
    prediction_key: str = "predicted_dq",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad per-group position-sorted rows into (pred, dq, savings)."""
    order: dict[tuple[object, ...], list[tuple[int, float, float, float]]]
    order = {}
    for row in rows:
        key = tuple(row.get(field) for field in GROUP_FIELDS)
        s = int(row["s"])
        ref_len = float(row["ref_len"])
        order.setdefault(key, []).append(
            (
                s,
                float(row[prediction_key]),
                float(row["dq"]),
                max(0.0, 1.0 - s / ref_len),
            )
        )
    width = max(len(members) for members in order.values())
    n = len(order)
    pred = np.full((n, width), np.inf, dtype=np.float64)
    dq = np.zeros((n, width), dtype=np.float64)
    savings = np.zeros((n, width), dtype=np.float64)
    for i, members in enumerate(order.values()):
        members.sort()
        for j, (_, p, d, sav) in enumerate(members):
            pred[i, j] = p
            dq[i, j] = d
            savings[i, j] = sav
    return pred, dq, savings


def policy_means(
    pred: np.ndarray,
    dq: np.ndarray,
    savings: np.ndarray,
    tau: float,
) -> tuple[float, float]:
    """Mean savings and cost of the earliest-below-tau policy."""
    hit = pred <= tau
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    idx = np.arange(pred.shape[0])
    sav = np.where(any_hit, savings[idx, first], 0.0)
    cost = np.where(any_hit, dq[idx, first], 0.0)
    return float(sav.mean()), float(cost.mean())


def fast_select_tau(
    rows: Sequence[dict[str, Any]],
    *,
    prediction_key: str = "predicted_dq",
    epsilon: float = EPSILON,
    grid_size: int = TAU_GRID_SIZE,
) -> float:
    """Exact vectorized equivalent of the locked ``select_tau``."""
    predictions = np.asarray(
        [float(row[prediction_key]) for row in rows], dtype=np.float64
    )
    if predictions.size == 0:
        raise ValueError("cannot select tau without rows")
    pred, dq, savings = group_matrices(rows, prediction_key=prediction_key)
    quantiles = np.quantile(predictions, np.linspace(0.0, 1.0, grid_size))
    below_min = float(predictions.min()) - 1.0
    grid = sorted({below_min, *(float(q) for q in quantiles)})
    best_tau = below_min
    best_savings = -1.0
    for tau in grid:
        sav, cost = policy_means(pred, dq, savings, tau)
        if cost <= epsilon and sav > best_savings:
            best_savings = sav
            best_tau = tau
    return best_tau
