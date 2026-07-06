"""Budget-aware tau calibration and deployable candidate selection.

Promotes the settled cross-fit pipeline pieces to the package: tau
calibration on out-of-fold scores (point or bootstrap-quantile cost
criterion) and the selection rule for scorer/variant/method
candidates. Raw OOF savings is not a safe selector: it picks
candidates whose test cost lands over budget. A candidate is
admitted only if the bootstrap upper bound of its OOF group-mean
cost at its frozen tau clears the budget; admissible candidates are
then ranked by OOF savings. Rationale and the empirical failure
cases: ``docs/implementation/online_forecasting.md``.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from herald.controller_metrics import EPSILON, TAU_GRID_SIZE
from herald.fast_controller import group_matrices

DEFAULT_BOOTSTRAPS = 2000


@dataclass(frozen=True)
class TauCalibration:
    """A frozen threshold and its out-of-fold operating point."""

    tau: float
    method: str
    oof_savings: float
    oof_cost: float


@dataclass(frozen=True)
class Candidate:
    """One scorer/variant/method candidate with its OOF evidence."""

    name: str
    rows: Sequence[dict[str, Any]]
    scores: Sequence[float]
    tau: float


@dataclass(frozen=True)
class SelectionReport:
    """Selection-relevant OOF metrics for one candidate."""

    name: str
    tau: float
    oof_savings: float
    oof_cost: float
    cost_bound: float
    admissible: bool


def calibrate_tau(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    *,
    method: str = "point",
    epsilon: float = EPSILON,
    grid_size: int = TAU_GRID_SIZE,
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = 0,
) -> TauCalibration:
    """Freeze tau on out-of-fold scores: max savings within budget.

    ``method`` is ``point`` (mean OOF cost <= epsilon, the locked
    criterion) or ``bootNN`` (the NN-th percentile of the bootstrap
    distribution of group-mean cost <= epsilon). The grid matches the
    locked ``select_tau``: score quantiles plus one value below the
    minimum, so never-switch is always feasible. Rows are not
    mutated.
    """
    pred, dq, savings = _matrices(rows, scores)
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot calibrate tau without rows")
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, grid_size))
    below_min = float(values.min()) - 1.0
    grid = sorted({below_min, *(float(q) for q in quantiles)})
    boot_quantile = _method_quantile(method)
    boot: tuple[np.ndarray, float] | None
    if boot_quantile is None:
        boot = None
    else:
        rng = np.random.default_rng(_stable_seed(seed, "calibrate"))
        boot_idx = rng.integers(
            0, pred.shape[0], size=(n_bootstraps, pred.shape[0])
        )
        boot = (boot_idx, boot_quantile)
    best = TauCalibration(
        tau=below_min, method=method, oof_savings=0.0, oof_cost=0.0
    )
    best_savings = -1.0
    for tau in grid:
        sav, cost = _per_group(pred, dq, savings, tau)
        mean_savings = float(sav.mean())
        mean_cost = float(cost.mean())
        if boot is None:
            criterion = mean_cost
        else:
            boot_means = cost[boot[0]].mean(axis=1)
            criterion = float(np.quantile(boot_means, boot[1]))
        if criterion <= epsilon and mean_savings > best_savings:
            best_savings = mean_savings
            best = TauCalibration(
                tau=tau,
                method=method,
                oof_savings=mean_savings,
                oof_cost=mean_cost,
            )
    return best


def bootstrap_cost_bound(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[float],
    tau: float,
    *,
    quantile: float = 0.90,
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = 0,
) -> float:
    """Bootstrap upper bound of the group-mean cost at a frozen tau."""
    pred, dq, savings = _matrices(rows, scores)
    _, cost = _per_group(pred, dq, savings, tau)
    rng = np.random.default_rng(_stable_seed(seed, "bound"))
    boot_idx = rng.integers(
        0, cost.shape[0], size=(n_bootstraps, cost.shape[0])
    )
    boot_means = cost[boot_idx].mean(axis=1)
    return float(np.quantile(boot_means, quantile))


def select_by_cost_bound(
    candidates: Sequence[Candidate],
    *,
    epsilon: float = EPSILON,
    quantile: float = 0.90,
    n_bootstraps: int = DEFAULT_BOOTSTRAPS,
    seed: int = 0,
) -> tuple[SelectionReport | None, list[SelectionReport]]:
    """Pick the admissible candidate with the largest OOF savings.

    Admissibility is the bootstrap cost bound at the candidate's own
    frozen tau, not its mean OOF cost: a point-calibrated tau sits at
    the budget by construction, and the bound is what separates
    holds-on-test from lands-over-budget. Returns the winner (None
    when nothing is admissible; deploy never-switch) and per-
    candidate reports for logging.
    """
    reports: list[SelectionReport] = []
    for candidate in candidates:
        pred, dq, savings = _matrices(candidate.rows, candidate.scores)
        sav, cost = _per_group(pred, dq, savings, candidate.tau)
        bound = bootstrap_cost_bound(
            candidate.rows,
            candidate.scores,
            candidate.tau,
            quantile=quantile,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        reports.append(
            SelectionReport(
                name=candidate.name,
                tau=candidate.tau,
                oof_savings=float(sav.mean()),
                oof_cost=float(cost.mean()),
                cost_bound=bound,
                admissible=bound <= epsilon,
            )
        )
    admissible = [report for report in reports if report.admissible]
    winner = max(
        admissible, key=lambda report: report.oof_savings, default=None
    )
    return winner, reports


def _matrices(
    rows: Sequence[dict[str, Any]], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group matrices from rows plus external scores, no mutation."""
    scored = [
        {**row, "predicted_dq": float(score)}
        for row, score in zip(rows, scores, strict=True)
    ]
    return group_matrices(scored)


def _per_group(
    pred: np.ndarray,
    dq: np.ndarray,
    savings: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-group savings and cost of the earliest-below-tau policy."""
    hit = pred <= tau
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    idx = np.arange(pred.shape[0])
    sav = np.where(any_hit, savings[idx, first], 0.0)
    cost = np.where(any_hit, dq[idx, first], 0.0)
    return sav, cost


def _method_quantile(method: str) -> float | None:
    """Bootstrap quantile for ``bootNN`` methods, None for point."""
    if method == "point":
        return None
    if method.startswith("boot"):
        suffix = method.removeprefix("boot")
        if suffix.isdigit() and 0 < int(suffix) < 100:
            return int(suffix) / 100.0
    raise ValueError(f"unknown calibration method {method!r}")


def _stable_seed(seed: int, *parts: str) -> int:
    """Derive a deterministic NumPy seed from labels."""
    digest = hashlib.sha256("\0".join([str(seed), *parts]).encode()).digest()
    return int.from_bytes(digest[:4], "big")
