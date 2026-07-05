"""Cell position-gate controller candidate.

Per (task, ratio) cell, fit the earliest switch position whose
realized quality cost stays within a tolerance for EVERY fit
compressor; cells with no feasible position are vetoed. The score is
graded by position past the threshold and collapses to coarse tied
blocks before it, so the locked tau selection cannot creep into risky
rows by admitting them a few at a time: admitting any tied block
wholesale busts the train budget and is rejected.

Model inputs are task, ratio, and position (``feat__position``, the
same online-known quantity behind the locked baseline's
``position_bucket``). No forbidden inputs (compressor, prompt_id,
q_ref, q_hybrid, damaged, major_damage, raw s, relative_s, ref_len)
are read at scoring time. The tolerance knob is selected by internal
donor cross-validation: hold one training-only donor compressor out
of the fit pool, run the unchanged locked tau selection on the
canonical train rows, and require the budget to hold on the held-out
donor. The held-out primary compressor never influences anything.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

CELL_KEYS: tuple[str, ...] = ("task", "ratio")
GROUP_FIELDS: tuple[str, ...] = ("model", "task", "prompt_id", "ratio")
SAFE_SCALE = 2048.0
POSITION_BUCKET = 16
PRE_THRESHOLD_BLOCK = 2.0
VETO_BLOCK = 3.0
VETO_GRADE_SCALE = 0.002
VETO_EARLINESS_SCALE = 0.0005
UNSEEN_SCORE = 4.5
DEFAULT_TOLERANCES: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04)


@dataclass(frozen=True)
class PositionGate:
    """Fitted per-cell switch thresholds and risk levels."""

    tolerance: float
    thresholds: dict[tuple[object, ...], float | None]
    cell_risk: dict[tuple[object, ...], float]
    worst_mean: dict[tuple[object, ...], float]
    global_worst_mean: float


def fit_position_gate(
    fit_rows: Sequence[dict[str, Any]],
    *,
    tolerance: float,
    min_groups: int = 5,
) -> PositionGate:
    """Fit per-cell thresholds on the fit compressors' rows.

    For each (task, ratio) cell and each candidate position p (the
    sampled positions present in the cell), the policy "switch at the
    first sampled position >= p" is scored per fit compressor as the
    mean realized dq over that compressor's decision groups. The
    threshold is the earliest p whose worst-compressor cost is at most
    ``tolerance``; a cell with no feasible p is vetoed.
    """
    cells: dict[
        tuple[object, ...], dict[str, list[list[tuple[float, float]]]]
    ]
    cells = defaultdict(lambda: defaultdict(list))
    groups: dict[tuple[object, ...], list[tuple[float, float]]] = defaultdict(
        list
    )
    group_comp: dict[tuple[object, ...], tuple[tuple[object, ...], str]] = {}
    for row in fit_rows:
        key = tuple(row.get(field) for field in GROUP_FIELDS) + (
            str(row.get("compressor")),
        )
        groups[key].append((_position(row), float(row["dq"])))
        group_comp[key] = (
            tuple(row.get(field) for field in CELL_KEYS),
            str(row.get("compressor")),
        )
    for key, members in groups.items():
        members.sort()
        cell, comp = group_comp[key]
        cells[cell][comp].append(members)

    thresholds: dict[tuple[object, ...], float | None] = {}
    cell_risk: dict[tuple[object, ...], float] = {}
    for cell, by_comp in cells.items():
        positions = sorted(
            {
                pos
                for curves in by_comp.values()
                for curve in curves
                for pos, _ in curve
            }
        )
        best: float | None = None
        for p in positions:
            worst = 0.0
            for curves in by_comp.values():
                if len(curves) < min_groups:
                    continue
                costs = [_cost_at(curve, p) for curve in curves]
                worst = max(worst, float(np.mean(costs)))
            if worst <= tolerance:
                best = p
                break
        thresholds[cell] = best
        per_comp_first = [
            float(np.mean([curve[0][1] for curve in curves]))
            for curves in by_comp.values()
        ]
        cell_risk[cell] = float(np.clip(max(per_comp_first), 0.0, 1.0))

    worst_mean, global_worst_mean = _worst_mean_tables(fit_rows)
    return PositionGate(
        tolerance=tolerance,
        thresholds=thresholds,
        cell_risk=cell_risk,
        worst_mean=worst_mean,
        global_worst_mean=global_worst_mean,
    )


def score_rows(
    gate: PositionGate, rows: Sequence[dict[str, Any]]
) -> list[float]:
    """Score rows: graded when past the cell threshold, blocks before.

    Past the threshold the score is (threshold - position) / scale,
    which is <= 0 and decreases with later positions, so tau tunes
    switch lateness on the train rows. Every pre-threshold row in a
    kept cell collapses to ONE flat tied value: admitting it would
    switch every kept-cell group at its first sampled position, which
    always busts the train budget, so tau creep is structurally dead.
    Vetoed and unseen cells sit above the flat block (unreachable for
    the policy) and are graded by cell risk, the grouped worst-case
    mean, and earliness, purely for catastrophe ranking.
    """
    out: list[float] = []
    for row in rows:
        cell = tuple(row.get(field) for field in CELL_KEYS)
        position = _position(row)
        if cell not in gate.thresholds:
            out.append(UNSEEN_SCORE + _grade(gate, row, position))
            continue
        threshold = gate.thresholds[cell]
        if threshold is None:
            risk = gate.cell_risk[cell]
            out.append(VETO_BLOCK + risk + _grade(gate, row, position))
            continue
        if position >= threshold:
            out.append((threshold - position) / SAFE_SCALE)
        else:
            out.append(PRE_THRESHOLD_BLOCK)
    return out


def _grade(gate: PositionGate, row: dict[str, Any], position: float) -> float:
    """Sub-block catastrophe grading; tiny next to block separation."""
    bucket = int(position // POSITION_BUCKET) * POSITION_BUCKET
    key = (row.get("task"), row.get("ratio"), bucket)
    worst = gate.worst_mean.get(key, gate.global_worst_mean)
    earliness = min(max(1.0 - position / SAFE_SCALE, 0.0), 1.0)
    return (
        VETO_GRADE_SCALE * float(np.clip(worst, 0.0, 1.0))
        + VETO_EARLINESS_SCALE * earliness
    )


def select_tolerance(
    outcomes: Mapping[float, tuple[bool, float]],
) -> float:
    """Pick the tolerance from internal donor-CV outcomes.

    ``outcomes`` maps tolerance -> (all folds within budget, worst
    internal savings). Among feasible tolerances, take the best worst
    savings, breaking ties toward the smaller tolerance. If nothing is
    feasible, take the smallest (most conservative) tolerance.
    """
    if not outcomes:
        raise ValueError("no tolerance outcomes")
    feasible = {tol: sav for tol, (ok, sav) in outcomes.items() if ok}
    if not feasible:
        return min(outcomes)
    return min(feasible, key=lambda tol: (-feasible[tol], tol))


def _worst_mean_tables(
    fit_rows: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[object, ...], float], float]:
    """Grouped mean of worst-compressor dq per (task, ratio, bucket)."""
    worst_by_point: dict[tuple[object, ...], float] = {}
    for row in fit_rows:
        point = tuple(row.get(field) for field in GROUP_FIELDS) + (
            _position(row),
        )
        dq = float(row["dq"])
        prev = worst_by_point.get(point)
        worst_by_point[point] = dq if prev is None else max(prev, dq)
    bucket_values: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for point, worst in worst_by_point.items():
        task, ratio = point[1], point[3]
        position = cast(float, point[-1])
        bucket = int(position // POSITION_BUCKET) * POSITION_BUCKET
        bucket_values[(task, ratio, bucket)].append(worst)
    worst_mean = {
        key: float(np.mean(values)) for key, values in bucket_values.items()
    }
    return worst_mean, float(np.mean(list(worst_by_point.values())))


def _position(row: dict[str, Any]) -> float:
    """Online-known generation position for gating."""
    value = row.get("feat__position")
    if value is None:
        value = (
            row["position_bucket"] if "position_bucket" in row else (row["s"])
        )
    return float(value)


def _cost_at(curve: Sequence[tuple[float, float]], threshold: float) -> float:
    """Realized dq when switching at the first position >= threshold."""
    for position, dq in curve:
        if position >= threshold:
            return dq
    return 0.0
