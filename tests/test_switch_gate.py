"""Tests for the cell position-gate controller candidate."""

import math

from herald.switch_gate import (
    PositionGate,
    fit_position_gate,
    score_rows,
    select_tolerance,
)


def _row(
    *,
    task: str = "t",
    ratio: float = 0.5,
    compressor: str = "a",
    prompt_id: str = "p0",
    s: int = 0,
    dq: float = 0.0,
    ref_len: int = 128,
) -> dict[str, object]:
    return {
        "model": "m",
        "task": task,
        "ratio": ratio,
        "compressor": compressor,
        "prompt_id": prompt_id,
        "s": s,
        "feat__position": float(s),
        "dq": dq,
        "ref_len": ref_len,
    }


def _cell_rows(
    *,
    task: str,
    ratio: float,
    compressor: str,
    prompt_id: str,
    dq_by_s: dict[int, float],
) -> list[dict[str, object]]:
    return [
        _row(
            task=task,
            ratio=ratio,
            compressor=compressor,
            prompt_id=prompt_id,
            s=s,
            dq=dq,
        )
        for s, dq in dq_by_s.items()
    ]


def test_fit_gate_picks_earliest_feasible_threshold() -> None:
    """P* is the earliest grid position whose worst-comp cost fits."""
    rows: list[dict[str, object]] = []
    # compressor a: damage at s=0, safe from 16 onwards
    # compressor b: damage through s=16, safe from 32 onwards
    for pid in ("p0", "p1"):
        rows += _cell_rows(
            task="t",
            ratio=0.5,
            compressor="a",
            prompt_id=pid,
            dq_by_s={0: 1.0, 16: 0.0, 32: 0.0, 48: 0.0},
        )
        rows += _cell_rows(
            task="t",
            ratio=0.5,
            compressor="b",
            prompt_id=pid,
            dq_by_s={0: 1.0, 16: 0.5, 32: 0.0, 48: 0.0},
        )
    gate = fit_position_gate(rows, tolerance=0.01, min_groups=1)
    assert gate.thresholds[("t", 0.5)] == 32


def test_fit_gate_vetoes_hopeless_cell() -> None:
    """A cell damaged at every position gets vetoed (None)."""
    rows: list[dict[str, object]] = []
    for pid in ("p0", "p1"):
        rows += _cell_rows(
            task="bad",
            ratio=0.5,
            compressor="a",
            prompt_id=pid,
            dq_by_s={0: 1.0, 16: 1.0, 32: 1.0},
        )
    gate = fit_position_gate(rows, tolerance=0.01, min_groups=1)
    assert gate.thresholds[("bad", 0.5)] is None


def test_scores_are_graded_past_threshold_and_tied_blocks_before() -> None:
    """Safe side is graded by position; risky rows form tied blocks."""
    rows: list[dict[str, object]] = []
    for pid in ("p0", "p1"):
        rows += _cell_rows(
            task="t",
            ratio=0.5,
            compressor="a",
            prompt_id=pid,
            dq_by_s={0: 1.0, 16: 0.0, 32: 0.0},
        )
        rows += _cell_rows(
            task="bad",
            ratio=0.5,
            compressor="a",
            prompt_id=pid,
            dq_by_s={0: 1.0, 16: 1.0},
        )
    gate = fit_position_gate(rows, tolerance=0.01, min_groups=1)
    assert gate.thresholds[("t", 0.5)] == 16

    safe_16 = score_rows(gate, [_row(task="t", s=16)])[0]
    safe_32 = score_rows(gate, [_row(task="t", s=32)])[0]
    early_0 = score_rows(gate, [_row(task="t", s=0)])[0]
    early_8 = score_rows(gate, [_row(task="t", s=8)])[0]
    veto_0 = score_rows(gate, [_row(task="bad", s=0)])[0]
    veto_16 = score_rows(gate, [_row(task="bad", s=16)])[0]

    assert safe_16 <= 0.0
    assert safe_32 < safe_16
    assert early_0 >= 1.0
    assert early_0 == early_8  # pre-threshold rows are hard ties
    assert veto_0 > early_0
    assert veto_0 > veto_16  # vetoed cells grade earlier as riskier


def test_cell_risk_is_worst_compressor_not_average() -> None:
    """A cell benign on one compressor but toxic on another is risky."""
    rows: list[dict[str, object]] = []
    for pid in ("p0", "p1"):
        rows += _cell_rows(
            task="t",
            ratio=0.5,
            compressor="mild",
            prompt_id=pid,
            dq_by_s={0: 0.0, 16: 0.0},
        )
        rows += _cell_rows(
            task="t",
            ratio=0.5,
            compressor="toxic",
            prompt_id=pid,
            dq_by_s={0: 0.8, 16: 0.0},
        )
    gate = fit_position_gate(rows, tolerance=0.01, min_groups=1)
    assert gate.cell_risk[("t", 0.5)] == 0.8


def test_unknown_cell_is_treated_as_vetoed() -> None:
    """Cells absent from fit rows must never be switched into."""
    rows = _cell_rows(
        task="t",
        ratio=0.5,
        compressor="a",
        prompt_id="p0",
        dq_by_s={0: 0.0},
    )
    gate = fit_position_gate(rows, tolerance=0.01, min_groups=1)
    score = score_rows(gate, [_row(task="unseen", ratio=0.25, s=0)])[0]
    assert score >= 3.0


def test_select_tolerance_prefers_feasible_then_savings() -> None:
    """Feasible tolerances win; ties break to the smaller tolerance."""
    outcomes = {
        0.002: (True, 0.05),
        0.005: (True, 0.20),
        0.010: (False, 0.40),
    }
    assert select_tolerance(outcomes) == 0.005
    all_infeasible = {
        0.002: (False, 0.0),
        0.005: (False, 0.1),
    }
    assert select_tolerance(all_infeasible) == 0.002


def test_gate_uses_feat_position_not_raw_s() -> None:
    """Scoring reads feat__position; raw s may be absent."""
    rows: list[dict[str, object]] = []
    for pid in ("p0", "p1"):
        rows += _cell_rows(
            task="t",
            ratio=0.5,
            compressor="a",
            prompt_id=pid,
            dq_by_s={0: 1.0, 16: 0.0, 32: 0.0},
        )
    gate = fit_position_gate(rows, tolerance=0.01, min_groups=1)
    probe = _row(task="t", s=32)
    del probe["s"]
    score = score_rows(gate, [probe])[0]
    assert score < 0.0
    assert isinstance(gate, PositionGate)
    assert not math.isnan(score)
