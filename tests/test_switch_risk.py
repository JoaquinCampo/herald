"""Tests for the per-row worst-case risk score shaping."""

import numpy as np

from herald.switch_risk import (
    PRE_WALL,
    RISKY_BASE,
    build_worst_points,
    shape_scores,
)


def _row(
    *,
    compressor: str = "a",
    prompt_id: str = "p0",
    s: int = 0,
    dq: float = 0.0,
    entropy: float = 1.0,
) -> dict[str, object]:
    return {
        "model": "m",
        "task": "t",
        "ratio": 0.5,
        "compressor": compressor,
        "prompt_id": prompt_id,
        "s": s,
        "feat__position": float(s),
        "feat__entropy": entropy,
        "dq": dq,
        "ref_len": 128,
    }


def test_worst_points_take_max_over_compressors() -> None:
    """One point per grid cell, labelled with the worst compressor dq."""
    rows = [
        _row(compressor="a", s=0, dq=0.1),
        _row(compressor="b", s=0, dq=0.7),
        _row(compressor="a", s=16, dq=-0.2),
        _row(compressor="b", s=16, dq=0.0),
    ]
    points = build_worst_points(rows)
    labels = {
        float(point["feat__position"]): point["worst_dq"] for point in points
    }
    assert labels[0.0] == 0.7
    assert labels[16.0] == 0.0
    assert len(points) == 2


def test_shape_scores_regions() -> None:
    """Safe region graded by risk; wall at position 0; risky graded."""
    risk = np.asarray([-0.05, 0.01, 0.5, 0.4])
    positions = np.asarray([32.0, 0.0, 0.0, 48.0])
    scores = shape_scores(risk, positions, cut=0.02)
    # r <= cut: score is the risk itself
    assert scores[0] == -0.05
    assert scores[1] == 0.01
    # r > cut at position 0: the wall
    assert scores[2] == PRE_WALL
    # r > cut later: graded above the wall
    assert scores[3] > RISKY_BASE
    # graded risky rows order by risk
    higher = shape_scores(np.asarray([0.9]), np.asarray([48.0]), cut=0.02)[0]
    assert higher > scores[3]


def test_shape_scores_risky_grades_stay_below_unreachable_top() -> None:
    """Risky grading stays within its band."""
    risk = np.asarray([5.0])
    positions = np.asarray([16.0])
    score = shape_scores(risk, positions, cut=0.0)[0]
    assert RISKY_BASE < score < RISKY_BASE + 1.2
