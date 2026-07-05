"""Tests for the consensus-damage model helpers."""

import numpy as np

from herald.switch_risk import (
    build_consensus_points,
    fit_consensus_model,
    predict_risk,
)


def _row(
    prompt_id: str,
    compressor: str,
    position: float,
    dq: float,
    signal: float,
) -> dict[str, object]:
    return {
        "model": "llama",
        "task": "gsm8k",
        "prompt_id": prompt_id,
        "ratio": 0.5,
        "compressor": compressor,
        "s": int(position),
        "dq": dq,
        "feat__position": position,
        "feat__entropy": signal,
    }


def test_build_consensus_points_means_over_compressors() -> None:
    rows = [
        _row("p1", "a", 0.0, 0.0, 0.3),
        _row("p1", "b", 0.0, 0.5, 0.3),
        _row("p1", "a", 16.0, 1.0, 0.9),
    ]
    points = build_consensus_points(rows)
    assert len(points) == 2
    by_pos = {p["feat__position"]: p for p in points}
    assert by_pos[0.0]["consensus_dq"] == 0.25
    assert by_pos[16.0]["consensus_dq"] == 1.0


def test_fit_consensus_model_classifier_orders_risk() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        signal = float(rng.uniform(0.0, 1.0))
        dq = 0.5 if signal > 0.5 else 0.0
        rows.append(_row(f"p{i}", "a", 0.0, dq, signal))
    points = build_consensus_points(rows)
    model = fit_consensus_model(
        points, ["feat__entropy"], kind="classifier", seed=0
    )
    safe = [_row("q0", "a", 0.0, 0.0, 0.1)]
    risky = [_row("q1", "a", 0.0, 0.0, 0.9)]
    p_safe = predict_risk(model, safe, ["feat__entropy"])[0]
    p_risky = predict_risk(model, risky, ["feat__entropy"])[0]
    assert 0.0 <= p_safe <= 1.0
    assert 0.0 <= p_risky <= 1.0
    assert p_risky > p_safe


def test_fit_consensus_model_magnitude_regresses_dq() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(400):
        signal = float(rng.uniform(0.0, 1.0))
        rows.append(_row(f"p{i}", "a", 0.0, signal, signal))
    points = build_consensus_points(rows)
    model = fit_consensus_model(
        points, ["feat__entropy"], kind="magnitude", seed=0
    )
    low = predict_risk(
        model, [_row("q0", "a", 0.0, 0.0, 0.1)], ["feat__entropy"]
    )[0]
    high = predict_risk(
        model, [_row("q1", "a", 0.0, 0.0, 0.9)], ["feat__entropy"]
    )[0]
    assert high > low
