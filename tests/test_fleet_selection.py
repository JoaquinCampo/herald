"""Tests for budget-aware calibration and fleet selection."""

import math

import pytest

from herald.controller_metrics import (
    evaluate_controller_split,
    evaluate_frozen_tau,
    select_tau,
)
from herald.fleet_selection import (
    Candidate,
    bootstrap_cost_bound,
    calibrate_tau,
    select_by_cost_bound,
)


def make_row(
    prompt_id: str,
    s: int,
    dq: float,
    *,
    ratio: float = 0.5,
    ref_len: float = 100.0,
) -> dict[str, object]:
    return {
        "model": "m",
        "task": "t",
        "prompt_id": prompt_id,
        "ratio": ratio,
        "s": s,
        "ref_len": ref_len,
        "dq": dq,
    }


def two_group_case() -> tuple[list[dict[str, object]], list[float]]:
    """Group a is safe early; group b is damaged early, safe late."""
    rows = [
        make_row("a", 10, 0.0),
        make_row("a", 50, 0.0),
        make_row("b", 10, 0.5),
        make_row("b", 50, 0.0),
    ]
    scores = [0.10, 0.20, 0.30, 0.15]
    return rows, scores


def test_calibrate_tau_point_maximizes_savings_in_budget() -> None:
    rows, scores = two_group_case()
    result = calibrate_tau(rows, scores, method="point", epsilon=0.01)
    # tau must admit a@10 (0.9 savings) and b@50 (0.5 savings) but
    # exclude b@10 (dq 0.5 blows the budget).
    assert result.oof_savings == pytest.approx(0.7)
    assert result.oof_cost == pytest.approx(0.0)
    assert 0.15 <= result.tau < 0.30


def test_calibrate_tau_never_switch_when_nothing_fits() -> None:
    rows = [make_row("a", 10, 0.9), make_row("b", 10, 0.8)]
    scores = [0.1, 0.2]
    result = calibrate_tau(rows, scores, method="point", epsilon=0.01)
    assert result.oof_savings == 0.0
    assert result.tau < 0.1


def test_calibrate_tau_does_not_mutate_rows() -> None:
    rows, scores = two_group_case()
    calibrate_tau(rows, scores, method="point")
    assert all("predicted_dq" not in row for row in rows)


def variable_cost_case() -> tuple[list[dict[str, object]], list[float]]:
    """Many groups; low scores are cheap, high scores borderline."""
    rows: list[dict[str, object]] = []
    scores: list[float] = []
    for i in range(40):
        pid = f"p{i}"
        rows.append(make_row(pid, 10, 0.3 if i % 4 == 0 else 0.0))
        scores.append(0.5)
        rows.append(make_row(pid, 80, 0.0))
        scores.append(0.1)
    return rows, scores


def test_calibrate_tau_boot_is_more_conservative_than_point() -> None:
    rows, scores = variable_cost_case()
    point = calibrate_tau(rows, scores, method="point", epsilon=0.08)
    boot = calibrate_tau(rows, scores, method="boot90", epsilon=0.08, seed=0)
    assert boot.oof_savings <= point.oof_savings
    assert boot.tau <= point.tau


def test_calibrate_tau_rejects_unknown_method() -> None:
    rows, scores = two_group_case()
    with pytest.raises(ValueError):
        calibrate_tau(rows, scores, method="magic")


def test_bootstrap_cost_bound_at_least_mean_cost() -> None:
    rows, scores = variable_cost_case()
    tau = 0.5  # switch every group at s=10; 1 in 4 costs 0.3
    bound = bootstrap_cost_bound(rows, scores, tau, quantile=0.90, seed=0)
    mean_cost = 0.3 / 4
    assert bound >= mean_cost
    again = bootstrap_cost_bound(rows, scores, tau, quantile=0.90, seed=0)
    assert bound == again


def test_select_by_cost_bound_prefers_admissible_savings() -> None:
    safe_rows, safe_scores = two_group_case()
    risky_rows, risky_scores = variable_cost_case()
    risky = Candidate(
        name="risky",
        rows=risky_rows,
        scores=risky_scores,
        tau=0.5,
    )
    safe = Candidate(
        name="safe", rows=safe_rows, scores=safe_scores, tau=0.20
    )
    winner, reports = select_by_cost_bound(
        [risky, safe], epsilon=0.01, quantile=0.90, seed=0
    )
    assert winner is not None
    assert winner.name == "safe"
    by_name = {report.name: report for report in reports}
    assert not by_name["risky"].admissible
    assert by_name["risky"].oof_savings > by_name["safe"].oof_savings
    assert by_name["safe"].cost_bound <= 0.01


def test_select_by_cost_bound_none_admissible() -> None:
    rows, scores = variable_cost_case()
    candidate = Candidate(name="only", rows=rows, scores=scores, tau=0.5)
    winner, reports = select_by_cost_bound(
        [candidate], epsilon=0.01, quantile=0.90, seed=0
    )
    assert winner is None
    assert len(reports) == 1
    assert not reports[0].admissible


def scored_rows(
    rows: list[dict[str, object]], scores: list[float]
) -> list[dict[str, object]]:
    return [
        {**row, "predicted_dq": score}
        for row, score in zip(rows, scores, strict=True)
    ]


def test_evaluate_frozen_tau_hand_computed() -> None:
    rows, scores = two_group_case()
    test_rows = scored_rows(rows, scores)
    report = evaluate_frozen_tau(
        test_rows, prediction_key="predicted_dq", tau=0.20
    )
    assert report["tau"] == 0.20
    assert report["mean_savings"] == pytest.approx(0.7)
    assert report["mean_cost"] == pytest.approx(0.0)
    assert report["budget_respected"]
    assert report["n_test_groups"] == 2
    assert report["switch_rate"] == pytest.approx(1.0)


def test_evaluate_frozen_tau_matches_locked_evaluator() -> None:
    train_rows, train_scores = variable_cost_case()
    train = scored_rows(train_rows, train_scores)
    test_rows, test_scores = two_group_case()
    test = scored_rows(test_rows, test_scores)
    tau = select_tau(train, prediction_key="predicted_dq")
    frozen = evaluate_frozen_tau(
        test, prediction_key="predicted_dq", tau=tau, seed=3
    )
    full = evaluate_controller_split(
        train, test, prediction_key="predicted_dq", seed=3
    )
    assert frozen["tau"] == pytest.approx(full["tau"])
    for key in ("mean_savings", "mean_cost", "switch_rate"):
        assert frozen[key] == pytest.approx(full[key])
    assert frozen["bootstrap_ci"] == full["bootstrap_ci"]
    assert frozen["oracle"] == full["oracle"]
    assert math.isfinite(frozen["mean_savings"])
