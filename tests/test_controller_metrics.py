import pytest

from herald.controller_metrics import (
    decision_groups,
    evaluate_controller_split,
    oracle_choice,
    policy_choice,
    recall_at_fpr,
    select_tau,
)


def _row(
    prompt_id: str,
    s: int,
    dq: float,
    pred: float,
    *,
    task: str = "ifeval",
    ratio: float = 0.5,
    ref_len: int = 100,
    compressor: str = "knorm",
) -> dict[str, object]:
    """Build a minimal switch row for controller tests."""
    return {
        "model": "llama",
        "task": task,
        "prompt_id": prompt_id,
        "compressor": compressor,
        "ratio": ratio,
        "s": s,
        "ref_len": ref_len,
        "dq": dq,
        "predicted_dq": pred,
    }


def test_decision_groups_split_by_prompt_and_ratio() -> None:
    """Groups separate prompts and ratios, never mix them."""
    rows = [
        _row("p1", 0, 0.0, 0.1),
        _row("p1", 16, 0.0, 0.1),
        _row("p1", 0, 0.0, 0.1, ratio=0.75),
        _row("p2", 0, 0.0, 0.1),
    ]
    groups = decision_groups(rows)
    assert len(groups) == 3
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 1, 2]


def test_policy_switches_at_earliest_qualifying_position() -> None:
    """The policy takes the first position predicted safe enough."""
    group = [
        _row("p1", 0, 1.0, 0.9),
        _row("p1", 16, 0.0, 0.05),
        _row("p1", 32, 0.0, 0.01),
    ]
    choice = policy_choice(group, prediction_key="predicted_dq", tau=0.1)
    assert choice.switched
    assert choice.s == 16
    assert choice.savings == pytest.approx(1.0 - 16.0 / 100.0)
    assert choice.cost == 0.0


def test_policy_never_switches_when_nothing_qualifies() -> None:
    """No qualifying position means zero savings and zero cost."""
    group = [
        _row("p1", 0, 1.0, 0.9),
        _row("p1", 16, 1.0, 0.8),
    ]
    choice = policy_choice(group, prediction_key="predicted_dq", tau=0.1)
    assert not choice.switched
    assert choice.savings == 0.0
    assert choice.cost == 0.0


def test_oracle_takes_earliest_zero_damage_position() -> None:
    """The oracle switches at the first truly safe position."""
    group = [
        _row("p1", 0, 0.5, 0.0),
        _row("p1", 16, 0.0, 0.9),
    ]
    choice = oracle_choice(group)
    assert choice.switched
    assert choice.s == 16
    assert choice.cost == 0.0


def test_select_tau_respects_budget_on_train() -> None:
    """Tau selection maximizes savings subject to the cost budget."""
    rows = []
    # p1: switching at 0 is safe (dq 0, pred 0.0)
    rows.extend([_row("p1", 0, 0.0, 0.0), _row("p1", 16, 0.0, 0.0)])
    # p2: switching at 0 is catastrophic and predicted risky (0.9);
    # position 16 is safe and predicted safe (0.1)
    rows.extend([_row("p2", 0, 1.0, 0.9), _row("p2", 16, 0.0, 0.1)])
    tau = select_tau(rows, prediction_key="predicted_dq", epsilon=0.01)
    # tau must admit 0.1 (captures both safe switches) but not 0.9
    assert 0.1 <= tau < 0.9


def test_evaluate_controller_split_reports_policies() -> None:
    """The split evaluator returns savings, cost, and references."""
    train = []
    test = []
    for i in range(8):
        pid = f"tr{i}"
        train.extend(
            [
                _row(pid, 0, 1.0 if i % 2 else 0.0, 0.8 if i % 2 else 0.0),
                _row(pid, 16, 0.0, 0.1),
            ]
        )
    for i in range(6):
        pid = f"te{i}"
        test.extend(
            [
                _row(pid, 0, 1.0 if i % 2 else 0.0, 0.8 if i % 2 else 0.0),
                _row(pid, 16, 0.0, 0.1),
            ]
        )
    out = evaluate_controller_split(
        train,
        test,
        prediction_key="predicted_dq",
        epsilon=0.01,
        seed=0,
        bootstrap_resamples=50,
    )
    assert out["n_test_groups"] == 6
    assert 0.0 <= out["mean_savings"] <= 1.0
    assert out["mean_cost"] <= 0.01 + 1e-9
    assert out["budget_respected"] is True
    assert out["oracle"]["mean_savings"] >= out["mean_savings"]
    assert out["never"]["mean_savings"] == 0.0
    assert out["always_s0"]["mean_savings"] == pytest.approx(1.0)
    ci = out["bootstrap_ci"]
    assert ci["savings_low"] is not None
    assert ci["savings_low"] <= out["mean_savings"] <= ci["savings_high"]


def test_recall_at_fpr_matches_manual_case() -> None:
    """Recall at bounded FPR follows the canonical definition."""
    labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.9, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    # ranked: 0.9(pos), 0.8(neg), 0.2(pos), ... at fpr<=0.10 we may
    # accept at most 0 negatives before... 1 neg of 8 = 0.125 > 0.10,
    # so only the first positive is reachable.
    value = recall_at_fpr(labels, scores, max_fpr=0.10)
    assert value == pytest.approx(0.5)
