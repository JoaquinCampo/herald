import pytest

from herald.deployment_contract import (
    DeploymentContract,
    DeploymentMeasurement,
    evaluate_deployment,
)


def _measurement(
    prompt_id: str,
    *,
    quality_reference: float = 1.0,
    quality_candidate: float = 1.0,
    baseline_wall_s: float = 10.0,
    candidate_wall_s: float = 10.2,
    baseline_tokens: int = 100,
    candidate_tokens: int = 100,
    baseline_peak_kv_bytes: int | None = 1_000,
    candidate_peak_kv_bytes: int | None = 500,
    baseline_kv_byte_tokens: float | None = 100_000.0,
    candidate_kv_byte_tokens: float | None = 50_000.0,
) -> DeploymentMeasurement:
    return DeploymentMeasurement(
        prompt_id=prompt_id,
        quality_reference=quality_reference,
        quality_candidate=quality_candidate,
        baseline_wall_s=baseline_wall_s,
        candidate_wall_s=candidate_wall_s,
        baseline_tokens=baseline_tokens,
        candidate_tokens=candidate_tokens,
        baseline_peak_kv_bytes=baseline_peak_kv_bytes,
        candidate_peak_kv_bytes=candidate_peak_kv_bytes,
        baseline_kv_byte_tokens=baseline_kv_byte_tokens,
        candidate_kv_byte_tokens=candidate_kv_byte_tokens,
    )


def _contract() -> DeploymentContract:
    return DeploymentContract(
        min_pairs=3,
        bootstrap_resamples=200,
    )


def test_feasible_configuration_passes_all_constraints() -> None:
    report = evaluate_deployment(
        [_measurement(f"p{i}") for i in range(3)],
        contract=_contract(),
    )

    assert report.feasible
    assert report.quality_pass
    assert report.speed_pass
    assert report.memory_verified
    assert report.peak_kv_savings is not None
    assert report.peak_kv_savings.mean == pytest.approx(0.5)
    assert report.kv_byte_token_savings is not None
    assert report.kv_byte_token_savings.mean == pytest.approx(0.5)
    assert report.failures == ()


def test_quality_noninferiority_uses_confidence_bound() -> None:
    rows = [_measurement(f"p{i}", quality_candidate=0.95) for i in range(3)]

    report = evaluate_deployment(rows, contract=_contract())

    assert not report.quality_pass
    assert not report.feasible
    assert "quality_noninferiority" in report.failures


def test_speed_constraint_uses_total_end_to_end_wall_time() -> None:
    rows = [_measurement(f"p{i}", candidate_wall_s=10.6) for i in range(3)]

    report = evaluate_deployment(rows, contract=_contract())

    assert not report.speed_pass
    assert not report.feasible
    assert "end_to_end_slowdown" in report.failures


def test_missing_isolated_kv_measurements_cannot_pass() -> None:
    rows = [
        _measurement(
            f"p{i}",
            baseline_peak_kv_bytes=None,
            candidate_peak_kv_bytes=None,
        )
        for i in range(3)
    ]

    report = evaluate_deployment(rows, contract=_contract())

    assert not report.memory_verified
    assert not report.feasible
    assert report.peak_kv_savings is None
    assert "isolated_kv_memory" in report.failures


def test_kv_byte_token_area_is_optional_when_peak_kv_is_measured() -> None:
    rows = [
        _measurement(
            f"p{i}",
            baseline_kv_byte_tokens=None,
            candidate_kv_byte_tokens=None,
        )
        for i in range(3)
    ]

    report = evaluate_deployment(rows, contract=_contract())

    assert report.memory_verified
    assert report.memory_pass
    assert report.kv_byte_token_savings is None
    assert report.feasible


def test_minimum_sample_size_is_enforced() -> None:
    report = evaluate_deployment(
        [_measurement("p0"), _measurement("p1")],
        contract=_contract(),
    )

    assert not report.feasible
    assert "minimum_pairs" in report.failures


def test_invalid_measurement_is_rejected() -> None:
    with pytest.raises(ValueError, match="baseline_wall_s"):
        _measurement("p0", baseline_wall_s=0.0)
