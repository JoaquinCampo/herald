import pytest

from herald.deployment_contract import (
    DeploymentContract,
    DeploymentMeasurement,
    evaluate_deployment,
    measurement_from_live_records,
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
    assert report.tail_quality_pass
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


def test_major_quality_failures_are_a_separate_tail_constraint() -> None:
    rows = [_measurement("p0", quality_candidate=0.0)] + [
        _measurement(f"p{i}") for i in range(1, 3)
    ]

    report = evaluate_deployment(rows, contract=_contract())

    assert not report.tail_quality_pass
    assert not report.quality_pass
    assert not report.feasible
    assert "major_damage_rate" in report.failures


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


def test_duplicate_prompt_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique prompt IDs"):
        evaluate_deployment(
            [_measurement("p0"), _measurement("p0"), _measurement("p1")],
            contract=_contract(),
        )


def test_invalid_measurement_is_rejected() -> None:
    with pytest.raises(ValueError, match="baseline_wall_s"):
        _measurement("p0", baseline_wall_s=0.0)


def test_live_record_adapter_uses_live_baseline_and_isolated_kv() -> None:
    baseline = {
        "prompt_id": "p0",
        "wall_s": 10.0,
        "ref_len": 100,
        "q_ref_live": 0.8,
        "q_ref_recorded": 0.2,
        "kv_measurement_scope": "end_to_end_retained_kv_cache",
        "peak_kv_cache_bytes": 1_000,
    }
    episode = {
        "prompt_id": "p0",
        "commit_s": 20,
        "n_new_ids": 70,
        "q_live": 0.75,
        "total_wall_s": 10.2,
        "kv_measurement_scope": "end_to_end_retained_kv_cache",
        "peak_kv_cache_bytes": 500,
    }

    row = measurement_from_live_records(episode, baseline)

    assert row.quality_reference == 0.8
    assert row.quality_candidate == 0.75
    assert row.baseline_tokens == 100
    assert row.candidate_tokens == 90
    assert row.baseline_peak_kv_bytes == 1_000
    assert row.candidate_peak_kv_bytes == 500


def test_live_record_adapter_requires_end_to_end_kv_scope() -> None:
    baseline = {
        "prompt_id": "p0",
        "wall_s": 10.0,
        "ref_len": 100,
        "q_ref_live": 1.0,
        "peak_kv_cache_bytes": 1_000,
    }
    episode = {
        "prompt_id": "p0",
        "commit_s": None,
        "ref_len_live": 100,
        "n_new_ids": 0,
        "q_live": 1.0,
        "total_wall_s": 10.0,
        "peak_kv_cache_bytes": 900,
    }

    with pytest.raises(ValueError, match="kv_measurement_scope"):
        measurement_from_live_records(episode, baseline)


def test_live_record_adapter_rejects_allocator_peak_as_kv_measurement() -> (
    None
):
    baseline = {
        "prompt_id": "p0",
        "wall_s": 10.0,
        "ref_len": 100,
        "q_ref_live": 1.0,
        "kv_measurement_scope": "end_to_end_retained_kv_cache",
        "peak_mem_bytes": 10_000,
    }
    episode = {
        "prompt_id": "p0",
        "commit_s": None,
        "ref_len_live": 100,
        "n_new_ids": 0,
        "q_live": 1.0,
        "total_wall_s": 10.0,
        "kv_measurement_scope": "end_to_end_retained_kv_cache",
        "peak_mem_bytes": 9_000,
    }

    row = measurement_from_live_records(episode, baseline)

    assert row.baseline_peak_kv_bytes is None
    assert row.candidate_peak_kv_bytes is None
