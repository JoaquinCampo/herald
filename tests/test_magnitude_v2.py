"""Deterministic tests for locked v2 protocol and leakage boundaries."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import herald.magnitude_v2 as magnitude_v2
from herald.magnitude import trajectory_weights
from herald.magnitude_v2 import (
    DEFAULT_EVIDENCE,
    DEFAULT_LOCK,
    bootstrap_locked_mean_skill,
    cross_fitted_platt,
    fit_development,
    load_development_rows,
    make_folds,
    paired_prompt_bootstrap,
    prompt_inner_folds,
    signed_mixture_prediction,
    validate_lock_and_evidence,
)


def test_locked_real_partition_and_folds() -> None:
    evidence, lock = validate_lock_and_evidence(
        DEFAULT_EVIDENCE, DEFAULT_LOCK
    )
    folds = make_folds(evidence["split"]["train_prompt_ids"], lock)
    assert [len(fold.prompt_ids) for fold in folds] == [31, 31, 31, 31, 30]
    assert len(set().union(*(set(fold.prompt_ids) for fold in folds))) == 154
    assert not set(evidence["split"]["train_prompt_ids"]) & set(
        evidence["split"]["test_prompt_ids"]
    )


def test_protocol_rejects_tampered_source_evidence(tmp_path: Path) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text())
    evidence["split"]["train_prompt_ids"] = []
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    lock = json.loads(DEFAULT_LOCK.read_text())
    with pytest.raises(ValueError, match="source evidence hash"):
        validate_lock_and_evidence(evidence_path, DEFAULT_LOCK)
    lock["source_evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        validate_lock_and_evidence(evidence_path, lock_path)


def test_inner_folds_are_prompt_disjoint() -> None:
    prompts = [f"p-{index}" for index in range(17)]
    folds = prompt_inner_folds(prompts, 4)
    assert sum(map(len, folds)) == len(prompts)
    assert not any(
        set(left) & set(right)
        for index, left in enumerate(folds)
        for right in folds[index + 1 :]
    )


def test_signed_mixture_preserves_negative_lift_and_empty_cases() -> None:
    result = signed_mixture_prediction(
        np.asarray([1.0, 0.5]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.8, 0.8]),
        np.asarray([0.2, 0.2]),
    )
    np.testing.assert_allclose(result, [-0.2, 0.4])
    np.testing.assert_allclose(
        signed_mixture_prediction(0.0, 0.5, 2.0, 3.0), 0.0
    )
    np.testing.assert_allclose(
        signed_mixture_prediction(1.0, 0.0, 0.0, 0.0), 0.0
    )


def test_trajectory_mass_is_equal_per_prompt_ratio() -> None:
    rows = [
        {"prompt_id": "a", "ratio": 0.5, "dq": 1.0},
        {"prompt_id": "a", "ratio": 0.5, "dq": 2.0},
        {"prompt_id": "b", "ratio": 0.5, "dq": 3.0},
        {"prompt_id": "b", "ratio": 0.7, "dq": 4.0},
    ]
    weights = trajectory_weights(rows)
    assert np.isclose(weights[:2].sum(), 1.0)
    assert np.isclose(weights[2], 1.0)
    assert np.isclose(weights[3], 1.0)


def test_paired_bootstrap_is_deterministic_and_clustered() -> None:
    rows = [
        {"prompt_id": "a", "ratio": 0.5, "dq": 0.0},
        {"prompt_id": "a", "ratio": 0.7, "dq": 1.0},
        {"prompt_id": "b", "ratio": 0.5, "dq": -1.0},
        {"prompt_id": "b", "ratio": 0.7, "dq": 0.0},
    ]
    m0 = np.zeros(4)
    m1 = np.asarray([0.1, 0.9, -0.8, 0.2])
    first = paired_prompt_bootstrap(rows, m0, m1, resamples=32, seed=7)
    second = paired_prompt_bootstrap(rows, m0, m1, resamples=32, seed=7)
    assert first == second
    assert first["metric"] == "ratio_macro_mse_m1_minus_m0"


def test_protocol_pins_bootstrap_count(tmp_path: Path) -> None:
    lock = json.loads(DEFAULT_LOCK.read_text())
    lock["development_config"]["bootstrap_resamples"] = 99
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="boosting/calibration"):
        validate_lock_and_evidence(DEFAULT_EVIDENCE, lock_path)


def test_loader_rejects_wrong_parquet_before_scanning(tmp_path: Path) -> None:
    parquet_path = tmp_path / "wrong.parquet"
    parquet_path.write_bytes(b"not the frozen parquet")
    with pytest.raises(ValueError, match="authoritative parquet SHA256"):
        load_development_rows(parquet_path)


def test_platt_weights_match_expanded_observations() -> None:
    raw = np.asarray([-2.0, -0.5, 0.75, 2.0])
    target = np.asarray([0.0, 1.0, 0.0, 1.0])
    weights = np.asarray([5, 1, 2, 4])
    weighted = magnitude_v2._fit_platt(
        raw,
        target,
        weights=weights,
    )
    expanded = magnitude_v2._fit_platt(
        np.repeat(raw, weights.astype(int)),
        np.repeat(target, weights.astype(int)),
    )
    probe = np.linspace(-3.0, 3.0, 13)
    np.testing.assert_allclose(
        weighted.predict(probe),
        expanded.predict(probe),
        rtol=1e-5,
        atol=1e-6,
    )


def test_platt_single_class_is_constant_prevalence() -> None:
    calibration = magnitude_v2._fit_platt(
        np.asarray([-100.0, 100.0]),
        np.ones(2),
    )
    prediction = calibration.predict(np.asarray([-50.0, 0.0, 50.0]))
    np.testing.assert_allclose(prediction, prediction[0])
    assert prediction[0] == pytest.approx(1.0, abs=2e-6)


def test_cross_fitted_platt_uses_every_seed_and_raw_margins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained_seeds: list[int] = []
    margin_flags: list[bool] = []

    def fake_train(
        rows: list[dict[str, object]],
        features: tuple[str, ...],
        target: np.ndarray,
        *,
        objective: str,
        seed: int,
        rounds: int,
        weights: np.ndarray,
    ) -> int:
        del rows, features, target, objective, rounds, weights
        trained_seeds.append(seed)
        return seed

    def fake_raw(
        model: int,
        rows: list[dict[str, object]],
        features: tuple[str, ...],
        *,
        output_margin: bool = False,
    ) -> np.ndarray:
        del features
        margin_flags.append(output_margin)
        return np.full(len(rows), float(model))

    monkeypatch.setattr(magnitude_v2, "_xgb_train", fake_train)
    monkeypatch.setattr(magnitude_v2, "_xgb_raw", fake_raw)
    rows = [
        {
            "prompt_id": f"p-{index}",
            "ratio": 0.5,
            "s": 0,
            "feat__x": float(index),
        }
        for index in range(12)
    ]
    cross_fitted_platt(
        rows,
        np.asarray([index % 2 for index in range(12)], dtype=float),
        ("feat__x",),
        folds=4,
        rounds=1,
        seeds=(0, 1, 2),
    )
    assert trained_seeds.count(0) == 4
    assert trained_seeds.count(1) == 4
    assert trained_seeds.count(2) == 4
    assert margin_flags == [True] * 12


def test_locked_mean_bootstrap_has_correct_skill_direction() -> None:
    rows = [
        {"prompt_id": f"p-{index}", "ratio": 0.5, "dq": target}
        for index, target in enumerate((1.0, -1.0, 1.0, -1.0))
    ]
    target = np.asarray([row["dq"] for row in rows])
    result = bootstrap_locked_mean_skill(
        rows,
        target,
        np.zeros(len(rows)),
        resamples=64,
        seed=7,
    )
    assert result["estimate"] == pytest.approx(1.0)
    assert result["lower"] == pytest.approx(1.0)
    assert result["upper"] == pytest.approx(1.0)


def test_fit_rejects_unvalidated_provenance() -> None:
    _, lock = validate_lock_and_evidence(DEFAULT_EVIDENCE, DEFAULT_LOCK)
    with pytest.raises(ValueError, match="validated loader provenance"):
        fit_development(
            [{"compressor": "expected_attention"}],
            lock=lock,
            provenance=None,
        )
