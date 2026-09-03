"""Synthetic contract tests for the locked M3 helpers."""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from herald import magnitude_v2 as v2
from herald import magnitude_v3 as m3
from herald.magnitude import KNOWN_COMPRESSORS


def _row(
    prompt: str,
    compressor: str = "knorm",
    ratio: float = 0.5,
    s: int = 0,
    dq: float = 0.1,
) -> dict[str, object]:
    return {
        "model": "llama",
        "task": "ifeval",
        "prompt_id": prompt,
        "compressor": compressor,
        "ratio": ratio,
        "s": s,
        "dq": dq,
    }


def _protocol_lock() -> dict[str, Any]:
    return {
        "schema_version": "herald.magnitude_v3_layer_band_lock.v1",
        "status": "locked_before_m3_layer_sensor_results",
        "candidate_suite": copy.deepcopy(m3.CANDIDATE_SUITE),
        "nested_selection": copy.deepcopy(m3.NESTED_SELECTION),
        "simultaneous_inference": copy.deepcopy(m3.SIMULTANEOUS_INFERENCE),
        "development_gate": copy.deepcopy(m3.DEVELOPMENT_GATE),
        "stop_rule": m3.STOP_RULE,
        "diagnostics_only": list(m3.DIAGNOSTICS_ONLY),
        "replay": {
            "state_semantics": {
                "expected_attention": "herald.matched_reprefill_v1",
                "knorm": "herald.cache_native_pending_v1",
                "streaming_llm": "herald.cache_native_pending_v1",
            }
        },
    }


def _joined_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    base = [_row("p", dq=0.2), _row("q", dq=-0.2)]
    m2_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    for index, row in enumerate(base):
        m2_rows.append(
            {
                **row,
                "m2_prediction": float(row["dq"]) + 0.1,
                "locked_baseline_mean": 0.0,
                "locked_baseline_median": 0.0,
                "fold": index,
                "fold_train_prompt_ids_sha256": "a" * 64,
                "fold_test_prompt_ids_sha256": "b" * 64,
                **dict.fromkeys(m3.GLOBAL_FEATURES, 1.0),
            }
        )
        band_rows.append(
            {
                **row,
                **dict.fromkeys(m3.BAND_FEATURES, 2.0),
            }
        )
    return base, m2_rows, band_rows


def test_exact_integer_keys_reject_bool_and_fraction() -> None:
    row = _row("p")
    assert m3.exact_key(row)[-1] == 0
    row["s"] = True
    with pytest.raises(ValueError):
        m3.exact_key(row)
    row["s"] = 1.5
    with pytest.raises(ValueError):
        m3.exact_key(row)


def test_join_is_exact_and_carries_frozen_m2_fields() -> None:
    base, m2_rows, band_rows = _joined_rows()
    joined = m3.join_m3_features(base, m2_rows, band_rows)
    assert len(joined) == len(base)
    assert joined[0]["m2_prediction"] == pytest.approx(0.3)
    assert joined[0]["m2_locked_baseline_median"] == 0.0
    assert set(m3.GLOBAL_FEATURES) <= joined[0].keys()
    assert set(m3.BAND_FEATURES) <= joined[0].keys()
    with pytest.raises(ValueError, match="duplicate M2 exact key"):
        m3.join_m3_features(
            base,
            [*m2_rows, dict(m2_rows[0])],
            band_rows,
        )


def test_feature_schema_does_not_duplicate_ratio_or_position() -> None:
    c1 = m3._features("C1")
    c2 = m3._features("C2")
    assert "ratio" not in c1
    assert "s" not in c1
    assert len(c1) == 146
    assert len(c2) == 182
    assert len(c2) == len(set(c2))


def test_protocol_validator_requires_exact_candidate_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "m3-lock.json"
    lock = _protocol_lock()
    path.write_text(json.dumps(lock))
    assert m3.validate_m3_protocol_lock(path) == lock
    lock["candidate_suite"]["C4"]["alpha_candidates"] = [1.0]
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="candidate-suite"):
        m3.validate_m3_protocol_lock(path)


def test_band_loader_rejects_wrong_state_semantics(tmp_path: Path) -> None:
    lock_path = tmp_path / "m3-lock.json"
    lock_path.write_text(json.dumps(_protocol_lock()))
    sidecar_root = tmp_path / "bands"
    sidecar = sidecar_root / "sensor_sidecars" / "llama__ifeval.jsonl"
    sidecar.parent.mkdir(parents=True)
    evidence_hash = "a" * 64
    parent_lock_hash = "b" * 64
    m3_lock_hash = "c" * 64
    record = {
        "prompt_id": "p",
        "compressor": "knorm",
        "ratio": 0.25,
        "s": 0,
        "sensors": dict.fromkeys(m3.LAYER_BAND_FEATURE_NAMES, 1.0),
        "prefix_hash": "d" * 64,
        "protocol_version": m3.PROTOCOL_VERSION,
        "feature_names": list(m3.LAYER_BAND_FEATURE_NAMES),
        "evidence_sha256": evidence_hash,
        "lock_sha256": parent_lock_hash,
        "sensor_lock_sha256": m3_lock_hash,
        "model_key": "llama",
        "task": "ifeval",
        "state_semantics": "wrong",
    }
    sidecar.write_text(json.dumps(record) + "\n")
    manifest = sidecar.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "model": "llama",
                "task": "ifeval",
                "protocol_version": m3.PROTOCOL_VERSION,
                "feature_names": list(m3.LAYER_BAND_FEATURE_NAMES),
                "evidence_sha256": evidence_hash,
                "lock_sha256": parent_lock_hash,
                "sensor_lock_sha256": m3_lock_hash,
                "sidecar_sha256": m3.sha256_file(sidecar),
            }
        )
    )
    with pytest.raises(ValueError, match="state semantics"):
        m3.load_band_sidecar(
            sidecar_root,
            manifest,
            expected_keys={("p", "knorm", 0.25, 0)},
            evidence_sha256=evidence_hash,
            lock_sha256=parent_lock_hash,
            sensor_lock_sha256=m3_lock_hash,
            band_lock_path=lock_path,
        )


def test_shrinkage_is_training_only_closed_form_and_clipped() -> None:
    assert m3.fit_shrinkage([2, 2], [1, 1], [0, 0]) == pytest.approx(0.5)
    assert m3.fit_shrinkage([2, 2], [-1, -1], [0, 0]) == 0.0
    assert m3.fit_shrinkage([-2, -2], [1, 1], [0, 0]) == 0.0
    assert np.allclose(
        m3.apply_shrinkage([2, 0], [0, 2], 0.5),
        [1, 1],
    )


def test_ratio_position_baseline_falls_back_to_ratio_mean() -> None:
    train = [
        _row("p", ratio=0.25, s=0, dq=0.2),
        _row("q", ratio=0.25, s=16, dq=0.4),
        _row("p", ratio=0.5, s=0, dq=10.0),
    ]
    fit = m3._ratio_position_mean(
        train,
        np.asarray([0.2, 0.4, 10.0]),
    )
    prediction = m3._baseline(
        [_row("z", ratio=0.25, s=64)],
        fit,
    )
    assert prediction[0] == pytest.approx(0.3)


def test_c4_prediction_returns_shared_plus_locked_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row("p", ratio=0.25), _row("q", ratio=0.25)]
    shared = (object(),)
    residual = (object(),)

    def fake_predict(
        models: object,
        rows: object,
        features: object,
    ) -> np.ndarray:
        del rows, features
        value = 1.0 if models is shared else 4.0
        return np.full(2, value)

    monkeypatch.setattr(m3, "_predict", fake_predict)
    fit = (shared, {0.25: residual})
    assert np.allclose(
        m3._predict_candidate(fit, rows, "C4", alpha=0.75),
        2.0,
    )
    assert np.allclose(
        m3._predict_candidate(fit, rows, "C4", alpha=1.0),
        1.0,
    )


def test_positive_risk_predicts_from_raw_margins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_raw(
        model: object,
        rows: object,
        features: object,
        *,
        output_margin: bool = False,
    ) -> np.ndarray:
        del model, rows, features
        calls.append(output_margin)
        return np.asarray([2.0])

    monkeypatch.setattr(v2, "_xgb_raw", fake_raw)
    result = m3.predict_positive_risk(
        {
            "models": (object(), object()),
            "features": (),
            "calibration": v2.PlattCalibration(1.0, 0.0, 0.5),
        },
        [_row("p")],
    )
    assert calls == [True, True]
    assert result[0] == pytest.approx(1 / (1 + np.exp(-2)))


def test_simplest_candidate_within_one_se() -> None:
    scores = dict(
        zip(
            m3.CANDIDATES,
            [1.0, 0.95, 0.7, 0.71, 0.72],
            strict=True,
        )
    )
    errors = dict.fromkeys(m3.CANDIDATES, 0.03)
    assert m3.select_simplest_one_se(scores, errors) == "C2"


def _inference_rows() -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for compressor in sorted(KNOWN_COMPRESSORS):
        rows: list[dict[str, object]] = []
        for prompt_index in range(6):
            for ratio in m3.RATIOS:
                target = 0.4 if prompt_index % 2 else -0.2
                rows.append(
                    {
                        **_row(
                            f"p{prompt_index}",
                            compressor,
                            ratio,
                            dq=target,
                        ),
                        "m3_prediction": target + 0.02,
                        "m2_prediction": target + 0.1,
                        "locked_baseline_mean": 0.0,
                        "positive_risk": 0.7 if target > 0 else 0.3,
                        "positive_risk_baseline": 0.5,
                    }
                )
        output[compressor] = rows
    return output


def test_max_t_is_prompt_clustered_and_simultaneous() -> None:
    rows = [
        _row(f"p{index}", ratio=ratio)
        for index in range(4)
        for ratio in m3.RATIOS
    ]
    model = [0.0] * len(rows)
    baseline = [1.0] * len(rows)
    result = m3.max_t_bootstrap(
        {"knorm": rows, "streaming_llm": rows},
        {"knorm": model, "streaming_llm": model},
        {"knorm": baseline, "streaming_llm": baseline},
        resamples=30,
        seed=3,
    )
    assert result["resamples"] == 30
    assert result["multiplicity"] == "simultaneous max-T over 2 claims"
    assert set(result["intervals"]) == {"knorm", "streaming_llm"}
    assert all(
        "lower" in interval for interval in result["intervals"].values()
    )


def test_all_four_simultaneous_families_have_locked_claim_counts() -> None:
    result = m3._simultaneous_development_inference(
        _inference_rows(),
        resamples=50,
        seed=9,
    )
    families = result["families"]
    assert len(families["macro_mse_skill"]["intervals"]) == 3
    assert len(families["paired_m3_minus_m2"]["intervals"]) == 3
    assert len(families["positive_risk_brier_skill"]["intervals"]) == 3
    assert len(families["per_ratio_mse_skill"]["intervals"]) == 12
    assert all(
        "upper" in interval
        for interval in families["paired_m3_minus_m2"]["intervals"].values()
    )


def test_gate_requires_every_locked_condition() -> None:
    good = m3.evaluate_gate(
        primary_lower={"knorm": 0.05},
        per_ratio_skills={"knorm": dict.fromkeys(map(str, m3.RATIOS), 0.01)},
        paired_upper={"knorm": -0.01},
        brier_lower={"knorm": 0.001},
        calibration_slope={"knorm": 0.2},
        selected_by_fold={"knorm": ["C2"] * 5},
    )
    assert good["any_compressor_pass"] is True
    bad = m3.evaluate_gate(
        primary_lower={"knorm": 0.05},
        per_ratio_skills={"knorm": {"0.25": 0.01, "0.5": -0.01}},
        paired_upper={"knorm": -0.01},
        brier_lower={"knorm": 0.001},
        calibration_slope={"knorm": 0.2},
        selected_by_fold={"knorm": ["C2"] * 5},
    )
    assert bad["any_compressor_pass"] is False


def test_complete_selection_procedure_emits_nested_oof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_ids = [f"p{index}" for index in range(5)]
    folds = tuple(
        v2.Fold(index, (prompt,), v2.hash_prompt_ids([prompt]))
        for index, prompt in enumerate(prompt_ids)
    )
    rows: list[dict[str, Any]] = []
    for compressor in sorted(KNOWN_COMPRESSORS):
        for prompt_index, prompt in enumerate(prompt_ids):
            for ratio in m3.RATIOS:
                target = (prompt_index - 2) * 0.4 + (ratio - 0.5) * 0.1
                rows.append(
                    {
                        **_row(
                            prompt,
                            compressor,
                            ratio,
                            dq=target,
                        ),
                        "m2_prediction": target + 0.15,
                    }
                )
        compressor_rows = [
            row for row in rows if row["compressor"] == compressor
        ]
        for fold in folds:
            test_prompts = set(fold.prompt_ids)
            train = [
                row
                for row in compressor_rows
                if row["prompt_id"] not in test_prompts
            ]
            test = [
                row
                for row in compressor_rows
                if row["prompt_id"] in test_prompts
            ]
            target = np.asarray([float(row["dq"]) for row in train])
            baseline_fit = m3._ratio_position_mean(train, target)
            baseline = m3._baseline(test, baseline_fit)
            train_hash = v2.hash_prompt_ids(
                sorted({str(row["prompt_id"]) for row in train})
            )
            for row, value in zip(test, baseline, strict=True):
                row.update(
                    {
                        "m2_locked_baseline_mean": float(value),
                        "m2_locked_baseline_median": float(value),
                        "m2_fold": fold.index,
                        "m2_fold_train_prompt_ids_sha256": train_hash,
                        "m2_fold_test_prompt_ids_sha256": fold.hash,
                    }
                )

    monkeypatch.setattr(v2, "make_folds", lambda prompt_ids, lock: folds)
    monkeypatch.setattr(
        m3,
        "_fit_candidate",
        lambda rows, candidate, seeds, shared_models=None: (candidate,),
    )
    errors = {
        "C0": 0.5,
        "C1": 0.2,
        "C2": 0.0,
        "C3": 0.3,
        "C4": 0.4,
    }

    def fake_candidate_predict(
        fit: object,
        rows: list[dict[str, Any]],
        candidate: str,
        *,
        alpha: float = 1.0,
    ) -> np.ndarray:
        del fit
        c4_extra = 0.05 if candidate == "C4" and alpha < 1 else 0.0
        return np.asarray(
            [float(row["dq"]) + errors[candidate] + c4_extra for row in rows]
        )

    monkeypatch.setattr(m3, "_predict_candidate", fake_candidate_predict)
    monkeypatch.setattr(
        m3,
        "fit_positive_risk",
        lambda rows, candidate, seeds: {
            "calibration": v2.PlattCalibration(1.0, 0.0, 0.5)
        },
    )
    monkeypatch.setattr(
        m3,
        "predict_positive_risk",
        lambda fit, rows: np.asarray(
            [0.75 if float(row["dq"]) > 0 else 0.25 for row in rows]
        ),
    )
    report, oof = m3.fit_development(rows, lock=_protocol_lock())
    assert len(oof) == len(rows)
    assert {row["selected_candidate"] for row in oof} == {"C2"}
    assert report["development_gate"]["any_compressor_pass"] is True
    assert set(report["simultaneous_inference"]["families"]) == {
        "macro_mse_skill",
        "paired_m3_minus_m2",
        "positive_risk_brier_skill",
        "per_ratio_mse_skill",
    }


def test_frozen_m2_loader_binds_current_parent_files(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    protocol_lock = tmp_path / "v2-lock.json"
    sensor_lock = tmp_path / "sensor-lock.json"
    base_parquet = tmp_path / "base.parquet"
    for path, content in (
        (evidence, "evidence"),
        (protocol_lock, "protocol"),
        (sensor_lock, "sensor"),
        (base_parquet, "base"),
    ):
        path.write_text(content)
    base = [_row("p", dq=0.2)]
    oof = tmp_path / "m2.parquet"
    m3.write_oof_parquet(
        [
            {
                **base[0],
                "m2_prediction": 0.1,
                "locked_baseline_mean": 0.0,
                "locked_baseline_median": 0.0,
                "fold": 0,
            }
        ],
        oof,
    )
    reference_manifest_hash = "a" * 64
    reference_inputs_hash = "b" * 64
    base_provenance = {
        "parquet_sha256": m3.sha256_file(base_parquet),
    }
    report = tmp_path / "m2-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "herald.magnitude_v2_sensors.v1",
                "base_provenance": {
                    "parquet_sha256": m3.sha256_file(base_parquet),
                    "source_evidence_sha256": m3.sha256_file(evidence),
                    "protocol_lock_sha256": m3.sha256_file(protocol_lock),
                },
                "sensor_provenance": {
                    "evidence_sha256": m3.sha256_file(evidence),
                    "lock_sha256": m3.sha256_file(protocol_lock),
                    "sensor_lock_sha256": m3.sha256_file(sensor_lock),
                    "reference_manifest_sha256": reference_manifest_hash,
                    "reference_inputs_sha256": reference_inputs_hash,
                },
            }
        )
    )
    freeze = tmp_path / "m2-freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": m3.M2_FREEZE_SCHEMA_VERSION,
                "status": "frozen_before_m2_results",
                "base_parquet_sha256": m3.sha256_file(base_parquet),
                "source_evidence_sha256": m3.sha256_file(evidence),
                "protocol_lock_sha256": m3.sha256_file(protocol_lock),
                "sensor_lock_sha256": m3.sha256_file(sensor_lock),
                "reference_manifest_sha256": reference_manifest_hash,
                "reference_inputs_sha256": reference_inputs_hash,
            }
        )
    )
    result_freeze = tmp_path / "m2-result-freeze.json"
    result_freeze.write_text(
        json.dumps(
            {
                "schema_version": ("herald.magnitude_v2_m2_result_freeze.v1"),
                "status": "frozen_before_m3_layer_sensor_results",
                "m2_prefit_freeze_sha256": m3.sha256_file(freeze),
                "report": {
                    "sha256": m3.sha256_file(report),
                    "oof_sha256": m3.sha256_file(oof),
                },
                "oof": {"sha256": m3.sha256_file(oof)},
                "development_decision": {
                    "any_compressor_pass": False,
                    "passing_compressors": [],
                },
            }
        )
    )
    rows, _ = m3.load_frozen_m2(
        report,
        oof,
        freeze,
        result_freeze_path=result_freeze,
        base_rows=base,
        base_provenance=base_provenance,
        evidence_path=evidence,
        protocol_lock_path=protocol_lock,
        sensor_lock_path=sensor_lock,
    )
    assert rows[0]["m2_prediction"] == pytest.approx(0.1)
    sensor_lock.write_text("changed")
    with pytest.raises(ValueError, match="sensor_lock_sha256"):
        m3.load_frozen_m2(
            report,
            oof,
            freeze,
            result_freeze_path=result_freeze,
            base_rows=base,
            base_provenance=base_provenance,
            evidence_path=evidence,
            protocol_lock_path=protocol_lock,
            sensor_lock_path=sensor_lock,
        )


def test_freeze_validator_binds_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("a")
    payload = {
        "schema_version": m3.M3_FREEZE_SCHEMA_VERSION,
        "hash": m3.sha256_file(source),
    }
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(payload))
    assert m3.validate_m3_freeze(freeze, payload) == payload
    source.write_text("b")
    with pytest.raises(ValueError):
        m3.validate_m3_freeze(
            freeze,
            {
                "schema_version": m3.M3_FREEZE_SCHEMA_VERSION,
                "hash": m3.sha256_file(source),
            },
        )
