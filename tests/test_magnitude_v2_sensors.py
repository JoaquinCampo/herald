"""Synthetic, no-fit contracts for the locked sensor comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from herald.magnitude_v2_sensors import (
    _as_int,
    _development_and_quarantine,
    _load_reference_inputs,
    _reject_quarantine,
    assemble_gates,
    expected_m2_freeze_payload,
    join_sensor_features,
    paired_prompt_bootstrap_m2_minus_m1,
    sha256_file,
    validate_m2_freeze,
)
from herald.press_sensors import sensor_feature_names


def _base(prompt: str = "p-1") -> dict[str, object]:
    return {
        "model": "llama",
        "task": "ifeval",
        "prompt_id": prompt,
        "compressor": "knorm",
        "ratio": 0.5,
        "s": 0,
        "dq": 1.0,
        "feat__x": 2.0,
    }


def _sensor(base: dict[str, object]) -> dict[str, object]:
    row = {
        key: base[key] for key in ("prompt_id", "compressor", "ratio", "s")
    }
    row.update(
        {
            f"sensor__{name}": float(index)
            for index, name in enumerate(sensor_feature_names())
        }
    )
    return row


def _m1(base: dict[str, object]) -> dict[str, object]:
    row = {
        key: base[key]
        for key in ("model", "task", "prompt_id", "compressor", "ratio", "s")
    }
    row.update(
        {
            "dq": base["dq"],
            "m1_prediction": 0.5,
            "locked_baseline_mean": 0.0,
            "locked_baseline_median": 0.0,
        }
    )
    return row


def test_join_exposes_ordered_sensor_columns_and_rejects_missing_key() -> (
    None
):
    base, sensor, m1 = _base(), _sensor(_base()), _m1(_base())
    joined = join_sensor_features([base], [sensor], [m1])
    names = [name for name in joined[0] if name.startswith("sensor__")]
    assert names == [f"sensor__{name}" for name in sensor_feature_names()]
    assert "prompt_id" not in names
    with pytest.raises(ValueError, match="one-to-one"):
        join_sensor_features([base], [sensor], [])


def test_join_rejects_dq_disagreement_and_nonfinite_sensor() -> None:
    base, sensor, m1 = _base(), _sensor(_base()), _m1(_base())
    m1["dq"] = 0.0
    with pytest.raises(ValueError, match="dq disagree"):
        join_sensor_features([base], [sensor], [m1])
    m1["dq"] = 1.0
    sensor["sensor__removed_k_norm_mass_fraction_lmean"] = np.nan

    with pytest.raises(ValueError, match="nonfinite"):
        join_sensor_features([base], [sensor], [m1])


@pytest.mark.parametrize("value", [0.5, True, np.bool_(True), np.inf])
def test_exact_switch_position_rejects_coercion(value: object) -> None:
    with pytest.raises(ValueError):
        _as_int(value)


def test_reference_inputs_bind_manifest_and_prefixes(tmp_path: Path) -> None:
    task_root = tmp_path / "llama" / "ifeval"
    reference_path = task_root / "references" / "p-1.json"
    reference_path.parent.mkdir(parents=True)
    reference_path.write_text(
        json.dumps(
            {
                "prompt_id": "p-1",
                "prompt_input_ids": [1, 2, 3, 4, 5],
                "gen_ids": [6, 7],
            }
        )
    )
    manifest_path = task_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_files": [
                    {
                        "path": reference_path.name,
                        "sha256": sha256_file(reference_path),
                    }
                ]
            }
        )
    )
    prefix = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    record = {
        "prompt_id": "p-1",
        "s": 1,
        "prefix_hash": hashlib.sha256(prefix.tobytes()).hexdigest(),
    }
    evidence = {
        "source_manifest": {"sha256": sha256_file(manifest_path)},
        "split": {"train_prompt_ids": ["p-1"]},
    }
    manifest_hash, inputs_hash = _load_reference_inputs(
        tmp_path,
        evidence,
        [record],
    )
    assert manifest_hash == sha256_file(manifest_path)
    assert len(inputs_hash) == 64

    record["prefix_hash"] = "0" * 64
    with pytest.raises(ValueError, match="prefix hash"):
        _load_reference_inputs(tmp_path, evidence, [record])


def test_m2_freeze_binds_every_prefit_artifact(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "base.parquet",
            "m1-freeze.json",
            "m1-report.json",
            "m1-oof.parquet",
            "sensor-manifest.json",
            "sensor-sidecar.jsonl",
            "evidence.json",
            "lock.json",
            "sensor-lock.json",
        )
    }
    for index, path in enumerate(paths.values()):
        path.write_text(str(index))
    payload = expected_m2_freeze_payload(
        base_parquet_path=paths["base.parquet"],
        m1_freeze_path=paths["m1-freeze.json"],
        m1_report_path=paths["m1-report.json"],
        m1_oof_path=paths["m1-oof.parquet"],
        sensor_manifest_path=paths["sensor-manifest.json"],
        sensor_sidecar_path=paths["sensor-sidecar.jsonl"],
        evidence_path=paths["evidence.json"],
        lock_path=paths["lock.json"],
        sensor_lock_path=paths["sensor-lock.json"],
        reference_manifest_sha256="a" * 64,
        reference_inputs_sha256="b" * 64,
    )
    freeze_path = tmp_path / "m2-freeze.json"
    freeze_path.write_text(json.dumps(payload))
    assert validate_m2_freeze(freeze_path, payload) == payload

    payload["status"] = "changed"
    with pytest.raises(ValueError, match="does not match"):
        validate_m2_freeze(freeze_path, payload)


def test_evidence_partition_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _development_and_quarantine(
            {
                "split": {
                    "train_prompt_ids": ["dev"],
                    "test_prompt_ids": ["dev"],
                }
            }
        )


def test_quarantine_is_rejection_only() -> None:
    with pytest.raises(ValueError, match="quarantined"):
        _reject_quarantine(
            [_base("test-only")], {"test-only"}, "synthetic rows"
        )


def test_m2_minus_m1_bootstrap_direction_is_deterministic() -> None:
    rows = [
        {"prompt_id": "a", "ratio": 0.5, "s": 0, "dq": 1.0},
        {"prompt_id": "a", "ratio": 0.75, "s": 0, "dq": 1.0},
        {"prompt_id": "b", "ratio": 0.5, "s": 0, "dq": -1.0},
        {"prompt_id": "b", "ratio": 0.75, "s": 0, "dq": -1.0},
    ]
    m1, m2 = np.zeros(4), np.asarray([1.0, 1.0, -1.0, -1.0])
    first = paired_prompt_bootstrap_m2_minus_m1(
        rows, m1, m2, resamples=64, seed=7
    )
    second = paired_prompt_bootstrap_m2_minus_m1(
        rows, m1, m2, resamples=64, seed=7
    )
    assert first == second
    assert first["estimate"] < 0.0
    assert first["upper"] < 0.0


def _metric(mse: float, mae: float = 1.0) -> dict[str, object]:
    return {
        "ratio_macro": {"mse": mse, "mae": mae},
        "positive_rows": {"mse": mse, "mae": mae},
        "major_rows": {"mse": mse, "mae": mae},
        "per_ratio": {"0.5": {"mse": mse}},
    }


def test_gate_assembly_is_independent_per_compressor() -> None:
    passing = {
        "m1": _metric(2.0),
        "m2": _metric(0.5),
        "locked_mean": _metric(1.0),
        "locked_median": _metric(1.0),
        "m2_vs_locked_median": {"mae": 0.5},
        "bootstrap_locked_mean": {"lower": 0.1},
        "paired_bootstrap": {"upper": -0.1},
        "positive_risk_brier_skill": 0.1,
    }
    failing = {**passing, "m2": _metric(1.5)}
    result = assemble_gates({"knorm": passing, "streaming_llm": failing})
    assert result["per_compressor"]["knorm"]["all_criteria_pass"]
    assert not result["per_compressor"]["streaming_llm"]["all_criteria_pass"]
    assert result["passing_compressors"] == ["knorm"]
