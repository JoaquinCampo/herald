"""Strict nested M3 magnitude development pipeline."""

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from herald import magnitude_v2 as v2
from herald.magnitude import (
    CAUSAL_FEATURE_COLUMNS,
    KNOWN_COMPRESSORS,
    POSITION_BUCKET,
    _feature_matrix,
    _ratio_metrics,
    trajectory_weights,
)
from herald.magnitude_v2_sensors import M2_FREEZE_SCHEMA_VERSION
from herald.press_sensors import RATIOS, sensor_feature_names
from herald.press_sensors_m3 import LAYER_BAND_FEATURE_NAMES
from herald.storage import (
    read_sensor_records,
    sensor_manifest_path,
    sensor_sidecar_path,
    validate_sensor_records,
)

SCHEMA_VERSION = "herald.magnitude_v3.v1"
PROTOCOL_VERSION = "herald.m3.layer_band_replay.v1"
M3_FREEZE_SCHEMA_VERSION = "herald.magnitude_v3_freeze.v1"
MODEL_KEY, TASK = "llama", "ifeval"
KEY_COLUMNS = ("prompt_id", "compressor", "ratio", "s")
CANDIDATES = ("C0", "C1", "C2", "C3", "C4")
SEEDS = (0, 1, 2)
BASE_FEATURES = tuple(sorted(CAUSAL_FEATURE_COLUMNS))
BASE_INPUT_FEATURES = BASE_FEATURES + ("ratio", "s")
GLOBAL_FEATURES = tuple(f"sensor__{name}" for name in sensor_feature_names())
BAND_FEATURES = tuple(f"band__{name}" for name in LAYER_BAND_FEATURE_NAMES)
DIRECT_XGB_CONFIG: dict[str, float | int] = {
    "colsample_bytree": 0.7,
    "eta": 0.03,
    "max_depth": 2,
    "min_child_weight": 10,
    "num_boost_round": 600,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "subsample": 0.8,
}
RESIDUAL_XGB_CONFIG: dict[str, float | int] = {
    "colsample_bytree": 0.7,
    "eta": 0.03,
    "max_depth": 1,
    "min_child_weight": 12,
    "num_boost_round": 300,
    "reg_alpha": 0.0,
    "reg_lambda": 20.0,
    "subsample": 0.8,
}
CANDIDATE_SUITE: dict[str, Any] = {
    "C0": {
        "calibration": "prompt-cross-fitted Platt",
        "features": "base112_plus_global36",
        "kind": "frozen_M2_hurdle_reproduction",
        "xgboost": {
            "colsample_bytree": 0.8,
            "eta": 0.05,
            "max_depth": 4,
            "min_child_weight": 1,
            "num_boost_round": 300,
            "subsample": 0.8,
        },
    },
    "C1": {
        "features": "base112_plus_global36",
        "kind": "direct_mean",
        "objective": "reg:squarederror",
        "xgboost": DIRECT_XGB_CONFIG,
    },
    "C2": {
        "features": "base112_plus_global36_plus_layer_bands36",
        "kind": "direct_mean",
        "objective": "reg:squarederror",
        "xgboost": DIRECT_XGB_CONFIG,
    },
    "C3": {
        "features": "base112_plus_global36_plus_layer_bands36",
        "kind": "all_row_positive_negative_severity_difference",
        "prediction": "positive_head_minus_negative_head",
        "targets": ["max(dq,0)", "max(-dq,0)"],
        "xgboost": DIRECT_XGB_CONFIG,
    },
    "C4": {
        "alpha_candidates": [0.75, 1.0],
        "features": {
            "ratio_residual": "base112_plus_global36",
            "shared": "base112_plus_global36_plus_layer_bands36",
        },
        "kind": "weak_ratio_residual",
        "prediction": "shared + (1-alpha)*ratio_residual",
        "residual_xgboost": RESIDUAL_XGB_CONFIG,
        "shared_xgboost": DIRECT_XGB_CONFIG,
    },
    "common": {
        "compressor_independence": True,
        "model_seeds": list(SEEDS),
        "trajectory_weighting": (
            "equal total weight per prompt-ratio trajectory"
        ),
    },
    "order": list(CANDIDATES),
    "positive_risk": {
        "calibration": "prompt-cross-fitted Platt",
        "never_multiplied_into_magnitude": True,
        "objective": "binary:logistic",
        "separate_output": True,
        "xgboost": DIRECT_XGB_CONFIG,
    },
    "postfit_shrinkage": {
        "applies_to": ["C1", "C2", "C3", "C4"],
        "formula": "baseline_mean + gamma*(prediction-baseline_mean)",
        "free_intercept": False,
        "gamma": (
            "inner-OOF weighted closed-form MSE minimizer clipped to [0,1]"
        ),
    },
}
NESTED_SELECTION: dict[str, Any] = {
    "inner_folds": 4,
    "outer_folds": 5,
    "outer_oof": "predictions of the complete selection procedure",
    "rule": "choose first/simplest candidate within best_MSE + one_SE(best)",
    "selection_metric": "trajectory-weighted ratio-macro MSE",
    "stability": (
        "same non-C0 candidate must be selected in all five outer folds"
    ),
    "standard_error": (
        "1000 prompt-cluster bootstrap resamples of best candidate, seed 2718"
    ),
}
SIMULTANEOUS_INFERENCE: dict[str, Any] = {
    "confidence": 0.95,
    "families": {
        "macro_mse_skill": 3,
        "paired_m3_minus_m2": 3,
        "per_ratio_mse_skill": 12,
        "positive_risk_brier_skill": 3,
    },
    "method": "prompt-cluster max-T Westfall-Young",
    "resamples": 10000,
    "seed": 314159,
    "sidedness": "one-sided",
}
DEVELOPMENT_GATE = {
    "every_ratio_mse_skill_gt_zero": True,
    "macro_mse_skill_fwer_lower_min": 0.05,
    "paired_m3_minus_m2_fwer_upper_lt_zero": True,
    "positive_risk_brier_skill_fwer_lower_gt_zero": True,
    "positive_risk_calibration_slope_gt_zero": True,
    "same_non_c0_candidate_all_outer_folds": True,
}
DIAGNOSTICS_ONLY = [
    "MAE versus locked median",
    "positive-row MSE",
    "major-row MSE",
    "signed error",
    "rank correlation",
    "calibration intercept and reliability bins",
]
STOP_RULE = (
    "No M4: if no compressor passes every frozen M3 gate, stop development "
    "and keep confirmation sealed."
)
M3_CONFIG: dict[str, Any] = {
    "candidate_order": list(CANDIDATES),
    "model_seeds": list(SEEDS),
    "outer_folds": 5,
    "inner_folds": 4,
    "trajectory_weighting": (
        "equal total weight per prompt-ratio trajectory"
    ),
    "base_features": 112,
    "global_features": 36,
    "band_features": 36,
    "direct_xgboost": DIRECT_XGB_CONFIG,
    "ratio_residual": RESIDUAL_XGB_CONFIG,
    "c4_alpha_candidates": [0.75, 1.0],
    "postfit_shrinkage": ("baseline_mean + gamma*(prediction-baseline_mean)"),
    "nested_selection": {
        "standard_error": "1000 prompt-cluster bootstrap resamples",
        "standard_error_seed": 2718,
    },
    "simultaneous_inference": SIMULTANEOUS_INFERENCE,
    "positive_risk": {
        **DIRECT_XGB_CONFIG,
        "calibration": "prompt-cross-fitted Platt",
        "never_multiplied_into_magnitude": True,
        "objective": "binary:logistic",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact {path} must contain an object")
    return value


def validate_m3_protocol_lock(path: Path) -> dict[str, Any]:
    """Require the exact candidate and inference contract before replay."""
    lock = _load_json(path)
    if (
        lock.get("schema_version") != "herald.magnitude_v3_layer_band_lock.v1"
        or lock.get("status") != "locked_before_m3_layer_sensor_results"
    ):
        raise ValueError("M3 protocol lock schema/status is invalid")
    expected_sections = {
        "candidate_suite": CANDIDATE_SUITE,
        "nested_selection": NESTED_SELECTION,
        "simultaneous_inference": SIMULTANEOUS_INFERENCE,
        "development_gate": DEVELOPMENT_GATE,
        "diagnostics_only": DIAGNOSTICS_ONLY,
        "stop_rule": STOP_RULE,
    }
    for name, expected in expected_sections.items():
        if lock.get(name) != expected:
            raise ValueError(
                f"M3 {name.replace('_', '-')} section differs from lock"
            )
    return lock


def _as_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("value is not numeric") from error
    if not np.isfinite(number):
        raise ValueError("value is not finite")
    return number


def _as_int(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean is not an integer")
    number = _as_float(value)
    if not number.is_integer():
        raise ValueError("value is not an integer")
    return int(number)


def exact_key(row: Mapping[str, object]) -> tuple[str, str, float, int]:
    try:
        key = (
            str(row["prompt_id"]),
            str(row["compressor"]),
            _as_float(row["ratio"]),
            _as_int(row["s"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("row has an invalid exact key") from error
    if (
        not key[0]
        or key[1] not in KNOWN_COMPRESSORS
        or not 0 < key[2] < 1
        or key[3] < 0
    ):
        raise ValueError("row has an invalid exact key")
    return key


def exact_keys(
    rows: Sequence[Mapping[str, object]], label: str = "rows"
) -> set[tuple[str, str, float, int]]:
    values = [exact_key(row) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} exact key")
    return set(values)


def _require_model_task(
    rows: Sequence[Mapping[str, object]], label: str
) -> None:
    for row in rows:
        if str(row.get("model")) != MODEL_KEY or str(row.get("task")) != TASK:
            raise ValueError(f"{label} model/task is not locked")


def _finite_fields(
    row: Mapping[str, object], names: Sequence[str], label: str
) -> None:
    try:
        values = np.asarray(
            [_as_float(row[name]) for name in names], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label} has missing or invalid numeric fields"
        ) from error
    if not np.isfinite(values).all():
        raise ValueError(f"{label} has nonfinite fields")


def load_base_rows(
    parquet_path: Path,
    *,
    evidence_path: Path = v2.DEFAULT_EVIDENCE,
    lock_path: Path = v2.DEFAULT_LOCK,
    expected_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, provenance = v2.load_development_rows(
        parquet_path, evidence_path=evidence_path, lock_path=lock_path
    )
    if (
        expected_sha256 is not None
        and provenance["parquet_sha256"] != expected_sha256
    ):
        raise ValueError("base parquet hash differs from M3 lock")
    features = tuple(
        name
        for name in provenance.get("features", ())
        if name not in {"ratio", "s"}
    )
    if len(features) != 110 or set(features) != set(BASE_FEATURES):
        raise ValueError(
            "base parquet does not contain exactly 110 causal features "
            "(112 including ratio and s)"
        )
    _require_model_task(rows, "base")
    exact_keys(rows, "base")
    output = dict(provenance)
    output.update(
        {
            "base_feature_names": list(BASE_FEATURES),
            "base_parquet_sha256": provenance["parquet_sha256"],
        }
    )
    return rows, output


def _validate_post_result_freeze(
    path: Path, prefit_path: Path, report_path: Path, oof_path: Path
) -> None:
    result = _load_json(path)
    if (
        result.get("schema_version")
        != "herald.magnitude_v2_m2_result_freeze.v1"
        or result.get("status") != "frozen_before_m3_layer_sensor_results"
    ):
        raise ValueError("M2 result freeze schema/status is invalid")
    if result.get("m2_prefit_freeze_sha256") != sha256_file(prefit_path):
        raise ValueError("M2 result freeze prefit hash mismatch")
    report_binding, oof_binding = result.get("report"), result.get("oof")
    if not isinstance(report_binding, Mapping) or not isinstance(
        oof_binding, Mapping
    ):
        raise ValueError("M2 result freeze bindings are missing")
    if (
        report_binding.get("sha256") != sha256_file(report_path)
        or oof_binding.get("sha256") != sha256_file(oof_path)
        or report_binding.get("oof_sha256") != sha256_file(oof_path)
    ):
        raise ValueError("M2 result freeze report/OOF hash mismatch")
    decision = result.get("development_decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("any_compressor_pass") is not False
        or decision.get("passing_compressors") != []
    ):
        raise ValueError("M2 result freeze must record no passing compressor")


def load_frozen_m2(
    report_path: Path,
    oof_path: Path,
    freeze_path: Path,
    *,
    result_freeze_path: Path,
    base_rows: Sequence[Mapping[str, object]],
    base_provenance: Mapping[str, Any],
    evidence_path: Path,
    protocol_lock_path: Path,
    sensor_lock_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    freeze = _load_json(freeze_path)
    if (
        freeze.get("schema_version") != M2_FREEZE_SCHEMA_VERSION
        or freeze.get("status") != "frozen_before_m2_results"
    ):
        raise ValueError("M2 pre-fit freeze schema/status is invalid")
    _validate_post_result_freeze(
        result_freeze_path,
        freeze_path,
        report_path,
        oof_path,
    )
    expected_bindings = {
        "base_parquet_sha256": base_provenance.get("parquet_sha256"),
        "source_evidence_sha256": sha256_file(evidence_path),
        "protocol_lock_sha256": sha256_file(protocol_lock_path),
        "sensor_lock_sha256": sha256_file(sensor_lock_path),
    }
    for name, expected in expected_bindings.items():
        if not isinstance(expected, str) or freeze.get(name) != expected:
            raise ValueError(f"M2 freeze {name} differs from current inputs")

    report = _load_json(report_path)
    if report.get("schema_version") != "herald.magnitude_v2_sensors.v1":
        raise ValueError("M2 report schema is invalid")
    report_base = report.get("base_provenance")
    report_sensors = report.get("sensor_provenance")
    if not isinstance(report_base, Mapping) or not isinstance(
        report_sensors,
        Mapping,
    ):
        raise ValueError("M2 report provenance is incomplete")
    report_base_bindings = {
        "parquet_sha256": expected_bindings["base_parquet_sha256"],
        "source_evidence_sha256": expected_bindings["source_evidence_sha256"],
        "protocol_lock_sha256": expected_bindings["protocol_lock_sha256"],
    }
    report_sensor_bindings = {
        "evidence_sha256": expected_bindings["source_evidence_sha256"],
        "lock_sha256": expected_bindings["protocol_lock_sha256"],
        "sensor_lock_sha256": expected_bindings["sensor_lock_sha256"],
        "reference_manifest_sha256": freeze.get("reference_manifest_sha256"),
        "reference_inputs_sha256": freeze.get("reference_inputs_sha256"),
    }
    if any(
        report_base.get(name) != expected
        for name, expected in report_base_bindings.items()
    ) or any(
        report_sensors.get(name) != expected
        for name, expected in report_sensor_bindings.items()
    ):
        raise ValueError("M2 report provenance differs from frozen inputs")

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for frozen M2 OOF") from error
    rows = [
        dict(row)
        for row in pq.read_table(oof_path).to_pylist()  # type: ignore[no-untyped-call]
    ]
    if not rows:
        raise ValueError("M2 OOF is empty")
    _require_model_task(rows, "M2 OOF")
    exact_keys(rows, "M2 OOF")
    for row in rows:
        _finite_fields(
            row,
            (
                "dq",
                "m2_prediction",
                "locked_baseline_mean",
                "locked_baseline_median",
                "fold",
            ),
            "M2 OOF",
        )
    if exact_keys(base_rows, "base") != exact_keys(rows, "M2 OOF"):
        raise ValueError("M2 OOF keys do not exactly match base")
    base_by_key = {exact_key(row): row for row in base_rows}
    for row in rows:
        if not math.isclose(
            _as_float(row["dq"]),
            _as_float(base_by_key[exact_key(row)]["dq"]),
            abs_tol=1e-12,
            rel_tol=0,
        ):
            raise ValueError("M2 OOF dq disagrees with base")
    return rows, {
        "freeze_sha256": sha256_file(freeze_path),
        "result_freeze_sha256": sha256_file(result_freeze_path),
        "report_sha256": sha256_file(report_path),
        "oof_sha256": sha256_file(oof_path),
        "sensor_lock_sha256": expected_bindings["sensor_lock_sha256"],
        "protocol_lock_sha256": expected_bindings["protocol_lock_sha256"],
        "reference_manifest_sha256": freeze["reference_manifest_sha256"],
        "reference_inputs_sha256": freeze["reference_inputs_sha256"],
    }


def load_band_sidecar(
    sidecar_root: Path,
    manifest_path: Path,
    *,
    expected_keys: set[tuple[str, str, float, int]],
    evidence_sha256: str,
    lock_sha256: str,
    sensor_lock_sha256: str,
    band_lock_path: Path,
    reference_results_root: Path | None = None,
    evidence_path: Path | None = None,
    reference_manifest_sha256: str | None = None,
    reference_inputs_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        manifest_path.resolve()
        != sensor_manifest_path(sidecar_root, MODEL_KEY, TASK).resolve()
    ):
        raise ValueError("band manifest path does not match sidecar root")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("model") != MODEL_KEY
        or manifest.get("task") != TASK
        or manifest.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ValueError("band manifest model/task/protocol is not locked")
    band_lock = validate_m3_protocol_lock(band_lock_path)
    from herald.sensor_replay_m3 import STATE_SEMANTICS

    replay = band_lock.get("replay")
    if (
        not isinstance(replay, Mapping)
        or replay.get("state_semantics") != STATE_SEMANTICS
    ):
        raise ValueError("M3 layer-band state semantics differ from lock")
    names = tuple(map(str, manifest.get("feature_names", ())))
    if (
        names != tuple(LAYER_BAND_FEATURE_NAMES)
        or len(names) != 36
        or len(set(names)) != 36
    ):
        raise ValueError(
            "band feature order is not exactly the locked 36 values"
        )
    if (
        manifest.get("evidence_sha256") != evidence_sha256
        or manifest.get("lock_sha256") != lock_sha256
        or manifest.get("sensor_lock_sha256") != sensor_lock_sha256
    ):
        raise ValueError("band manifest provenance mismatch")
    sidecar = sensor_sidecar_path(sidecar_root, MODEL_KEY, TASK)
    if manifest.get("sidecar_sha256") != sha256_file(sidecar):
        raise ValueError("band sidecar hash mismatch")
    records = read_sensor_records(sidecar_root, MODEL_KEY, TASK)
    observed = validate_sensor_records(
        records,
        model=MODEL_KEY,
        task=TASK,
        evidence_sha256=evidence_sha256,
        lock_sha256=lock_sha256,
        sensor_lock_sha256=sensor_lock_sha256,
        protocol_version=PROTOCOL_VERSION,
        feature_names=names,
    )
    for record in records:
        compressor = str(record.get("compressor", ""))
        if record.get("state_semantics") != STATE_SEMANTICS.get(compressor):
            raise ValueError("band sidecar record state semantics mismatch")
    if observed != expected_keys:
        raise ValueError("band sidecar keys do not exactly match base keys")
    reference_hashes: tuple[str, str] | None = None
    if reference_results_root is not None:
        if evidence_path is None:
            raise ValueError(
                "evidence path is required for reference validation"
            )
        from herald.magnitude_v2_sensors import _load_reference_inputs

        reference_hashes = _load_reference_inputs(
            reference_results_root, v2._load_json(evidence_path), records
        )
        if (
            reference_manifest_sha256 is not None
            and reference_hashes[0] != reference_manifest_sha256
        ) or (
            reference_inputs_sha256 is not None
            and reference_hashes[1] != reference_inputs_sha256
        ):
            raise ValueError("band reference hashes differ from frozen M2")
    result = []
    for record in records:
        sensors = cast(Mapping[str, object], record["sensors"])
        result.append(
            {
                "model": MODEL_KEY,
                "task": TASK,
                **{column: record[column] for column in KEY_COLUMNS},
                **{
                    f"band__{name}": _as_float(sensors[name])
                    for name in names
                },
                "prefix_hash": record["prefix_hash"],
            }
        )
    return result, {
        "manifest_sha256": sha256_file(manifest_path),
        "sidecar_sha256": sha256_file(sidecar),
        "sensor_lock_sha256": sensor_lock_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "feature_names": list(names),
        "reference_manifest_sha256": reference_hashes[0]
        if reference_hashes
        else manifest.get("reference_manifest_sha256"),
        "reference_inputs_sha256": reference_hashes[1]
        if reference_hashes
        else manifest.get("reference_inputs_sha256"),
    }


def join_m3_features(
    base_rows: Sequence[Mapping[str, object]],
    m2_rows: Sequence[Mapping[str, object]],
    band_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    _require_model_task(base_rows, "base")
    _require_model_task(m2_rows, "M2")
    _require_model_task(band_rows, "bands")
    base_keys = exact_keys(base_rows, "base")
    m2_keys = exact_keys(m2_rows, "M2")
    band_keys = exact_keys(band_rows, "bands")
    if not (base_keys == m2_keys == band_keys):
        raise ValueError(
            "base, M2, and bands are not exact one-to-one coverage"
        )
    m2_by_key = {exact_key(row): row for row in m2_rows}
    band_by_key = {exact_key(row): row for row in band_rows}
    result: list[dict[str, Any]] = []
    for base in base_rows:
        key = exact_key(base)
        m2 = m2_by_key[key]
        band = band_by_key[key]
        if not math.isclose(
            _as_float(base["dq"]),
            _as_float(m2["dq"]),
            abs_tol=1e-12,
            rel_tol=0,
        ):
            raise ValueError("M2 dq disagrees with base")
        row = dict(base)
        for source, destination in (
            ("m2_prediction", "m2_prediction"),
            ("locked_baseline_mean", "m2_locked_baseline_mean"),
            ("locked_baseline_median", "m2_locked_baseline_median"),
            ("fold", "m2_fold"),
        ):
            if source not in m2:
                raise ValueError(f"M2 lacks required field {source}")
            row[destination] = _as_float(m2[source])
        for source in (
            "fold_train_prompt_ids_sha256",
            "fold_test_prompt_ids_sha256",
        ):
            value = m2.get(source)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"M2 lacks valid {source}")
            row[f"m2_{source}"] = value
        for name in GLOBAL_FEATURES:
            if name not in m2:
                raise ValueError(f"M2 lacks global feature {name}")
            row[name] = _as_float(m2[name])
        for name in BAND_FEATURES:
            if name not in band:
                raise ValueError(f"bands lacks feature {name}")
            row[name] = _as_float(band[name])
        result.append(row)
    return result


def fit_shrinkage(
    raw: ArrayLike,
    target: ArrayLike,
    baseline: ArrayLike,
    weights: ArrayLike | None = None,
) -> float:
    raw_a, target_a, base_a = (
        np.asarray(x, dtype=np.float64) for x in (raw, target, baseline)
    )
    if not len(raw_a) == len(target_a) == len(base_a):
        raise ValueError("shrinkage arrays differ in length")
    w = (
        np.ones(len(raw_a))
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    delta = raw_a - base_a
    denominator = float(np.sum(w * delta * delta))
    if denominator <= 0:
        return 0.0
    return float(
        np.clip(np.sum(w * delta * (target_a - base_a)) / denominator, 0, 1)
    )


def apply_shrinkage(
    raw: ArrayLike,
    baseline: ArrayLike,
    lam: float,
) -> NDArray[np.float64]:
    if not 0 <= float(lam) <= 1:
        raise ValueError("shrinkage lambda must be in [0,1]")
    return np.asarray(baseline, dtype=np.float64) + float(lam) * (
        np.asarray(raw, dtype=np.float64)
        - np.asarray(baseline, dtype=np.float64)
    )


def _features(candidate: str) -> tuple[str, ...]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate}")
    values = BASE_FEATURES + GLOBAL_FEATURES
    return values if candidate in {"C0", "C1"} else values + BAND_FEATURES


def _custom_train(
    rows: Sequence[Mapping[str, object]],
    features: tuple[str, ...],
    target: np.ndarray,
    *,
    seed: int,
    config: Mapping[str, Any],
    objective: str = "reg:squarederror",
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError("xgboost is required for M3") from error
    params = {
        "objective": objective,
        "max_depth": int(config["max_depth"]),
        "eta": float(config["eta"]),
        "subsample": float(config.get("subsample", 0.8)),
        "colsample_bytree": float(config.get("colsample_bytree", 0.7)),
        "min_child_weight": float(config["min_child_weight"]),
        "reg_lambda": float(config.get("reg_lambda", 0)),
        "reg_alpha": float(config.get("reg_alpha", 0)),
        "seed": int(seed),
        "nthread": -1,
    }
    return xgb.train(
        params,
        xgb.DMatrix(
            _feature_matrix([dict(row) for row in rows], features),
            label=target,
            weight=trajectory_weights(rows),
        ),
        num_boost_round=int(config["num_boost_round"]),
        verbose_eval=False,
    )


def _predict(
    models: Sequence[Any],
    rows: Sequence[Mapping[str, object]],
    features: tuple[str, ...],
) -> NDArray[np.float64]:
    if not models:
        return np.zeros(len(rows), dtype=np.float64)
    values = [dict(row) for row in rows]
    return np.asarray(
        np.mean(
            np.stack(
                [v2._xgb_raw(model, values, features) for model in models]
            ),
            axis=0,
        ),
        dtype=np.float64,
    )


def _fit_candidate(
    rows: list[dict[str, Any]],
    candidate: str,
    seeds: tuple[int, ...] = SEEDS,
    *,
    shared_models: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    features = _features(candidate)
    target = np.asarray([_as_float(row["dq"]) for row in rows])
    if candidate == "C0":
        return (
            v2._fit_hurdle(
                rows,
                features,
                seed=seeds[0],
                seeds=seeds,
                rounds=300,
                inner_folds=4,
            ),
        )
    config = M3_CONFIG["direct_xgboost"]
    if candidate == "C3":
        return tuple(
            (
                _custom_train(
                    rows,
                    features,
                    np.maximum(target, 0),
                    seed=seed,
                    config=config,
                ),
                _custom_train(
                    rows,
                    features,
                    np.maximum(-target, 0),
                    seed=seed,
                    config=config,
                ),
            )
            for seed in seeds
        )
    if shared_models is not None and candidate != "C4":
        raise ValueError("shared models are valid only for C4")
    shared = shared_models or tuple(
        _custom_train(rows, features, target, seed=seed, config=config)
        for seed in seeds
    )
    if candidate != "C4":
        return shared
    shared_prediction = _predict(shared, rows, features)
    residual_features = BASE_FEATURES + GLOBAL_FEATURES
    residuals: dict[float, tuple[Any, ...]] = {}
    residual_target = target - shared_prediction
    for ratio in sorted({_as_float(row["ratio"]) for row in rows}):
        subset = [row for row in rows if _as_float(row["ratio"]) == ratio]
        indexes = np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if _as_float(row["ratio"]) == ratio
            ],
            dtype=np.int64,
        )
        residuals[ratio] = tuple(
            _custom_train(
                subset,
                residual_features,
                residual_target[indexes],
                seed=seed,
                config=M3_CONFIG["ratio_residual"],
            )
            for seed in seeds
        )
    return (shared, residuals)


def _predict_candidate(
    fit: tuple[Any, ...],
    rows: list[dict[str, Any]],
    candidate: str,
    *,
    alpha: float = 1.0,
) -> NDArray[np.float64]:
    features = _features(candidate)
    if candidate == "C0":
        return v2._hurdle_predict(fit[0], rows)[0]
    if candidate == "C3":
        return _predict([item[0] for item in fit], rows, features) - _predict(
            [item[1] for item in fit], rows, features
        )
    if candidate == "C4":
        if float(alpha) not in M3_CONFIG["c4_alpha_candidates"]:
            raise ValueError("C4 alpha is outside the locked candidates")
        shared, residuals = fit
        output = _predict(shared, rows, features)
        residual_features = BASE_FEATURES + GLOBAL_FEATURES
        for ratio, models in residuals.items():
            indexes = [
                i
                for i, row in enumerate(rows)
                if _as_float(row["ratio"]) == ratio
            ]
            if indexes:
                subset = [rows[i] for i in indexes]
                output[indexes] += (1.0 - float(alpha)) * _predict(
                    models,
                    subset,
                    residual_features,
                )
        return output
    return _predict(fit, rows, features)


def fit_positive_risk(
    rows: Sequence[Mapping[str, object]],
    candidate: str,
    *,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Fit a candidate-specific shallow risk ensemble and cross-fit Platt."""
    values = [dict(row) for row in rows]
    features = _features(candidate)
    target = np.asarray(
        [_as_float(row["dq"]) > 0 for row in values], dtype=np.float64
    )
    risk_config = M3_CONFIG["positive_risk"]
    prompts = sorted({str(row["prompt_id"]) for row in values})
    folds = v2.prompt_inner_folds(prompts, 4)
    margins = np.full(len(values), np.nan)
    for validation_prompts in folds:
        validation = set(validation_prompts)
        train_mask = np.asarray(
            [str(row["prompt_id"]) not in validation for row in values],
            dtype=bool,
        )
        validation_rows = [
            row for row, keep in zip(values, ~train_mask, strict=True) if keep
        ]
        train_rows = [
            row for row, keep in zip(values, train_mask, strict=True) if keep
        ]
        if len(np.unique(target[train_mask])) < 2:
            continue
        models = tuple(
            _custom_train(
                train_rows,
                features,
                target[train_mask],
                seed=seed,
                config=risk_config,
                objective="binary:logistic",
            )
            for seed in seeds
        )
        margins[~train_mask] = np.mean(
            np.stack(
                [
                    v2._xgb_raw(
                        model, validation_rows, features, output_margin=True
                    )
                    for model in models
                ]
            ),
            axis=0,
        )
    observed = np.isfinite(margins)
    prevalence = (
        float(
            np.average(
                target[observed], weights=trajectory_weights(values)[observed]
            )
        )
        if np.any(observed)
        else float(np.average(target, weights=trajectory_weights(values)))
    )
    calibration = v2._fit_platt(
        margins[observed],
        target[observed],
        prevalence,
        trajectory_weights(values)[observed],
    )
    models = tuple(
        _custom_train(
            values,
            features,
            target,
            seed=seed,
            config=risk_config,
            objective="binary:logistic",
        )
        for seed in seeds
    )
    return {
        "candidate": candidate,
        "models": models,
        "features": features,
        "calibration": calibration,
    }


def predict_positive_risk(
    fit: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
) -> NDArray[np.float64]:
    values = [dict(row) for row in rows]
    models = tuple(fit["models"])
    features = tuple(fit["features"])
    margins = (
        np.mean(
            np.stack(
                [
                    v2._xgb_raw(
                        model,
                        values,
                        features,
                        output_margin=True,
                    )
                    for model in models
                ]
            ),
            axis=0,
        )
        if models
        else np.zeros(len(values), dtype=np.float64)
    )
    return np.asarray(
        fit["calibration"].predict(margins),
        dtype=np.float64,
    )


def fit_development(
    rows: Sequence[Mapping[str, object]],
    *,
    lock: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    m2_provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the locked five-by-four prompt-nested candidate selection."""
    if not rows:
        raise ValueError("M3 rows are empty")
    if (
        lock.get("schema_version") != "herald.magnitude_v3_layer_band_lock.v1"
        or lock.get("status") != "locked_before_m3_layer_sensor_results"
    ):
        raise ValueError("M3 protocol lock schema/status is invalid")
    for name, expected in (
        ("candidate_suite", CANDIDATE_SUITE),
        ("nested_selection", NESTED_SELECTION),
        ("simultaneous_inference", SIMULTANEOUS_INFERENCE),
        ("development_gate", DEVELOPMENT_GATE),
        ("diagnostics_only", DIAGNOSTICS_ONLY),
        ("stop_rule", STOP_RULE),
    ):
        if lock.get(name) != expected:
            raise ValueError(f"M3 {name.replace('_', '-')} is not locked")
    values = [dict(row) for row in rows]
    _require_model_task(values, "M3")
    exact_keys(values, "M3")
    compressors = tuple(sorted({str(row["compressor"]) for row in values}))
    if compressors != tuple(sorted(KNOWN_COMPRESSORS)):
        raise ValueError("M3 rows do not contain every locked compressor")
    if (
        len(BASE_INPUT_FEATURES) != 112
        or len(BASE_FEATURES) != 110
        or len(GLOBAL_FEATURES) != 36
        or len(BAND_FEATURES) != 36
        or len(_features("C1")) != 146
        or len(_features("C2")) != 182
    ):
        raise ValueError("M3 feature schema cardinality is not locked")
    folds = v2.make_folds(
        sorted({str(row["prompt_id"]) for row in values}),
        dict(lock),
    )
    seeds = tuple(int(seed) for seed in M3_CONFIG["model_seeds"])
    output: list[dict[str, Any]] = []
    selected_folds: dict[str, list[str]] = defaultdict(list)

    for compressor in compressors:
        comp_rows = [
            row for row in values if str(row["compressor"]) == compressor
        ]
        for fold in folds:
            test_prompts = set(fold.prompt_ids)
            train = [
                row
                for row in comp_rows
                if str(row["prompt_id"]) not in test_prompts
            ]
            test = [
                row
                for row in comp_rows
                if str(row["prompt_id"]) in test_prompts
            ]
            if not train or not test:
                raise ValueError(
                    "M3 outer fold has empty train/test partition"
                )
            train_hash = v2.hash_prompt_ids(
                sorted({str(row["prompt_id"]) for row in train})
            )
            for row in test:
                if (
                    _as_int(row["m2_fold"]) != fold.index
                    or row.get("m2_fold_train_prompt_ids_sha256")
                    != train_hash
                    or row.get("m2_fold_test_prompt_ids_sha256") != fold.hash
                ):
                    raise ValueError("M2 OOF fold provenance differs from M3")

            inner_folds = v2.prompt_inner_folds(
                sorted({str(row["prompt_id"]) for row in train}),
                int(M3_CONFIG["inner_folds"]),
            )
            inner_predictions: dict[str, list[float]] = {
                candidate: [] for candidate in CANDIDATES if candidate != "C4"
            }
            c4_predictions: dict[float, list[float]] = {
                float(alpha): [] for alpha in M3_CONFIG["c4_alpha_candidates"]
            }
            inner_rows: list[dict[str, Any]] = []
            inner_bases: list[float] = []
            for validation_prompts in inner_folds:
                validation = set(validation_prompts)
                inner_train = [
                    row
                    for row in train
                    if str(row["prompt_id"]) not in validation
                ]
                inner_test = [
                    row
                    for row in train
                    if str(row["prompt_id"]) in validation
                ]
                if not inner_train or not inner_test:
                    raise ValueError(
                        "M3 inner fold has empty train/test partition"
                    )
                inner_target = np.asarray(
                    [_as_float(row["dq"]) for row in inner_train],
                    dtype=np.float64,
                )
                baseline_fit = _ratio_position_mean(
                    inner_train,
                    inner_target,
                )
                fitted: dict[str, tuple[Any, ...]] = {}
                for candidate in CANDIDATES:
                    fitted[candidate] = _fit_candidate(
                        inner_train,
                        candidate,
                        seeds,
                        shared_models=(
                            fitted["C2"] if candidate == "C4" else None
                        ),
                    )
                for candidate in CANDIDATES:
                    if candidate == "C4":
                        for alpha in c4_predictions:
                            raw = _predict_candidate(
                                fitted[candidate],
                                inner_test,
                                candidate,
                                alpha=alpha,
                            )
                            c4_predictions[alpha].extend(raw.tolist())
                    else:
                        raw = _predict_candidate(
                            fitted[candidate],
                            inner_test,
                            candidate,
                        )
                        inner_predictions[candidate].extend(raw.tolist())
                inner_bases.extend(
                    _baseline(inner_test, baseline_fit).tolist()
                )
                inner_rows.extend(inner_test)

            selection_target = np.asarray(
                [_as_float(row["dq"]) for row in inner_rows],
                dtype=np.float64,
            )
            selection_base = np.asarray(inner_bases, dtype=np.float64)
            selection_weights = trajectory_weights(inner_rows)
            c4_alpha, c4_alpha_scores = _choose_c4_alpha(
                c4_predictions,
                inner_rows,
                selection_target,
                selection_base,
                selection_weights,
            )
            inner_predictions["C4"] = c4_predictions[c4_alpha]

            scores: dict[str, float] = {}
            lambdas: dict[str, float] = {}
            clusters: dict[str, np.ndarray] = {}
            for candidate in CANDIDATES:
                raw = np.asarray(
                    inner_predictions[candidate],
                    dtype=np.float64,
                )
                gamma = (
                    1.0
                    if candidate == "C0"
                    else fit_shrinkage(
                        raw,
                        selection_target,
                        selection_base,
                        selection_weights,
                    )
                )
                prediction = apply_shrinkage(
                    raw,
                    selection_base,
                    gamma,
                )
                cluster_scores = _prompt_cluster_scores(
                    inner_rows,
                    selection_target,
                    prediction,
                )
                scores[candidate] = float(np.mean(cluster_scores))
                lambdas[candidate] = gamma
                clusters[candidate] = cluster_scores

            best = min(CANDIDATES, key=lambda name: scores[name])
            best_se = _cluster_bootstrap_standard_error(
                clusters[best],
                resamples=1000,
                seed=2718,
            )
            standard_errors = dict.fromkeys(CANDIDATES, best_se)
            chosen = select_simplest_one_se(scores, standard_errors)
            selected_folds[compressor].append(chosen)

            outer_target = np.asarray(
                [_as_float(row["dq"]) for row in train],
                dtype=np.float64,
            )
            outer_baseline_fit = _ratio_position_mean(
                train,
                outer_target,
            )
            candidate_fit = _fit_candidate(train, chosen, seeds)
            raw_test = _predict_candidate(
                candidate_fit,
                test,
                chosen,
                alpha=c4_alpha,
            )
            if chosen == "C0":
                frozen_m2 = np.asarray(
                    [_as_float(row["m2_prediction"]) for row in test],
                    dtype=np.float64,
                )
                if not np.allclose(
                    raw_test,
                    frozen_m2,
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError(
                        "C0 does not reproduce frozen M2 predictions"
                    )
            baseline_test = _baseline(test, outer_baseline_fit)
            prediction = apply_shrinkage(
                raw_test,
                baseline_test,
                lambdas[chosen],
            )
            for row, baseline in zip(test, baseline_test, strict=True):
                if not math.isclose(
                    float(baseline),
                    _as_float(row["m2_locked_baseline_mean"]),
                    abs_tol=1e-12,
                    rel_tol=0,
                ):
                    raise ValueError(
                        "recomputed locked mean differs from frozen M2"
                    )

            risk_fit = fit_positive_risk(train, chosen, seeds=seeds)
            risk_test = predict_positive_risk(risk_fit, test)
            event_target = np.asarray(
                [_as_float(row["dq"]) > 0 for row in train],
                dtype=np.float64,
            )
            event_baseline_fit = _ratio_position_mean(
                train,
                event_target,
            )
            risk_baseline = _baseline(test, event_baseline_fit)
            risk_calibration = risk_fit["calibration"]
            for row, pred, baseline, risk, risk_base in zip(
                test,
                prediction,
                baseline_test,
                risk_test,
                risk_baseline,
                strict=True,
            ):
                output.append(
                    {
                        "model": MODEL_KEY,
                        "task": TASK,
                        **{column: row[column] for column in KEY_COLUMNS},
                        "dq": _as_float(row["dq"]),
                        "fold": fold.index,
                        "selected_candidate": chosen,
                        "m3_prediction": float(pred),
                        "locked_baseline_mean": float(baseline),
                        "locked_baseline_median": _as_float(
                            row["m2_locked_baseline_median"]
                        ),
                        "c4_alpha": c4_alpha,
                        "c4_alpha_scores": dict(c4_alpha_scores),
                        "m2_prediction": _as_float(row["m2_prediction"]),
                        "positive_risk": float(risk),
                        "positive_risk_baseline": float(risk_base),
                        "risk_train_calibration_slope": float(
                            risk_calibration.slope
                        ),
                        "risk_train_calibration_intercept": float(
                            risk_calibration.intercept
                        ),
                        "candidate_scores": dict(scores),
                        "candidate_standard_errors": dict(standard_errors),
                        "candidate_lambdas": dict(lambdas),
                        "fold_train_prompt_ids_sha256": train_hash,
                        "fold_test_prompt_ids_sha256": fold.hash,
                    }
                )

    if exact_keys(output, "M3 OOF") != exact_keys(values, "M3"):
        raise ValueError("M3 OOF does not exactly cover input cells")

    rows_by_compressor = {
        compressor: [
            row for row in output if str(row["compressor"]) == compressor
        ]
        for compressor in compressors
    }
    simultaneous = _simultaneous_development_inference(
        rows_by_compressor,
        resamples=int(SIMULTANEOUS_INFERENCE["resamples"]),
        seed=int(SIMULTANEOUS_INFERENCE["seed"]),
    )
    primary_lower = {
        compressor: float(
            simultaneous["families"]["macro_mse_skill"]["intervals"][
                compressor
            ]["lower"]
        )
        for compressor in compressors
    }
    paired_upper = {
        compressor: float(
            simultaneous["families"]["paired_m3_minus_m2"]["intervals"][
                compressor
            ]["upper"]
        )
        for compressor in compressors
    }
    brier_lower = {
        compressor: float(
            simultaneous["families"]["positive_risk_brier_skill"][
                "intervals"
            ][compressor]["lower"]
        )
        for compressor in compressors
    }
    ratio_skills = {
        compressor: {
            str(ratio): float(
                simultaneous["families"]["per_ratio_mse_skill"]["intervals"][
                    f"{compressor}|{ratio}"
                ]["estimate"]
            )
            for ratio in RATIOS
        }
        for compressor in compressors
    }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input": dict(provenance or {}),
        "m2_provenance": dict(m2_provenance or {}),
        "protocol": {
            "lock": dict(lock),
            "config": M3_CONFIG,
            "features": {
                "base": list(BASE_FEATURES),
                "global": list(GLOBAL_FEATURES),
                "bands": list(BAND_FEATURES),
            },
        },
        "candidate_selection_frequencies": {
            compressor: {
                candidate: selected_folds[compressor].count(candidate)
                for candidate in CANDIDATES
            }
            for compressor in compressors
        },
        "diagnostics_only": list(DIAGNOSTICS_ONLY),
        "simultaneous_inference": simultaneous,
        "compressors": {},
    }
    calibration_slopes: dict[str, float] = {}
    for compressor in compressors:
        comp = rows_by_compressor[compressor]
        target = np.asarray(
            [_as_float(row["dq"]) for row in comp],
            dtype=np.float64,
        )
        prediction = np.asarray(
            [_as_float(row["m3_prediction"]) for row in comp],
            dtype=np.float64,
        )
        baseline_mean = np.asarray(
            [_as_float(row["locked_baseline_mean"]) for row in comp],
            dtype=np.float64,
        )
        baseline_median = np.asarray(
            [_as_float(row["locked_baseline_median"]) for row in comp],
            dtype=np.float64,
        )
        m2_prediction = np.asarray(
            [_as_float(row["m2_prediction"]) for row in comp],
            dtype=np.float64,
        )
        risk = np.asarray(
            [_as_float(row["positive_risk"]) for row in comp],
            dtype=np.float64,
        )
        risk_diagnostics = _risk_diagnostics(comp, risk)
        calibration_slopes[compressor] = float(
            risk_diagnostics["calibration_slope"]
        )
        selected_metrics = v2._model_metrics(
            comp,
            target,
            prediction,
        )
        selected_metrics["per_ratio"] = _ratio_metrics(
            target,
            prediction,
            trajectory_weights(comp),
            comp,
        )["per_ratio"]
        selected_metrics["signed_error"] = float(
            np.average(
                prediction - target,
                weights=trajectory_weights(comp),
            )
        )
        report["compressors"][compressor] = {
            "selected_m3": selected_metrics,
            "frozen_m2": v2._model_metrics(
                comp,
                target,
                m2_prediction,
            ),
            "locked_mean": v2._model_metrics(
                comp,
                target,
                baseline_mean,
            ),
            "locked_median": v2._model_metrics(
                comp,
                target,
                baseline_median,
            ),
            "positive_risk": risk_diagnostics,
            "selected_candidates": selected_folds[compressor],
        }

    gate = evaluate_gate(
        primary_lower=primary_lower,
        per_ratio_skills=ratio_skills,
        paired_upper=paired_upper,
        brier_lower=brier_lower,
        calibration_slope=calibration_slopes,
        selected_by_fold=selected_folds,
    )
    gate["specification"] = DEVELOPMENT_GATE
    report["development_gate"] = gate
    return report, output


@dataclass(frozen=True, slots=True)
class _RatioPositionMean:
    by_position: Mapping[tuple[float, int], float]
    by_ratio: Mapping[float, float]
    global_value: float


@dataclass(frozen=True, slots=True)
class _PromptRatioLoss:
    prompts: tuple[str, ...]
    ratios: tuple[float, ...]
    values: NDArray[np.float64]


def _ratio_position_mean(
    rows: Sequence[Mapping[str, object]],
    target: ArrayLike,
) -> _RatioPositionMean:
    target = np.asarray(target, dtype=np.float64)
    if not rows or len(rows) != len(target):
        raise ValueError("baseline rows and target differ in length")
    weights = trajectory_weights(rows)
    position_groups: dict[tuple[float, int], list[int]] = defaultdict(list)
    ratio_groups: dict[float, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        ratio = _as_float(row["ratio"])
        position_groups[(ratio, _as_int(row["s"]) // POSITION_BUCKET)].append(
            index
        )
        ratio_groups[ratio].append(index)
    by_position = {
        key: float(
            np.average(
                target[np.asarray(indices, dtype=np.int64)],
                weights=weights[np.asarray(indices, dtype=np.int64)],
            )
        )
        for key, indices in position_groups.items()
    }
    by_ratio = {
        ratio: float(
            np.average(
                target[np.asarray(indices, dtype=np.int64)],
                weights=weights[np.asarray(indices, dtype=np.int64)],
            )
        )
        for ratio, indices in ratio_groups.items()
    }
    return _RatioPositionMean(
        by_position=by_position,
        by_ratio=by_ratio,
        global_value=float(np.average(target, weights=weights)),
    )


def _baseline(
    rows: Sequence[Mapping[str, object]],
    fit: _RatioPositionMean,
) -> NDArray[np.float64]:
    return np.asarray(
        [
            fit.by_position.get(
                (
                    _as_float(row["ratio"]),
                    _as_int(row["s"]) // POSITION_BUCKET,
                ),
                fit.by_ratio.get(
                    _as_float(row["ratio"]),
                    fit.global_value,
                ),
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def _prompt_ratio_squared_loss(
    rows: Sequence[Mapping[str, object]],
    prediction: ArrayLike,
    target: ArrayLike | None = None,
) -> _PromptRatioLoss:
    values = list(rows)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = (
        np.asarray([_as_float(row["dq"]) for row in values], dtype=np.float64)
        if target is None
        else np.asarray(target, dtype=np.float64)
    )
    if (
        not values
        or len(values) != len(prediction_array)
        or len(values) != len(target_array)
        or not np.isfinite(prediction_array).all()
        or not np.isfinite(target_array).all()
    ):
        raise ValueError(
            "loss-table inputs are empty, nonfinite, or misaligned"
        )
    prompts = tuple(sorted({str(row["prompt_id"]) for row in values}))
    ratios = tuple(sorted({_as_float(row["ratio"]) for row in values}))
    if ratios != tuple(RATIOS):
        raise ValueError("loss table does not contain every locked ratio")
    squared_error = (target_array - prediction_array) ** 2
    prompt_indexes = {prompt: index for index, prompt in enumerate(prompts)}
    ratio_indexes = {ratio: index for index, ratio in enumerate(ratios)}
    row_prompt_indexes = np.fromiter(
        (prompt_indexes[str(row["prompt_id"])] for row in values),
        dtype=np.int64,
        count=len(values),
    )
    row_ratio_indexes = np.fromiter(
        (ratio_indexes[_as_float(row["ratio"])] for row in values),
        dtype=np.int64,
        count=len(values),
    )
    table = np.zeros((len(prompts), len(ratios)), dtype=np.float64)
    counts = np.zeros_like(table, dtype=np.int64)
    np.add.at(
        table,
        (row_prompt_indexes, row_ratio_indexes),
        squared_error,
    )
    np.add.at(
        counts,
        (row_prompt_indexes, row_ratio_indexes),
        1,
    )
    if np.any(counts == 0):
        raise ValueError("prompt cluster lacks one or more locked ratios")
    table /= counts
    return _PromptRatioLoss(prompts, ratios, table)


def _prompt_cluster_scores(
    rows: Sequence[Mapping[str, object]],
    target: ArrayLike,
    prediction: ArrayLike,
) -> NDArray[np.float64]:
    table = _prompt_ratio_squared_loss(rows, prediction, target)
    return np.asarray(np.mean(table.values, axis=1), dtype=np.float64)


def _choose_c4_alpha(
    predictions: Mapping[float, Sequence[float]],
    rows: Sequence[Mapping[str, object]],
    target: ArrayLike,
    baseline: ArrayLike,
    weights: ArrayLike,
) -> tuple[float, dict[str, float]]:
    expected = {float(alpha) for alpha in M3_CONFIG["c4_alpha_candidates"]}
    if set(predictions) != expected:
        raise ValueError("C4 inner-OOF alpha predictions are incomplete")
    choices: list[tuple[float, int, float]] = []
    scores: dict[str, float] = {}
    for alpha in sorted(predictions):
        raw = np.asarray(predictions[alpha], dtype=np.float64)
        gamma = fit_shrinkage(raw, target, baseline, weights)
        prediction = apply_shrinkage(raw, baseline, gamma)
        score = float(
            np.mean(_prompt_cluster_scores(rows, target, prediction))
        )
        scores[str(alpha)] = score
        simplicity = 0 if alpha == 1.0 else 1
        choices.append((score, simplicity, alpha))
    return min(choices)[2], scores


def _cluster_bootstrap_standard_error(
    cluster_scores: ArrayLike,
    *,
    resamples: int,
    seed: int,
) -> float:
    scores = np.asarray(cluster_scores, dtype=np.float64)
    if not len(scores) or resamples < 2:
        raise ValueError("cluster bootstrap needs data and two resamples")
    if len(scores) == 1:
        return 0.0
    draws = np.random.default_rng(seed).integers(
        0,
        len(scores),
        size=(resamples, len(scores)),
    )
    return float(np.std(np.mean(scores[draws], axis=1), ddof=1))


def select_simplest_one_se(
    scores: Mapping[str, float],
    standard_errors: Mapping[str, float],
    *,
    order: Sequence[str] = CANDIDATES,
) -> str:
    if set(order) != set(scores) or set(order) != set(standard_errors):
        raise ValueError("selection score is incomplete")
    best = min(order, key=lambda candidate: float(scores[candidate]))
    threshold = float(scores[best]) + float(standard_errors[best])
    return next(
        candidate
        for candidate in order
        if float(scores[candidate]) <= threshold
    )


def _max_t_intervals(
    estimates: Mapping[str, float],
    bootstrap_values: Mapping[str, NDArray[np.float64]],
    *,
    direction: str,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if (
        direction not in {"lower", "upper"}
        or resamples < 2
        or not 0 < confidence < 1
        or not estimates
        or set(estimates) != set(bootstrap_values)
    ):
        raise ValueError("invalid max-T bootstrap inputs")
    claims = tuple(sorted(estimates))
    matrix = np.column_stack(
        [
            np.asarray(bootstrap_values[claim], dtype=np.float64)
            for claim in claims
        ]
    )
    if matrix.shape != (resamples, len(claims)):
        raise ValueError("max-T bootstrap draw shape differs from contract")
    point = np.asarray([float(estimates[claim]) for claim in claims])
    standard_errors = np.std(matrix, axis=0, ddof=1)
    centered = matrix - point
    standardized = np.divide(
        centered if direction == "lower" else -centered,
        standard_errors,
        out=np.zeros_like(centered),
        where=standard_errors > 0,
    )
    critical = max(
        0.0,
        float(
            np.quantile(
                np.max(standardized, axis=1),
                confidence,
            )
        ),
    )
    intervals: dict[str, dict[str, float]] = {}
    for column, claim in enumerate(claims):
        interval = {
            "estimate": float(point[column]),
            "standard_error": float(standard_errors[column]),
        }
        bound = critical * standard_errors[column]
        if direction == "lower":
            interval["lower"] = float(point[column] - bound)
        else:
            interval["upper"] = float(point[column] + bound)
        intervals[claim] = interval
    return {
        "resamples": resamples,
        "seed": seed,
        "confidence": confidence,
        "direction": direction,
        "sidedness": "one-sided",
        "multiplicity": f"simultaneous max-T over {len(claims)} claims",
        "max_t_critical": critical,
        "intervals": intervals,
    }


def _loss_family_bootstrap(
    rows_by_compressor: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
    model_by_compressor: Mapping[str, Sequence[float]],
    reference_by_compressor: Mapping[str, Sequence[float]],
    *,
    metric: str,
    per_ratio: bool,
    target_by_compressor: Mapping[str, ArrayLike] | None = None,
    resamples: int,
    seed: int,
    direction: str,
) -> dict[str, Any]:
    compressors = tuple(sorted(rows_by_compressor))
    if (
        not compressors
        or set(compressors) != set(model_by_compressor)
        or set(compressors) != set(reference_by_compressor)
        or metric not in {"skill", "difference"}
    ):
        raise ValueError("loss-family inputs are incomplete")
    if target_by_compressor is not None and set(compressors) != set(
        target_by_compressor
    ):
        raise ValueError("loss-family targets are incomplete")
    rng = np.random.default_rng(seed)
    estimates: dict[str, float] = {}
    bootstrap_values: dict[str, NDArray[np.float64]] = {}
    prompt_draws: NDArray[np.int64] | None = None
    expected_prompts: tuple[str, ...] | None = None

    def compare(
        model_loss: NDArray[np.float64],
        reference_loss: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if metric == "difference":
            return model_loss - reference_loss
        return np.divide(
            reference_loss - model_loss,
            reference_loss,
            out=np.zeros_like(model_loss),
            where=reference_loss != 0,
        )

    for compressor in compressors:
        rows = rows_by_compressor[compressor]
        target = (
            None
            if target_by_compressor is None
            else target_by_compressor[compressor]
        )
        model_table = _prompt_ratio_squared_loss(
            rows,
            model_by_compressor[compressor],
            target,
        )
        reference_table = _prompt_ratio_squared_loss(
            rows,
            reference_by_compressor[compressor],
            target,
        )
        if (
            model_table.prompts != reference_table.prompts
            or model_table.ratios != reference_table.ratios
        ):
            raise ValueError("loss-family model/reference tables differ")
        if expected_prompts is None:
            expected_prompts = model_table.prompts
            prompt_draws = rng.integers(
                0,
                len(expected_prompts),
                size=(resamples, len(expected_prompts)),
            )
        elif model_table.prompts != expected_prompts:
            raise ValueError("compressor prompt clusters differ")
        if prompt_draws is None:
            raise AssertionError("prompt draws were not initialized")
        if per_ratio:
            for ratio_index, ratio in enumerate(model_table.ratios):
                claim = f"{compressor}|{ratio}"
                point_model = np.asarray(
                    np.mean(model_table.values[:, ratio_index])
                )
                point_reference = np.asarray(
                    np.mean(reference_table.values[:, ratio_index])
                )
                estimates[claim] = float(
                    compare(point_model, point_reference)
                )
                drawn_model = np.mean(
                    model_table.values[
                        prompt_draws,
                        ratio_index,
                    ],
                    axis=1,
                )
                drawn_reference = np.mean(
                    reference_table.values[
                        prompt_draws,
                        ratio_index,
                    ],
                    axis=1,
                )
                bootstrap_values[claim] = compare(
                    drawn_model,
                    drawn_reference,
                )
        else:
            estimates[compressor] = float(
                compare(
                    np.asarray(np.mean(model_table.values)),
                    np.asarray(np.mean(reference_table.values)),
                )
            )
            drawn_model = np.mean(
                model_table.values[prompt_draws],
                axis=(1, 2),
            )
            drawn_reference = np.mean(
                reference_table.values[prompt_draws],
                axis=(1, 2),
            )
            bootstrap_values[compressor] = compare(
                drawn_model,
                drawn_reference,
            )
    return _max_t_intervals(
        estimates,
        bootstrap_values,
        direction=direction,
        resamples=resamples,
        seed=seed,
    )


def max_t_bootstrap(
    rows_by_compressor: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
    model_by_compressor: Mapping[str, Sequence[float]],
    baseline_by_compressor: Mapping[str, Sequence[float]],
    *,
    resamples: int = 10000,
    seed: int = 314159,
    direction: str = "lower",
) -> dict[str, Any]:
    """Simultaneous prompt-cluster MSE-skill intervals."""
    return _loss_family_bootstrap(
        rows_by_compressor,
        model_by_compressor,
        baseline_by_compressor,
        metric="skill",
        per_ratio=False,
        resamples=resamples,
        seed=seed,
        direction=direction,
    )


def _simultaneous_development_inference(
    rows_by_compressor: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    m3 = {
        compressor: [_as_float(row["m3_prediction"]) for row in rows]
        for compressor, rows in rows_by_compressor.items()
    }
    m2 = {
        compressor: [_as_float(row["m2_prediction"]) for row in rows]
        for compressor, rows in rows_by_compressor.items()
    }
    mean = {
        compressor: [_as_float(row["locked_baseline_mean"]) for row in rows]
        for compressor, rows in rows_by_compressor.items()
    }
    risk = {
        compressor: [_as_float(row["positive_risk"]) for row in rows]
        for compressor, rows in rows_by_compressor.items()
    }
    risk_baseline = {
        compressor: [_as_float(row["positive_risk_baseline"]) for row in rows]
        for compressor, rows in rows_by_compressor.items()
    }
    event = {
        compressor: [_as_float(row["dq"]) > 0 for row in rows]
        for compressor, rows in rows_by_compressor.items()
    }
    return {
        "method": SIMULTANEOUS_INFERENCE["method"],
        "resamples": resamples,
        "seed": seed,
        "confidence": SIMULTANEOUS_INFERENCE["confidence"],
        "sidedness": SIMULTANEOUS_INFERENCE["sidedness"],
        "families": {
            "macro_mse_skill": _loss_family_bootstrap(
                rows_by_compressor,
                m3,
                mean,
                metric="skill",
                per_ratio=False,
                resamples=resamples,
                seed=seed,
                direction="lower",
            ),
            "paired_m3_minus_m2": _loss_family_bootstrap(
                rows_by_compressor,
                m3,
                m2,
                metric="difference",
                per_ratio=False,
                resamples=resamples,
                seed=seed,
                direction="upper",
            ),
            "positive_risk_brier_skill": _loss_family_bootstrap(
                rows_by_compressor,
                risk,
                risk_baseline,
                metric="skill",
                per_ratio=False,
                target_by_compressor=event,
                resamples=resamples,
                seed=seed,
                direction="lower",
            ),
            "per_ratio_mse_skill": _loss_family_bootstrap(
                rows_by_compressor,
                m3,
                mean,
                metric="skill",
                per_ratio=True,
                resamples=resamples,
                seed=seed,
                direction="lower",
            ),
        },
    }


def _risk_diagnostics(
    rows: Sequence[Mapping[str, object]],
    risk: ArrayLike,
) -> dict[str, Any]:
    prediction = np.asarray(risk, dtype=np.float64)
    event = np.asarray(
        [_as_float(row["dq"]) > 0 for row in rows],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [_as_float(row["positive_risk_baseline"]) for row in rows],
        dtype=np.float64,
    )
    weights = trajectory_weights(rows)
    if (
        len(prediction) != len(rows)
        or not np.isfinite(prediction).all()
        or np.any((prediction < 0) | (prediction > 1))
    ):
        raise ValueError("positive-risk predictions are invalid")
    clipped = np.clip(prediction, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    calibration = v2._fit_platt(
        logits,
        event,
        float(np.average(event, weights=weights)),
        weights,
    )
    brier = float(np.average((prediction - event) ** 2, weights=weights))
    baseline_brier = float(
        np.average((baseline - event) ** 2, weights=weights)
    )
    bin_indices = np.minimum((prediction * 10).astype(np.int64), 9)
    reliability_bins: list[dict[str, Any]] = []
    for index in range(10):
        selected = bin_indices == index
        reliability_bins.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "n": int(np.sum(selected)),
                "weight": float(np.sum(weights[selected])),
                "mean_prediction": (
                    float(
                        np.average(
                            prediction[selected],
                            weights=weights[selected],
                        )
                    )
                    if np.any(selected)
                    else None
                ),
                "observed_frequency": (
                    float(
                        np.average(
                            event[selected],
                            weights=weights[selected],
                        )
                    )
                    if np.any(selected)
                    else None
                ),
            }
        )
    return {
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": (
            (baseline_brier - brier) / baseline_brier
            if baseline_brier
            else 0.0
        ),
        "calibration_slope": float(calibration.slope),
        "calibration_intercept": float(calibration.intercept),
        "reliability_bins": reliability_bins,
    }


bootstrap_max_t = max_t_bootstrap


def evaluate_gate(
    *,
    primary_lower: Mapping[str, float],
    per_ratio_skills: Mapping[str, Mapping[str, float]],
    paired_upper: Mapping[str, float],
    brier_lower: Mapping[str, float],
    calibration_slope: Mapping[str, float],
    selected_by_fold: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    compressors = set(primary_lower)
    if (
        not compressors
        or set(per_ratio_skills) != compressors
        or set(paired_upper) != compressors
        or set(brier_lower) != compressors
        or set(calibration_slope) != compressors
        or set(selected_by_fold) != compressors
    ):
        raise ValueError("M3 gate inputs are incomplete")
    expected_ratios = {str(ratio) for ratio in RATIOS}
    per_compressor: dict[str, Any] = {}
    for compressor in sorted(compressors):
        selected = tuple(selected_by_fold[compressor])
        ratio_values = per_ratio_skills[compressor]
        every_ratio_positive = set(ratio_values) == expected_ratios and all(
            float(value) > 0 for value in ratio_values.values()
        )
        criteria = {
            "primary_skill_lower_ge_5_points": (
                float(primary_lower[compressor]) >= 0.05
            ),
            "every_ratio_positive_skill": every_ratio_positive,
            "paired_m3_minus_m2_upper_lt_zero": (
                float(paired_upper[compressor]) < 0
            ),
            "positive_brier_lower_gt_zero": (
                float(brier_lower[compressor]) > 0
            ),
            "non_inverted_calibration": (
                float(calibration_slope[compressor]) > 0
            ),
            "same_non_c0_candidate_all_outer_folds": (
                len(selected) == 5
                and len(set(selected)) == 1
                and selected[0] != "C0"
            ),
        }
        per_compressor[compressor] = {
            "criteria": criteria,
            "all_criteria_pass": all(criteria.values()),
        }
    passing = [
        name
        for name, value in per_compressor.items()
        if value["all_criteria_pass"]
    ]
    return {
        "per_compressor": per_compressor,
        "passing_compressors": passing,
        "any_compressor_pass": bool(passing),
        "status": "evaluated",
    }


def expected_m3_freeze_payload(
    *,
    base_parquet_path: Path,
    m2_report_path: Path,
    m2_oof_path: Path,
    m2_prefit_freeze_path: Path,
    m2_result_freeze_path: Path,
    m2_sensor_lock_path: Path,
    band_manifest_path: Path,
    band_sidecar_path: Path,
    evidence_path: Path,
    protocol_lock_path: Path,
    reference_manifest_sha256: str,
    reference_inputs_sha256: str,
    implementation_paths: Mapping[str, Path],
) -> dict[str, Any]:
    for name, value in (
        ("reference manifest", reference_manifest_sha256),
        ("reference inputs", reference_inputs_sha256),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{name} hash is invalid")
    if not implementation_paths:
        raise ValueError("M3 implementation bindings are empty")
    paths = {
        "base_parquet": base_parquet_path,
        "m2_report": m2_report_path,
        "m2_oof": m2_oof_path,
        "m2_prefit_freeze": m2_prefit_freeze_path,
        "m2_result_freeze": m2_result_freeze_path,
        "m2_sensor_lock": m2_sensor_lock_path,
        "band_manifest": band_manifest_path,
        "band_sidecar": band_sidecar_path,
        "evidence": evidence_path,
        "m3_protocol_lock": protocol_lock_path,
    }
    return {
        "schema_version": M3_FREEZE_SCHEMA_VERSION,
        "status": "frozen_before_m3_results",
        "input_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "reference_manifest_sha256": reference_manifest_sha256,
        "reference_inputs_sha256": reference_inputs_sha256,
        "implementation_sha256": {
            name: sha256_file(path)
            for name, path in implementation_paths.items()
        },
        "candidate_config": M3_CONFIG,
        "protocol_version": PROTOCOL_VERSION,
        "feature_names": {
            "base": list(BASE_FEATURES),
            "global": list(GLOBAL_FEATURES),
            "bands": list(BAND_FEATURES),
        },
    }


def validate_m3_freeze(
    freeze_path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    actual = _load_json(freeze_path)
    if actual != dict(expected):
        raise ValueError("M3 pre-fit freeze does not match expected bindings")
    return actual


def write_oof_parquet(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to write M3 OOF") from error
    exact_keys(rows, "M3 OOF")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path)  # type: ignore[no-untyped-call]


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
