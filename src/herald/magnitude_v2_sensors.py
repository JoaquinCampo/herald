"""Locked development-only M2 comparison with label-free sensor features.

M2 consumes frozen M1 predictions and label-free compressor-action sensors. It
never reads a confirmation/test partition and never pools compressors.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from herald import magnitude_v2 as v2
from herald.magnitude import (
    CAUSAL_FEATURE_COLUMNS,
    KNOWN_COMPRESSORS,
    _ratio_metrics,
    trajectory_weights,
)
from herald.press_sensors import sensor_feature_names
from herald.sensor_replay import read_allowed_reference, validate_sensor_lock
from herald.storage import (
    read_sensor_records,
    sensor_manifest_path,
    sensor_sidecar_path,
    validate_sensor_records,
)

SCHEMA_VERSION = "herald.magnitude_v2_sensors.v1"
SENSOR_PROTOCOL_VERSION = "herald.m2.sensor_replay.v2"
MODEL_KEY = "llama"
TASK = "ifeval"
KEY_COLUMNS = ("prompt_id", "compressor", "ratio", "s")
M1_REQUIRED_COLUMNS = (
    "model",
    "task",
    *KEY_COLUMNS,
    "dq",
    "m1_prediction",
    "locked_baseline_mean",
    "locked_baseline_median",
    "fold_occurrence_prevalence",
    "fold_positive_prevalence",
    "fold",
    "fold_train_prompt_ids_sha256",
    "fold_test_prompt_ids_sha256",
)
M1_NUMERIC_COLUMNS = (
    "dq",
    "m1_prediction",
    "locked_baseline_mean",
    "locked_baseline_median",
    "fold_occurrence_prevalence",
    "fold_positive_prevalence",
    "fold",
)
EXPECTED_M2_MODEL_AND_GATE = {
    "absolute_gate": [
        "90% prompt-bootstrap lower MSE skill versus locked mean > 0",
        "MAE skill versus locked median > 0",
        "positive and major-row MSE not worse than locked mean",
        "every ratio MSE better than locked mean",
        "positive-risk Brier skill > 0",
    ],
    "compressor_independence": True,
    "incremental_gate": [
        "ratio-macro MSE lower than frozen M1",
        "positive and major-row MSE not worse than frozen M1",
        "90% paired prompt-bootstrap upper M2-minus-M1 MSE < 0",
    ],
    "inner_folds": 4,
    "model": (
        "M1 hierarchical hurdle boosted trees with the frozen causal base "
        "features plus exactly 36 sensor features"
    ),
    "model_seeds": [0, 1, 2],
    "num_boost_round": 300,
    "outer_folds": 5,
    "selection_rule": (
        "per compressor; favor frozen M1 unless every absolute and "
        "incremental M2 criterion passes"
    ),
    "trajectory_weighting": "equal total weight per prompt-ratio trajectory",
}
M2_FREEZE_SCHEMA_VERSION = "herald.magnitude_v2_m2_freeze.v1"


def expected_m2_freeze_payload(
    *,
    base_parquet_path: Path,
    m1_freeze_path: Path,
    m1_report_path: Path,
    m1_oof_path: Path,
    sensor_manifest_path: Path,
    sensor_sidecar_path: Path,
    evidence_path: Path,
    lock_path: Path,
    sensor_lock_path: Path,
    reference_manifest_sha256: str,
    reference_inputs_sha256: str,
) -> dict[str, Any]:
    """Build the exact immutable pre-fit M2 freeze object."""
    repository_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": M2_FREEZE_SCHEMA_VERSION,
        "status": "frozen_before_m2_results",
        "source_evidence_sha256": sha256_file(evidence_path),
        "protocol_lock_sha256": sha256_file(lock_path),
        "sensor_lock_sha256": sha256_file(sensor_lock_path),
        "base_parquet_sha256": sha256_file(base_parquet_path),
        "m1_freeze_sha256": sha256_file(m1_freeze_path),
        "m1_report_sha256": sha256_file(m1_report_path),
        "m1_oof_sha256": sha256_file(m1_oof_path),
        "sensor_manifest_sha256": sha256_file(sensor_manifest_path),
        "sensor_sidecar_sha256": sha256_file(sensor_sidecar_path),
        "reference_manifest_sha256": reference_manifest_sha256,
        "reference_inputs_sha256": reference_inputs_sha256,
        "implementation_sha256": {
            "magnitude.py": sha256_file(
                repository_root / "src" / "herald" / "magnitude.py"
            ),
            "magnitude_v2_sensors.py": sha256_file(Path(__file__)),
            "compare_magnitude_v2_sensors.py": sha256_file(
                repository_root
                / "scripts"
                / "compare_magnitude_v2_sensors.py"
            ),
            "uv.lock": sha256_file(repository_root / "uv.lock"),
        },
    }


def validate_m2_freeze(
    freeze_path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a byte-parsed freeze object to equal every expected binding."""
    actual = _load_json(freeze_path)
    if actual != dict(expected):
        raise ValueError("M2 pre-fit freeze does not match expected bindings")
    return actual


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


def _as_float(value: object) -> float:
    return float(value)  # type: ignore[arg-type]


def _require_model_task(
    rows: Sequence[Mapping[str, object]], label: str
) -> None:
    for row in rows:
        if row.get("model") != MODEL_KEY or row.get("task") != TASK:
            raise ValueError(f"{label} model/task is not locked")


def _as_int(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("boolean is not an integer")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("value is not numeric") from error
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError("value is not a finite integer")
    return int(number)


def _key(row: Mapping[str, object]) -> tuple[str, str, float, int]:
    try:
        result = (
            str(row["prompt_id"]),
            str(row["compressor"]),
            _as_float(row["ratio"]),
            _as_int(row["s"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("row has an invalid exact key") from error
    if (
        not result[0]
        or not result[1]
        or not np.isfinite(result[2])
        or result[3] < 0
    ):
        raise ValueError("row has an invalid exact key")
    return result


def _keys(
    rows: Sequence[Mapping[str, object]], label: str
) -> set[tuple[str, str, float, int]]:
    values = [_key(row) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} exact key")
    return set(values)


def _finite(
    row: Mapping[str, object], names: Sequence[str], label: str
) -> None:
    try:
        values = np.asarray(
            [_as_float(row[name]) for name in names],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label} has missing or nonnumeric fields"
        ) from error
    if not np.isfinite(values).all():
        raise ValueError(f"{label} has nonfinite fields")


def _development_and_quarantine(
    evidence: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    split = evidence.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("evidence split is missing")
    development = {str(value) for value in split.get("train_prompt_ids", ())}
    quarantine = {str(value) for value in split.get("test_prompt_ids", ())}
    if development & quarantine:
        raise ValueError("development and quarantined prompt IDs overlap")
    return development, quarantine


def _reject_quarantine(
    rows: Sequence[Mapping[str, object]], quarantine: set[str], label: str
) -> None:
    found = sorted(
        {str(row.get("prompt_id", "")) for row in rows} & quarantine
    )
    if found:
        raise ValueError(f"quarantined prompt entered {label}: {found[:3]}")


def _feature_hash(features: Sequence[str]) -> str:
    payload = json.dumps(list(features), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _sensor_lock_default(lock_path: Path) -> Path:
    return lock_path.with_name("magnitude_v2_sensor_lock.json")


def _load_reference_inputs(
    reference_results_root: Path,
    evidence: Mapping[str, Any],
    records: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    manifest_path = (
        reference_results_root / MODEL_KEY / TASK / "manifest.json"
    )
    source_manifest = evidence.get("source_manifest")
    if not isinstance(source_manifest, Mapping) or source_manifest.get(
        "sha256"
    ) != sha256_file(manifest_path):
        raise ValueError("reference manifest hash disagrees with evidence")
    manifest = _load_json(manifest_path)
    artifacts = manifest.get("artifact_files")
    if not isinstance(artifacts, list):
        raise ValueError("reference manifest artifact_files is missing")
    declared: dict[str, str] = {}
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            raise ValueError("reference manifest artifact entry is invalid")
        path = str(entry.get("path", ""))
        digest = str(entry.get("sha256", ""))
        if (
            not path
            or path in declared
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("reference manifest artifact entry is invalid")
        declared[path] = digest
    split = evidence.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("evidence split is missing")
    allowed = {str(value) for value in split.get("train_prompt_ids", ())}
    positions_by_prompt: dict[str, set[int]] = {}
    for record in records:
        prompt_id = str(record["prompt_id"])
        if prompt_id not in allowed:
            raise ValueError("sensor record is outside reference allowlist")
        positions_by_prompt.setdefault(prompt_id, set()).add(
            _as_int(record["s"])
        )
    references: dict[str, Any] = {}
    for prompt_id in sorted(allowed):
        reference = read_allowed_reference(
            reference_results_root,
            MODEL_KEY,
            TASK,
            prompt_id,
            allowed,
        )
        relative = str(reference.path.relative_to(manifest_path.parent))
        declared_hash = declared.get(relative) or declared.get(
            reference.path.name
        )
        if (
            declared_hash is None
            or sha256_file(reference.path) != declared_hash
        ):
            raise ValueError(
                f"reference manifest digest mismatch: {prompt_id}"
            )
        references[prompt_id] = reference
    prefix_hashes: dict[tuple[str, int], str] = {}
    for prompt_id, positions in positions_by_prompt.items():
        reference = references[prompt_id]
        for position in positions:
            if position < 0 or position >= len(reference.gen_ids):
                raise ValueError("sensor position is outside generation IDs")
            prefix = reference.prompt_input_ids + reference.gen_ids[:position]
            prefix_hashes[(prompt_id, position)] = hashlib.sha256(
                np.asarray(prefix, dtype=np.int64).tobytes()
            ).hexdigest()
    for record in records:
        key = (str(record["prompt_id"]), _as_int(record["s"]))
        if record.get("prefix_hash") != prefix_hashes.get(key):
            raise ValueError("sensor prefix hash disagrees with reference")
    canonical = [
        {
            "prompt_id": prompt_id,
            "prompt_input_ids": list(references[prompt_id].prompt_input_ids),
            "gen_ids": list(references[prompt_id].gen_ids),
        }
        for prompt_id in sorted(references)
    ]
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return sha256_file(manifest_path), digest


def load_sensor_sidecar(
    sidecar_root: Path,
    manifest_path: Path,
    *,
    evidence_path: Path,
    lock_path: Path,
    expected_keys: set[tuple[str, str, float, int]],
    reference_results_root: Path,
    sensor_lock_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the content-bound sensor sidecar and require the sensor lock."""
    evidence, _ = v2.validate_lock_and_evidence(evidence_path, lock_path)
    development, quarantine = _development_and_quarantine(evidence)
    sensor_lock_path = sensor_lock_path or _sensor_lock_default(lock_path)
    sensor_lock_sha = validate_sensor_lock(
        sensor_lock_path,
        evidence_path,
        lock_path,
    )
    sensor_lock = _load_json(sensor_lock_path)
    expected_features = tuple(sensor_feature_names())
    if len(expected_features) != 36 or len(set(expected_features)) != 36:
        raise ValueError("sensor schema must contain exactly 36 features")
    if sensor_lock.get("m2_model_and_gate") != EXPECTED_M2_MODEL_AND_GATE:
        raise ValueError("sensor lock M2 model and gate differs from lock")
    expected_manifest = sensor_manifest_path(sidecar_root, MODEL_KEY, TASK)
    if manifest_path.resolve() != expected_manifest.resolve():
        raise ValueError("sensor manifest path does not match sidecar root")
    manifest = _load_json(manifest_path)
    if manifest.get("model") != MODEL_KEY or manifest.get("task") != TASK:
        raise ValueError("sensor manifest model/task is not locked")
    if manifest.get("protocol_version") != SENSOR_PROTOCOL_VERSION:
        raise ValueError("sensor manifest protocol is not locked")
    evidence_sha, lock_sha = (
        sha256_file(evidence_path),
        sha256_file(lock_path),
    )
    if (
        manifest.get("evidence_sha256") != evidence_sha
        or manifest.get("lock_sha256") != lock_sha
    ):
        raise ValueError("sensor manifest evidence/lock hash mismatch")
    if manifest.get("sensor_lock_sha256") != sensor_lock_sha:
        raise ValueError("sensor manifest sensor-lock hash mismatch")
    if (
        tuple(map(str, manifest.get("feature_names", ())))
        != expected_features
    ):
        raise ValueError("sensor manifest feature order mismatch")
    if manifest.get("feature_names_sha256") != _feature_hash(
        expected_features
    ):
        raise ValueError("sensor manifest feature hash mismatch")
    sidecar = sensor_sidecar_path(sidecar_root, MODEL_KEY, TASK)
    if not sidecar.is_file() or manifest.get("sidecar_sha256") != sha256_file(
        sidecar
    ):
        raise ValueError("sensor manifest sidecar hash mismatch")
    records = read_sensor_records(sidecar_root, MODEL_KEY, TASK)
    if int(manifest.get("record_count", -1)) != len(records):
        raise ValueError("sensor manifest record count mismatch")
    observed = validate_sensor_records(
        records,
        model=MODEL_KEY,
        task=TASK,
        evidence_sha256=evidence_sha,
        lock_sha256=lock_sha,
        sensor_lock_sha256=sensor_lock_sha,
        protocol_version=SENSOR_PROTOCOL_VERSION,
        feature_names=expected_features,
    )
    _reject_quarantine(records, quarantine, "sensor records")
    if not {str(record["prompt_id"]) for record in records} <= development:
        raise ValueError(
            "sensor record is outside the frozen development IDs"
        )
    if observed != expected_keys:
        raise ValueError("sensor keys do not exactly match base keys")
    key_digest = "\n".join(
        "\0".join(map(str, key)) for key in sorted(observed)
    )
    reference_manifest_sha, reference_inputs_sha = _load_reference_inputs(
        reference_results_root, evidence, records
    )
    if (
        manifest.get("record_keys_sha256")
        != hashlib.sha256(key_digest.encode()).hexdigest()
    ):
        raise ValueError("sensor manifest key hash mismatch")
    result: list[dict[str, Any]] = []
    for record in records:
        sensors = record["sensors"]
        if not isinstance(sensors, Mapping):
            raise ValueError("sensor record payload is not a mapping")
        row = {
            "prompt_id": str(record["prompt_id"]),
            "compressor": str(record["compressor"]),
            "ratio": _as_float(record["ratio"]),
            "s": _as_int(record["s"]),
        }
        for name in expected_features:
            value = _as_float(sensors[name])
            if not np.isfinite(value):
                raise ValueError("sensor record contains nonfinite values")
            row[f"sensor__{name}"] = value
        result.append(row)
    return result, {
        "manifest_sha256": sha256_file(manifest_path),
        "sidecar_sha256": str(manifest["sidecar_sha256"]),
        "sensor_lock_sha256": sensor_lock_sha,
        "evidence_sha256": evidence_sha,
        "lock_sha256": lock_sha,
        "reference_manifest_sha256": reference_manifest_sha,
        "reference_inputs_sha256": reference_inputs_sha,
        "protocol_version": SENSOR_PROTOCOL_VERSION,
        "feature_names": list(expected_features),
    }


def load_frozen_m1(
    report_path: Path,
    oof_path: Path,
    freeze_path: Path,
    *,
    base_rows: Sequence[Mapping[str, object]],
    base_provenance: Mapping[str, Any],
    evidence_path: Path,
    lock_path: Path,
    sensor_lock_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load M1 only after validating its immutable freeze artifact."""
    evidence, lock = v2.validate_lock_and_evidence(evidence_path, lock_path)
    development, quarantine = _development_and_quarantine(evidence)
    sensor_lock_path = sensor_lock_path or _sensor_lock_default(lock_path)
    validate_sensor_lock(sensor_lock_path, evidence_path, lock_path)
    sensor_lock = _load_json(sensor_lock_path)
    freeze_binding = sensor_lock.get("m1_freeze")
    if not isinstance(freeze_binding, Mapping):
        raise ValueError("sensor lock lacks the M1 freeze binding")
    freeze = _load_json(freeze_path)
    if (
        freeze.get("schema_version") != "herald.magnitude_v2_m1_freeze.v1"
        or freeze.get("status") != "frozen_before_m2_sidecar_results"
        or freeze.get("source_evidence_sha256") != sha256_file(evidence_path)
        or freeze.get("protocol_lock_sha256") != sha256_file(lock_path)
        or freeze_binding.get("freeze_sha256") != sha256_file(freeze_path)
    ):
        raise ValueError("M1 freeze provenance is invalid")
    frozen_report = freeze.get("report")
    frozen_oof = freeze.get("oof")
    if not isinstance(frozen_report, Mapping) or not isinstance(
        frozen_oof,
        Mapping,
    ):
        raise ValueError("M1 freeze artifact bindings are missing")
    report_hash = sha256_file(report_path)
    oof_hash = sha256_file(oof_path)
    if (
        frozen_report.get("sha256") != report_hash
        or frozen_oof.get("sha256") != oof_hash
        or freeze_binding.get("report_sha256") != report_hash
        or freeze_binding.get("oof_sha256") != oof_hash
    ):
        raise ValueError("M1 report/OOF differs from the frozen artifacts")

    report = _load_json(report_path)
    if report.get("schema_version") != v2.SCHEMA_VERSION:
        raise ValueError("frozen M1 report schema is invalid")
    input_provenance = report.get("input")
    if not isinstance(input_provenance, Mapping):
        raise ValueError("frozen M1 report input provenance is missing")
    expected_evidence = sha256_file(evidence_path)
    expected_lock = sha256_file(lock_path)
    if input_provenance.get("parquet_sha256") != base_provenance.get(
        "parquet_sha256"
    ):
        raise ValueError("M1 report base parquet hash mismatch")
    if input_provenance.get("source_evidence_sha256") != expected_evidence:
        raise ValueError("M1 report evidence hash mismatch")
    if input_provenance.get("protocol_lock_sha256") != expected_lock:
        raise ValueError("M1 report lock hash mismatch")
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("lock") != lock:
        raise ValueError("M1 report protocol lock is not frozen")
    expected_features = tuple(
        str(name)
        for name in base_provenance.get("features", ())
        if str(name) not in {"ratio", "s"}
    )
    if set(expected_features) != set(CAUSAL_FEATURE_COLUMNS) or tuple(
        map(str, protocol.get("features", ()))
    ) != tuple(sorted(expected_features)):
        raise ValueError("M1 report causal feature schema mismatch")
    fold_payload = report.get("folds")
    locked_partition = lock.get("prompt_partitions")
    locked_counts = (
        locked_partition.get("fold_counts")
        if isinstance(locked_partition, Mapping)
        else None
    )
    locked_hashes = (
        locked_partition.get("fold_prompt_ids_sha256")
        if isinstance(locked_partition, Mapping)
        else None
    )
    if (
        not isinstance(fold_payload, list)
        or not isinstance(locked_counts, list)
        or not isinstance(locked_hashes, list)
        or len(fold_payload) != 5
        or locked_counts != [31, 31, 31, 31, 30]
        or len(locked_hashes) != 5
    ):
        raise ValueError(
            "M1 report folds are not the locked five-fold partition"
        )
    fold_by_index: dict[int, tuple[tuple[str, ...], str]] = {}
    prompt_fold: dict[str, int] = {}
    for item in fold_payload:
        if not isinstance(item, Mapping):
            raise ValueError("M1 report fold entry is invalid")
        try:
            index = _as_int(item["index"])
            values = item["prompt_ids"]
            count = _as_int(item["count"])
            declared_hash = str(item["prompt_ids_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("M1 report fold entry is invalid") from error
        if (
            not isinstance(values, list)
            or index in fold_by_index
            or index < 0
            or index >= len(locked_hashes)
        ):
            raise ValueError("M1 report fold entry is invalid")
        prompt_ids = tuple(str(value) for value in values)
        if (
            count != len(prompt_ids)
            or len(prompt_ids) != len(set(prompt_ids))
            or declared_hash != v2.hash_prompt_ids(prompt_ids)
            or declared_hash != str(locked_hashes[index])
        ):
            raise ValueError("M1 report fold prompt hash is invalid")
        fold_by_index[index] = (prompt_ids, declared_hash)
        for prompt_id in prompt_ids:
            if prompt_id in prompt_fold:
                raise ValueError("M1 report folds overlap")
            prompt_fold[prompt_id] = index
    if set(fold_by_index) != set(range(5)) or set(prompt_fold) != development:
        raise ValueError(
            "M1 report folds are not an exact development partition"
        )
    if [len(fold_by_index[index][0]) for index in range(5)] != locked_counts:
        raise ValueError("M1 report fold counts differ from protocol lock")
    try:
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for frozen M1 OOF") from error
    schema = pq.read_schema(oof_path)  # type: ignore[no-untyped-call]
    missing = set(M1_REQUIRED_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"M1 OOF lacks locked columns: {sorted(missing)}")
    dataset = ds.dataset(  # type: ignore[no-untyped-call]
        oof_path,
        format="parquet",
    )
    predicate = ds.field("prompt_id").isin(  # type: ignore[attr-defined, no-untyped-call]
        sorted(development)
    )
    table = dataset.to_table(
        columns=list(M1_REQUIRED_COLUMNS),
        filter=predicate,
    )
    total_rows = pq.ParquetFile(  # type: ignore[no-untyped-call]
        oof_path
    ).metadata.num_rows
    if table.num_rows != total_rows:
        raise ValueError("M1 OOF contains a non-development prompt")
    m1 = [dict(row) for row in table.to_pylist()]
    _require_model_task(base_rows, "base")
    _require_model_task(m1, "M1 OOF")
    base_by_key = {_key(row): row for row in base_rows}
    for row in m1:
        _finite(row, M1_NUMERIC_COLUMNS, "M1 OOF")
        prompt_id = str(row["prompt_id"])
        row_fold = _as_int(row["fold"])
        if prompt_fold.get(prompt_id) != row_fold:
            raise ValueError("M1 OOF row prompt does not map to its fold")
        if row["fold_test_prompt_ids_sha256"] != fold_by_index[row_fold][1]:
            raise ValueError("M1 OOF test-fold hash disagrees with report")
        expected_train_hash = v2.hash_prompt_ids(
            sorted(set(development) - set(fold_by_index[row_fold][0]))
        )
        if row["fold_train_prompt_ids_sha256"] != expected_train_hash:
            raise ValueError("M1 OOF train-fold hash disagrees with report")
        if not np.isclose(
            _as_float(row["dq"]),
            _as_float(base_by_key[_key(row)]["dq"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("M1 OOF dq disagrees with base")
    return m1, {
        "freeze_sha256": sha256_file(freeze_path),
        "report_sha256": report_hash,
        "oof_sha256": oof_hash,
        "parquet_sha256": str(input_provenance["parquet_sha256"]),
        "evidence_sha256": expected_evidence,
        "lock_sha256": expected_lock,
        "schema_version": report["schema_version"],
    }


def join_sensor_features(
    base_rows: Sequence[Mapping[str, object]],
    sensor_rows: Sequence[Mapping[str, object]],
    m1_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    """Join exact base, sensor, and M1 cells using only sensor features."""
    _require_model_task(base_rows, "base")
    _require_model_task(m1_rows, "M1")
    base_keys, sensor_keys, m1_keys = (
        _keys(base_rows, "base"),
        _keys(sensor_rows, "sensor"),
        _keys(m1_rows, "M1"),
    )
    if not (base_keys == sensor_keys == m1_keys):
        raise ValueError(
            "base, sensor, and M1 keys are not exact one-to-one coverage"
        )
    sensor_names = tuple(f"sensor__{name}" for name in sensor_feature_names())
    m1_by_key, sensor_by_key = (
        {_key(row): row for row in m1_rows},
        {_key(row): row for row in sensor_rows},
    )
    result: list[dict[str, Any]] = []
    for base in base_rows:
        key = _key(base)
        m1 = m1_by_key[key]
        sensor = sensor_by_key[key]
        if not np.isclose(
            _as_float(base["dq"]),
            _as_float(m1["dq"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("base and M1 dq disagree")
        _finite(
            m1,
            ("locked_baseline_mean", "locked_baseline_median"),
            "M1 locked baseline",
        )
        _finite(sensor, sensor_names, "sensor join")
        joined = dict(base)
        joined.update(
            {name: _as_float(sensor[name]) for name in sensor_names}
        )
        result.append(joined)
    return result


def paired_prompt_bootstrap_m2_minus_m1(
    rows: Sequence[Mapping[str, object]],
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    resamples: int = 1000,
    seed: int = 1729,
) -> dict[str, Any]:
    """Cluster paired ratio-macro MSE deltas by prompt (90% interval)."""
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    groups = {
        prompt: np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if str(row["prompt_id"]) == prompt
            ],
            dtype=np.int64,
        )
        for prompt in prompts
    }
    target, weights = (
        np.asarray(
            [_as_float(row["dq"]) for row in rows],
            dtype=np.float64,
        ),
        trajectory_weights(rows),
    )
    rng, deltas = np.random.default_rng(seed), []
    for draw in rng.integers(0, len(prompts), size=(resamples, len(prompts))):
        indices = np.concatenate([groups[prompts[int(i)]] for i in draw])
        sampled = [rows[int(i)] for i in indices]
        m2_mse = _ratio_metrics(
            target[indices], m2[indices], weights[indices], sampled
        )["overall"]["mse"]
        m1_mse = _ratio_metrics(
            target[indices], m1[indices], weights[indices], sampled
        )["overall"]["mse"]
        deltas.append(float(m2_mse - m1_mse))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "resamples": resamples,
        "seed": seed,
        "metric": "m2_minus_m1_ratio_macro_mse",
        "estimate": float(values.mean()),
        "lower": float(np.quantile(values, 0.05)),
        "upper": float(np.quantile(values, 0.95)),
    }


def assemble_gates(
    metrics_by_compressor: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble independent absolute and incremental gates per compressor."""
    criteria, per_compressor = {}, {}
    for compressor in sorted(metrics_by_compressor):
        value, m2, m1, mean = (
            metrics_by_compressor[compressor],
            metrics_by_compressor[compressor]["m2"],
            metrics_by_compressor[compressor]["m1"],
            metrics_by_compressor[compressor]["locked_mean"],
        )
        absolute = {
            "mse_bootstrap_lower_gt_zero": float(
                value["bootstrap_locked_mean"]["lower"]
            )
            > 0.0,
            "mae_skill_gt_zero": float(value["m2_vs_locked_median"]["mae"])
            > 0.0,
            "positive_mse_not_worse": float(m2["positive_rows"]["mse"])
            <= float(mean["positive_rows"]["mse"]),
            "major_mse_not_worse": float(m2["major_rows"]["mse"])
            <= float(mean["major_rows"]["mse"]),
            "every_ratio_mse_better": all(
                float(m2["per_ratio"][r]["mse"])
                < float(mean["per_ratio"][r]["mse"])
                for r in m2["per_ratio"]
            ),
            "positive_risk_brier_skill_gt_zero": float(
                value["positive_risk_brier_skill"]
            )
            > 0.0,
        }
        incremental = {
            "ratio_macro_mse_lower_than_m1": float(m2["ratio_macro"]["mse"])
            < float(m1["ratio_macro"]["mse"]),
            "positive_mse_not_worse_than_m1": float(
                m2["positive_rows"]["mse"]
            )
            <= float(m1["positive_rows"]["mse"]),
            "major_mse_not_worse_than_m1": float(m2["major_rows"]["mse"])
            <= float(m1["major_rows"]["mse"]),
            "paired_bootstrap_upper_lt_zero": float(
                value["paired_bootstrap"]["upper"]
            )
            < 0.0,
        }
        local = {**absolute, **incremental}
        criteria.update(
            {f"{compressor}.{name}": passed for name, passed in local.items()}
        )
        per_compressor[compressor] = {
            "criteria": local,
            "absolute": absolute,
            "incremental": incremental,
            "all_criteria_pass": all(local.values()),
        }
    passing = [
        name
        for name, value in per_compressor.items()
        if value["all_criteria_pass"]
    ]
    return {
        "criteria": criteria,
        "per_compressor": per_compressor,
        "passing_compressors": passing,
        "any_compressor_pass": bool(passing),
        "all_comparison_criteria_pass": all(criteria.values())
        if criteria
        else False,
        "status": "evaluated",
    }


def _skill(model: float, baseline: float) -> float:
    return (baseline - model) / baseline if baseline else 0.0


def fit_development_sensors(
    rows: Sequence[Mapping[str, object]],
    m1_rows: Sequence[Mapping[str, object]],
    *,
    lock: Mapping[str, Any],
    base_provenance: Mapping[str, Any],
    sensor_provenance: Mapping[str, Any],
    m1_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit nested prompt OOF M2 hurdles independently for each compressor."""
    if not rows:
        raise ValueError("joined M2 rows are empty")
    joined_keys = _keys(rows, "joined")
    m1_keys = _keys(m1_rows, "M1")
    if joined_keys != m1_keys:
        raise ValueError(
            "joined and M1 keys are not exact one-to-one coverage"
        )
    compressors = tuple(sorted({str(row["compressor"]) for row in rows}))
    if compressors != tuple(sorted(KNOWN_COMPRESSORS)):
        raise ValueError("M2 rows do not contain every locked compressor")
    folds = v2.make_folds(
        sorted({str(row["prompt_id"]) for row in rows}), dict(lock)
    )
    base_features = tuple(
        str(name)
        for name in base_provenance.get("features", ())
        if str(name) not in {"ratio", "s"}
    )
    if set(base_features) != set(CAUSAL_FEATURE_COLUMNS):
        raise ValueError(
            "M2 base features differ from locked causal features"
        )
    sensor_features = tuple(
        f"sensor__{name}" for name in sensor_feature_names()
    )
    if len(sensor_features) != 36 or len(set(sensor_features)) != 36:
        raise ValueError("M2 does not contain exactly 36 sensor features")
    features, m1_by_key = (
        tuple(sorted(base_features)) + sensor_features,
        {_key(row): row for row in m1_rows},
    )
    config = lock["development_config"]
    oof: list[dict[str, Any]] = []
    seeds, rounds, inner = (
        tuple(int(value) for value in config["model_seeds"]),
        int(config["num_boost_round"]),
        int(config["inner_folds"]),
    )
    for compressor in compressors:
        comp_rows = [
            dict(row) for row in rows if str(row["compressor"]) == compressor
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
                    f"fold {fold.index} has empty train/test partition"
                )
            fit = v2._fit_hurdle(
                train,
                features,
                seed=seeds[0],
                seeds=seeds,
                rounds=rounds,
                inner_folds=inner,
            )
            prediction, p_nz, p_sign, m_plus, m_minus = v2._hurdle_predict(
                fit, test
            )
            for row, pred, pn, ps, mp, mm in zip(
                test, prediction, p_nz, p_sign, m_plus, m_minus, strict=True
            ):
                frozen = m1_by_key[_key(row)]
                output = {
                    "model": row.get("model", MODEL_KEY),
                    "task": row.get("task", TASK),
                    **{column: row[column] for column in KEY_COLUMNS},
                    "dq": _as_float(row["dq"]),
                    "fold": fold.index,
                    "fold_train_prompt_ids_sha256": v2.hash_prompt_ids(
                        sorted({str(r["prompt_id"]) for r in train})
                    ),
                    "fold_test_prompt_ids_sha256": fold.hash,
                    "m1_prediction": _as_float(frozen["m1_prediction"]),
                    "m2_prediction": float(pred),
                    "m2_p_nonzero": float(pn),
                    "m2_p_positive": float(pn * ps),
                    "m2_m_plus": float(mp),
                    "m2_m_minus": float(mm),
                    "locked_baseline_mean": _as_float(
                        frozen["locked_baseline_mean"]
                    ),
                    "locked_baseline_median": _as_float(
                        frozen["locked_baseline_median"]
                    ),
                    "fold_occurrence_prevalence": _as_float(
                        frozen["fold_occurrence_prevalence"]
                    ),
                    "fold_positive_prevalence": _as_float(
                        frozen["fold_positive_prevalence"]
                    ),
                }
                output.update(
                    {name: _as_float(row[name]) for name in sensor_features}
                )
                oof.append(output)
    if _keys(oof, "M2 OOF") != _keys(rows, "joined"):
        raise ValueError("M2 OOF is not exact-once key coverage")
    metrics_by_compressor = {}
    for compressor in compressors:
        comp = [row for row in oof if row["compressor"] == compressor]
        target, m1, m2 = (
            np.asarray([float(row["dq"]) for row in comp]),
            np.asarray([float(row["m1_prediction"]) for row in comp]),
            np.asarray([float(row["m2_prediction"]) for row in comp]),
        )
        bmean, bmedian, weights = (
            np.asarray([float(row["locked_baseline_mean"]) for row in comp]),
            np.asarray(
                [float(row["locked_baseline_median"]) for row in comp]
            ),
            trajectory_weights(comp),
        )
        m1_metrics, m2_metrics = (
            v2._model_metrics(comp, target, m1),
            v2._model_metrics(comp, target, m2),
        )
        mean_metrics, median_metrics = (
            v2._model_metrics(comp, target, bmean),
            v2._model_metrics(comp, target, bmedian),
        )
        for metric, values in (
            (m1_metrics, m1),
            (m2_metrics, m2),
            (mean_metrics, bmean),
            (median_metrics, bmedian),
        ):
            metric["per_ratio"] = _ratio_metrics(
                target, values, weights, comp
            )["per_ratio"]
        positive = (target > 0).astype(float)
        risk, risk_base = (
            np.asarray([float(row["m2_p_positive"]) for row in comp]),
            np.asarray(
                [float(row["fold_positive_prevalence"]) for row in comp]
            ),
        )
        brier, brier_base = (
            float(np.average((risk - positive) ** 2, weights=weights)),
            float(np.average((risk_base - positive) ** 2, weights=weights)),
        )
        metrics_by_compressor[compressor] = {
            "m1": m1_metrics,
            "m2": m2_metrics,
            "locked_mean": mean_metrics,
            "locked_median": median_metrics,
            "m2_vs_locked_mean": {
                "mse": _skill(
                    m2_metrics["ratio_macro"]["mse"],
                    mean_metrics["ratio_macro"]["mse"],
                ),
                "positive_mse": _skill(
                    m2_metrics["positive_rows"]["mse"],
                    mean_metrics["positive_rows"]["mse"],
                ),
                "major_mse": _skill(
                    m2_metrics["major_rows"]["mse"],
                    mean_metrics["major_rows"]["mse"],
                ),
            },
            "m2_vs_locked_median": {
                "mae": _skill(
                    m2_metrics["ratio_macro"]["mae"],
                    median_metrics["ratio_macro"]["mae"],
                )
            },
            "bootstrap_locked_mean": v2.bootstrap_locked_mean_skill(
                comp,
                m2,
                bmean,
                resamples=int(config["bootstrap_resamples"]),
                seed=1729,
            ),
            "paired_bootstrap": paired_prompt_bootstrap_m2_minus_m1(
                comp,
                m1,
                m2,
                resamples=int(config["bootstrap_resamples"]),
                seed=1729,
            ),
            "positive_risk_brier": brier,
            "positive_risk_brier_baseline": brier_base,
            "positive_risk_brier_skill": _skill(brier, brier_base),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "base_provenance": dict(base_provenance),
        "m1_provenance": dict(m1_provenance),
        "sensor_provenance": dict(sensor_provenance),
        "protocol": {
            "lock": dict(lock),
            "features": list(features),
            "base_features": list(sorted(base_features)),
            "sensor_features": list(sensor_features),
        },
        "folds": [
            {
                "index": fold.index,
                "count": len(fold.prompt_ids),
                "prompt_ids_sha256": fold.hash,
                "prompt_ids": list(fold.prompt_ids),
            }
            for fold in folds
        ],
        "compressors": metrics_by_compressor,
    }
    report["development_gate"] = assemble_gates(metrics_by_compressor)
    return report, oof


def write_oof_parquet(
    rows: Sequence[Mapping[str, object]], path: Path
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to write M2 OOF") from error
    if not rows or len(_keys(rows, "M2 OOF")) != len(rows):
        raise ValueError("M2 OOF must contain exact unique keys")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path)  # type: ignore[no-untyped-call]


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
