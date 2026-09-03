"""Atomic, idempotent, resumable on-disk storage for HERALD sweep results.

Layout under results_dir:
    {model}/{task}/references/{safe_id(prompt_id)}.json
    {model}/{task}/references/{safe_id(prompt_id)}.npy
    {model}/{task}/references/_done.jsonl          (append-only manifest)
    {model}/{task}/hybrids/{compressor}__{ratio:.4f}.jsonl
    {model}/{task}/hybrid_features/{compressor}__{ratio:.4f}/
        {safe_id(prompt_id)}__s{s}.npy

Atomicity: JSON and NPY files are written to a temp file in the same
directory, fsynced, then renamed via os.replace (atomic on POSIX). The
manifest line is only appended AFTER both renames succeed, so a half-written
pair never appears as complete.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast
from urllib.parse import quote

import numpy as np
import numpy.typing as npt


def safe_id(prompt_id: str) -> str:
    """Encode a prompt_id as an injective filename-safe stem."""
    return quote(prompt_id, safe="")


def legacy_safe_id(prompt_id: str) -> str:
    """Pre-feature-collection filename convention for old artifacts."""
    return prompt_id.replace("/", "_")


def _ref_dir(results_dir: Path, model: str, task: str) -> Path:
    return results_dir / model / task / "references"


def _hybrid_dir(results_dir: Path, model: str, task: str) -> Path:
    return results_dir / model / task / "hybrids"


def _hybrid_feature_dir(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
) -> Path:
    return (
        results_dir
        / model
        / task
        / "hybrid_features"
        / _shard_name(compressor, ratio).removesuffix(".jsonl")
    )


def _shard_name(compressor: str, ratio: float) -> str:
    return f"{compressor}__{ratio:.4f}.jsonl"


def hybrid_feature_path(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    prompt_id: str,
    s: int,
) -> Path:
    """Return the stored compressed-stream feature path for a hybrid."""
    return (
        _hybrid_feature_dir(results_dir, model, task, compressor, ratio)
        / f"{safe_id(prompt_id)}__s{s}.npy"
    )


def reference_done(results_dir: Path, model: str, task: str) -> set[str]:
    """Return the set of prompt_ids whose reference is fully written.

    Reads _done.jsonl; skips any torn final line. Returns an empty set
    when the manifest does not exist.
    """
    manifest = _ref_dir(results_dir, model, task) / "_done.jsonl"
    if not manifest.exists():
        return set()
    done: set[str] = set()
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["prompt_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def ensure_reference_done(
    results_dir: Path,
    model: str,
    task: str,
    prompt_id: str,
) -> None:
    """Idempotently commit an existing complete reference to its manifest."""
    if prompt_id in reference_done(results_dir, model, task):
        return
    ref_dir = _ref_dir(results_dir, model, task)
    safe_stem = safe_id(prompt_id)
    legacy_stem = legacy_safe_id(prompt_id)
    json_path = ref_dir / f"{safe_stem}.json"
    npy_path = ref_dir / f"{safe_stem}.npy"
    if not json_path.exists() and safe_stem != legacy_stem:
        json_path = ref_dir / f"{legacy_stem}.json"
    if not npy_path.exists() and safe_stem != legacy_stem:
        npy_path = ref_dir / f"{legacy_stem}.npy"
    if not json_path.exists() or not npy_path.exists():
        raise FileNotFoundError(
            f"cannot commit incomplete reference {prompt_id!r}"
        )
    manifest = ref_dir / "_done.jsonl"
    with manifest.open("a") as stream:
        stream.write(json.dumps({"prompt_id": prompt_id}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_reference(
    results_dir: Path,
    model: str,
    task: str,
    *,
    prompt_id: str,
    prompt_input_ids: list[int],
    gen_ids: list[int],
    text: str,
    q: float,
    features: npt.NDArray[np.float32],
    feature_names: list[str] | None = None,
    q_strict: float | None = None,
) -> None:
    """Atomically write reference files, then record in the manifest.

    ``q_strict`` is optional so references written by older sweeps remain
    loadable and retain their original schema.
    """
    ref_dir = _ref_dir(results_dir, model, task)
    ref_dir.mkdir(parents=True, exist_ok=True)

    sid = safe_id(prompt_id)
    json_path = ref_dir / f"{sid}.json"
    npy_path = ref_dir / f"{sid}.npy"

    # Write JSON atomically. feature_names describes the npy columns
    # when the superset is wider than the legacy FEATURE_NAMES order.
    record: dict[str, object] = {
        "prompt_id": prompt_id,
        "prompt_input_ids": prompt_input_ids,
        "gen_ids": gen_ids,
        "text": text,
        "q": q,
    }
    if feature_names is not None:
        record["feature_names"] = feature_names
    if q_strict is not None:
        record["q_strict"] = q_strict
    payload = json.dumps(record).encode()
    with tempfile.NamedTemporaryFile(delete=False, dir=ref_dir) as tf:
        tf.write(payload)
        tf.flush()
        os.fsync(tf.fileno())
        json_tmp = tf.name
    os.replace(json_tmp, json_path)

    # Write NPY atomically via a file object so np.save does not
    # append ".npy" to the name (it only does that with string paths).
    f16: npt.NDArray[np.float16] = features.astype(np.float16)
    with tempfile.NamedTemporaryFile(delete=False, dir=ref_dir) as tf:
        np.save(tf, f16)
        tf.flush()
        os.fsync(tf.fileno())
        npy_tmp = tf.name
    os.replace(npy_tmp, npy_path)

    # Commit: append to manifest only after both renames succeeded.
    manifest = ref_dir / "_done.jsonl"
    line = json.dumps({"prompt_id": prompt_id}) + "\n"
    with manifest.open("a") as mf:
        mf.write(line)
        mf.flush()
        os.fsync(mf.fileno())


def load_reference(
    results_dir: Path,
    model: str,
    task: str,
    prompt_id: str,
) -> dict[str, object]:
    """Load JSON metadata for a reference (does not load .npy features).

    Raises FileNotFoundError if the JSON file is missing.
    """
    ref_dir = _ref_dir(results_dir, model, task)
    json_path = ref_dir / f"{safe_id(prompt_id)}.json"
    if not json_path.exists():
        legacy_path = ref_dir / f"{legacy_safe_id(prompt_id)}.json"
        if legacy_path.exists():
            json_path = legacy_path
    if not json_path.exists():
        raise FileNotFoundError(f"Reference not found: {json_path}")
    with json_path.open() as f:
        data: dict[str, object] = json.load(f)
        return data


def hybrid_done(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    *,
    require_features: bool = False,
) -> set[tuple[str, int]]:
    """Return distinct valid (prompt_id, s) pairs in one hybrid shard."""
    return set(
        hybrid_record_keys(
            results_dir,
            model,
            task,
            compressor,
            ratio,
            require_features=require_features,
        )
    )


def hybrid_record_keys(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    *,
    require_features: bool = False,
) -> list[tuple[str, int]]:
    """Return every valid hybrid key, preserving duplicate records.

    Tolerates a torn or invalid final line by skipping it. When
    ``require_features`` is true, only keys with their canonical feature
    artifact are returned.
    """
    shard = _hybrid_dir(results_dir, model, task) / _shard_name(
        compressor, ratio
    )
    if not shard.exists():
        return []
    records: list[tuple[str, int]] = []
    with shard.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                prompt_id = str(rec["prompt_id"])
                s = int(rec["s"])
                if require_features:
                    if "features_path" not in rec:
                        continue
                    fpath = hybrid_feature_path(
                        results_dir,
                        model,
                        task,
                        compressor,
                        ratio,
                        prompt_id,
                        s,
                    )
                    stored = Path(str(rec["features_path"]))
                    stored_path = (
                        stored
                        if stored.is_absolute()
                        else (results_dir / stored)
                    )
                    if stored_path != fpath or not fpath.exists():
                        continue
                records.append((prompt_id, s))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return records


def save_hybrid_features(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    *,
    prompt_id: str,
    s: int,
    features: npt.NDArray[np.float32],
) -> Path:
    """Atomically write compressed-stream features for one hybrid."""
    feat_dir = _hybrid_feature_dir(
        results_dir, model, task, compressor, ratio
    )
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_path = hybrid_feature_path(
        results_dir, model, task, compressor, ratio, prompt_id, s
    )
    f16: npt.NDArray[np.float16] = features.astype(np.float16)
    with tempfile.NamedTemporaryFile(delete=False, dir=feat_dir) as tf:
        np.save(tf, f16)
        tf.flush()
        os.fsync(tf.fileno())
        tmp = tf.name
    os.replace(tmp, out_path)
    return out_path


def append_hybrid(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    *,
    prompt_id: str,
    s: int,
    new_ids: list[int],
    text: str,
    q: float,
    dq: float,
    features: npt.NDArray[np.float32] | None = None,
    press_features: dict[str, float] | None = None,
    q_control: float | None = None,
    q_control_strict: float | None = None,
    q_hybrid_strict: float | None = None,
    dq_strict: float | None = None,
    intervention_semantics: str | None = None,
    generation_provenance: dict[str, object] | None = None,
) -> None:
    """Append one JSON line to a shard, preserving legacy optional fields."""
    hyb_dir = _hybrid_dir(results_dir, model, task)
    hyb_dir.mkdir(parents=True, exist_ok=True)

    shard = hyb_dir / _shard_name(compressor, ratio)
    rec: dict[str, object] = {
        "prompt_id": prompt_id,
        "s": s,
        "new_ids": new_ids,
        "text": text,
        "q": q,
        "dq": dq,
    }
    if q_control is not None:
        rec["q_control"] = q_control
    if q_control_strict is not None:
        rec["q_control_strict"] = q_control_strict
    if q_hybrid_strict is not None:
        rec["q_hybrid_strict"] = q_hybrid_strict
    if dq_strict is not None:
        rec["dq_strict"] = dq_strict
    if intervention_semantics is not None:
        rec["intervention_semantics"] = intervention_semantics
    if generation_provenance is not None:
        rec["generation_provenance"] = generation_provenance
    if press_features is not None:
        rec["press_features"] = press_features
    if features is not None:
        feat_path = save_hybrid_features(
            results_dir,
            model,
            task,
            compressor,
            ratio,
            prompt_id=prompt_id,
            s=s,
            features=features,
        )
        rec["features_path"] = str(feat_path.relative_to(results_dir))
    record = json.dumps(rec)
    with shard.open("a") as f:
        f.write(record + "\n")
        f.flush()
        os.fsync(f.fileno())


# Sensor replay sidecars are intentionally separate from outcome JSONL shards.
_SENSOR_REQUIRED_KEYS = frozenset(
    {
        "prompt_id",
        "compressor",
        "ratio",
        "s",
        "sensors",
        "prefix_hash",
        "protocol_version",
        "feature_names",
        "evidence_sha256",
        "lock_sha256",
        "sensor_lock_sha256",
        "model_key",
        "task",
    }
)


def sensor_sidecar_path(output_root: Path, model: str, task: str) -> Path:
    """Return the isolated JSONL path used by label-free replay."""
    return (
        output_root
        / "sensor_sidecars"
        / f"{safe_id(model)}__{safe_id(task)}.jsonl"
    )


def sensor_manifest_path(output_root: Path, model: str, task: str) -> Path:
    return sensor_sidecar_path(output_root, model, task).with_suffix(
        ".manifest.json"
    )


def _sensor_key(
    record: Mapping[str, object],
) -> tuple[str, str, float, int]:
    try:
        prompt_id = str(record["prompt_id"])
        compressor = str(record["compressor"])
        ratio = float(cast(float, record["ratio"]))
        position = int(cast(int, record["s"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("sensor record has an invalid exact key") from error
    if (
        not prompt_id
        or not compressor
        or not np.isfinite(ratio)
        or position < 0
    ):
        raise ValueError("sensor record has an invalid exact key")
    return prompt_id, compressor, ratio, position


def read_sensor_records(
    output_root: Path,
    model: str,
    task: str,
) -> list[dict[str, object]]:
    """Read complete records, repairing only a torn final append."""
    path = sensor_sidecar_path(output_root, model, task)
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    keys: set[tuple[str, str, float, int]] = set()
    with path.open("r+b") as stream:
        data = stream.read()
        if data and not data.endswith(b"\n"):
            last_newline = data.rfind(b"\n")
            end = last_newline + 1
            stream.seek(end)
            stream.truncate()
            data = data[:end]
        for line_number, line in enumerate(data.splitlines(), start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid complete sensor JSON at line {line_number}"
                ) from error
            if not isinstance(raw, dict):
                raise ValueError(
                    f"sensor line {line_number} is not an object"
                )
            record = cast(dict[str, object], raw)
            missing = _SENSOR_REQUIRED_KEYS.difference(record)
            if missing:
                raise ValueError(
                    f"sensor line {line_number} missing fields: "
                    f"{sorted(missing)}"
                )
            if record["model_key"] != model or record["task"] != task:
                raise ValueError(
                    f"sensor line {line_number} has wrong model/task"
                )
            key = _sensor_key(record)
            if key in keys:
                raise ValueError(f"duplicate sensor record key: {key}")
            keys.add(key)
            records.append(record)
    return records


def validate_sensor_records(
    records: Sequence[Mapping[str, object]],
    *,
    model: str,
    task: str,
    evidence_sha256: str,
    lock_sha256: str,
    sensor_lock_sha256: str,
    protocol_version: str,
    feature_names: Sequence[str],
) -> set[tuple[str, str, float, int]]:
    """Validate provenance, schema, finiteness, and prefix agreement."""
    expected_features = tuple(str(name) for name in feature_names)
    if not expected_features or len(expected_features) != len(
        set(expected_features)
    ):
        raise ValueError("sensor feature schema is empty or duplicated")
    expected_feature_set = set(expected_features)
    prefixes: dict[tuple[str, int], str] = {}
    keys: set[tuple[str, str, float, int]] = set()
    for record in records:
        if (
            record.get("model_key") != model
            or record.get("task") != task
            or record.get("evidence_sha256") != evidence_sha256
            or record.get("lock_sha256") != lock_sha256
            or record.get("sensor_lock_sha256") != sensor_lock_sha256
            or record.get("protocol_version") != protocol_version
        ):
            raise ValueError("sensor record provenance mismatch")
        names = record.get("feature_names")
        if not isinstance(names, list) or tuple(map(str, names)) != (
            expected_features
        ):
            raise ValueError("sensor record feature order mismatch")
        sensors = record.get("sensors")
        if not isinstance(sensors, Mapping) or set(sensors) != (
            expected_feature_set
        ):
            raise ValueError("sensor record feature set mismatch")
        try:
            values = np.asarray(
                [float(sensors[name]) for name in expected_features],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "sensor record contains nonnumeric values"
            ) from error
        if not np.isfinite(values).all():
            raise ValueError("sensor record contains nonfinite values")
        prefix_hash = record.get("prefix_hash")
        if (
            not isinstance(prefix_hash, str)
            or len(prefix_hash) != 64
            or any(char not in "0123456789abcdef" for char in prefix_hash)
        ):
            raise ValueError("sensor record prefix hash is invalid")
        key = _sensor_key(record)
        if key in keys:
            raise ValueError(f"duplicate sensor record key: {key}")
        keys.add(key)
        boundary = (key[0], key[3])
        previous_hash = prefixes.setdefault(boundary, prefix_hash)
        if previous_hash != prefix_hash:
            raise ValueError(
                f"sensor records disagree on prefix at {boundary}"
            )
    return keys


def sensor_record_keys(
    output_root: Path,
    model: str,
    task: str,
) -> set[tuple[str, str, float, int]]:
    """Read exact committed sidecar keys, repairing a torn final append."""
    return {
        _sensor_key(record)
        for record in read_sensor_records(output_root, model, task)
    }


def append_sensor_record(
    output_root: Path,
    model: str,
    task: str,
    record: dict[str, object],
    *,
    existing_keys: set[tuple[str, str, float, int]] | None = None,
) -> None:
    """Atomically append one exact-keyed sensor record, idempotently."""
    missing = _SENSOR_REQUIRED_KEYS.difference(record)
    if missing:
        raise ValueError(f"sensor record missing fields: {sorted(missing)}")
    if record["model_key"] != model or record["task"] != task:
        raise ValueError("sensor record provenance does not match sidecar")
    key = _sensor_key(record)
    path = sensor_sidecar_path(output_root, model, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    known = (
        sensor_record_keys(output_root, model, task)
        if existing_keys is None
        else existing_keys
    )
    if key in known:
        return
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if existing_keys is not None:
        existing_keys.add(key)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sensor_manifest(
    output_root: Path,
    model: str,
    task: str,
    *,
    expected_keys: set[tuple[str, str, float, int]],
    evidence_sha256: str,
    lock_sha256: str,
    sensor_lock_sha256: str,
    protocol_version: str,
    feature_names: Sequence[str],
) -> Path:
    """Commit a content-bound manifest only for exact valid coverage."""
    records = read_sensor_records(output_root, model, task)
    observed = validate_sensor_records(
        records,
        model=model,
        task=task,
        evidence_sha256=evidence_sha256,
        lock_sha256=lock_sha256,
        sensor_lock_sha256=sensor_lock_sha256,
        protocol_version=protocol_version,
        feature_names=feature_names,
    )
    if observed != expected_keys:
        missing = sorted(expected_keys - observed)
        extra = sorted(observed - expected_keys)
        raise ValueError(
            f"incomplete sensor coverage "
            f"(missing={missing[:3]}, extra={extra[:3]})"
        )
    record_keys = "\n".join(
        "\0".join(map(str, key)) for key in sorted(observed)
    )
    feature_payload = json.dumps(
        list(feature_names),
        separators=(",", ":"),
    )
    sidecar = sensor_sidecar_path(output_root, model, task)
    manifest = {
        "model": model,
        "task": task,
        "protocol_version": protocol_version,
        "evidence_sha256": evidence_sha256,
        "lock_sha256": lock_sha256,
        "sensor_lock_sha256": sensor_lock_sha256,
        "feature_names": list(feature_names),
        "feature_names_sha256": hashlib.sha256(
            feature_payload.encode()
        ).hexdigest(),
        "record_count": len(observed),
        "record_keys_sha256": hashlib.sha256(
            record_keys.encode()
        ).hexdigest(),
        "sidecar_sha256": _file_sha256(sidecar),
    }
    path = sensor_manifest_path(output_root, model, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path
