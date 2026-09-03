"""Safe, label-blind audit helpers for the H5-R0 AttentionTap pilot.

This module deliberately contains no model execution.  The parser and audit
functions are usable on a CPU and never need to inspect decoded reference text
or outcome-bearing fields.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from herald.attention_features import tap_feature_names
from herald.features import FEATURE_NAMES

PROTOCOL_SCHEMA_VERSION = "herald.attention_tap_pilot_protocol_lock.v1"
PROTOCOL_STATUS = "locked_before_h5_r0_gpu"
PILOT_ROSTER: tuple[tuple[int, str], ...] = (
    (0, "ifeval-168"),
    (1, "ifeval-179"),
    (2, "ifeval-163"),
    (3, "ifeval-1773"),
    (4, "ifeval-2035"),
)
TAP_LAYER_INDICES: tuple[int, ...] = (8, 16, 24)


class AttentionReplayError(ValueError):
    """Raised when a label-blind pilot input or audit is invalid."""


@dataclass(frozen=True, slots=True)
class LegacyIdentity:
    """The identity prefix of one immutable reference."""

    prompt_id: str
    prompt_input_ids: tuple[int, ...]
    gen_ids: tuple[int, ...]
    path: Path


def sha256_file(path: Path) -> str:
    """Hash a file without decoding or materializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_prompt_ids(prompt_ids: Sequence[str]) -> str:
    """Return the canonical sorted-ID hash used by locked protocols."""
    values = sorted(str(value) for value in prompt_ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _read_byte(stream: Any) -> bytes:
    value: bytes = stream.read(1)
    if value == b"":
        raise AttentionReplayError(
            "legacy identity prefix ended unexpectedly"
        )
    return value


def _skip_space(stream: Any) -> bytes:
    value = _read_byte(stream)
    while value in b" \t\r\n":
        value = _read_byte(stream)
    return value


def _read_json_value(stream: Any, first: bytes) -> object:
    """Read exactly one JSON value, stopping at its final byte."""
    decoder = json.JSONDecoder()
    raw = bytearray(first)
    while True:
        try:
            value, end = decoder.raw_decode(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raw.extend(_read_byte(stream))
            continue
        if raw[end:].strip():
            raise AttentionReplayError(
                "legacy identity value has trailing bytes"
            )
        return value


def _expect(stream: Any, expected: bytes) -> None:
    value = _skip_space(stream)
    if value != expected:
        raise AttentionReplayError(
            f"legacy identity prefix expected {expected!r}, got {value!r}"
        )


def _read_field(stream: Any, expected_name: str) -> object:
    first = _skip_space(stream)
    value = _read_json_value(stream, first)
    if value != expected_name:
        raise AttentionReplayError(
            f"legacy identity field is not {expected_name!r}"
        )
    _expect(stream, b":")
    return _read_json_value(stream, _skip_space(stream))


def _tokens(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise AttentionReplayError(f"legacy {field} must be a non-empty list")
    result: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise AttentionReplayError(
                f"legacy {field} contains invalid token"
            )
        result.append(token)
    return tuple(result)


def read_legacy_identity_prefix(
    path: Path, prompt_id: str | None = None
) -> LegacyIdentity:
    """Read only ``prompt_id``, ``prompt_input_ids`` and ``gen_ids``.

    The stream is closed immediately after the final byte of ``gen_ids`` is
    consumed.  In particular, the following comma and ``text`` value are not
    read or parsed, allowing an invalid or secret tail to remain inaccessible.
    """
    try:
        with path.open("rb") as stream:
            _expect(stream, b"{")
            raw_prompt = _read_field(stream, "prompt_id")
            if not isinstance(raw_prompt, str):
                raise AttentionReplayError("legacy prompt_id is not a string")
            _expect(stream, b",")
            raw_prompt_ids = _read_field(stream, "prompt_input_ids")
            _expect(stream, b",")
            raw_gen_ids = _read_field(stream, "gen_ids")
    except OSError as error:
        raise AttentionReplayError(
            f"cannot read legacy reference {path}"
        ) from error
    if prompt_id is not None and raw_prompt != prompt_id:
        raise AttentionReplayError(
            "legacy prompt_id does not match requested ID"
        )
    return LegacyIdentity(
        raw_prompt,
        _tokens(raw_prompt_ids, "prompt_input_ids"),
        _tokens(raw_gen_ids, "gen_ids"),
        path,
    )


def prefix_hash(
    prompt_input_ids: Sequence[int], gen_ids: Sequence[int], position: int
) -> str:
    """Hash one canonical int64 prompt-plus-generation prefix.

    This is intentionally the same representation as ``sensor_replay_m3``:
    ``np.asarray(prefix, dtype=np.int64).tobytes()``.
    """
    if position < 0 or position > len(gen_ids):
        raise AttentionReplayError(
            f"prefix position is outside generation: {position}"
        )
    prefix = tuple(int(value) for value in prompt_input_ids) + tuple(
        int(value) for value in gen_ids[:position]
    )
    if not prefix:
        raise AttentionReplayError("prefix cannot be empty")
    if any(value < 0 for value in prefix):
        raise AttentionReplayError("prefix contains a negative token")
    tokens = np.asarray(prefix, dtype=np.int64)
    return hashlib.sha256(tokens.tobytes()).hexdigest()


_FORBIDDEN_SIDECAR_FIELD = re.compile(
    r'"(?:text|q|dq|gold|grader|output|response|prompt|instruction|'
    r'score|label|target)"\s*:',
    flags=re.IGNORECASE,
)
_ESCAPED_SIDECAR_FIELD = re.compile(
    r'"(?:\\.|[^"\\])*\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})'
    r'(?:\\.|[^"\\])*"\s*:'
)
_SIDECAR_FIELDS = frozenset(
    {
        "compressor",
        "evidence_sha256",
        "feature_names",
        "lock_sha256",
        "model_key",
        "prefix_hash",
        "prompt_id",
        "protocol_version",
        "ratio",
        "s",
        "sensor_lock_sha256",
        "sensors",
        "state_semantics",
        "task",
    }
)


def _selected_sidecar_identity(
    raw: object, selected: set[str], line_number: int
) -> tuple[str, int, str]:
    if not isinstance(raw, Mapping) or set(raw) != _SIDECAR_FIELDS:
        raise AttentionReplayError(
            f"selected sidecar schema is invalid on line {line_number}"
        )
    prompt_id = raw["prompt_id"]
    if not isinstance(prompt_id, str) or prompt_id not in selected:
        raise AttentionReplayError(
            f"selected sidecar line {line_number} has invalid prompt ID"
        )
    position = raw["s"]
    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or position < 0
    ):
        raise AttentionReplayError(
            f"invalid sidecar position on line {line_number}"
        )
    digest = raw["prefix_hash"]
    if not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None:
        raise AttentionReplayError(
            f"invalid sidecar hash on line {line_number}"
        )
    for key in (
        "compressor",
        "evidence_sha256",
        "lock_sha256",
        "model_key",
        "protocol_version",
        "sensor_lock_sha256",
        "state_semantics",
        "task",
    ):
        if not isinstance(raw[key], str):
            raise AttentionReplayError(
                f"selected sidecar schema is invalid on line {line_number}"
            )
    for key in ("evidence_sha256", "lock_sha256", "sensor_lock_sha256"):
        value = cast(str, raw[key])
        if _HASH_RE.fullmatch(value) is None:
            raise AttentionReplayError(
                f"selected sidecar schema is invalid on line {line_number}"
            )
    feature_names = raw["feature_names"]
    sensors = raw["sensors"]
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or any(not isinstance(name, str) for name in feature_names)
        or len(feature_names) != len(set(feature_names))
        or not isinstance(sensors, Mapping)
        or set(sensors) != set(feature_names)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in sensors.values()
        )
    ):
        raise AttentionReplayError(
            f"selected sidecar sensors are invalid on line {line_number}"
        )
    ratio = raw["ratio"]
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio)
        or ratio <= 0
        or ratio > 1
    ):
        raise AttentionReplayError(
            f"selected sidecar ratio is invalid on line {line_number}"
        )
    if raw["model_key"] != "llama" or raw["task"] != "ifeval":
        raise AttentionReplayError(
            f"selected sidecar identity is invalid on line {line_number}"
        )
    return prompt_id, position, digest


def load_prefix_hash_sidecar(
    path: Path, selected_ids: Sequence[str]
) -> dict[tuple[str, int], str]:
    """Load selected safe records without parsing nonselected payloads."""
    selected_values = tuple(selected_ids)
    if any(not isinstance(prompt_id, str) for prompt_id in selected_values):
        raise AttentionReplayError("selected sidecar IDs must be strings")
    selected = set(selected_values)
    if len(selected) != len(selected_values):
        raise AttentionReplayError("selected sidecar IDs contain duplicates")
    values: dict[tuple[str, int], str] = {}
    seen_ids: set[str] = set()
    try:
        stream = path.open()
    except OSError as error:
        raise AttentionReplayError(
            f"cannot read prefix sidecar {path}"
        ) from error
    with stream:
        for line_number, line in enumerate(stream, 1):
            matching_ids = [
                prompt_id
                for prompt_id in selected
                if re.search(
                    rf'"prompt_id"\s*:\s*{re.escape(json.dumps(prompt_id))}',
                    line,
                )
            ]
            if not matching_ids:
                continue
            if _ESCAPED_SIDECAR_FIELD.search(
                line
            ) or _FORBIDDEN_SIDECAR_FIELD.search(line):
                raise AttentionReplayError(
                    f"forbidden field on selected sidecar line {line_number}"
                )
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise AttentionReplayError(
                    f"malformed selected sidecar line {line_number}"
                ) from error
            prompt_id, position, digest = _selected_sidecar_identity(
                raw, selected, line_number
            )
            seen_ids.add(prompt_id)
            key = (prompt_id, position)
            prior = values.setdefault(key, digest)
            if prior != digest:
                raise AttentionReplayError(
                    f"conflicting sidecar hash for {key}"
                )
    missing = selected - seen_ids
    if missing:
        raise AttentionReplayError(
            f"sidecar has no boundary for {sorted(missing)!r}"
        )
    return values


def validate_hashes(
    paths: Mapping[str, Path], expected: Mapping[str, str]
) -> dict[str, str]:
    """Validate an immutable name-to-path hash map exactly."""
    if set(paths) != set(expected):
        raise AttentionReplayError(
            "protocol hash names do not match input paths"
        )
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != dict(expected):
        raise AttentionReplayError("immutable protocol input hash mismatch")
    return actual


def validate_protocol_lock(
    lock: Mapping[str, object],
    *,
    schema: str = PROTOCOL_SCHEMA_VERSION,
    status: str = PROTOCOL_STATUS,
) -> None:
    """Validate the exact immutable protocol-lock shape."""
    required = {
        "schema_version",
        "status",
        "pilot_roster",
        "pilot_prompt_ids_sha256",
        "generation",
        "paths",
        "input_hashes",
        "access_policy",
        "hard_gates",
    }
    if set(lock) != required:
        raise AttentionReplayError("protocol lock keys are not exact")
    if lock.get("schema_version") != schema or lock.get("status") != status:
        raise AttentionReplayError("protocol lock schema/status is invalid")
    roster = lock["pilot_roster"]
    if not isinstance(roster, list) or len(roster) != len(PILOT_ROSTER):
        raise AttentionReplayError("protocol lock pilot roster is malformed")
    parsed_roster: list[tuple[int, str]] = []
    for item in roster:
        if not isinstance(item, Mapping) or set(item) != {
            "fold",
            "prompt_id",
        }:
            raise AttentionReplayError(
                "protocol lock pilot roster is malformed"
            )
        fold = item["fold"]
        prompt_id = item["prompt_id"]
        if (
            isinstance(fold, bool)
            or not isinstance(fold, int)
            or not isinstance(prompt_id, str)
        ):
            raise AttentionReplayError(
                "protocol lock pilot roster is malformed"
            )
        parsed_roster.append((fold, prompt_id))
    if tuple(parsed_roster) != PILOT_ROSTER:
        raise AttentionReplayError("protocol lock pilot roster is not exact")
    prompt_ids_hash = lock["pilot_prompt_ids_sha256"]
    if not isinstance(
        prompt_ids_hash, str
    ) or prompt_ids_hash != hash_prompt_ids(
        [prompt_id for _, prompt_id in PILOT_ROSTER]
    ):
        raise AttentionReplayError("pilot prompt-ID hash does not match lock")
    generation = lock["generation"]
    generation_keys = {
        "model_key",
        "model_id",
        "dtype",
        "device",
        "attn_implementation",
        "batch_size",
        "max_new_tokens",
        "tap_layer_indices",
    }
    if (
        not isinstance(generation, Mapping)
        or set(generation) != generation_keys
    ):
        raise AttentionReplayError(
            "protocol lock generation keys are not exact"
        )
    for key in (
        "model_key",
        "model_id",
        "dtype",
        "device",
        "attn_implementation",
    ):
        if not isinstance(generation[key], str):
            raise AttentionReplayError(
                "protocol lock generation value is malformed"
            )
    for key in ("batch_size", "max_new_tokens"):
        if isinstance(generation[key], bool) or not isinstance(
            generation[key], int
        ):
            raise AttentionReplayError(
                "protocol lock generation value is malformed"
            )
    layers = generation["tap_layer_indices"]
    if not isinstance(layers, list) or any(
        isinstance(layer, bool) or not isinstance(layer, int)
        for layer in layers
    ):
        raise AttentionReplayError("protocol lock tap layers are malformed")
    paths = lock["paths"]
    path_keys = {
        "attention_source",
        "generate_source",
        "ifeval_source",
        "attention_replay_source",
        "features_source",
        "script_source",
        "protocol_lock",
        "source_oof",
        "legacy_config",
        "legacy_reference_dir",
        "legacy_manifest",
        "legacy_prefix_sidecar",
        "legacy_prefix_manifest",
        "output_root",
    }
    if (
        not isinstance(paths, Mapping)
        or set(paths) != path_keys
        or any(not isinstance(value, str) for value in paths.values())
    ):
        raise AttentionReplayError("protocol lock path keys are not exact")
    input_hashes = lock["input_hashes"]
    input_keys = {
        "source_oof_sha256",
        "legacy_config_sha256",
        "legacy_manifest_sha256",
        "prefix_sidecar_sha256",
        "prefix_manifest_sha256",
        "attention_source_sha256",
        "generate_source_sha256",
        "ifeval_source_sha256",
        "attention_replay_source_sha256",
        "features_source_sha256",
        "script_sha256",
    }
    if (
        not isinstance(input_hashes, Mapping)
        or set(input_hashes) != input_keys
        or any(
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in input_hashes.values()
        )
    ):
        raise AttentionReplayError(
            "protocol lock input hash keys are not exact"
        )
    access_policy = lock["access_policy"]
    expected_access = {
        "decode_text": False,
        "load_gold": False,
        "scorer": False,
        "outcomes": False,
        "protected_prompts": False,
        "resume": False,
    }
    if (
        not isinstance(access_policy, Mapping)
        or set(access_policy) != set(expected_access)
        or any(
            not isinstance(value, bool) for value in access_policy.values()
        )
        or dict(access_policy) != expected_access
    ):
        raise AttentionReplayError("protocol lock access policy is not exact")
    hard_gates = lock["hard_gates"]
    expected_gates = {
        "max_tapped_control_ratio": 2.0,
        "require_bitwise_identity": True,
        "require_tap_finite": True,
        "require_tap_nonconstant_within_prompt": True,
    }
    if (
        not isinstance(hard_gates, Mapping)
        or set(hard_gates) != set(expected_gates)
        or not isinstance(hard_gates.get("max_tapped_control_ratio"), float)
        or any(
            not isinstance(hard_gates.get(name), bool)
            for name in (
                "require_bitwise_identity",
                "require_tap_finite",
                "require_tap_nonconstant_within_prompt",
            )
        )
        or dict(hard_gates) != expected_gates
    ):
        raise AttentionReplayError("protocol lock hard gates are not exact")


def _array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def audit_arrays(
    control: np.ndarray,
    tapped: np.ndarray,
    legacy: np.ndarray,
    *,
    expected_gen_ids: Sequence[int] | None = None,
    control_gen_ids: Sequence[int] | None = None,
    tapped_gen_ids: Sequence[int] | None = None,
    expected_prompt_ids: Sequence[int] | None = None,
    control_prompt_ids: Sequence[int] | None = None,
    tapped_prompt_ids: Sequence[int] | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, object]:
    """Audit one paired prompt without inspecting text or outcome fields."""
    control = np.asarray(control)
    tapped = np.asarray(tapped)
    legacy = np.asarray(legacy)
    names = list(feature_names) if feature_names is not None else []
    expected_names = [*FEATURE_NAMES, *tap_feature_names()]
    control_shape = (
        control.dtype == np.dtype(np.float16)
        and legacy.dtype == np.dtype(np.float16)
        and control.ndim == legacy.ndim == 2
        and control.shape[1:] == (len(FEATURE_NAMES),)
        and legacy.shape == control.shape
    )
    tapped_shape_names = (
        control.ndim == 2
        and tapped.dtype == np.dtype(np.float16)
        and tapped.ndim == 2
        and tapped.shape
        == (control.shape[0], len(FEATURE_NAMES) + len(tap_feature_names()))
        and names == expected_names
    )
    shapes_names = control_shape and tapped_shape_names
    control_legacy = control_shape and _array_equal(control, legacy)
    tapped_control = (
        tapped_shape_names
        and control.ndim == 2
        and _array_equal(tapped[:, : len(FEATURE_NAMES)], control)
    )
    control_gen_match = (
        expected_gen_ids is not None
        and control_gen_ids is not None
        and list(expected_gen_ids) == list(control_gen_ids)
    )
    tap_gen_match = (
        control_gen_ids is not None
        and tapped_gen_ids is not None
        and list(control_gen_ids) == list(tapped_gen_ids)
    )
    control_prompt_match = (
        expected_prompt_ids is not None
        and control_prompt_ids is not None
        and list(expected_prompt_ids) == list(control_prompt_ids)
    )
    tap_prompt_match = (
        control_prompt_ids is not None
        and tapped_prompt_ids is not None
        and list(control_prompt_ids) == list(tapped_prompt_ids)
    )
    tap = (
        tapped[:, len(FEATURE_NAMES) :]
        if tapped.ndim == 2 and tapped.shape[1] >= len(FEATURE_NAMES)
        else np.empty((0, 0))
    )
    finite = bool(np.isfinite(tap).all()) if tap.size else False
    nonconstant = (
        bool(any(np.unique(tap[:, i]).size >= 2 for i in range(tap.shape[1])))
        if tap.size
        else False
    )
    return {
        "control_legacy_match": bool(control_legacy),
        "tapped_control_match": bool(tapped_control),
        "control_gen_ids_match": bool(control_gen_match),
        "tap_gen_ids_match": bool(tap_gen_match),
        "control_prompt_ids_match": bool(control_prompt_match),
        "tap_prompt_ids_match": bool(tap_prompt_match),
        "shapes_names_match": bool(shapes_names),
        "tap_finite": finite,
        "tap_nonconstant": nonconstant,
    }


def classify_decision(
    audits: Sequence[Mapping[str, object]],
    *,
    aggregate_cost_ok: bool,
) -> str:
    """Classify only after all explicit prompt-level gates are available."""
    if not audits:
        raise AttentionReplayError("cannot classify an empty pilot")
    if not isinstance(aggregate_cost_ok, bool):
        raise AttentionReplayError("aggregate cost gate must be a boolean")
    required = {
        "control_legacy_match",
        "control_gen_ids_match",
        "tap_gen_ids_match",
        "control_prompt_ids_match",
        "tap_prompt_ids_match",
        "prefix_hashes_match",
        "tapped_control_match",
        "shapes_names_match",
        "tap_finite",
        "tap_nonconstant",
        "no_oom",
        "cost_ok",
    }
    for audit in audits:
        if not required <= set(audit):
            missing = sorted(required - set(audit))
            raise AttentionReplayError(
                f"prompt audit is missing fields: {missing}"
            )
        if any(not isinstance(audit[key], bool) for key in required):
            raise AttentionReplayError(
                "prompt audit gate fields must be booleans"
            )
    if any(not audit["no_oom"] for audit in audits):
        return "retire_attention_tap"
    control_ok = all(
        bool(audit["control_legacy_match"])
        and bool(audit["control_gen_ids_match"])
        and bool(audit["control_prompt_ids_match"])
        and bool(audit["prefix_hashes_match"])
        and bool(audit["no_oom"])
        for audit in audits
    )
    tap_ok = all(
        bool(audit["tapped_control_match"])
        and bool(audit["tap_gen_ids_match"])
        and bool(audit["tap_prompt_ids_match"])
        and bool(audit["shapes_names_match"])
        and bool(audit["tap_finite"])
        and bool(audit["tap_nonconstant"])
        and bool(audit["no_oom"])
        and bool(audit["cost_ok"])
        for audit in audits
    )
    if not control_ok:
        return "inconclusive_environment_mismatch"
    if not tap_ok or not aggregate_cost_ok:
        return "retire_attention_tap"
    return "license_full_label_blind_replay"


_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "decision",
        "pilot_count",
        "pilot_prompt_ids_sha256",
        "feature_names_sha256",
        "tap_layers",
        "tap_feature_count",
        "control_time_seconds",
        "tapped_time_seconds",
        "tapped_control_time_ratio",
        "aggregate_cost_ok",
        "prompt_audits",
        "no_oom",
        "input_hashes",
        "protocol_lock_sha256",
    }
)
_AUDIT_KEYS = frozenset(
    {
        "fold",
        "control_rows",
        "tapped_rows",
        "control_time_seconds",
        "tapped_time_seconds",
        "control_peak_allocated_bytes",
        "tapped_peak_allocated_bytes",
        "control_legacy_match",
        "tapped_control_match",
        "control_gen_ids_match",
        "tap_gen_ids_match",
        "control_prompt_ids_match",
        "tap_prompt_ids_match",
        "prefix_hashes_match",
        "shapes_names_match",
        "tap_finite",
        "tap_nonconstant",
        "no_oom",
        "cost_ok",
        "legacy_prompt_ids_sha256",
        "legacy_gen_ids_sha256",
        "control_prompt_ids_sha256",
        "control_gen_ids_sha256",
        "tapped_prompt_ids_sha256",
        "tapped_gen_ids_sha256",
        "control_array_sha256",
        "tapped_array_sha256",
    }
)
_REPORT_INPUT_HASH_KEYS = frozenset(
    {
        "source_oof_sha256",
        "legacy_config_sha256",
        "legacy_manifest_sha256",
        "prefix_sidecar_sha256",
        "prefix_manifest_sha256",
        "attention_source_sha256",
        "generate_source_sha256",
        "ifeval_source_sha256",
        "attention_replay_source_sha256",
        "features_source_sha256",
        "script_sha256",
    }
)
_REPORT_REQUIRED = _REPORT_KEYS
_AUDIT_REQUIRED = _AUDIT_KEYS
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _report_value(value: object, *, key: str, nested: bool = False) -> object:
    if isinstance(value, Mapping):
        if key == "prompt_audits":
            if not value or any(
                not isinstance(raw_key, str)
                or re.fullmatch(r"fold_\d+", raw_key) is None
                or not isinstance(raw_value, Mapping)
                or set(raw_value) != _AUDIT_REQUIRED
                for raw_key, raw_value in value.items()
            ):
                raise AttentionReplayError("prompt audits are not exact")
            return {
                raw_key: _report_value(raw_value, key=raw_key, nested=True)
                for raw_key, raw_value in value.items()
            }
        if key == "input_hashes":
            if set(value) != _REPORT_INPUT_HASH_KEYS or any(
                not isinstance(raw_key, str)
                or not isinstance(raw_value, str)
                or _HASH_RE.fullmatch(raw_value) is None
                for raw_key, raw_value in value.items()
            ):
                raise AttentionReplayError("input hashes are malformed")
            return dict(value)
        allowed = _AUDIT_KEYS if nested else _REPORT_KEYS
        unknown = set(value) - allowed
        if unknown or (nested and set(value) != _AUDIT_REQUIRED):
            raise AttentionReplayError(
                f"unexpected report keys: {sorted(unknown)}"
            )
        if key == "report" and set(value) != _REPORT_REQUIRED:
            raise AttentionReplayError("report keys are not exact")
        return {
            raw_key: _report_value(raw_value, key=raw_key, nested=nested)
            for raw_key, raw_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_report_value(item, key=key) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AttentionReplayError("report contains a non-finite number")
        return value
    if isinstance(value, str):
        if key in {"decision", "schema_version"}:
            return value
        if key.endswith("_sha256") and _HASH_RE.fullmatch(value):
            return value
        raise AttentionReplayError(
            f"unexpected string-bearing report field: {key}"
        )
    raise AttentionReplayError(
        f"unsupported report value: {type(value).__name__}"
    )


def validate_report(report: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact label-blind report schema and gate consistency."""
    if not isinstance(report, Mapping):
        raise AttentionReplayError("report must be a JSON object")
    clean = _report_value(report, key="report")
    if not isinstance(clean, dict):
        raise AttentionReplayError("report must be a JSON object")
    if clean.get("schema_version") != "herald.attention_tap_pilot_report.v1":
        raise AttentionReplayError("report schema version is invalid")
    expected_names_hash = hashlib.sha256(
        json.dumps(
            [*FEATURE_NAMES, *tap_feature_names()], separators=(",", ":")
        ).encode()
    ).hexdigest()
    if (
        clean.get("pilot_count") != len(PILOT_ROSTER)
        or clean.get("pilot_prompt_ids_sha256")
        != hash_prompt_ids([prompt_id for _, prompt_id in PILOT_ROSTER])
        or clean.get("feature_names_sha256") != expected_names_hash
        or clean.get("tap_layers") != list(TAP_LAYER_INDICES)
        or clean.get("tap_feature_count") != len(tap_feature_names())
    ):
        raise AttentionReplayError(
            "report frozen identity fields are invalid"
        )
    prompt_audits = clean.get("prompt_audits")
    if not isinstance(prompt_audits, dict) or set(prompt_audits) != {
        f"fold_{fold}" for fold, _ in PILOT_ROSTER
    }:
        raise AttentionReplayError("report prompt roster is invalid")
    audits: list[Mapping[str, object]] = []
    for fold, _ in PILOT_ROSTER:
        audit = prompt_audits[f"fold_{fold}"]
        if not isinstance(audit, dict) or audit.get("fold") != fold:
            raise AttentionReplayError("report prompt fold is invalid")
        for key in ("control_rows", "tapped_rows"):
            value = audit[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise AttentionReplayError(f"report {key} is invalid")
        for key in (
            "control_peak_allocated_bytes",
            "tapped_peak_allocated_bytes",
        ):
            value = audit[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise AttentionReplayError(f"report {key} is invalid")
        for key in ("control_time_seconds", "tapped_time_seconds"):
            value = audit[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise AttentionReplayError(f"report {key} is invalid")
        for key in (
            "control_legacy_match",
            "tapped_control_match",
            "control_gen_ids_match",
            "tap_gen_ids_match",
            "control_prompt_ids_match",
            "tap_prompt_ids_match",
            "prefix_hashes_match",
            "shapes_names_match",
            "tap_finite",
            "tap_nonconstant",
            "no_oom",
            "cost_ok",
        ):
            if not isinstance(audit[key], bool):
                raise AttentionReplayError(f"report {key} is invalid")
        audits.append(audit)
    for key in ("control_time_seconds", "tapped_time_seconds"):
        value = clean.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value < 0
        ):
            raise AttentionReplayError(f"report {key} is invalid")
    ratio = clean.get("tapped_control_time_ratio")
    if ratio is not None and (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or ratio < 0
    ):
        raise AttentionReplayError("report timing ratio is invalid")
    no_oom = clean.get("no_oom")
    aggregate_cost_ok = clean.get("aggregate_cost_ok")
    if not isinstance(no_oom, bool) or not isinstance(
        aggregate_cost_ok, bool
    ):
        raise AttentionReplayError("report aggregate gates are invalid")
    if not no_oom and ratio is not None:
        raise AttentionReplayError("OOM report timing ratio must be null")
    control_time = float(clean["control_time_seconds"])
    tapped_time = float(clean["tapped_time_seconds"])
    audit_control_time = math.fsum(
        float(cast(int | float, audit["control_time_seconds"]))
        for audit in audits
    )
    audit_tapped_time = math.fsum(
        float(cast(int | float, audit["tapped_time_seconds"]))
        for audit in audits
    )
    if not math.isclose(
        control_time, audit_control_time, rel_tol=1e-12, abs_tol=0.0
    ) or not math.isclose(
        tapped_time, audit_tapped_time, rel_tol=1e-12, abs_tol=0.0
    ):
        raise AttentionReplayError(
            "report aggregate timings are inconsistent"
        )
    expected_ratio = (
        tapped_time / control_time if no_oom and control_time > 0 else None
    )
    if expected_ratio is None:
        if ratio is not None:
            raise AttentionReplayError("report timing ratio must be null")
    elif ratio is None or not math.isclose(
        float(ratio), expected_ratio, rel_tol=1e-12, abs_tol=0.0
    ):
        raise AttentionReplayError("report timing ratio is inconsistent")
    expected_no_oom = all(bool(audit["no_oom"]) for audit in audits)
    expected_cost_ok = (
        expected_no_oom
        and expected_ratio is not None
        and expected_ratio <= 2.0
    )
    if (
        no_oom != expected_no_oom
        or aggregate_cost_ok != expected_cost_ok
        or any(audit["cost_ok"] != expected_cost_ok for audit in audits)
    ):
        raise AttentionReplayError("report aggregate gates are inconsistent")
    decision = clean.get("decision")
    if decision != classify_decision(
        audits, aggregate_cost_ok=expected_cost_ok
    ):
        raise AttentionReplayError("report decision is inconsistent")
    return clean


def write_report(path: Path, report: Mapping[str, object]) -> None:
    """Validate and write an exact report schema."""
    clean = validate_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def write_float16_array(path: Path, values: np.ndarray) -> None:
    """Persist a float16 audit array; legacy NaNs are permitted."""
    array = np.asarray(values)
    if array.dtype != np.dtype(np.float16) or array.ndim != 2:
        raise AttentionReplayError(
            "pilot arrays must be two-dimensional float16"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
