"""Resumable, label-free M3 layer-band replay.

This module deliberately owns the replay loop and layer-band measurement while
reusing only immutable M2 tensor primitives.  It emits no outcomes or labels.
"""

import hashlib
import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from herald.config import MODELS
from herald.generate import LoadedModel, load_model
from herald.magnitude_v2 import validate_lock_and_evidence
from herald.press_sensors import (
    COMPRESSORS,
    RATIOS,
    RECENT_WINDOW,
    SINK_TOKENS,
    SensorCaptureError,
    causal_recent_reliance,
    eviction_mask,
    layer_sensor_values,
)
from herald.press_sensors_m3 import (
    LAYER_BAND_FEATURE_NAMES,
    LAYER_BAND_PROTOCOL,
    LAYER_COUNT,
    layer_band_schema,
    layer_band_values,
)
from herald.presses import get_press
from herald.sensor_replay import (
    _attention_modules,
    _expected_attention_score,
    _position_embeddings,
    _reference_path,
    _rotated_kv,
    validate_sensor_lock,
)
from herald.storage import (
    append_sensor_record,
    read_sensor_records,
    sensor_sidecar_path,
    validate_sensor_records,
    write_sensor_manifest,
)

PROTOCOL_VERSION = LAYER_BAND_PROTOCOL
M3_LOCK_SCHEMA_VERSION = "herald.magnitude_v3_layer_band_lock.v1"
STRIDE = 16
STATE_SEMANTICS = {
    "expected_attention": "herald.matched_reprefill_v1",
    "knorm": "herald.cache_native_pending_v1",
    "streaming_llm": "herald.cache_native_pending_v1",
}


def _guard_gpu(allowed_gpu_pids: tuple[int, ...]) -> None:
    from herald.intervention_sweep import guard_gpu_contention

    guard_gpu_contention(allowed_gpu_pids)


class ReplayProtocolError(ValueError):
    """Raised when M3 replay inputs violate the frozen protocol."""


@dataclass(frozen=True, slots=True)
class ReplayReference:
    prompt_id: str
    prompt_input_ids: tuple[int, ...]
    gen_ids: tuple[int, ...]
    path: Path


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    results_root: Path
    evidence_path: Path
    lock_path: Path
    sensor_lock_path: Path
    m3_lock_path: Path
    m2_result_freeze_path: Path
    output_root: Path
    model_key: str = "llama"
    task: str = "ifeval"
    model_id: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    stride: int = STRIDE
    compressors: tuple[str, ...] = COMPRESSORS
    ratios: tuple[float, ...] = RATIOS
    allowed_gpu_pids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.stride != STRIDE:
            raise ReplayProtocolError("M3 replay is locked to stride 16")
        if tuple(self.compressors) != COMPRESSORS:
            raise ReplayProtocolError(
                f"compressors are locked to {COMPRESSORS}"
            )
        if tuple(float(ratio) for ratio in self.ratios) != RATIOS:
            raise ReplayProtocolError(f"ratios are locked to {RATIOS}")
        if self.model_key != "llama" or self.task != "ifeval":
            raise ReplayProtocolError("M3 replay is locked to llama/ifeval")
        if self.model_id not in (None, MODELS["llama"]):
            raise ReplayProtocolError("M3 replay model identity is locked")
        if self.dtype != "bfloat16" or self.attn_implementation != "sdpa":
            raise ReplayProtocolError("M3 dtype/attention backend is locked")
        if self.output_root.resolve() == self.results_root.resolve():
            raise ReplayProtocolError(
                "M3 output root must be isolated from references"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_prompt_ids_m3(
    evidence_path: Path,
    lock_path: Path,
) -> tuple[set[str], str, str]:
    """Validate the parent and expose only its 154 development IDs."""
    try:
        evidence, _ = validate_lock_and_evidence(evidence_path, lock_path)
        ids = tuple(
            str(item) for item in evidence["split"]["train_prompt_ids"]
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ReplayProtocolError(
            "unable to validate frozen evidence/lock"
        ) from error
    if len(ids) != 154 or not ids or len(set(ids)) != len(ids):
        raise ReplayProtocolError(
            "parent protocol does not expose 154 unique development IDs"
        )
    return set(ids), _sha256(evidence_path), _sha256(lock_path)


def m3_implementation_paths() -> dict[str, Path]:
    """Return every source file whose bytes define M3 replay or fitting."""
    root = Path(__file__).resolve().parents[2]
    return {
        "src/herald/config.py": root / "src/herald/config.py",
        "src/herald/generate.py": root / "src/herald/generate.py",
        "src/herald/intervention_sweep.py": (
            root / "src/herald/intervention_sweep.py"
        ),
        "src/herald/magnitude.py": root / "src/herald/magnitude.py",
        "src/herald/magnitude_v2.py": root / "src/herald/magnitude_v2.py",
        "src/herald/magnitude_v2_sensors.py": (
            root / "src/herald/magnitude_v2_sensors.py"
        ),
        "src/herald/magnitude_v3.py": root / "src/herald/magnitude_v3.py",
        "src/herald/presses.py": root / "src/herald/presses.py",
        "src/herald/press_sensors.py": root / "src/herald/press_sensors.py",
        "src/herald/sensor_replay.py": root / "src/herald/sensor_replay.py",
        "src/herald/storage.py": root / "src/herald/storage.py",
        "src/herald/press_sensors_m3.py": (
            root / "src/herald/press_sensors_m3.py"
        ),
        "src/herald/sensor_replay_m3.py": Path(__file__).resolve(),
        "scripts/compare_magnitude_v3.py": (
            root / "scripts/compare_magnitude_v3.py"
        ),
        "scripts/replay_press_sensors_m3.py": (
            root / "scripts/replay_press_sensors_m3.py"
        ),
        "uv.lock": root / "uv.lock",
    }


def validate_m3_lock(
    m3_lock_path: Path,
    evidence_path: Path,
    lock_path: Path,
    sensor_lock_path: Path,
    m2_result_freeze_path: Path,
) -> str:
    """Validate parent provenance, M3 schema, replay, and source hashes."""
    try:
        raw = json.loads(m3_lock_path.read_text())
        _, parent_lock = validate_lock_and_evidence(evidence_path, lock_path)
        validate_sensor_lock(sensor_lock_path, evidence_path, lock_path)
        parent_sensor = json.loads(sensor_lock_path.read_text())
        m2_result = json.loads(m2_result_freeze_path.read_text())
        from herald.magnitude_v3 import validate_m3_protocol_lock

        validate_m3_protocol_lock(m3_lock_path)
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise ReplayProtocolError(
            "unable to validate M3 lock and parent locks"
        ) from error
    if (
        not isinstance(raw, Mapping)
        or not isinstance(parent_sensor, Mapping)
        or not isinstance(m2_result, Mapping)
    ):
        raise ReplayProtocolError(
            "M3 lock and parent freezes must be objects"
        )
    decision = m2_result.get("development_decision")
    if (
        m2_result.get("schema_version")
        != "herald.magnitude_v2_m2_result_freeze.v1"
        or m2_result.get("status") != "frozen_before_m3_layer_sensor_results"
        or not isinstance(decision, Mapping)
        or decision.get("any_compressor_pass") is not False
        or decision.get("passing_compressors") != []
    ):
        raise ReplayProtocolError("M2 result freeze does not authorize M3")
    try:
        parent_sensor_hash = _sha256(sensor_lock_path)
        expected_parent_protocol = _sha256(lock_path)
        expected_evidence = _sha256(evidence_path)
        expected_m2_result = _sha256(m2_result_freeze_path)
        dev_hash = str(
            parent_lock["prompt_partitions"]["development_prompt_ids_sha256"]
        )
    except (KeyError, TypeError) as error:
        raise ReplayProtocolError(
            "parent lock partition is incomplete"
        ) from error
    if (
        raw.get("schema_version") != M3_LOCK_SCHEMA_VERSION
        or raw.get("status") != "locked_before_m3_layer_sensor_results"
        or raw.get("source_evidence_sha256") != expected_evidence
        or raw.get("parent_protocol_lock_sha256") != expected_parent_protocol
        or raw.get("parent_sensor_lock_sha256") != parent_sensor_hash
        or raw.get("parent_m2_result_freeze_sha256") != expected_m2_result
        or raw.get("development_prompt_ids_sha256") != dev_hash
        or raw.get("prompt_partitions")
        != parent_lock.get("prompt_partitions")
    ):
        raise ReplayProtocolError("M3 lock parent provenance is invalid")
    if raw.get("parent_sensor_lock_schema_version") != parent_sensor.get(
        "schema_version"
    ):
        raise ReplayProtocolError(
            "M3 lock does not bind parent sensor lock schema"
        )
    if raw.get("layer_band_schema") != layer_band_schema():
        raise ReplayProtocolError("M3 layer-band schema differs from lock")
    expected_replay = {
        "compressors": list(COMPRESSORS),
        "ratios": list(RATIOS),
        "stride": STRIDE,
        "sink_tokens": SINK_TOKENS,
        "recent_window": RECENT_WINDOW,
        "model_key": "llama",
        "model_id": MODELS["llama"],
        "task": "ifeval",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "state_semantics": STATE_SEMANTICS,
        "protocol_version": PROTOCOL_VERSION,
    }
    if raw.get("replay") != expected_replay:
        raise ReplayProtocolError("M3 replay configuration differs from lock")
    actual = {
        name: _sha256(path)
        for name, path in m3_implementation_paths().items()
    }
    locked = raw.get("implementation_sha256")
    if locked != actual:
        raise ReplayProtocolError("M3 implementation differs from lock")
    return _sha256(m3_lock_path)


def read_allowed_reference_m3(
    results_root: Path,
    model_key: str,
    task: str,
    prompt_id: str,
    allowed_ids: set[str],
) -> ReplayReference:
    """Read one development reference, never quarantine or sidecars."""
    if prompt_id not in allowed_ids:
        raise ReplayProtocolError(
            f"prompt is quarantined or unexpected: {prompt_id}"
        )
    path = _reference_path(results_root, model_key, task, prompt_id)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayProtocolError(
            f"invalid allowed reference {prompt_id}"
        ) from error
    if not isinstance(raw, Mapping) or raw.get("prompt_id") != prompt_id:
        raise ReplayProtocolError(f"reference key mismatch for {prompt_id}")
    try:
        prompt_ids = tuple(int(value) for value in raw["prompt_input_ids"])
        gen_ids = tuple(int(value) for value in raw["gen_ids"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayProtocolError(
            f"reference inputs missing for {prompt_id}"
        ) from error
    if (
        not prompt_ids
        or not gen_ids
        or any(value < 0 for value in prompt_ids + gen_ids)
    ):
        raise ReplayProtocolError(f"invalid token IDs for {prompt_id}")
    return ReplayReference(prompt_id, prompt_ids, gen_ids, path)


def _exact_position(record: Mapping[str, object]) -> int:
    value = record.get("s")
    if isinstance(value, bool):
        raise ReplayProtocolError("sensor record boundary is not an integer")
    if isinstance(value, int):
        position = value
    elif isinstance(value, float) and isfinite(value) and value.is_integer():
        position = int(value)
    else:
        raise ReplayProtocolError(
            "sensor record boundary is not an exact integer"
        )
    if position < 0:
        raise ReplayProtocolError("sensor record boundary is negative")
    return position


def _validate_sidecar_positions(
    output_root: Path,
    model: str,
    task: str,
) -> None:
    """Reject malformed complete ``s`` values before generic key parsing."""
    path = sensor_sidecar_path(output_root, model, task)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, Mapping) and "s" in raw:
            _exact_position(raw)


def _prefix_hash(reference: ReplayReference, position: int) -> str:
    if position < 0 or position >= len(reference.gen_ids):
        raise ReplayProtocolError(
            f"boundary is outside reference: {position}"
        )
    tokens = torch.tensor(
        reference.prompt_input_ids + reference.gen_ids[:position],
        dtype=torch.long,
    )
    return hashlib.sha256(tokens.numpy().tobytes()).hexdigest()


def _measure_layer_bands(
    model: Any,
    captured: dict[
        int, list[tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]]
    ],
    prompt_len: int,
    prefix_len: int,
    presses: Mapping[str, Any],
    compressors: Sequence[str],
) -> dict[str, dict[str, float]]:
    selected = tuple(compressors)
    if not selected or not set(selected) <= set(COMPRESSORS):
        raise ReplayProtocolError("invalid M3 compressor subset")
    modules = _attention_modules(model)
    if set(captured) != {int(module.layer_idx) for module in modules}:
        raise SensorCaptureError("not every attention layer was captured")
    rows: dict[str, list[Mapping[str, float]]] = {}
    rotary = None
    expected_length = prompt_len + prefix_len
    for module in modules:
        chunks = captured[int(module.layer_idx)]
        if not chunks:
            raise SensorCaptureError("layer capture is empty")
        hidden = torch.cat([chunk[0] for chunk in chunks], dim=1)
        keys = torch.cat([chunk[1] for chunk in chunks], dim=2)
        values = torch.cat([chunk[2] for chunk in chunks], dim=2)
        position = (
            torch.cat([chunk[3][0] for chunk in chunks], dim=1),
            torch.cat([chunk[3][1] for chunk in chunks], dim=1),
        )
        if (
            hidden.shape[1] != expected_length
            or keys.shape[2] != expected_length
        ):
            raise SensorCaptureError("captured prefix length mismatch")
        score_by_key: dict[tuple[str, float], Tensor] = {}
        for compressor in ("expected_attention", "knorm"):
            if compressor not in selected:
                continue
            try:
                score = presses[compressor].score(
                    module,
                    hidden,
                    keys,
                    values,
                    None,
                    {"position_embeddings": position},
                )
            except AttributeError as error:
                if rotary is None:
                    for candidate in model.modules():
                        if (
                            candidate.__class__.__name__
                            == "LlamaRotaryEmbedding"
                        ):
                            rotary = candidate
                            break
                    if rotary is None:
                        raise SensorCaptureError(
                            "model has no rotary embedding"
                        ) from error
                score = _expected_attention_score(
                    presses[compressor], module, hidden, keys, values, rotary
                )
            score_by_key[(compressor, 0.0)] = torch.as_tensor(score)
        if "streaming_llm" in selected:
            for ratio in RATIOS:
                score_by_key[("streaming_llm", ratio)] = torch.as_tensor(
                    presses[f"streaming_llm@{ratio}"].score(
                        module,
                        hidden,
                        keys,
                        values,
                        None,
                        {"position_embeddings": position},
                    )
                )
        reliance = causal_recent_reliance(module, hidden, keys, position)
        for (compressor, score_ratio), scores in score_by_key.items():
            ratios = (
                RATIOS if compressor != "streaming_llm" else (score_ratio,)
            )
            for ratio in ratios:
                row = layer_sensor_values(
                    keys=keys,
                    values=values,
                    scores=scores,
                    evicted=eviction_mask(scores, ratio),
                    prompt_len=prompt_len,
                    reliance=reliance,
                    sink_tokens=SINK_TOKENS,
                    recent_window=RECENT_WINDOW,
                )
                rows.setdefault(f"{compressor}\0{ratio:.4f}", []).append(row)
    return {key: layer_band_values(values) for key, values in rows.items()}


def replay_prompt_m3(
    lm: LoadedModel,
    reference: ReplayReference,
    *,
    stride: int = STRIDE,
    compressors: Sequence[str] = COMPRESSORS,
    ratios: Sequence[float] = RATIOS,
    on_record: Callable[[dict[str, object]], None] | None = None,
    before_boundary: Callable[[], None] | None = None,
    before_reprefill: Callable[[], None] | None = None,
    before_model_forward: Callable[[], None] | None = None,
    positions: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Replay one reference and emit 12 layer-band records per boundary."""
    if (
        tuple(compressors) != COMPRESSORS
        or tuple(float(ratio) for ratio in ratios) != RATIOS
    ):
        raise ReplayProtocolError("M3 compressor/ratio scope is locked")
    if stride != STRIDE:
        raise ReplayProtocolError("M3 replay stride is locked to 16")
    expected_positions = tuple(range(0, len(reference.gen_ids), stride))
    selected = expected_positions if positions is None else tuple(positions)
    if len(selected) != len(set(selected)) or not set(selected) <= set(
        expected_positions
    ):
        raise ReplayProtocolError(
            "replay positions are outside the locked grid"
        )
    device = next(lm.model.parameters()).device
    presses: dict[str, Any] = {
        "expected_attention": get_press("expected_attention", 0.0),
        "knorm": get_press("knorm", 0.0),
    }
    presses.update(
        {
            f"streaming_llm@{ratio}": get_press("streaming_llm", ratio)
            for ratio in RATIOS
        }
    )
    captured: dict[
        int, list[tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]]
    ] = {}
    capture_target = captured

    def hook(
        module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, Tensor):
            raise SensorCaptureError("attention layer hidden state missing")
        position = _position_embeddings(args, kwargs)
        keys, values = _rotated_kv(module, hidden, position)
        capture_target.setdefault(int(module.layer_idx), []).append(
            (hidden, keys, values, position)
        )

    modules = _attention_modules(lm.model)
    handles = [
        module.register_forward_pre_hook(hook, with_kwargs=True)
        for module in modules
    ]
    if before_model_forward is not None:

        def guard_forward(
            _module: Any,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
        ) -> None:
            before_model_forward()

        handles.append(
            lm.model.register_forward_pre_hook(
                guard_forward,
                with_kwargs=True,
            )
        )
    ids: list[int] = []
    cache: Any = None
    records: list[dict[str, object]] = []
    try:
        while len(ids) < len(reference.gen_ids):
            if before_boundary is not None:
                before_boundary()
            target = (
                1
                if not ids
                else min(
                    len(reference.gen_ids),
                    (((len(ids) - 1) // stride) + 1) * stride + 1,
                )
            )
            full = torch.cat(
                [
                    torch.tensor(
                        reference.prompt_input_ids,
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.tensor(ids, dtype=torch.long, device=device),
                ]
            ).unsqueeze(0)
            with torch.no_grad():
                output = lm.model.generate(  # type: ignore[operator]
                    input_ids=full,
                    attention_mask=torch.ones_like(full),
                    generation_config=getattr(lm, "gen_config", None),
                    max_new_tokens=target - len(ids),
                    past_key_values=cache,
                    return_dict_in_generate=True,
                )
            new_ids = [
                int(token)
                for token in output.sequences[0, full.shape[1] :].tolist()
            ]
            expected = list(
                reference.gen_ids[len(ids) : len(ids) + len(new_ids)]
            )
            if new_ids != expected:
                raise ReplayProtocolError(
                    f"parity failed for {reference.prompt_id} at s={len(ids)}"
                )
            if not new_ids:
                raise ReplayProtocolError(
                    f"generation stopped early for {reference.prompt_id}"
                )
            ids.extend(new_ids)
            cache = getattr(output, "past_key_values", None)
            if cache is None:
                raise SensorCaptureError(
                    "cache-native generation returned no cache"
                )
            position = len(ids) - 1
            if (
                position % stride == 0
                and position < len(reference.gen_ids)
                and position in selected
            ):
                rows = _measure_layer_bands(
                    lm.model,
                    captured,
                    len(reference.prompt_input_ids),
                    position,
                    presses,
                    compressors=("knorm", "streaming_llm"),
                )
                if before_reprefill is not None:
                    before_reprefill()
                ref_capture: dict[
                    int,
                    list[
                        tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]
                    ],
                ] = {}
                capture_target = ref_capture
                ref_full = torch.tensor(
                    reference.prompt_input_ids + reference.gen_ids[:position],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                try:
                    with torch.no_grad():
                        kwargs: dict[str, object] = {"use_cache": False}
                        if (
                            "logits_to_keep"
                            in inspect.signature(lm.model.forward).parameters
                        ):
                            kwargs["logits_to_keep"] = 1
                        lm.model(ref_full, **kwargs)
                finally:
                    capture_target = captured
                rows.update(
                    _measure_layer_bands(
                        lm.model,
                        ref_capture,
                        len(reference.prompt_input_ids),
                        position,
                        presses,
                        compressors=("expected_attention",),
                    )
                )
                for key, sensors in rows.items():
                    compressor, ratio_text = key.split("\0", 1)
                    record: dict[str, object] = {
                        "prompt_id": reference.prompt_id,
                        "compressor": compressor,
                        "ratio": float(ratio_text),
                        "s": position,
                        "sensors": sensors,
                        "prefix_hash": _prefix_hash(reference, position),
                        "protocol_version": PROTOCOL_VERSION,
                        "feature_names": list(LAYER_BAND_FEATURE_NAMES),
                        "state_semantics": STATE_SEMANTICS[compressor],
                    }
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
    finally:
        for handle in handles:
            handle.remove()
    return records


def replay_locked_m3(
    config: ReplayConfig,
    *,
    prompt_ids: Iterable[str] | None = None,
    model_loader: Callable[..., LoadedModel] = load_model,
) -> int:
    """Replay the development subset with validated resume state."""
    allowed, evidence_hash, lock_hash = locked_prompt_ids_m3(
        config.evidence_path, config.lock_path
    )
    m3_lock_hash = validate_m3_lock(
        config.m3_lock_path,
        config.evidence_path,
        config.lock_path,
        config.sensor_lock_path,
        config.m2_result_freeze_path,
    )
    requested = tuple(sorted(allowed) if prompt_ids is None else prompt_ids)
    if len(requested) != len(set(requested)) or any(
        prompt_id not in allowed for prompt_id in requested
    ):
        bad = next(
            (
                prompt_id
                for prompt_id in requested
                if prompt_id not in allowed
            ),
            "duplicate",
        )
        raise ReplayProtocolError(
            f"prompt is quarantined or unexpected: {bad}"
        )
    references = [
        read_allowed_reference_m3(
            config.results_root,
            config.model_key,
            config.task,
            prompt_id,
            allowed,
        )
        for prompt_id in requested
    ]
    expected = {
        (reference.prompt_id, compressor, float(ratio), position)
        for reference in references
        for position in range(0, len(reference.gen_ids), STRIDE)
        for compressor in COMPRESSORS
        for ratio in RATIOS
    }
    _validate_sidecar_positions(
        config.output_root, config.model_key, config.task
    )
    existing_records = read_sensor_records(
        config.output_root, config.model_key, config.task
    )
    for record in existing_records:
        _exact_position(record)
    existing = validate_sensor_records(
        existing_records,
        model=config.model_key,
        task=config.task,
        evidence_sha256=evidence_hash,
        lock_sha256=lock_hash,
        sensor_lock_sha256=m3_lock_hash,
        protocol_version=PROTOCOL_VERSION,
        feature_names=LAYER_BAND_FEATURE_NAMES,
    )
    if extra := existing - expected:
        raise ReplayProtocolError(
            f"M3 sidecar contains out-of-scope keys: {sorted(extra)[:3]}"
        )
    references_by_id = {
        reference.prompt_id: reference for reference in references
    }
    for record in existing_records:
        prompt_id = str(record["prompt_id"])
        position = _exact_position(record)
        compressor = str(record["compressor"])
        reference = references_by_id.get(prompt_id)
        if reference is None:
            raise ReplayProtocolError(
                f"prefix hash mismatch for {prompt_id} at s={position}"
            )
        if record.get("state_semantics") != STATE_SEMANTICS.get(compressor):
            raise ReplayProtocolError(
                f"state semantics mismatch for {prompt_id} at s={position}"
            )
        if record.get("prefix_hash") != _prefix_hash(reference, position):
            raise ReplayProtocolError(
                f"prefix hash mismatch for {prompt_id} at s={position}"
            )
    missing = expected - existing
    guard: Callable[[], None] | None = None
    if config.device.startswith("cuda"):

        def guard_gpu() -> None:
            _guard_gpu(config.allowed_gpu_pids)

        guard = guard_gpu
    lm: LoadedModel | None = None
    if missing:
        if guard is not None:
            guard()
        lm = model_loader(
            config.model_key,
            dtype=config.dtype,
            device=config.device,
            attn_implementation=config.attn_implementation,
            model_id=config.model_id,
        )
        try:
            modules = _attention_modules(lm.model)
        except SensorCaptureError as error:
            raise ReplayProtocolError(
                "M3 model exposes no supported attention layers"
            ) from error
        indices = [int(module.layer_idx) for module in modules]
        if len(modules) != LAYER_COUNT or indices != list(range(LAYER_COUNT)):
            raise ReplayProtocolError(
                f"M3 requires exactly {LAYER_COUNT} ordered attention layers"
            )
    written = 0
    for reference in references:
        missing_positions = [
            position
            for position in range(0, len(reference.gen_ids), STRIDE)
            if any(
                (reference.prompt_id, compressor, float(ratio), position)
                not in existing
                for compressor in COMPRESSORS
                for ratio in RATIOS
            )
        ]
        if not missing_positions:
            continue
        if lm is None:
            raise ReplayProtocolError(
                "model was not loaded for missing M3 sensors"
            )

        def save(record: dict[str, object]) -> None:
            nonlocal written
            record.update(
                {
                    "evidence_sha256": evidence_hash,
                    "lock_sha256": lock_hash,
                    "sensor_lock_sha256": m3_lock_hash,
                    "model_key": config.model_key,
                    "task": config.task,
                }
            )
            key = (
                str(record["prompt_id"]),
                str(record["compressor"]),
                float(cast(float, record["ratio"])),
                int(cast(int, record["s"])),
            )
            if key in existing:
                return
            append_sensor_record(
                config.output_root,
                config.model_key,
                config.task,
                record,
                existing_keys=existing,
            )
            written += 1

        replay_prompt_m3(
            lm,
            reference,
            on_record=save,
            before_model_forward=guard,
            positions=missing_positions,
        )
    write_sensor_manifest(
        config.output_root,
        config.model_key,
        config.task,
        expected_keys=expected,
        evidence_sha256=evidence_hash,
        lock_sha256=lock_hash,
        sensor_lock_sha256=m3_lock_hash,
        protocol_version=PROTOCOL_VERSION,
        feature_names=LAYER_BAND_FEATURE_NAMES,
    )
    return written


# M2-parallel public names make the M3 CLI and callers discoverable without
# importing the mutable v2 replay implementation as their entry point.
locked_prompt_ids = locked_prompt_ids_m3
read_allowed_reference = read_allowed_reference_m3
replay_prompt = replay_prompt_m3
replay_locked = replay_locked_m3
