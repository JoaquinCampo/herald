"""Label-free M2 sensor replay over locked development prompts."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from herald.config import MODELS
from herald.generate import LoadedModel, load_model
from herald.magnitude_v2 import validate_lock_and_evidence
from herald.press_sensors import (
    COMPRESSORS,
    RATIOS,
    RECENT_WINDOW,
    SENSOR_NAMES,
    SENSOR_STATS,
    SINK_TOKENS,
    SensorCaptureError,
    aggregate_layer_sensors,
    causal_recent_reliance,
    layer_sensor_values,
    sensor_feature_names,
)
from herald.presses import get_press
from herald.storage import (
    append_sensor_record,
    read_sensor_records,
    validate_sensor_records,
    write_sensor_manifest,
)

PROTOCOL_VERSION = "herald.m2.sensor_replay.v2"
SENSOR_LOCK_SCHEMA_VERSION = "herald.magnitude_v2_sensor_lock.v1"
STRIDE = 16


class ReplayProtocolError(ValueError):
    """Raised when replay inputs are outside the locked scope."""


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
            raise ReplayProtocolError("M2 replay is locked to stride 16")
        if tuple(self.compressors) != COMPRESSORS:
            raise ReplayProtocolError(
                f"compressors are locked to {COMPRESSORS}"
            )
        if tuple(float(r) for r in self.ratios) != RATIOS:
            raise ReplayProtocolError(f"ratios are locked to {RATIOS}")
        if self.model_key != "llama" or self.task != "ifeval":
            raise ReplayProtocolError("M2 replay is locked to llama/ifeval")
        if self.model_id not in (None, MODELS["llama"]):
            raise ReplayProtocolError("M2 replay model identity is locked")
        if self.dtype != "bfloat16" or self.attn_implementation != "sdpa":
            raise ReplayProtocolError(
                "M2 replay dtype/attention backend is locked"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_prompt_ids(
    evidence_path: Path,
    lock_path: Path,
) -> tuple[set[str], str, str]:
    """Validate the frozen protocol and return only development IDs."""
    try:
        evidence, _ = validate_lock_and_evidence(evidence_path, lock_path)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ReplayProtocolError(
            "unable to validate frozen evidence/lock"
        ) from error
    development_ids = tuple(
        str(item) for item in evidence["split"]["train_prompt_ids"]
    )
    if not development_ids:
        raise ReplayProtocolError("empty development allowlist")
    return set(development_ids), _sha256(evidence_path), _sha256(lock_path)


def validate_sensor_lock(
    sensor_lock_path: Path,
    evidence_path: Path,
    lock_path: Path,
) -> str:
    """Validate the immutable sensor addendum and implementation hashes."""
    try:
        sensor_lock = json.loads(sensor_lock_path.read_text())
        _, parent_lock = validate_lock_and_evidence(
            evidence_path,
            lock_path,
        )
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise ReplayProtocolError("unable to validate sensor lock") from error
    if not isinstance(sensor_lock, Mapping):
        raise ReplayProtocolError("sensor lock is not an object")
    if (
        sensor_lock.get("schema_version") != SENSOR_LOCK_SCHEMA_VERSION
        or sensor_lock.get("status") != "locked_before_sensor_replay_results"
        or sensor_lock.get("source_evidence_sha256") != _sha256(evidence_path)
        or sensor_lock.get("parent_protocol_lock_sha256")
        != _sha256(lock_path)
        or sensor_lock.get("development_prompt_ids_sha256")
        != parent_lock["prompt_partitions"]["development_prompt_ids_sha256"]
    ):
        raise ReplayProtocolError("sensor lock provenance is invalid")
    expected_schema = {
        "names": list(SENSOR_NAMES),
        "stats": list(SENSOR_STATS),
        "feature_names": sensor_feature_names(),
    }
    if sensor_lock.get("sensor_schema") != expected_schema:
        raise ReplayProtocolError("sensor schema differs from lock")
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
        "state_semantics": {
            "expected_attention": "herald.matched_reprefill_v1",
            "knorm": "herald.cache_native_pending_v1",
            "streaming_llm": "herald.cache_native_pending_v1",
        },
        "protocol_version": PROTOCOL_VERSION,
    }
    if sensor_lock.get("replay") != expected_replay:
        raise ReplayProtocolError(
            "sensor replay configuration differs from lock"
        )
    repository_root = Path(__file__).resolve().parents[2]
    implementation_paths = {
        "src/herald/config.py": repository_root / "src/herald/config.py",
        "src/herald/generate.py": repository_root / "src/herald/generate.py",
        "src/herald/intervention_sweep.py": repository_root
        / "src/herald/intervention_sweep.py",
        "src/herald/magnitude_v2.py": repository_root
        / "src/herald/magnitude_v2.py",
        "src/herald/presses.py": repository_root / "src/herald/presses.py",
        "src/herald/press_sensors.py": repository_root
        / "src/herald/press_sensors.py",
        "src/herald/sensor_replay.py": Path(__file__).resolve(),
        "src/herald/storage.py": repository_root / "src/herald/storage.py",
        "scripts/replay_press_sensors.py": repository_root
        / "scripts/replay_press_sensors.py",
        "uv.lock": repository_root / "uv.lock",
    }
    actual_hashes = {
        name: _sha256(path) for name, path in implementation_paths.items()
    }
    if sensor_lock.get("implementation_sha256") != actual_hashes:
        raise ReplayProtocolError("sensor implementation differs from lock")
    return _sha256(sensor_lock_path)


def _reference_path(
    results_root: Path, model_key: str, task: str, prompt_id: str
) -> Path:
    from herald.storage import safe_id

    return (
        results_root
        / model_key
        / task
        / "references"
        / f"{safe_id(prompt_id)}.json"
    )


def read_allowed_reference(
    results_root: Path,
    model_key: str,
    task: str,
    prompt_id: str,
    allowed_ids: set[str],
) -> ReplayReference:
    """Open exactly one allowlisted reference and parse only replay inputs."""
    if prompt_id not in allowed_ids:
        raise ReplayProtocolError(
            f"prompt is quarantined or unexpected: {prompt_id}"
        )
    path = _reference_path(results_root, model_key, task, prompt_id)
    try:
        with path.open() as stream:
            raw = json.load(stream)
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


def _attention_modules(model: Any) -> list[Any]:
    modules = [
        module
        for module in model.modules()
        if all(
            hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")
        )
        and hasattr(module, "layer_idx")
    ]
    modules.sort(key=lambda module: int(module.layer_idx))
    if not modules:
        raise SensorCaptureError(
            "model exposes no supported attention layers"
        )
    return modules


def _position_embeddings(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Tensor, Tensor]:
    value = kwargs.get("position_embeddings")
    if value is None and len(args) > 1:
        value = args[1]
    if not isinstance(value, tuple) or len(value) != 2:
        raise SensorCaptureError(
            "attention layer did not expose position embeddings"
        )
    return value


def _rotated_kv(
    module: Any, hidden: Tensor, position: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    batch, length = hidden.shape[:2]
    head_dim = int(module.head_dim)
    key = (
        module.k_proj(hidden)
        .view(batch, length, -1, head_dim)
        .transpose(1, 2)
    )
    value = (
        module.v_proj(hidden)
        .view(batch, length, -1, head_dim)
        .transpose(1, 2)
    )
    zero_query = torch.zeros_like(key)
    _, key = apply_rotary_pos_emb(  # type: ignore[no-untyped-call]
        zero_query,
        key,
        *position,
    )
    key = cast(Tensor, key)
    return key, value


def _expected_attention_score(
    press: Any,
    module: Any,
    hidden: Tensor,
    keys: Tensor,
    values: Tensor,
    rotary: Any,
) -> Tensor:
    """Parity implementation for kvpress 0.5.2 on post-4.58 Transformers."""
    from kvpress.utils import get_prerope_query_states
    from transformers.models.llama.modeling_llama import repeat_kv

    sink = int(press.n_sink)
    if hidden.shape[1] <= sink or keys.shape[2] <= sink:
        raise SensorCaptureError(
            "expected-attention prefix is shorter than sink"
        )
    h = hidden[:, sink:]
    queries = get_prerope_query_states(module, h)
    mean = queries.mean(dim=2, keepdim=True)
    cov = None
    if press.use_covariance:
        centered = queries - mean
        cov = torch.einsum("bnsi,bnsj->bnij", centered, centered) / h.shape[1]
    mean = mean.squeeze(2)
    q_len = hidden.shape[1]
    position_ids = torch.arange(
        q_len, q_len + int(press.n_future_positions), device=mean.device
    ).unsqueeze(0)
    cos, sin = rotary(mean, position_ids)
    cos, sin = cos[0], sin[0]
    head_dim = int(module.head_dim)
    identity = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
    skew = torch.zeros_like(identity)
    half = head_dim // 2
    skew[half:, :half] = identity[:half, :half]
    skew[:half, half:] = -identity[:half, :half]
    rotation = (cos.unsqueeze(1) * identity + sin.unsqueeze(1) * skew).mean(
        dim=0
    )
    mean = mean @ rotation.T
    if cov is not None:
        cov = rotation @ cov @ rotation.T
    key_slice = keys[:, :, sink:]
    value_slice = values[:, :, sink:]
    batch, kv_heads, length, dim = key_slice.shape
    expanded = repeat_kv(
        key_slice, module.config.num_attention_heads // kv_heads
    ).transpose(2, 3)
    score = torch.matmul(mean.unsqueeze(2), expanded).squeeze(2) / dim**0.5
    if cov is not None:
        score = (
            score
            + torch.einsum("bhin,bhij,bhjn->bhn", expanded, cov, expanded)
            / dim
            / 2
        )
    score = F.softmax(score, dim=-1)
    groups = module.config.num_attention_heads // kv_heads
    score = score.view(batch, kv_heads, groups, length).mean(dim=2)
    if press.use_vnorm:
        score = (score + press.epsilon) * value_slice.norm(dim=-1)
    return F.pad(score, (sink, 0), value=score.max().item())


def _find_rotary(model: Any) -> Any:
    for candidate in model.modules():
        if candidate.__class__.__name__ == "LlamaRotaryEmbedding":
            return candidate
    raise SensorCaptureError(
        "model does not expose a parent rotary embedding"
    )


def _measure_captured(
    model: Any,
    captured: dict[
        int, list[tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]]
    ],
    prompt_len: int,
    prefix_len: int,
    presses: Mapping[str, Any],
    compressors: Sequence[str] = COMPRESSORS,
) -> dict[str, dict[str, float]]:
    selected = tuple(compressors)
    if not selected or not set(selected) <= set(COMPRESSORS):
        raise ReplayProtocolError("invalid sensor compressor subset")
    modules = _attention_modules(model)
    if set(captured) != {int(module.layer_idx) for module in modules}:
        raise SensorCaptureError("not every layer was captured")
    rotary = None
    layer_rows: dict[str, list[dict[str, float]]] = {}
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
            except AttributeError:
                if rotary is None:
                    rotary = _find_rotary(model)
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
            ratio_values = (
                RATIOS if compressor != "streaming_llm" else (score_ratio,)
            )
            for ratio in ratio_values:
                from herald.press_sensors import eviction_mask

                mask = eviction_mask(scores, ratio)
                values_for_layer = layer_sensor_values(
                    keys=keys,
                    values=values,
                    scores=scores,
                    evicted=mask,
                    prompt_len=prompt_len,
                    reliance=reliance,
                )
                key = f"{compressor}\0{ratio:.4f}"
                layer_rows.setdefault(key, []).append(values_for_layer)
    return {
        key: aggregate_layer_sensors(layer_values)
        for key, layer_values in layer_rows.items()
    }


def _prefix_hash(reference: ReplayReference, position: int) -> str:
    tokens = torch.tensor(
        reference.prompt_input_ids + reference.gen_ids[:position],
        dtype=torch.long,
    )
    return hashlib.sha256(tokens.numpy().tobytes()).hexdigest()


def replay_prompt(
    lm: LoadedModel,
    reference: ReplayReference,
    *,
    stride: int = STRIDE,
    compressors: Sequence[str] = COMPRESSORS,
    ratios: Sequence[float] = RATIOS,
    on_record: Callable[[dict[str, object]], None] | None = None,
    before_boundary: Callable[[], None] | None = None,
    before_reprefill: Callable[[], None] | None = None,
    positions: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    """Replay one prompt on the cache-native reference schedule."""
    if (
        tuple(compressors) != COMPRESSORS
        or tuple(float(r) for r in ratios) != RATIOS
    ):
        raise ReplayProtocolError("replay compressor/ratio scope is locked")
    if stride != STRIDE:
        raise ReplayProtocolError("replay stride is locked to 16")
    expected_positions = tuple(range(0, len(reference.gen_ids), stride))
    selected_positions = (
        expected_positions if positions is None else tuple(positions)
    )
    if len(selected_positions) != len(set(selected_positions)) or not set(
        selected_positions
    ) <= set(expected_positions):
        raise ReplayProtocolError(
            "replay positions are outside the locked grid"
        )
    device = next(lm.model.parameters()).device
    presses = {
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
            generation_config = getattr(lm, "gen_config", None)
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
                    generation_config=generation_config,
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
                    f"block parity failed for {reference.prompt_id} "
                    f"at s={len(ids)}"
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
                and position in selected_positions
            ):
                rows = _measure_captured(
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
                        tuple[
                            Tensor,
                            Tensor,
                            Tensor,
                            tuple[Tensor, Tensor],
                        ]
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
                        forward_kwargs: dict[str, object] = {
                            "use_cache": False
                        }
                        if (
                            "logits_to_keep"
                            in inspect.signature(lm.model.forward).parameters
                        ):
                            forward_kwargs["logits_to_keep"] = 1
                        lm.model(ref_full, **forward_kwargs)
                finally:
                    capture_target = captured
                rows.update(
                    _measure_captured(
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
                        "feature_names": sensor_feature_names(),
                    }
                    records.append(record)
                    if on_record is not None:
                        on_record(record)
            del full
    finally:
        for handle in handles:
            handle.remove()
    return records


def replay_locked(
    config: ReplayConfig,
    *,
    prompt_ids: Iterable[str] | None = None,
    model_loader: Callable[..., LoadedModel] = load_model,
) -> int:
    """Run resumably over an explicit subset of the locked 154 prompts."""
    allowed, evidence_hash, lock_hash = locked_prompt_ids(
        config.evidence_path,
        config.lock_path,
    )
    sensor_lock_hash = validate_sensor_lock(
        config.sensor_lock_path,
        config.evidence_path,
        config.lock_path,
    )
    requested = tuple(sorted(allowed) if prompt_ids is None else prompt_ids)
    if len(requested) != len(set(requested)):
        raise ReplayProtocolError("requested prompt IDs must be unique")
    if any(prompt_id not in allowed for prompt_id in requested):
        bad = next(
            prompt_id for prompt_id in requested if prompt_id not in allowed
        )
        raise ReplayProtocolError(
            f"prompt is quarantined or unexpected: {bad}"
        )
    # Validate all keys before model loading or opening any reference file.
    references = [
        read_allowed_reference(
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
    features = sensor_feature_names()
    existing_records = read_sensor_records(
        config.output_root,
        config.model_key,
        config.task,
    )
    existing = validate_sensor_records(
        existing_records,
        model=config.model_key,
        task=config.task,
        evidence_sha256=evidence_hash,
        lock_sha256=lock_hash,
        sensor_lock_sha256=sensor_lock_hash,
        protocol_version=PROTOCOL_VERSION,
        feature_names=features,
    )
    extra = existing - expected
    if extra:
        raise ReplayProtocolError(
            f"sensor sidecar contains out-of-scope keys: {sorted(extra)[:3]}"
        )
    references_by_id = {
        reference.prompt_id: reference for reference in references
    }
    for record in existing_records:
        prompt_id = str(record["prompt_id"])
        position = int(cast(int, record["s"]))
        reference = references_by_id[prompt_id]
        if record.get("prefix_hash") != _prefix_hash(reference, position):
            raise ReplayProtocolError(
                f"prefix hash mismatch for {prompt_id} at s={position}"
            )

    missing = expected - existing
    before_boundary: Callable[[], None] | None = None
    if config.device.startswith("cuda"):
        from herald.intervention_sweep import guard_gpu_contention

        def guard_boundary() -> None:
            guard_gpu_contention(config.allowed_gpu_pids)

        before_boundary = guard_boundary

    lm: LoadedModel | None = None
    if missing:
        if before_boundary is not None:
            before_boundary()
        lm = model_loader(
            config.model_key,
            dtype=config.dtype,
            device=config.device,
            attn_implementation=config.attn_implementation,
            model_id=config.model_id,
        )

    written = 0
    for reference in references:
        missing_positions = [
            position
            for position in range(0, len(reference.gen_ids), STRIDE)
            if any(
                (
                    reference.prompt_id,
                    compressor,
                    float(ratio),
                    position,
                )
                not in existing
                for compressor in COMPRESSORS
                for ratio in RATIOS
            )
        ]
        if not missing_positions:
            continue
        if lm is None:
            raise ReplayProtocolError(
                "model was not loaded for missing sensors"
            )

        def save(record: dict[str, object]) -> None:
            nonlocal written
            record["evidence_sha256"] = evidence_hash
            record["lock_sha256"] = lock_hash
            record["sensor_lock_sha256"] = sensor_lock_hash
            record["model_key"] = config.model_key
            record["task"] = config.task
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

        replay_prompt(
            lm,
            reference,
            on_record=save,
            before_boundary=before_boundary,
            before_reprefill=before_boundary,
            positions=missing_positions,
        )
    write_sensor_manifest(
        config.output_root,
        config.model_key,
        config.task,
        expected_keys=expected,
        evidence_sha256=evidence_hash,
        lock_sha256=lock_hash,
        sensor_lock_sha256=sensor_lock_hash,
        protocol_version=PROTOCOL_VERSION,
        feature_names=features,
    )
    return written
