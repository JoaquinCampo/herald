"""Faithful, resumable IFEval intervention regeneration.

This module deliberately keeps the authoritative reference cache alive while a
prompt is decoded.  StreamingLLM and Knorm branch from that cache; the
ExpectedAttention press is defined by matched no-press/pressed re-prefills.
"""

import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from herald.config import MODELS, RATIOS
from herald.features import FEATURE_NAMES, FeatureCollector
from herald.generate import LoadedModel, build_input_ids, prefill_continuation
from herald.ifeval import IFEvalScores, score_ifeval_robustness
from herald.live_controller import (
    _extend_reference,
    _fork_score_cache,
    continue_from_cache,
)
from herald.presses import get_press
from herald.storage import (
    append_hybrid,
    ensure_reference_done,
    load_reference,
    safe_id,
    save_reference,
)
from herald.tasks import PromptRecord

APPROVED_COMPRESSORS: tuple[str, ...] = (
    "streaming_llm",
    "knorm",
    "expected_attention",
)
CACHE_NATIVE_COMPRESSORS: tuple[str, ...] = ("streaming_llm", "knorm")
APPROVED_RATIOS: tuple[float, ...] = RATIOS
CACHE_NATIVE_SEMANTICS = "herald.cache_native_pending_v1"
EXPECTED_ATTENTION_SEMANTICS = "herald.matched_reprefill_v1"


@dataclass(frozen=True, slots=True)
class InterventionSweepConfig:
    """Immutable scope and generation settings for one regeneration."""

    model_key: str = "llama"
    task: str = "ifeval"
    prompt_count: int = 200
    stride: int = 16
    compressors: tuple[str, ...] = APPROVED_COMPRESSORS
    ratios: tuple[float, ...] = APPROVED_RATIOS
    max_new_tokens: int = 1024
    results_dir: Path = Path("results/intervention")
    model_id: str | None = None
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device: str = "cuda"
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            self.prompt_count < 1
            or self.stride < 1
            or self.max_new_tokens < 1
        ):
            raise ValueError(
                "prompt_count, stride, and max_new_tokens must be positive"
            )
        if not self.compressors or any(
            c not in APPROVED_COMPRESSORS for c in self.compressors
        ):
            raise ValueError(
                f"compressors must be drawn from {APPROVED_COMPRESSORS}"
            )
        if len(set(self.compressors)) != len(self.compressors):
            raise ValueError("compressors must be unique")
        if not self.ratios:
            raise ValueError("at least one ratio is required")
        for ratio in self.ratios:
            if ratio not in APPROVED_RATIOS:
                raise ValueError(
                    f"ratio {ratio!r} is outside the approved scope"
                )
        if len(set(self.ratios)) != len(self.ratios):
            raise ValueError("ratios must be unique")


def initialize_intervention_config(
    results_dir: Path,
    config: InterventionSweepConfig,
) -> Path:
    """Freeze intervention-only settings before writing any artifacts."""
    path = results_dir / "intervention_config.json"
    payload = json.dumps(
        {**asdict(config), "results_dir": str(config.results_dir)},
        sort_keys=True,
        indent=2,
        default=str,
    )
    if path.is_file():
        if path.read_text() != payload:
            raise ValueError(
                "existing intervention config does not match invocation"
            )
        return path
    results_dir.mkdir(parents=True, exist_ok=True)
    temporary = results_dir / ".intervention_config.tmp"
    temporary.write_text(payload)
    os.replace(temporary, path)
    return path


@dataclass(frozen=True, slots=True)
class GpuContentionEvent:
    """Structured evidence that another process owns the compute device."""

    own_pid: int
    allowed_pids: tuple[int, ...]
    unknown_pids: tuple[int, ...]


class GpuMonitoringError(RuntimeError):
    """Raised when active GPU ownership cannot be observed safely."""


class GpuContentionError(RuntimeError):
    """Raised before work when an unapproved GPU process is active."""

    def __init__(self, event: GpuContentionEvent) -> None:
        self.event = event
        super().__init__(
            f"GPU contention: unknown compute PIDs {event.unknown_pids}"
        )


def active_gpu_compute_pids() -> set[int]:
    """Return NVIDIA compute PIDs, without invoking a shell.

    A machine without ``nvidia-smi`` is treated as having no observable
    contention; the CLI separately rejects an unavailable requested CUDA
    device.  Unexpected command failures are surfaced rather than hidden.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise GpuMonitoringError("nvidia-smi is unavailable") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        raise GpuMonitoringError(f"nvidia-smi process query failed: {detail}")
    pids: set[int] = set()
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            pid = int(value)
        except ValueError as error:
            raise GpuMonitoringError(
                f"invalid nvidia-smi compute PID {value!r}"
            ) from error
        if pid > 0:
            pids.add(pid)
    return pids


def gpu_contention_event(
    pids: Iterable[int],
    *,
    allowed_pids: Iterable[int] = (),
    own_pid: int | None = None,
) -> GpuContentionEvent | None:
    """Exclude this process and approved keepalives from compute PIDs."""
    own = os.getpid() if own_pid is None else int(own_pid)
    allowed = {int(pid) for pid in allowed_pids}
    unknown = tuple(
        sorted({int(pid) for pid in pids if int(pid) > 0} - allowed - {own})
    )
    if not unknown:
        return None
    return GpuContentionEvent(
        own_pid=own, allowed_pids=tuple(sorted(allowed)), unknown_pids=unknown
    )


def guard_gpu_contention(allowed_pids: Iterable[int] = ()) -> None:
    """Fail closed unless every active compute PID is explicitly allowed."""
    _guard_gpu(active_gpu_compute_pids, tuple(allowed_pids))


def _guard_gpu(
    provider: Callable[[], Iterable[int]], allowed_pids: tuple[int, ...]
) -> None:
    event = gpu_contention_event(provider(), allowed_pids=allowed_pids)
    if event is not None:
        raise GpuContentionError(event)


def _score_output(
    scorer: Callable[..., Any] | None, record: PromptRecord, text: str
) -> tuple[float, float]:
    """Return loose and strict instruction-level quality scores."""
    if scorer is None:
        if record.task != "ifeval" or not record.gold:
            raise ValueError(f"missing prompt gold for {record.prompt_id}")
        scores = score_ifeval_robustness(text, record.gold, mode="both")
        return scores.loose, scores.strict
    try:
        n_params = len(inspect.signature(scorer).parameters)
    except (TypeError, ValueError):
        n_params = 0
    if n_params == 1:
        value = scorer(text)
    elif n_params >= 3:
        value = scorer(record.task, text, record.gold)
    else:
        value = scorer(record, text)
    if isinstance(value, IFEvalScores):
        return float(value.loose), float(value.strict)
    if isinstance(value, Mapping):
        if "loose" not in value or "strict" not in value:
            raise ValueError("scorer mapping must contain loose and strict")
        return float(value["loose"]), float(value["strict"])
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(
        "intervention scorer must return loose and strict scores"
    )


def _decode(lm: LoadedModel, ids: list[int]) -> str:
    return lm.tokenizer.decode(ids, skip_special_tokens=True)


def _cache_signature(cache: Any) -> tuple[tuple[Any, ...], ...]:
    """Capture cache tensor identity metadata without copying tensor data."""
    signature: list[tuple[Any, ...]] = []
    for layer in getattr(cache, "layers", ()):
        row: list[Any] = []
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor):
                row.extend(
                    (
                        tensor.untyped_storage().data_ptr(),
                        tuple(tensor.shape),
                        tensor._version,
                    )
                )
            else:
                row.extend((None, None, None))
        signature.append(tuple(row))
    return tuple(signature)


def _reference_incremental(
    lm: LoadedModel,
    record: PromptRecord,
    max_new_tokens: int,
    stride: int,
    on_boundary: Callable[[int, list[int], int, Any, torch.Tensor], None]
    | None = None,
) -> tuple[list[int], np.ndarray, list[int]]:
    """Generate a reference and process each pending boundary immediately."""
    prompt = build_input_ids(lm, record)
    collector = FeatureCollector()
    ids: list[int] = []
    cache: Any = None
    while len(ids) < max_new_tokens:
        target = (
            1
            if not ids
            else min(
                max_new_tokens,
                (((len(ids) - 1) // stride) + 1) * stride + 1,
            )
        )
        new_ids, cache = _extend_reference(
            lm, prompt, ids, cache, collector, target - len(ids)
        )
        if not new_ids:
            break
        ids.extend(int(token) for token in new_ids)
        if len(ids) > max_new_tokens:
            ids = ids[:max_new_tokens]
        position = len(ids) - 1
        if position % stride == 0 and position < max_new_tokens:
            if cache is None:
                raise RuntimeError(
                    "reference generation did not return a live KV cache"
                )
            if on_boundary is not None:
                on_boundary(
                    position,
                    list(ids[:position]),
                    ids[position],
                    cache,
                    prompt,
                )
        if int(ids[-1]) in lm.eos_ids:
            break
    raw = collector.stacked()
    if raw.ndim == 3 and raw.shape[1]:
        features = raw[: len(ids), 0, :].astype(np.float32, copy=True)
    else:
        features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return ids, features, prompt.tolist()


def _strict_score_pair(
    scorer: Callable[..., Any] | None,
    record: PromptRecord,
    text: str,
) -> tuple[float, float]:
    loose, strict = _score_output(scorer, record, text)
    if not math.isfinite(loose) or not math.isfinite(strict):
        raise ValueError("quality scores must be finite")
    return loose, strict


def _validate_reference_features(path: Path, expected_rows: int) -> None:
    """Require an intact base-logit feature matrix for one reference."""
    try:
        matrix = np.load(path, mmap_mode="r")
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid reference feature artifact: {path}"
        ) from error
    expected_shape = (expected_rows, len(FEATURE_NAMES))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"reference feature shape {matrix.shape} != {expected_shape}"
        )


def _existing_cells(
    task_dir: Path,
    compressor: str,
    ratio: float,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Read a shard strictly; malformed or duplicate cells fail closed."""
    shard = task_dir / "hybrids" / f"{compressor}__{ratio:.4f}.jsonl"
    if not shard.exists():
        return {}
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    file_size = shard.stat().st_size
    with shard.open("r+b") as stream:
        line_number = 0
        while line := stream.readline():
            line_number += 1
            offset_after = stream.tell()
            if not line.strip():
                continue
            try:
                data = json.loads(line.decode())
                prompt_id = str(data["prompt_id"])
                s = int(data["s"])
                float(data["q"])
                float(data["dq"])
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                torn_final_line = (
                    offset_after == file_size and not line.endswith(b"\n")
                )
                if torn_final_line:
                    stream.truncate(offset_after - len(line))
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ValueError(
                    f"invalid hybrid row {shard}:{line_number}"
                ) from exc
            key = (prompt_id, s)
            if key in cells:
                raise ValueError(
                    f"duplicate completed hybrid cell {key} in {shard}"
                )
            cells[key] = data
    return cells


def _validate_existing(
    rows: dict[tuple[str, int], dict[str, Any]],
    *,
    ref_q: float,
    ref_q_strict: float,
    semantics: str,
    validate_legacy: bool = True,
) -> None:
    """Validate fields before or after the authoritative reference score."""
    for key, row in rows.items():
        if "q_control" in row:
            control = float(row["q_control"])
        elif validate_legacy:
            control = ref_q
        else:
            control = None
        if control is not None:
            hybrid = float(row["q"])
            if not math.isclose(
                float(row["dq"]), control - hybrid, abs_tol=1e-6, rel_tol=0.0
            ):
                raise ValueError(f"existing loose delta mismatch for {key}")
        if row.get("intervention_semantics") not in (None, semantics):
            raise ValueError(
                f"existing intervention semantics mismatch for {key}"
            )
        if "dq_strict" in row:
            if "q_hybrid_strict" not in row:
                raise ValueError(
                    f"existing strict delta lacks treatment score for {key}"
                )
            if "q_control_strict" in row:
                strict_control = float(row["q_control_strict"])
            elif validate_legacy:
                strict_control = ref_q_strict
            else:
                strict_control = None
            if strict_control is not None:
                strict_hybrid = float(row["q_hybrid_strict"])
                if not math.isclose(
                    float(row["dq_strict"]),
                    strict_control - strict_hybrid,
                    abs_tol=1e-6,
                    rel_tol=0.0,
                ):
                    raise ValueError(
                        f"existing strict delta mismatch for {key}"
                    )


def _artifact_file_entries(path: Path) -> list[dict[str, Any]]:
    """Return deterministic streamed hashes for every artifact file."""
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*") if p.is_file())
    )
    for file_path in files:
        digest = hashlib.sha256()
        size = 0
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        entries.append(
            {
                "path": str(
                    file_path.relative_to(
                        path.parent if path.is_file() else path
                    )
                ),
                "size": size,
                "sha256": digest.hexdigest(),
            }
        )
    return entries


def _write_manifest(
    root: Path,
    config: InterventionSweepConfig,
    counts: dict[str, int],
    *,
    expected_prompts: int,
    expected_cells: int,
) -> None:
    cumulative = {
        "references": sum(
            1
            for path in (root / "references").glob("*.json")
            if path.is_file()
        ),
        # Strict shard reads also remove a torn final append before hashing.
        "cells": sum(
            len(_existing_cells(root, compressor, ratio))
            for compressor in config.compressors
            for ratio in config.ratios
        ),
    }
    reference_files = _artifact_file_entries(root / "references")
    hybrid_files = _artifact_file_entries(root / "hybrids")
    all_files = reference_files + hybrid_files
    aggregate = hashlib.sha256(
        json.dumps(all_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config_entries = _artifact_file_entries(
        config.results_dir / "config.json"
    )
    config_digest = config_entries[0]["sha256"] if config_entries else None
    intervention_entries = _artifact_file_entries(
        config.results_dir / "intervention_config.json"
    )
    intervention_digest = (
        intervention_entries[0]["sha256"] if intervention_entries else None
    )
    payload = {
        "schema_version": "herald.intervention_manifest.v1",
        "status": "complete"
        if cumulative["references"] >= expected_prompts
        and cumulative["cells"] >= expected_cells
        else "incomplete",
        "config": {**asdict(config), "results_dir": str(config.results_dir)},
        "model_id": config.model_id or MODELS.get(config.model_key),
        "dtype": config.dtype,
        "scorer": {
            "id": "herald.ifeval.instruction_level.v1",
            "modes": ["loose", "strict"],
        },
        "scorer_modes": ["loose", "strict"],
        "compressor_semantics": {
            "cache_native": CACHE_NATIVE_SEMANTICS,
            "expected_attention": EXPECTED_ATTENTION_SEMANTICS,
        },
        "compressors": list(config.compressors),
        "ratios": list(config.ratios),
        "seed": config.seed,
        "reference_artifacts": reference_files,
        "hybrid_artifacts": hybrid_files,
        "sweep_config_sha256": config_digest,
        "intervention_config_sha256": intervention_digest,
        "artifact_files": all_files,
        "artifact_aggregate_sha256": aggregate,
        "completion_counts": cumulative,
        "invocation_counts": counts,
    }
    tmp = root / ".manifest.tmp"
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str))
    os.replace(tmp, root / "manifest.json")


def run_intervention_sweep(
    lm: LoadedModel,
    records: Iterable[PromptRecord],
    config: InterventionSweepConfig,
    *,
    scorer: Callable[..., Any] | None = None,
    allowed_gpu_pids: Iterable[int] = (),
    gpu_pid_provider: Callable[[], Iterable[int]] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Regenerate one prompt-oriented raw tree, resuming completed cells."""
    initialize_intervention_config(config.results_dir, config)
    provider = gpu_pid_provider or active_gpu_compute_pids
    allowed = tuple(sorted({int(pid) for pid in allowed_gpu_pids}))
    root = config.results_dir / config.model_key / config.task
    root.mkdir(parents=True, exist_ok=True)
    records_slice = list(records)[: config.prompt_count]
    if len(records_slice) != config.prompt_count:
        actual = len(records_slice)
        raise ValueError(
            f"expected {config.prompt_count} prompts, got {actual}"
        )
    last_gpu_poll = float("-inf")
    cached_gpu_pids: tuple[int, ...] = ()

    def polled_gpu_pids() -> tuple[int, ...]:
        nonlocal last_gpu_poll, cached_gpu_pids
        now = time.monotonic()
        if now - last_gpu_poll >= 2.0:
            cached_gpu_pids = tuple(provider())
            last_gpu_poll = now
        return cached_gpu_pids

    existing_maps = {
        (compressor, ratio): _existing_cells(root, compressor, ratio)
        for compressor in config.compressors
        for ratio in config.ratios
    }
    counts = {"prompts": 0, "references": 0, "cells": 0, "skipped_cells": 0}
    expected_cells = 0
    for record in records_slice:
        _guard_gpu(polled_gpu_pids, allowed)
        if not record.gold:
            raise ValueError(f"missing prompt gold for {record.prompt_id}")
        try:
            stored_reference = load_reference(
                config.results_dir,
                config.model_key,
                config.task,
                record.prompt_id,
            )
        except FileNotFoundError:
            stored_reference = None
        if stored_reference is not None:
            stored_ids = stored_reference.get("gen_ids")
            stored_q = stored_reference.get("q")
            stored_q_strict = stored_reference.get("q_strict")
            if (
                not isinstance(stored_ids, list)
                or not stored_ids
                or not isinstance(stored_q, (int, float))
            ):
                raise ValueError(
                    f"invalid existing reference for {record.prompt_id}"
                )
            positions = range(0, len(stored_ids), config.stride)
            expected_for_prompt = [
                (record.prompt_id, position) for position in positions
            ]
            complete = isinstance(stored_q_strict, (int, float)) and all(
                all(
                    key in existing_maps[(compressor, ratio)]
                    for compressor in config.compressors
                    for ratio in config.ratios
                )
                for key in expected_for_prompt
            )
            reference_features = (
                root / "references" / f"{safe_id(record.prompt_id)}.npy"
            )
            if complete:
                try:
                    _validate_reference_features(
                        reference_features,
                        len(stored_ids),
                    )
                except ValueError:
                    complete = False
            if complete:
                if not isinstance(stored_q_strict, (int, float)):
                    raise RuntimeError(
                        "complete reference lacks a strict quality score"
                    )
                for compressor in config.compressors:
                    semantics = (
                        EXPECTED_ATTENTION_SEMANTICS
                        if compressor == "expected_attention"
                        else CACHE_NATIVE_SEMANTICS
                    )
                    for ratio in config.ratios:
                        cells = existing_maps[(compressor, ratio)]
                        for key in expected_for_prompt:
                            _validate_existing(
                                {key: cells[key]},
                                ref_q=float(stored_q),
                                ref_q_strict=float(stored_q_strict),
                                semantics=semantics,
                            )
                ensure_reference_done(
                    config.results_dir,
                    config.model_key,
                    config.task,
                    record.prompt_id,
                )
                counts["references"] += 1
                counts["prompts"] += 1
                counts["skipped_cells"] += (
                    len(expected_for_prompt)
                    * len(config.compressors)
                    * len(config.ratios)
                )
                expected_cells += (
                    len(expected_for_prompt)
                    * len(config.compressors)
                    * len(config.ratios)
                )
                if progress:
                    progress(
                        {
                            "event": "prompt_skipped",
                            "prompt_id": record.prompt_id,
                            "reason": "complete",
                        }
                    )
                continue
        treatments: list[dict[str, Any]] = []

        def on_boundary(
            s: int,
            prefix_ids: list[int],
            pending_id: int,
            source_cache: Any,
            prompt: torch.Tensor,
            current_record: PromptRecord = record,
            current_treatments: list[dict[str, Any]] = treatments,
        ) -> None:
            nonlocal expected_cells
            expected_cells += len(config.compressors) * len(config.ratios)
            sham_ids: list[int] | None = None
            sham_scores: tuple[float, float] | None = None
            for compressor in config.compressors:
                semantics = (
                    EXPECTED_ATTENTION_SEMANTICS
                    if compressor == "expected_attention"
                    else CACHE_NATIVE_SEMANTICS
                )
                for ratio in config.ratios:
                    _guard_gpu(polled_gpu_pids, allowed)
                    key = (current_record.prompt_id, s)
                    existing = existing_maps[(compressor, ratio)]
                    _validate_existing(
                        {key: existing[key]} if key in existing else {},
                        ref_q=0.0,
                        ref_q_strict=0.0,
                        semantics=semantics,
                        validate_legacy=False,
                    )
                    if key in existing:
                        counts["skipped_cells"] += 1
                        continue
                    if compressor == "expected_attention":
                        if sham_ids is None:
                            sham_ids = prefill_continuation(
                                lm,
                                prompt,
                                prefix_ids,
                                config.max_new_tokens - s,
                                None,
                                config.seed,
                            )
                            sham_scores = _strict_score_pair(
                                scorer,
                                current_record,
                                _decode(lm, prefix_ids + sham_ids),
                            )
                        pressed_ids = prefill_continuation(
                            lm,
                            prompt,
                            prefix_ids,
                            config.max_new_tokens - s,
                            get_press(compressor, ratio),
                            config.seed,
                        )
                        if sham_scores is None:
                            raise RuntimeError(
                                "ExpectedAttention sham was not computed"
                            )
                        treatment_scores = _strict_score_pair(
                            scorer,
                            current_record,
                            _decode(lm, prefix_ids + pressed_ids),
                        )
                        provenance = {
                            "control": "sham_reprefill",
                            "prefix_length": s,
                            "pending_token": pending_id,
                        }
                        treatment_ids = pressed_ids
                    else:
                        before = _cache_signature(source_cache)
                        compressed_cache = _fork_score_cache(
                            lm.model,
                            source_cache,
                            get_press(compressor, ratio),
                        )
                        if _cache_signature(source_cache) != before:
                            raise RuntimeError(
                                "cache-native intervention mutated the "
                                "source cache"
                            )
                        treatment_ids, _ = continue_from_cache(
                            lm,
                            prompt,
                            prefix_ids,
                            [pending_id],
                            compressed_cache,
                            config.max_new_tokens - s,
                        )
                        if _cache_signature(source_cache) != before:
                            raise RuntimeError(
                                "cache-native continuation mutated the "
                                "source cache"
                            )
                        treatment_scores = _strict_score_pair(
                            scorer,
                            current_record,
                            _decode(lm, prefix_ids + treatment_ids),
                        )
                        provenance = {
                            "control": "live_uncompressed_reference",
                            "prefix_length": s,
                            "pending_token": pending_id,
                        }
                    current_treatments.append(
                        {
                            "compressor": compressor,
                            "ratio": ratio,
                            "s": s,
                            "new_ids": treatment_ids,
                            "text": _decode(lm, prefix_ids + treatment_ids),
                            "scores": treatment_scores,
                            "control_scores": sham_scores,
                            "semantics": semantics,
                            "provenance": provenance,
                        }
                    )

        ref_ids, features, prompt_ids = _reference_incremental(
            lm,
            record,
            config.max_new_tokens,
            config.stride,
            on_boundary,
        )
        if not ref_ids:
            raise ValueError(
                f"reference produced no tokens for {record.prompt_id}"
            )
        reference_text = _decode(lm, ref_ids)
        q_ref, q_ref_strict = _strict_score_pair(
            scorer, record, reference_text
        )
        try:
            old = load_reference(
                config.results_dir,
                config.model_key,
                config.task,
                record.prompt_id,
            )
        except FileNotFoundError:
            old = None
        should_save_reference = old is None
        if old is not None:
            if (
                old.get("gen_ids") != ref_ids
                or old.get("text") != reference_text
                or old.get("q") != q_ref
            ):
                raise ValueError(
                    f"existing reference disagreement for {record.prompt_id}"
                )
            old_strict = old.get("q_strict")
            if old_strict is not None and old_strict != q_ref_strict:
                raise ValueError(
                    "existing strict reference disagreement for "
                    f"{record.prompt_id}"
                )
            ref_path = (
                root / "references" / f"{safe_id(record.prompt_id)}.npy"
            )
            try:
                _validate_reference_features(ref_path, len(ref_ids))
            except ValueError:
                should_save_reference = True
            if old_strict is None:
                should_save_reference = True
            if not should_save_reference:
                features = np.load(ref_path).astype(np.float32)
        if should_save_reference:
            save_reference(
                config.results_dir,
                config.model_key,
                config.task,
                prompt_id=record.prompt_id,
                prompt_input_ids=prompt_ids,
                gen_ids=ref_ids,
                text=reference_text,
                q=q_ref,
                q_strict=q_ref_strict,
                features=features,
                feature_names=list(FEATURE_NAMES),
            )
        ensure_reference_done(
            config.results_dir,
            config.model_key,
            config.task,
            record.prompt_id,
        )
        counts["references"] += 1
        for compressor in config.compressors:
            semantics = (
                EXPECTED_ATTENTION_SEMANTICS
                if compressor == "expected_attention"
                else CACHE_NATIVE_SEMANTICS
            )
            for ratio in config.ratios:
                # Validate all rows for this prompt once the reference score
                # is authoritative; no rows are rewritten on resume.
                for cell_key, cell in existing_maps[
                    (compressor, ratio)
                ].items():
                    if cell_key[0] == record.prompt_id:
                        _validate_existing(
                            {cell_key: cell},
                            ref_q=q_ref,
                            ref_q_strict=q_ref_strict,
                            semantics=semantics,
                        )
        counts["prompts"] += 1
        for treatment in treatments:
            q_hybrid, q_hybrid_strict = treatment["scores"]
            raw_control = treatment["control_scores"]
            q_control, q_control_strict = (
                raw_control
                if raw_control is not None
                else (q_ref, q_ref_strict)
            )
            append_hybrid(
                config.results_dir,
                config.model_key,
                config.task,
                treatment["compressor"],
                treatment["ratio"],
                prompt_id=record.prompt_id,
                s=treatment["s"],
                new_ids=treatment["new_ids"],
                text=treatment["text"],
                q=q_hybrid,
                dq=q_control - q_hybrid,
                q_control=q_control,
                q_control_strict=q_control_strict,
                q_hybrid_strict=q_hybrid_strict,
                dq_strict=q_control_strict - q_hybrid_strict,
                intervention_semantics=treatment["semantics"],
                generation_provenance=treatment["provenance"],
            )
            existing_maps[(treatment["compressor"], treatment["ratio"])][
                (record.prompt_id, treatment["s"])
            ] = treatment
            counts["cells"] += 1
        if progress:
            progress(
                {
                    "event": "reference_complete",
                    "prompt_id": record.prompt_id,
                    "n_tokens": len(ref_ids),
                }
            )
    _write_manifest(
        root,
        config,
        counts,
        expected_prompts=len(records_slice),
        expected_cells=expected_cells,
    )
    return counts
