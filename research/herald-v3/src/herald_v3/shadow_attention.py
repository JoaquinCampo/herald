"""Label-free shadow attention diagnostics for one SDPA continuation.

The wrapper in this module replaces the registered Hugging Face ``sdpa``
function temporarily.  It calls that function exactly once, before doing any
diagnostic work, and returns its object unchanged.  The diagnostic attention
row is a separate float32 calculation from the post-rotary query and key
tensors received by SDPA.  It never supplies values back to the model.
"""

import base64
import hashlib
import json
import random
import time
from collections.abc import (
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from herald_v3 import retrieval_discovery

MAX_NEW_TOKENS = 50
TIME_CAP_SECONDS = 300.0
SEED = 0
EXPECTED_CASE_ID = "discovery_t1_l1024_d20"
EXPECTED_CASE_GRID_SHA256 = (
    "992566f327e20acec06554f72e3f6aa0f1d755643042ae29c1f918bbc219dae1"
)
EXPECTED_QUERY_HEADS = 28
EXPECTED_KV_HEADS = 4
EXPECTED_GQA_GROUPS = 7
EXPECTED_HEAD_DIM = 128
DEFAULT_WITNESS_LIMIT = 2


class ShadowAttentionError(RuntimeError):
    """Raised when the SDPA shadow contract cannot be established."""


class ShadowAttentionDeadlineExceeded(ShadowAttentionError):
    """Raised when the paired slice reaches its fixed wall-clock cap."""


@dataclass
class ShadowRuntime:
    """Small synchronized runtime boundary shared by both continuations."""

    require_cuda: bool = True
    deadline_seconds: float = TIME_CAP_SECONDS
    clock: Callable[[], float] = time.perf_counter
    _deadline: float | None = field(default=None, init=False, repr=False)

    def seed(self) -> None:
        random.seed(SEED)
        try:
            import numpy as np
        except ImportError as error:
            raise ShadowAttentionError(
                "NumPy is required for deterministic shadow attention"
            ) from error
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    def validate_model(self, model: Any) -> torch.device:
        try:
            device = cast(torch.device, next(model.parameters()).device)
        except (AttributeError, StopIteration) as error:
            raise ShadowAttentionError("model has no parameters") from error
        if self.require_cuda and device.type != "cuda":
            raise ShadowAttentionError(
                "production shadow slice requires CUDA"
            )
        return device

    def start(self) -> None:
        if self._deadline is None:
            self._deadline = self.clock() + self.deadline_seconds

    def check_deadline(self) -> None:
        if self._deadline is not None and self.clock() >= self._deadline:
            raise ShadowAttentionDeadlineExceeded(
                "shadow attention time cap reached before the next model step"
            )

    def synchronize(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def reset_peak_memory(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def peak_memory(
        self, device: torch.device
    ) -> tuple[int | None, int | None]:
        if device.type != "cuda":
            return None, None
        return (
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.max_memory_reserved(device)),
        )


@dataclass(frozen=True)
class ContinuationEvidence:
    """Tokens, stop rule, cost, and final cache identity for one branch."""

    generated_token_ids: tuple[int, ...]
    stop_reason: str
    elapsed_seconds: float
    prefill_seconds: float
    decode_forward_seconds: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    final_cache_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_token_ids": list(self.generated_token_ids),
            "stop_reason": self.stop_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "prefill_seconds": self.prefill_seconds,
            "decode_forward_seconds": self.decode_forward_seconds,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "final_cache_fingerprint": self.final_cache_fingerprint,
        }


@dataclass(frozen=True)
class PairResult:
    """Paired native and shadow continuations with explicit parity gates."""

    case_id: str
    unwrapped: ContinuationEvidence
    wrapped: ContinuationEvidence
    diagnostics: dict[str, object]
    parity: dict[str, object]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "status": "completed" if self.passed else "failed",
            "passed": self.passed,
            "unwrapped": self.unwrapped.to_dict(),
            "wrapped": self.wrapped.to_dict(),
            "diagnostics": self.diagnostics,
            "parity": self.parity,
        }


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    detached = tensor.detach().contiguous().cpu()
    digest.update(f"{detached.dtype}|{tuple(detached.shape)}|".encode())
    digest.update(detached.view(torch.uint8).numpy().tobytes())


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    _hash_tensor(digest, tensor)
    return digest.hexdigest()


def _iter_cache_tensors(cache: Any) -> Iterable[torch.Tensor]:
    """Yield cache key/value tensors without retaining the cache itself."""
    if cache is None:
        return
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values", "key", "value"):
                tensor = getattr(layer, name, None)
                if isinstance(tensor, torch.Tensor):
                    yield tensor
        return
    for name in ("key_cache", "value_cache"):
        values = getattr(cache, name, None)
        if isinstance(values, (list, tuple)):
            for tensor in values:
                if isinstance(tensor, torch.Tensor):
                    yield tensor
            return
    if isinstance(cache, (list, tuple)):
        for value in cache:
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (list, tuple)):
                for tensor in value:
                    if isinstance(tensor, torch.Tensor):
                        yield tensor


def cache_fingerprint(cache: Any) -> str:
    """Hash cache tensor shapes, dtypes, and exact contents."""
    digest = hashlib.sha256()
    tensors = tuple(_iter_cache_tensors(cache))
    if not tensors:
        digest.update(repr(type(cache)).encode())
        get_length = getattr(cache, "get_seq_length", None)
        if callable(get_length):
            digest.update(str(int(get_length())).encode())
    for tensor in tensors:
        _hash_tensor(digest, tensor)
    return digest.hexdigest()


def model_state_fingerprint(model: Any) -> str:
    """Hash model tensors and training flags to detect diagnostic mutation."""
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        digest.update(f"parameter:{name}".encode())
        _hash_tensor(digest, tensor)
    for name, tensor in model.named_buffers():
        digest.update(f"buffer:{name}".encode())
        _hash_tensor(digest, tensor)
    digest.update(
        repr(
            tuple(
                (name, module.training)
                for name, module in model.named_modules()
            )
        ).encode()
    )
    return digest.hexdigest()


def rng_state_fingerprint() -> str:
    """Hash CPU and CUDA RNG states used by the paired branches."""
    digest = hashlib.sha256()
    _hash_tensor(digest, torch.random.get_rng_state())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            _hash_tensor(digest, state)
    digest.update(repr(random.getstate()).encode())
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        numpy_state = cast(
            tuple[str, Any, int, float, float], np.random.get_state()
        )
        digest.update(str(numpy_state[0]).encode())
        digest.update(numpy_state[1].tobytes())
        digest.update(repr(numpy_state[2:]).encode())
    return digest.hexdigest()


def _case_grid_sha256(cases: Sequence[Any]) -> str:
    payload = [case.to_dict() for case in cases]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(raw)


def select_fixed_case(
    tokenizer: Any,
    source_root: str | Any,
    *,
    require_pinned_grid: bool = True,
) -> tuple[Any, tuple[Any, ...], str]:
    """Build the released 24-case grid and return its first sorted case."""
    cases = tuple(retrieval_discovery.build_cases(tokenizer, source_root))
    if len(cases) != 24:
        raise ShadowAttentionError(
            "fixed shadow slice requires 24 discovery cases"
        )
    grid_sha256 = _case_grid_sha256(cases)
    first = sorted(cases, key=lambda case: case.case_id)[0]
    if first.case_id != EXPECTED_CASE_ID:
        raise ShadowAttentionError(
            "lexicographically first discovery case changed"
        )
    if require_pinned_grid and grid_sha256 != EXPECTED_CASE_GRID_SHA256:
        raise ShadowAttentionError(
            "24-case discovery grid differs from the pinned manifest"
        )
    return first, cases, grid_sha256


def first_divergence(
    unwrapped: Sequence[int], wrapped: Sequence[int]
) -> dict[str, object] | None:
    """Return the first differing token, including a length mismatch."""
    for index in range(max(len(unwrapped), len(wrapped))):
        left = unwrapped[index] if index < len(unwrapped) else None
        right = wrapped[index] if index < len(wrapped) else None
        if left != right:
            return {
                "generated_index": index,
                "unwrapped_token_id": left,
                "wrapped_token_id": right,
            }
    return None


def _record_partial_divergence(
    evidence: MutableMapping[str, object],
    unwrapped_state: Mapping[str, object],
    wrapped_state: Mapping[str, object],
) -> None:
    """Keep the first divergence available when a branch raises mid-run."""
    left = unwrapped_state.get("generated_token_ids")
    right = wrapped_state.get("generated_token_ids")
    if isinstance(left, list) and isinstance(right, list):
        evidence["first_divergence"] = first_divergence(left, right)


def _position_from_kwargs(kwargs: Mapping[str, object]) -> int | None:
    value = kwargs.get("position_ids")
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return int(value.detach().reshape(-1)[-1].item())


def _positional_or_keyword(
    args: Sequence[object],
    kwargs: Mapping[str, object],
    name: str,
    index: int,
    default: object,
) -> object:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _repeat_kv_for_shadow(
    key: torch.Tensor,
    query_heads: int,
    module: Any,
) -> tuple[torch.Tensor, int, int]:
    kv_heads = int(key.shape[1])
    groups_value = getattr(module, "num_key_value_groups", None)
    groups = int(groups_value) if groups_value is not None else 1
    if kv_heads == query_heads:
        return key, kv_heads, 1
    if groups <= 0 or kv_heads * groups != query_heads:
        raise ShadowAttentionError(
            "query and KV heads do not form valid GQA groups"
        )
    return torch.repeat_interleave(key, groups, dim=1), kv_heads, groups


def _right_aligned_mask(
    mask: torch.Tensor,
    *,
    batch_size: int,
    query_heads: int,
    query_length: int,
    key_length: int,
) -> torch.Tensor:
    """Broadcast an SDPA mask and crop leading excess positions like HF."""
    if mask.ndim == 2:
        aligned = mask[:, None, None, :]
    elif mask.ndim == 3:
        aligned = mask[:, None, :, :]
    elif mask.ndim == 4:
        aligned = mask
    else:
        raise ShadowAttentionError(
            "SDPA mask must have 2, 3, or 4 dimensions"
        )
    if aligned.shape[0] not in (1, batch_size):
        raise ShadowAttentionError(
            "SDPA mask batch dimension is incompatible"
        )
    if aligned.shape[-2] not in (1, query_length):
        if aligned.shape[-2] < query_length:
            raise ShadowAttentionError(
                "SDPA mask query dimension is too short"
            )
        aligned = aligned[..., -query_length:, :]
    if aligned.shape[-1] < key_length:
        raise ShadowAttentionError("SDPA mask key dimension is too short")
    aligned = aligned[..., :key_length]
    if aligned.shape[1] not in (1, query_heads):
        raise ShadowAttentionError("SDPA mask head dimension is incompatible")
    return aligned.expand(batch_size, query_heads, query_length, key_length)


def _encode_snapshot(tensor: torch.Tensor) -> dict[str, object]:
    """Encode an exact small CPU witness without keeping a device tensor."""
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(cpu.dtype).removeprefix("torch."),
        "shape": list(cpu.shape),
        "encoding": "base64_little_endian",
        "data": base64.b64encode(raw).decode("ascii"),
    }


class ShadowAttentionCollector:
    """Temporary SDPA replacement and bounded qlen-1 diagnostic collector."""

    def __init__(
        self,
        *,
        witness_limit: int = DEFAULT_WITNESS_LIMIT,
        strict_contract: bool = False,
    ) -> None:
        if witness_limit < 0:
            raise ValueError("witness_limit must be nonnegative")
        self.witness_limit = witness_limit
        self.strict_contract = strict_contract
        self.native_call_count = 0
        self.prefill_call_count = 0
        self.decode_call_count = 0
        self._identity_checks: list[bool] = []
        self.mapping_restored = False
        self._native: Callable[..., object] | None = None
        self._pending_step: int | None = None
        self._pending_records: list[dict[str, object]] = []
        self._records: list[dict[str, object]] = []
        self._witnesses: list[dict[str, object]] = []
        self._attention_digest = hashlib.sha256()

    def __enter__(self) -> "ShadowAttentionCollector":
        try:
            from transformers import AttentionInterface
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except ImportError as error:
            raise ShadowAttentionError(
                "installed Transformers has no AttentionInterface "
                "SDPA registry"
            ) from error
        try:
            self._native = ALL_ATTENTION_FUNCTIONS["sdpa"]
        except (KeyError, TypeError) as error:
            raise ShadowAttentionError(
                "native SDPA function is not registered"
            ) from error

        def wrapped(
            module: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None,
            *args: object,
            **kwargs: object,
        ) -> object:
            assert self._native is not None
            self.native_call_count += 1
            native_result = self._native(
                module, query, key, value, attention_mask, *args, **kwargs
            )
            returned_result = native_result
            self._identity_checks.append(returned_result is native_result)
            if query.ndim != 4 or int(query.shape[2]) != 1:
                self.prefill_call_count += 1
                return returned_result
            self.decode_call_count += 1
            self._collect_decode_row(
                module, query, key, value, attention_mask, args, kwargs
            )
            return returned_result

        self._wrapped = wrapped
        AttentionInterface.register("sdpa", wrapped)
        return self

    def __exit__(
        self, exc_type: object, exc_value: object, traceback: object
    ) -> None:
        del exc_type, exc_value, traceback
        if self._native is None:
            return
        from transformers import AttentionInterface

        AttentionInterface.register("sdpa", self._native)
        self.mapping_restored = True

    def begin_step(self, step_index: int) -> None:
        if self._pending_step is not None:
            raise ShadowAttentionError(
                "previous shadow attention step was not finalized"
            )
        self._pending_step = step_index
        self._pending_records = []

    def finish_step(self, generated_token_id: int) -> None:
        if self._pending_step is None:
            raise ShadowAttentionError(
                "shadow attention step was not started"
            )
        records = self._pending_records
        if self.strict_contract and len(records) != EXPECTED_QUERY_HEADS:
            raise ShadowAttentionError(
                "strict shadow slice did not observe exactly 28 layer "
                "calls per step"
            )
        layers = [int(cast(Any, record["layer_idx"])) for record in records]
        if self.strict_contract and sorted(layers) != list(range(28)):
            raise ShadowAttentionError(
                "strict shadow slice has unstable layer call accounting"
            )
        for record in records:
            record["generated_token_id"] = generated_token_id
            record["step_index"] = self._pending_step
            self._records.append(record)
        for witness in self._witnesses:
            if witness.get("step_index") == self._pending_step:
                witness["generated_token_id"] = generated_token_id
        self._pending_step = None
        self._pending_records = []

    def _collect_decode_row(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> None:
        del value
        if query.ndim != 4 or key.ndim != 4:
            raise ShadowAttentionError("SDPA query and key must be rank four")
        batch_size, query_heads, query_length, head_dim = query.shape
        key_batch, _, key_length, key_dim = key.shape
        if (
            key_batch != batch_size
            or key_dim != head_dim
            or query_length != 1
        ):
            raise ShadowAttentionError(
                "SDPA query and key shapes are incompatible"
            )
        repeated_key, kv_heads, groups = _repeat_kv_for_shadow(
            key, int(query_heads), module
        )
        dropout = _positional_or_keyword(args, kwargs, "dropout", 0, 0.0)
        scaling = _positional_or_keyword(args, kwargs, "scaling", 1, None)
        is_causal = _positional_or_keyword(args, kwargs, "is_causal", 2, None)
        if scaling is None:
            scale = float(head_dim) ** -0.5
        else:
            scale = float(cast(Any, scaling))
        if is_causal is None:
            is_causal = getattr(module, "is_causal", True)
        if self.strict_contract:
            if bool(getattr(module, "training", True)):
                raise ShadowAttentionError(
                    "strict shadow slice requires eval mode"
                )
            if (
                int(query_heads) != EXPECTED_QUERY_HEADS
                or kv_heads != EXPECTED_KV_HEADS
                or groups != EXPECTED_GQA_GROUPS
                or int(head_dim) != EXPECTED_HEAD_DIM
            ):
                raise ShadowAttentionError(
                    "strict shadow slice GQA dimensions differ"
                )
            if attention_mask is not None:
                raise ShadowAttentionError(
                    "strict SDPA decode unexpectedly received a mask"
                )
            if float(cast(Any, dropout)) != 0.0:
                raise ShadowAttentionError(
                    "strict SDPA decode dropout is not zero"
                )
            position = _position_from_kwargs(kwargs)
            if position is None or int(key_length) != position + 1:
                raise ShadowAttentionError(
                    "strict SDPA key length is not right-aligned to position"
                )

        query_float = query.detach().float()
        key_float = repeated_key.detach().float()
        scores = (
            torch.matmul(query_float, key_float.transpose(-2, -1)) * scale
        )
        effective_causal = (
            bool(is_causal)
            and int(query_length) > 1
            and attention_mask is None
        )
        mask_kind = "none"
        if attention_mask is not None:
            aligned = _right_aligned_mask(
                attention_mask,
                batch_size=int(batch_size),
                query_heads=int(query_heads),
                query_length=int(query_length),
                key_length=int(key_length),
            )
            if aligned.dtype == torch.bool:
                mask_kind = "boolean_allowed"
                scores = scores.masked_fill(~aligned, float("-inf"))
            elif aligned.is_floating_point():
                mask_kind = "additive"
                scores = scores + aligned.to(dtype=scores.dtype)
            else:
                raise ShadowAttentionError(
                    "SDPA mask must be boolean or floating point"
                )
        elif effective_causal:
            mask_kind = "causal"
            query_positions = torch.arange(
                int(query_length), device=scores.device
            ) + (int(key_length) - int(query_length))
            key_positions = torch.arange(
                int(key_length), device=scores.device
            )
            allowed = key_positions[None, :] <= query_positions[:, None]
            scores = scores.masked_fill(
                ~allowed[None, None, :, :], float("-inf")
            )

        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.where(
            torch.isfinite(probabilities),
            probabilities,
            torch.zeros_like(probabilities),
        )
        row_sums = probabilities.sum(dim=-1)
        finite = bool(torch.isfinite(row_sums).all().item())
        top_count = min(2, int(key_length))
        top_values, top_indices_tensor = torch.topk(
            probabilities, k=top_count, dim=-1
        )
        top_indices = top_indices_tensor[0, :, 0, 0].detach().cpu().tolist()
        top_probabilities = top_values[0, :, 0, 0].detach().cpu().tolist()
        if top_count == 2:
            margins = (
                (top_values[:, :, 0, 0] - top_values[:, :, 0, 1])[0]
                .detach()
                .cpu()
                .tolist()
            )
        else:
            margins = [0.0] * int(query_heads)
        sums = row_sums[0, :, 0].detach().cpu().tolist()
        normalized_error = max(
            (abs(float(value) - 1.0) for value in sums), default=0.0
        )
        row_checksum = _tensor_sha256(probabilities[0, :, 0])
        self._attention_digest.update(row_checksum.encode("ascii"))
        record: dict[str, object] = {
            "layer_idx": int(
                getattr(module, "layer_idx", len(self._records))
            ),
            "query_length": int(query_length),
            "key_length": int(key_length),
            "query_heads": int(query_heads),
            "kv_heads": kv_heads,
            "num_key_value_groups": groups,
            "scale": scale,
            "is_causal": bool(is_causal),
            "effective_is_causal": effective_causal,
            "mask_kind": mask_kind,
            "finite": finite,
            "normalized_attention_row_sums": [float(value) for value in sums],
            "row_normalization_error": float(normalized_error),
            "normalized_row_checksum": row_checksum,
            "top_indices": [int(value) for value in top_indices],
            "top_probabilities": [
                float(value) for value in top_probabilities
            ],
            "top1_top2_margin": [float(value) for value in margins],
            "index_checksum": _sha256(
                json.dumps(
                    [int(value) for value in top_indices],
                    separators=(",", ":"),
                ).encode()
            ),
        }
        self._pending_records.append(record)
        if (
            self._pending_step is not None
            and len(self._witnesses) < self.witness_limit
            and int(getattr(module, "layer_idx", -1)) == 0
        ):
            query_head = 0
            kv_head = query_head // groups
            mask_row: torch.Tensor | None = None
            if attention_mask is not None:
                mask_row = _right_aligned_mask(
                    attention_mask,
                    batch_size=int(batch_size),
                    query_heads=int(query_heads),
                    query_length=int(query_length),
                    key_length=int(key_length),
                )[0, query_head, 0]
            self._witnesses.append(
                {
                    "step_index": self._pending_step,
                    "layer_idx": int(getattr(module, "layer_idx", 0)),
                    "query_head": query_head,
                    "kv_head": kv_head,
                    "scale": scale,
                    "key_length": int(key_length),
                    "top_index": int(top_indices[query_head]),
                    "normalized_row_checksum": row_checksum,
                    "query_snapshot": _encode_snapshot(
                        query[0, query_head, 0]
                    ),
                    "key_snapshot": _encode_snapshot(key[0, kv_head]),
                    "mask_snapshot": (
                        _encode_snapshot(mask_row)
                        if mask_row is not None
                        else None
                    ),
                    "normalized_row_snapshot": _encode_snapshot(
                        probabilities[0, query_head, 0]
                    ),
                }
            )
        del scores, probabilities, query_float, key_float, repeated_key

    def to_dict(self) -> dict[str, object]:
        records = [*self._records, *self._pending_records]
        row_sums = [
            value
            for record in records
            for value in cast(
                Sequence[object], record["normalized_attention_row_sums"]
            )
        ]
        return {
            "native_call_count": self.native_call_count,
            "prefill_call_count": self.prefill_call_count,
            "decode_call_count": self.decode_call_count,
            "expected_native_once_per_wrapper_call": True,
            "native_call_count_matches_counter_partition": (
                self.native_call_count
                == self.prefill_call_count + self.decode_call_count
            ),
            "native_return_identity_preserved": bool(
                self._identity_checks and all(self._identity_checks)
            ),
            "attention_registry_restored": self.mapping_restored,
            "unfinished_step": self._pending_step,
            "record_count": len(records),
            "finite": all(bool(record["finite"]) for record in records),
            "normalized_row_sum_min": min(
                (float(cast(Any, value)) for value in row_sums), default=None
            ),
            "normalized_row_sum_max": max(
                (float(cast(Any, value)) for value in row_sums), default=None
            ),
            "attention_row_checksum": self._attention_digest.hexdigest(),
            "index_checksum": _sha256(
                json.dumps(
                    [record["top_indices"] for record in records],
                    separators=(",", ":"),
                ).encode()
            ),
            "records": self._records,
            "witnesses": self._witnesses,
        }


def _run_continuation(
    model: Any,
    tokenizer: Any,
    case: Any,
    *,
    runtime: ShadowRuntime,
    collector: ShadowAttentionCollector | None,
    state: MutableMapping[str, object],
) -> ContinuationEvidence:
    device = runtime.validate_model(model)
    runtime.check_deadline()
    state["generated_token_ids"] = []
    runtime.reset_peak_memory(device)
    started = runtime.clock()
    input_ids = torch.tensor(
        [case.prompt_ids], dtype=torch.long, device=device
    )
    if input_ids.shape[1] < 2:
        raise ShadowAttentionError("prompt must contain at least two tokens")
    prefill_started = runtime.clock()
    runtime.synchronize(device)
    with torch.no_grad():
        prefill = model(
            input_ids=input_ids[:, :-1],
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            logits_to_keep=1,
        )
    runtime.synchronize(device)
    prefill_seconds = runtime.clock() - prefill_started
    cache = prefill.past_key_values
    del prefill
    current = input_ids[:, -1:]
    generated: list[int] = []
    stop_reason = "max_tokens"
    seen_non_whitespace = False
    eos_ids = retrieval_discovery._eos_ids(model, tokenizer)
    decode_forward_seconds = 0.0
    for step_index in range(MAX_NEW_TOKENS):
        runtime.check_deadline()
        cache_length = int(cache.get_seq_length())
        attention_mask = torch.ones(
            (1, cache_length + 1), dtype=torch.long, device=device
        )
        position = torch.tensor(
            [[cache_length]], dtype=torch.long, device=device
        )
        if collector is not None:
            collector.begin_step(step_index)
        forward_started = runtime.clock()
        runtime.synchronize(device)
        with torch.no_grad():
            output = model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=position,
                cache_position=position[0],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
                logits_to_keep=1,
            )
        runtime.synchronize(device)
        decode_forward_seconds += runtime.clock() - forward_started
        next_cache = output.past_key_values
        token_id = int(output.logits[:, -1, :].argmax(dim=-1).item())
        if collector is not None:
            collector.finish_step(token_id)
        generated.append(token_id)
        state["generated_token_ids"] = list(generated)
        del output
        cache = next_cache
        if token_id in eos_ids:
            stop_reason = "eos"
            break
        piece = retrieval_discovery._decode(tokenizer, (token_id,))
        prior_text = retrieval_discovery._decode(tokenizer, generated)
        if piece.strip():
            seen_non_whitespace = True
        newline = prior_text.find("\n")
        if (
            seen_non_whitespace
            and newline >= 0
            and prior_text[:newline].strip()
        ):
            stop_reason = "line_break"
            break
        current = torch.tensor([[token_id]], dtype=torch.long, device=device)
    peak_allocated, peak_reserved = runtime.peak_memory(device)
    return ContinuationEvidence(
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
        elapsed_seconds=runtime.clock() - started,
        prefill_seconds=prefill_seconds,
        decode_forward_seconds=decode_forward_seconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        final_cache_fingerprint=cache_fingerprint(cache),
    )


def run_paired_continuations(
    model: Any,
    tokenizer: Any,
    case: Any,
    *,
    runtime: ShadowRuntime | None = None,
    strict_contract: bool = False,
    witness_limit: int = DEFAULT_WITNESS_LIMIT,
    state: MutableMapping[str, object] | None = None,
) -> PairResult:
    """Run native SDPA first, then wrapped SDPA, under one fixed deadline."""
    runtime = runtime or ShadowRuntime()
    evidence = state if state is not None else {}
    evidence.update(
        {
            "case_id": case.case_id,
            "case_prompt_token_ids": list(case.prompt_ids),
            "unwrapped_token_ids": None,
            "wrapped_token_ids": None,
            "first_divergence": None,
        }
    )
    runtime.validate_model(model)
    if (
        getattr(getattr(model, "config", None), "_attn_implementation", None)
        != "sdpa"
    ):
        raise ShadowAttentionError(
            "paired shadow slice requires the SDPA backend"
        )
    runtime.start()
    runtime.seed()
    before_unwrapped_model = model_state_fingerprint(model)
    before_unwrapped_rng = rng_state_fingerprint()
    unwrapped_state: dict[str, object] = {}
    try:
        unwrapped = _run_continuation(
            model,
            tokenizer,
            case,
            runtime=runtime,
            collector=None,
            state=unwrapped_state,
        )
    finally:
        evidence["unwrapped_token_ids"] = unwrapped_state.get(
            "generated_token_ids"
        )
    evidence["unwrapped_token_ids"] = list(unwrapped.generated_token_ids)
    after_unwrapped_model = model_state_fingerprint(model)
    after_unwrapped_rng = rng_state_fingerprint()
    runtime.check_deadline()

    runtime.seed()
    before_wrapped_model = model_state_fingerprint(model)
    before_wrapped_rng = rng_state_fingerprint()
    collector = ShadowAttentionCollector(
        witness_limit=witness_limit,
        strict_contract=strict_contract,
    )
    wrapped_state: dict[str, object] = {}
    try:
        with collector:
            wrapped = _run_continuation(
                model,
                tokenizer,
                case,
                runtime=runtime,
                collector=collector,
                state=wrapped_state,
            )
    finally:
        evidence["wrapped_token_ids"] = wrapped_state.get(
            "generated_token_ids"
        )
        _record_partial_divergence(evidence, unwrapped_state, wrapped_state)
        evidence["diagnostics"] = collector.to_dict()
    evidence["wrapped_token_ids"] = list(wrapped.generated_token_ids)
    after_wrapped_model = model_state_fingerprint(model)
    after_wrapped_rng = rng_state_fingerprint()
    evidence["diagnostics"] = collector.to_dict()
    runtime.check_deadline()

    divergence = first_divergence(
        unwrapped.generated_token_ids, wrapped.generated_token_ids
    )
    evidence["first_divergence"] = divergence
    token_equal = divergence is None
    stop_equal = unwrapped.stop_reason == wrapped.stop_reason
    cache_equal = (
        unwrapped.final_cache_fingerprint == wrapped.final_cache_fingerprint
    )
    model_equal = (
        before_unwrapped_model
        == after_unwrapped_model
        == before_wrapped_model
        == after_wrapped_model
    )
    rng_equal = (
        before_unwrapped_rng == before_wrapped_rng
        and after_unwrapped_rng == after_wrapped_rng
    )
    diagnostics = collector.to_dict()
    native_once = bool(
        diagnostics["native_call_count_matches_counter_partition"]
        and diagnostics["native_return_identity_preserved"]
        and diagnostics["attention_registry_restored"]
    )
    diagnostic_finite = bool(diagnostics["finite"])
    diagnostic_count = int(cast(int, diagnostics["record_count"]))
    expected_count = (
        len(wrapped.generated_token_ids) * EXPECTED_QUERY_HEADS
        if strict_contract
        else diagnostic_count
    )
    diagnostic_count_ok = diagnostic_count == expected_count
    parity: dict[str, object] = {
        "tokens_exact": token_equal,
        "first_divergence": divergence,
        "stop_reason_equal": stop_equal,
        "cache_fingerprint_equal": cache_equal,
        "model_state_equal": model_equal,
        "rng_state_equal": rng_equal,
        "native_invocation_once_and_identity": native_once,
        "diagnostics_finite": diagnostic_finite,
        "diagnostic_count_expected": diagnostic_count_ok,
        "model_state_fingerprints": {
            "before_unwrapped": before_unwrapped_model,
            "after_unwrapped": after_unwrapped_model,
            "before_wrapped": before_wrapped_model,
            "after_wrapped": after_wrapped_model,
        },
        "rng_state_fingerprints": {
            "before_unwrapped": before_unwrapped_rng,
            "after_unwrapped": after_unwrapped_rng,
            "before_wrapped": before_wrapped_rng,
            "after_wrapped": after_wrapped_rng,
        },
        "cache_fingerprints": {
            "unwrapped": unwrapped.final_cache_fingerprint,
            "wrapped": wrapped.final_cache_fingerprint,
        },
    }
    passed = (
        token_equal
        and stop_equal
        and cache_equal
        and model_equal
        and rng_equal
        and native_once
        and diagnostic_finite
        and diagnostic_count_ok
    )
    return PairResult(
        case_id=case.case_id,
        unwrapped=unwrapped,
        wrapped=wrapped,
        diagnostics=diagnostics,
        parity=parity,
        passed=passed,
    )
