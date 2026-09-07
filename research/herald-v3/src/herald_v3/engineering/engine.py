"""Paired KV-cache acceptance engine at the token-32 decision boundary.

The cache cloning and direct layer replacement are adapted from HERALD v2
``src/herald/live_controller.py`` at git revision
``b19d18bae5e81f07a1db9eac9afcc59e1cd5f25a``. The reviewed source file had
SHA-256 ``bc8a3aabb7aa8679d89acc229d37b8a071c40d534e639320c4ef20a8c22d8fbd``.

The Knorm selection exactly follows kvpress 0.5.2
``kvpress/presses/knorm_press.py`` (SHA-256
``26ea0d4f41e6eb120c474556d0b5461ad5ab9d029f4d74767b11f9683b8d824f``)
and ``kvpress/presses/scorer_press.py`` (SHA-256
``a5eb57a8d9defdaf1f46414fad1584e0c070643a85116d9a85f2e9321cb29728``):
score each key as negative L2 norm, retain
``int(length * (1 - ratio))`` entries with ``topk``, then gather keys and
values with the same per-head indices. This module keeps that small operation
local so it does not import the v2 controller or its policy stack.
"""

import hashlib
import json
import math
import time
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class ActionSpec:
    """One direct cache action."""

    name: str
    removal_fraction: float

    def __post_init__(self) -> None:
        if self.name != "knorm":
            raise ValueError(f"unsupported action: {self.name}")
        if not 0.0 <= self.removal_fraction < 1.0:
            raise ValueError("removal_fraction must be in [0, 1)")

    @property
    def action_id(self) -> str:
        return f"{self.name}:{self.removal_fraction:.6g}"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "removal_fraction": self.removal_fraction,
            "action_id": self.action_id,
        }


@dataclass(frozen=True)
class RngSnapshot:
    """Exact torch RNG state used to make independent branches repeatable."""

    cpu: torch.Tensor = field(repr=False)
    cuda: tuple[torch.Tensor, ...] = field(default=(), repr=False)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        _hash_tensor(digest, self.cpu)
        for state in self.cuda:
            _hash_tensor(digest, state)
        return digest.hexdigest()


@dataclass
class BoundaryState:
    """Decoder state before generated token index 31 is processed."""

    prompt_ids: torch.Tensor = field(repr=False)
    generated_ids: tuple[int, ...]
    cache: Any = field(repr=False)
    pending_token_id: int
    logical_position: int
    attention_mask: torch.Tensor = field(repr=False)
    rng_state: RngSnapshot = field(repr=False)
    state_fingerprint: str
    cache_lengths: tuple[int, ...]
    cache_bytes: int
    model_state_fingerprint: str = ""
    model_tensor_count: int = 0
    validation_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_token_ids": _tensor_token_ids(self.prompt_ids),
            "prompt_length": int(self.prompt_ids.shape[1]),
            "generated_token_ids": list(self.generated_ids),
            "generated_count": len(self.generated_ids),
            "pending_token_id": self.pending_token_id,
            "pending_generated_index": len(self.generated_ids) - 1,
            "logical_position": self.logical_position,
            "attention_mask": _tensor_token_ids(self.attention_mask),
            "cache_lengths": list(self.cache_lengths),
            "cache_bytes": self.cache_bytes,
            "state_fingerprint": self.state_fingerprint,
            "rng_fingerprint": self.rng_state.fingerprint,
            "model_state_fingerprint": self.model_state_fingerprint,
            "model_tensor_count": self.model_tensor_count,
            "validation_seconds": self.validation_seconds,
        }


@dataclass(frozen=True)
class EligibilityRecord:
    """Whether a prompt reaches a usable pending-token boundary."""

    eligible: bool
    reason: str
    requested_decision_tokens: int
    generated_token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "requested_decision_tokens": self.requested_decision_tokens,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_count": len(self.generated_token_ids),
        }


@dataclass(frozen=True)
class CompressionEvidence:
    """Physical effect and lineage of one direct Knorm transform."""

    action: ActionSpec
    kept_indices: tuple[tuple[tuple[int, ...], ...], ...]
    before_lengths: tuple[int, ...]
    after_lengths: tuple[int, ...]
    before_bytes: int
    after_bytes: int
    before_fingerprint: str
    after_fingerprint: str
    compression_seconds: float
    validation_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.to_dict(),
            "kept_indices": [
                [list(head) for head in layer] for layer in self.kept_indices
            ],
            "before_lengths": list(self.before_lengths),
            "after_lengths": list(self.after_lengths),
            "before_bytes": self.before_bytes,
            "after_bytes": self.after_bytes,
            "retained_byte_fraction": (
                self.after_bytes / self.before_bytes
                if self.before_bytes
                else None
            ),
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "compression_seconds": self.compression_seconds,
            "validation_seconds": self.validation_seconds,
        }


@dataclass(frozen=True)
class DistributionProbe:
    """Compact full-vocabulary comparison without retained distributions."""

    action: ActionSpec
    js_divergence: float
    reference_entropy: float
    action_entropy: float
    reference_top2_margin: float
    action_top2_margin: float
    argmax_match: bool
    reference_argmax: int
    action_argmax: int
    reference_probability_sum: float
    action_probability_sum: float
    max_probability_difference: float
    max_logit_difference: float
    finite: bool
    reduction_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.to_dict(),
            "js_divergence": self.js_divergence,
            "reference_entropy": self.reference_entropy,
            "action_entropy": self.action_entropy,
            "reference_top2_margin": self.reference_top2_margin,
            "action_top2_margin": self.action_top2_margin,
            "argmax_match": self.argmax_match,
            "reference_argmax": self.reference_argmax,
            "action_argmax": self.action_argmax,
            "reference_probability_sum": self.reference_probability_sum,
            "action_probability_sum": self.action_probability_sum,
            "max_probability_difference": self.max_probability_difference,
            "max_logit_difference": self.max_logit_difference,
            "finite": self.finite,
            "reduction_seconds": self.reduction_seconds,
        }


@dataclass(frozen=True)
class ContinuationResult:
    """A complete generated answer, including the shared first 32 tokens."""

    token_ids: tuple[int, ...]
    termination_reason: str
    forward_seconds: float
    first_forward_seconds: float
    final_cache_lengths: tuple[int, ...]
    final_cache_bytes: int
    final_cache_fingerprint: str
    validation_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "token_count": len(self.token_ids),
            "termination_reason": self.termination_reason,
            "forward_seconds": self.forward_seconds,
            "first_forward_seconds": self.first_forward_seconds,
            "final_cache_lengths": list(self.final_cache_lengths),
            "final_cache_bytes": self.final_cache_bytes,
            "final_cache_fingerprint": self.final_cache_fingerprint,
            "validation_seconds": self.validation_seconds,
        }


@dataclass(frozen=True)
class ArmResult:
    """One action outcome and its pre-decision evidence."""

    action: ActionSpec
    compression: CompressionEvidence
    continuation: ContinuationResult
    probe: DistributionProbe | None
    probe_enabled: bool
    state_copy_seconds: float
    device_memory_baseline_bytes: int | None
    device_peak_allocated_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.to_dict(),
            "compression": self.compression.to_dict(),
            "continuation": self.continuation.to_dict(),
            "probe": self.probe.to_dict() if self.probe is not None else None,
            "probe_enabled": self.probe_enabled,
            "state_copy_seconds": self.state_copy_seconds,
            "device_memory_baseline_bytes": self.device_memory_baseline_bytes,
            "device_peak_allocated_bytes": self.device_peak_allocated_bytes,
            "device_peak_increment_bytes": (
                self.device_peak_allocated_bytes
                - self.device_memory_baseline_bytes
                if self.device_peak_allocated_bytes is not None
                and self.device_memory_baseline_bytes is not None
                else None
            ),
        }


@dataclass(frozen=True)
class GateResult:
    """One explicit engineering acceptance assertion."""

    name: str
    passed: bool
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": to_builtin(self.details),
        }


@dataclass
class AcceptanceResult:
    """Serializable evidence for one prompt acceptance run."""

    model: dict[str, object]
    eligibility: EligibilityRecord
    boundary: BoundaryState | None = None
    uninterrupted: ContinuationResult | None = None
    noop_forks: tuple[ArmResult, ...] = ()
    action_arms: tuple[ArmResult, ...] = ()
    reverse_action_arms: tuple[ArmResult, ...] = ()
    gates: tuple[GateResult, ...] = ()
    timing_seconds: dict[str, float] = field(default_factory=dict)
    memory: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.eligibility.eligible
            and bool(self.gates)
            and all(gate.passed for gate in self.gates)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "passed": self.passed,
            "model": to_builtin(self.model),
            "eligibility": self.eligibility.to_dict(),
            "boundary": (
                self.boundary.to_dict() if self.boundary is not None else None
            ),
            "uninterrupted": (
                self.uninterrupted.to_dict()
                if self.uninterrupted is not None
                else None
            ),
            "noop_forks": [arm.to_dict() for arm in self.noop_forks],
            "action_arms": [arm.to_dict() for arm in self.action_arms],
            "reverse_action_arms": [
                arm.to_dict() for arm in self.reverse_action_arms
            ],
            "gates": [gate.to_dict() for gate in self.gates],
            "timing_seconds": dict(self.timing_seconds),
            "memory": to_builtin(self.memory),
        }


class EarlyBoundaryTermination(RuntimeError):
    """Raised internally when EOS prevents a decision-time probe."""

    def __init__(self, generated_ids: tuple[int, ...], reason: str) -> None:
        super().__init__(reason)
        self.generated_ids = generated_ids
        self.reason = reason


def to_builtin(value: Any) -> Any:
    """Convert compact evidence values to JSON-compatible built-ins."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_builtin(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        raise TypeError("raw non-scalar tensors are excluded from evidence")
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    raise TypeError(f"cannot serialize evidence value {type(value).__name__}")


def clone_cache(cache: Any) -> Any:
    """Clone a Transformers DynamicCache with independent tensor storage."""
    _require_dynamic_cache(cache)
    cloned = copy(cache)
    cloned.layers = [copy(layer) for layer in cache.layers]
    for source_layer, cloned_layer in zip(
        cache.layers, cloned.layers, strict=True
    ):
        for name in ("keys", "values"):
            value = getattr(source_layer, name, None)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"initialized cache layer has no tensor {name}"
                )
            setattr(cloned_layer, name, value.detach().clone())
    return cloned


def cache_storage_independent(source: Any, cloned: Any) -> bool:
    """Return whether corresponding cache tensors have distinct storage."""
    try:
        left = _cache_tensors(source)
        right = _cache_tensors(cloned)
    except TypeError:
        return False
    if len(left) != len(right):
        return False
    return all(
        a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
        for a, b in zip(left, right, strict=True)
        if a.numel() and b.numel()
    )


def cache_tensors_equal(left: Any, right: Any) -> bool:
    """Return whether corresponding cache tensors are exactly equal."""
    try:
        left_tensors = _cache_tensors(left)
        right_tensors = _cache_tensors(right)
    except TypeError:
        return False
    return len(left_tensors) == len(right_tensors) and all(
        torch.equal(a, b)
        for a, b in zip(left_tensors, right_tensors, strict=True)
    )


def compress_knorm(
    cache: Any, removal_fraction: float
) -> CompressionEvidence:
    """Apply the kvpress 0.5.2 Knorm transform directly and in place."""
    action = ActionSpec("knorm", removal_fraction)
    _require_dynamic_cache(cache)
    validation_started = time.perf_counter()
    before_lengths = cache_lengths(cache)
    if len(set(before_lengths)) != 1:
        raise ValueError(
            "Knorm acceptance requires equal full-attention lengths"
        )
    before_bytes = cache_nbytes(cache)
    before_fingerprint = cache_fingerprint(cache)
    validation_seconds = time.perf_counter() - validation_started
    device = _cache_tensors(cache)[0].device
    _sync_device(device)
    started = time.perf_counter()
    retained_tensors: list[torch.Tensor] = []
    with torch.no_grad():
        for layer in cache.layers:
            keys = layer.keys
            values = layer.values
            key_length = int(keys.shape[-2])
            kept_count = int(key_length * (1.0 - removal_fraction))
            if kept_count <= 0:
                raise ValueError("Knorm action would retain no cache entries")
            if removal_fraction == 0.0:
                indices = torch.arange(
                    key_length, dtype=torch.long, device=keys.device
                ).view(1, 1, key_length)
                indices = indices.expand(keys.shape[0], keys.shape[1], -1)
            else:
                scores = -keys.norm(dim=-1)
                indices = scores.topk(kept_count, dim=-1).indices
                gather_indices = indices.unsqueeze(-1).expand(
                    -1, -1, -1, keys.shape[-1]
                )
                layer.keys = keys.gather(2, gather_indices).contiguous()
                layer.values = values.gather(2, gather_indices).contiguous()
            if removal_fraction == 0.0:
                layer.keys = keys
                layer.values = values
            retained_tensors.append(indices)
    _sync_device(device)
    elapsed = time.perf_counter() - started
    validation_started = time.perf_counter()
    retained = tuple(_indices_to_tuple(item) for item in retained_tensors)
    after_lengths = cache_lengths(cache)
    after_bytes = cache_nbytes(cache)
    after_fingerprint = cache_fingerprint(cache)
    validation_seconds += time.perf_counter() - validation_started
    return CompressionEvidence(
        action=action,
        kept_indices=retained,
        before_lengths=before_lengths,
        after_lengths=after_lengths,
        before_bytes=before_bytes,
        after_bytes=after_bytes,
        before_fingerprint=before_fingerprint,
        after_fingerprint=after_fingerprint,
        compression_seconds=elapsed,
        validation_seconds=validation_seconds,
    )


def build_boundary(
    model: Any,
    input_ids: torch.Tensor,
    *,
    decision_tokens: int = 32,
    eos_ids: set[int] | frozenset[int] = frozenset(),
) -> BoundaryState:
    """Build KV for prompt plus indices 0..30, leaving index 31 pending."""
    _validate_model_and_input(model, input_ids)
    if decision_tokens < 2:
        raise ValueError("decision_tokens must be at least 2")
    device = _model_device(model)
    prompt = input_ids.to(device)
    prompt_mask = torch.ones_like(prompt)
    with torch.no_grad():
        output = model(
            input_ids=prompt,
            attention_mask=prompt_mask,
            use_cache=True,
            return_dict=True,
        )
    cache = output.past_key_values
    _require_dynamic_cache(cache)
    _require_full_attention_cache(cache)
    logits = output.logits[:, -1, :]
    generated: list[int] = []
    prompt_length = int(prompt.shape[1])
    for generated_index in range(decision_tokens):
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos_ids:
            raise EarlyBoundaryTermination(
                tuple(generated), "eos_at_or_before_pending_boundary"
            )
        if generated_index == decision_tokens - 1:
            break
        token_tensor = torch.tensor(
            [[token]], device=device, dtype=prompt.dtype
        )
        logical_position = prompt_length + generated_index
        output, _ = _pending_forward(
            model, token_tensor, cache, logical_position
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
    expected_length = prompt_length + decision_tokens - 1
    validation_started = time.perf_counter()
    lengths = cache_lengths(cache)
    if any(length != expected_length for length in lengths):
        raise RuntimeError(
            f"boundary cache lengths {lengths} != expected {expected_length}"
        )
    logical_position = expected_length
    attention_mask = torch.ones(
        (1, expected_length),
        device=device,
        dtype=prompt_mask.dtype,
    )
    rng = _capture_rng(device)
    model_state_fingerprint, model_tensor_count = (
        _model_runtime_state_fingerprint(model)
    )
    state_fingerprint = decoder_state_fingerprint(
        prompt,
        tuple(generated),
        cache,
        logical_position,
        attention_mask=attention_mask,
        pending_token_id=generated[-1],
        rng_fingerprint=rng.fingerprint,
        model_state_fingerprint=model_state_fingerprint,
    )
    cache_bytes = cache_nbytes(cache)
    validation_seconds = time.perf_counter() - validation_started
    return BoundaryState(
        prompt_ids=prompt.detach().clone(),
        generated_ids=tuple(generated),
        cache=cache,
        pending_token_id=generated[-1],
        logical_position=logical_position,
        attention_mask=attention_mask.detach().clone(),
        rng_state=rng,
        state_fingerprint=state_fingerprint,
        cache_lengths=lengths,
        cache_bytes=cache_bytes,
        model_state_fingerprint=model_state_fingerprint,
        model_tensor_count=model_tensor_count,
        validation_seconds=validation_seconds,
    )


def continue_from_boundary(
    model: Any,
    boundary: BoundaryState,
    *,
    max_new_tokens: int,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    action: ActionSpec | None = None,
    reference_logits: torch.Tensor | None = None,
    enable_probe: bool = False,
) -> ArmResult:
    """Run an independent complete arm whose IDs include the shared prefix."""
    if max_new_tokens <= len(boundary.generated_ids):
        raise ValueError("max_new_tokens must extend beyond the boundary")
    _restore_rng(boundary.rng_state, _model_device(model))
    device = _model_device(model)
    baseline_memory = _begin_peak_memory_measurement(device)
    _sync_device(device)
    clone_started = time.perf_counter()
    cache = clone_cache(boundary.cache)
    _sync_device(device)
    state_copy_seconds = time.perf_counter() - clone_started
    selected = action if action is not None else ActionSpec("knorm", 0.0)
    compression = compress_knorm(cache, selected.removal_fraction)
    observed_probe: DistributionProbe | None = None

    def observe(logits: torch.Tensor) -> None:
        nonlocal observed_probe
        if enable_probe:
            if reference_logits is None:
                raise ValueError("probe-enabled arm needs reference logits")
            observed_probe = full_vocabulary_js(
                reference_logits, logits, selected
            )

    continuation = _continue_cache(
        model,
        boundary,
        cache,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
        first_logits_observer=observe if enable_probe else None,
    )
    peak_memory = _finish_peak_memory_measurement(device)
    if enable_probe and observed_probe is None:
        raise RuntimeError("probe-enabled arm did not observe pending logits")
    return ArmResult(
        action=selected,
        compression=compression,
        continuation=continuation,
        probe=observed_probe,
        probe_enabled=enable_probe,
        state_copy_seconds=state_copy_seconds,
        device_memory_baseline_bytes=baseline_memory,
        device_peak_allocated_bytes=peak_memory,
    )


def probe_action(
    model: Any,
    boundary: BoundaryState,
    action: ActionSpec,
) -> DistributionProbe:
    """Measure one action from two independent sandbox cache copies."""
    reference_logits, _, _, _, _ = _probe_logits(model, boundary, None)
    action_logits, _, _, _, _ = _probe_logits(model, boundary, action)
    return full_vocabulary_js(reference_logits, action_logits, action)


def full_vocabulary_js(
    reference_logits: torch.Tensor,
    action_logits: torch.Tensor,
    action: ActionSpec | None = None,
) -> DistributionProbe:
    """Compute stable Jensen-Shannon divergence using torch only."""
    selected = action if action is not None else ActionSpec("knorm", 0.0)
    if reference_logits.shape != action_logits.shape:
        raise ValueError("probe logits must have identical shapes")
    if reference_logits.ndim == 2 and reference_logits.shape[0] == 1:
        reference_logits = reference_logits[0]
        action_logits = action_logits[0]
    if reference_logits.ndim != 1:
        raise ValueError("probe expects one full-vocabulary logit vector")
    device = reference_logits.device
    _sync_device(device)
    started = time.perf_counter()
    ref = reference_logits.to(dtype=torch.float64)
    act = action_logits.to(device=device, dtype=torch.float64)
    ref_log_p = torch.log_softmax(ref, dim=-1)
    act_log_p = torch.log_softmax(act, dim=-1)
    log_mid = torch.logaddexp(ref_log_p, act_log_p) - math.log(2.0)
    ref_p = ref_log_p.exp()
    act_p = act_log_p.exp()
    divergence = 0.5 * (
        torch.sum(ref_p * (ref_log_p - log_mid))
        + torch.sum(act_p * (act_log_p - log_mid))
    )
    ref_top2 = torch.topk(ref_p, k=min(2, ref_p.numel())).values
    act_top2 = torch.topk(act_p, k=min(2, act_p.numel())).values
    ref_margin = ref_top2[0] - (ref_top2[1] if ref_top2.numel() > 1 else 0.0)
    act_margin = act_top2[0] - (act_top2[1] if act_top2.numel() > 1 else 0.0)
    values = torch.stack(
        [
            divergence,
            -(ref_p * ref_log_p).sum(),
            -(act_p * act_log_p).sum(),
            ref_p.sum(),
            act_p.sum(),
            (ref_p - act_p).abs().max(),
            (ref - act).abs().max(),
        ]
    )
    finite = bool(torch.isfinite(values).all().item())
    _sync_device(device)
    elapsed = time.perf_counter() - started
    return DistributionProbe(
        action=selected,
        js_divergence=float(divergence.item()),
        reference_entropy=float(values[1].item()),
        action_entropy=float(values[2].item()),
        reference_top2_margin=float(ref_margin.item()),
        action_top2_margin=float(act_margin.item()),
        argmax_match=bool(ref.argmax().item() == act.argmax().item()),
        reference_argmax=int(ref.argmax().item()),
        action_argmax=int(act.argmax().item()),
        reference_probability_sum=float(values[3].item()),
        action_probability_sum=float(values[4].item()),
        max_probability_difference=float(values[5].item()),
        max_logit_difference=float(values[6].item()),
        finite=finite,
        reduction_seconds=elapsed,
    )


def run_acceptance(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    ratios: tuple[float, ...] = (0.25, 0.5),
    decision_tokens: int = 32,
    js_tolerance: float = 1e-6,
) -> AcceptanceResult:
    """Run boundary, parity, probe, compression and order acceptance gates."""
    if decision_tokens != 32:
        raise ValueError("engineering acceptance requires decision_tokens=32")
    if max_new_tokens <= decision_tokens:
        raise ValueError("max_new_tokens must exceed decision_tokens")
    started_total = time.perf_counter()
    initial_rng = _capture_rng(_model_device(model))
    model_evidence = model_signature(model)
    try:
        boundary = build_boundary(
            model,
            input_ids,
            decision_tokens=decision_tokens,
            eos_ids=eos_ids,
        )
    except EarlyBoundaryTermination as error:
        return AcceptanceResult(
            model=model_evidence,
            eligibility=EligibilityRecord(
                eligible=False,
                reason=error.reason,
                requested_decision_tokens=decision_tokens,
                generated_token_ids=error.generated_ids,
            ),
            timing_seconds={"total": time.perf_counter() - started_total},
        )
    eligibility = EligibilityRecord(
        eligible=True,
        reason="reached_pending_boundary",
        requested_decision_tokens=decision_tokens,
        generated_token_ids=boundary.generated_ids,
    )
    gates: list[GateResult] = []
    expected_length = int(input_ids.shape[1]) + decision_tokens - 1
    gates.append(
        GateResult(
            "boundary_exact",
            all(
                length == expected_length for length in boundary.cache_lengths
            )
            and boundary.pending_token_id == boundary.generated_ids[31]
            and boundary.logical_position == expected_length
            and tuple(boundary.attention_mask.shape) == (1, expected_length)
            and bool(boundary.attention_mask.all().item()),
            {
                "expected_cache_length": expected_length,
                "observed_cache_lengths": boundary.cache_lengths,
                "pending_generated_index": 31,
                "logical_position": boundary.logical_position,
                "attention_mask_shape": tuple(boundary.attention_mask.shape),
                "attention_mask_all_visible": bool(
                    boundary.attention_mask.all().item()
                ),
            },
        )
    )

    _restore_rng(initial_rng, _model_device(model))
    uninterrupted = _greedy_from_prompt(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
    )
    clone_started = time.perf_counter()
    clone_a = clone_cache(boundary.cache)
    clone_b = clone_cache(boundary.cache)
    clone_seconds = time.perf_counter() - clone_started
    validation_started = time.perf_counter()
    independent = (
        cache_storage_independent(boundary.cache, clone_a)
        and cache_storage_independent(boundary.cache, clone_b)
        and cache_storage_independent(clone_a, clone_b)
        and cache_tensors_equal(boundary.cache, clone_a)
        and cache_tensors_equal(boundary.cache, clone_b)
    )
    clone_a_fingerprint = cache_fingerprint(clone_a)
    clone_b_fingerprint = cache_fingerprint(clone_b)
    validation_seconds = time.perf_counter() - validation_started
    gates.append(
        GateResult(
            "independent_equal_forks",
            independent,
            {
                "source_fingerprint": boundary.state_fingerprint,
                "fork_a_cache_fingerprint": clone_a_fingerprint,
                "fork_b_cache_fingerprint": clone_b_fingerprint,
            },
        )
    )

    noop = ActionSpec("knorm", 0.0)
    noop_b = continue_from_boundary(
        model,
        boundary,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
        action=noop,
        enable_probe=False,
    )
    (
        reference_logits,
        reference_forward_seconds,
        reference_clone_seconds,
        reference_memory_baseline,
        reference_memory_peak,
    ) = _probe_logits(model, boundary, None)
    noop_a = continue_from_boundary(
        model,
        boundary,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
        action=noop,
        reference_logits=reference_logits,
        enable_probe=True,
    )
    noop_tokens_equal = (
        noop_a.continuation.token_ids
        == noop_b.continuation.token_ids
        == uninterrupted.token_ids
    )
    gates.append(
        GateResult(
            "noop_uninterrupted_token_parity",
            noop_tokens_equal,
            {
                "uninterrupted_token_ids": uninterrupted.token_ids,
                "fork_a_token_ids": noop_a.continuation.token_ids,
                "fork_b_token_ids": noop_b.continuation.token_ids,
            },
        )
    )
    noop_state_equal = _continuations_equal(
        noop_a.continuation, noop_b.continuation
    ) and _continuations_equal(noop_a.continuation, uninterrupted)
    gates.append(
        GateResult(
            "noop_uninterrupted_state_parity",
            noop_state_equal,
            {
                "uninterrupted_final_cache_fingerprint": (
                    uninterrupted.final_cache_fingerprint
                ),
                "fork_a_final_cache_fingerprint": (
                    noop_a.continuation.final_cache_fingerprint
                ),
                "fork_b_final_cache_fingerprint": (
                    noop_b.continuation.final_cache_fingerprint
                ),
                "termination_reasons": (
                    uninterrupted.termination_reason,
                    noop_a.continuation.termination_reason,
                    noop_b.continuation.termination_reason,
                ),
            },
        )
    )
    noop_probe = noop_a.probe
    assert noop_probe is not None
    noop_js_ok = (
        noop_probe.finite
        and abs(noop_probe.js_divergence) <= js_tolerance
        and abs(noop_probe.reference_probability_sum - 1.0) <= js_tolerance
        and abs(noop_probe.action_probability_sum - 1.0) <= js_tolerance
    )
    gates.append(
        GateResult(
            "noop_distribution_parity",
            noop_js_ok,
            {
                "js_divergence": noop_probe.js_divergence,
                "tolerance": js_tolerance,
                "max_probability_difference": (
                    noop_probe.max_probability_difference
                ),
                "max_logit_difference": noop_probe.max_logit_difference,
            },
        )
    )
    noop_identity = all(
        arm.compression.before_lengths == arm.compression.after_lengths
        and arm.compression.before_bytes == arm.compression.after_bytes
        and arm.compression.before_fingerprint
        == arm.compression.after_fingerprint
        for arm in (noop_a, noop_b)
    )
    gates.append(
        GateResult(
            "noop_transform_identity",
            noop_identity,
            {
                "fork_a_before": noop_a.compression.before_fingerprint,
                "fork_a_after": noop_a.compression.after_fingerprint,
                "fork_b_before": noop_b.compression.before_fingerprint,
                "fork_b_after": noop_b.compression.after_fingerprint,
            },
        )
    )
    (
        source_after_controls,
        model_after_controls,
        rng_after_controls,
        model_tensor_count,
        state_validation_seconds,
    ) = _inspect_boundary_state(model, boundary)
    validation_seconds += state_validation_seconds
    source_unchanged_by_controls = (
        source_after_controls == boundary.state_fingerprint
    )
    model_unchanged_by_controls = (
        model_after_controls == boundary.model_state_fingerprint
        and model_tensor_count == boundary.model_tensor_count
    )
    rng_unchanged_by_controls = (
        rng_after_controls == boundary.rng_state.fingerprint
    )
    gates.extend(
        (
            GateResult(
                "source_unchanged_by_controls",
                source_unchanged_by_controls,
                {
                    "before": boundary.state_fingerprint,
                    "after": source_after_controls,
                },
            ),
            GateResult(
                "model_state_unchanged_by_controls",
                model_unchanged_by_controls,
                {
                    "before": boundary.model_state_fingerprint,
                    "after": model_after_controls,
                    "expected_tensor_count": boundary.model_tensor_count,
                    "observed_tensor_count": model_tensor_count,
                },
            ),
            GateResult(
                "rng_state_unchanged_by_controls",
                rng_unchanged_by_controls,
                {
                    "before": boundary.rng_state.fingerprint,
                    "after": rng_after_controls,
                },
            ),
        )
    )
    if (
        not independent
        or not noop_tokens_equal
        or not noop_state_equal
        or not noop_js_ok
        or not noop_identity
        or not source_unchanged_by_controls
        or not model_unchanged_by_controls
        or not rng_unchanged_by_controls
    ):
        control_arms = (noop_a, noop_b)
        return AcceptanceResult(
            model=model_evidence,
            eligibility=eligibility,
            boundary=boundary,
            uninterrupted=uninterrupted,
            noop_forks=(noop_a, noop_b),
            gates=tuple(gates),
            timing_seconds={
                "clone_controls": clone_seconds,
                "reference_probe_clone": reference_clone_seconds,
                "reference_probe_forward": reference_forward_seconds,
                "validation_overhead": _validation_overhead_seconds(
                    boundary,
                    uninterrupted,
                    control_arms,
                    validation_seconds,
                ),
                "total": time.perf_counter() - started_total,
            },
            memory=_memory_evidence(
                boundary,
                reference_memory_baseline,
                reference_memory_peak,
                (noop_a,),
                (noop_b,),
            ),
        )

    actions = tuple(ActionSpec("knorm", ratio) for ratio in ratios)
    source_before_actions = boundary.state_fingerprint
    forward_arms = tuple(
        continue_from_boundary(
            model,
            boundary,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            action=action,
            reference_logits=reference_logits,
            enable_probe=True,
        )
        for action in actions
    )
    (
        source_after_actions,
        model_after_actions,
        rng_after_actions,
        model_tensor_count,
        state_validation_seconds,
    ) = _inspect_boundary_state(model, boundary)
    validation_seconds += state_validation_seconds
    gates.append(
        GateResult(
            "source_unchanged_by_actions",
            source_before_actions == source_after_actions,
            {
                "before": source_before_actions,
                "after": source_after_actions,
            },
        )
    )
    gates.extend(
        (
            GateResult(
                "model_state_unchanged_by_actions",
                model_after_actions == boundary.model_state_fingerprint
                and model_tensor_count == boundary.model_tensor_count,
                {
                    "before": boundary.model_state_fingerprint,
                    "after": model_after_actions,
                    "expected_tensor_count": boundary.model_tensor_count,
                    "observed_tensor_count": model_tensor_count,
                },
            ),
            GateResult(
                "rng_state_unchanged_by_actions",
                rng_after_actions == boundary.rng_state.fingerprint,
                {
                    "before": boundary.rng_state.fingerprint,
                    "after": rng_after_actions,
                },
            ),
        )
    )
    del reference_logits
    reverse_arms = tuple(
        continue_from_boundary(
            model,
            boundary,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            action=action,
            enable_probe=False,
        )
        for action in reversed(actions)
    )
    reverse_by_id = {arm.action.action_id: arm for arm in reverse_arms}
    for arm in forward_arms:
        reverse = reverse_by_id[arm.action.action_id]
        gates.append(
            GateResult(
                f"probe_and_order_transparency:{arm.action.action_id}",
                _continuations_equal(arm.continuation, reverse.continuation),
                {
                    "probe_enabled_token_ids": arm.continuation.token_ids,
                    "probe_disabled_reversed_token_ids": (
                        reverse.continuation.token_ids
                    ),
                    "probe_enabled_final_cache_fingerprint": (
                        arm.continuation.final_cache_fingerprint
                    ),
                    "probe_disabled_final_cache_fingerprint": (
                        reverse.continuation.final_cache_fingerprint
                    ),
                },
            )
        )
        lengths_decreased = all(
            after < before
            for before, after in zip(
                arm.compression.before_lengths,
                arm.compression.after_lengths,
                strict=True,
            )
        )
        bytes_decreased = (
            arm.compression.after_bytes < arm.compression.before_bytes
        )
        gates.append(
            GateResult(
                f"physical_eviction:{arm.action.action_id}",
                lengths_decreased and bytes_decreased,
                {
                    "before_lengths": arm.compression.before_lengths,
                    "after_lengths": arm.compression.after_lengths,
                    "before_bytes": arm.compression.before_bytes,
                    "after_bytes": arm.compression.after_bytes,
                },
            )
        )
        probe = arm.probe
        assert probe is not None
        gates.append(
            GateResult(
                f"finite_probe:{arm.action.action_id}",
                probe.finite
                and abs(probe.reference_probability_sum - 1.0) <= js_tolerance
                and abs(probe.action_probability_sum - 1.0) <= js_tolerance,
                {
                    "js_divergence": probe.js_divergence,
                    "reference_probability_sum": (
                        probe.reference_probability_sum
                    ),
                    "action_probability_sum": probe.action_probability_sum,
                    "tolerance": js_tolerance,
                },
            )
        )
    (
        source_after_all_arms,
        model_after_all_arms,
        rng_after_all_arms,
        model_tensor_count,
        state_validation_seconds,
    ) = _inspect_boundary_state(model, boundary)
    validation_seconds += state_validation_seconds
    gates.extend(
        (
            GateResult(
                "source_unchanged_after_all_arms",
                source_before_actions == source_after_all_arms,
                {
                    "before": source_before_actions,
                    "after": source_after_all_arms,
                },
            ),
            GateResult(
                "model_state_unchanged_after_all_arms",
                model_after_all_arms == boundary.model_state_fingerprint
                and model_tensor_count == boundary.model_tensor_count,
                {
                    "before": boundary.model_state_fingerprint,
                    "after": model_after_all_arms,
                    "expected_tensor_count": boundary.model_tensor_count,
                    "observed_tensor_count": model_tensor_count,
                },
            ),
            GateResult(
                "rng_state_unchanged_after_all_arms",
                rng_after_all_arms == boundary.rng_state.fingerprint,
                {
                    "before": boundary.rng_state.fingerprint,
                    "after": rng_after_all_arms,
                },
            ),
        )
    )
    all_arms = (noop_a, noop_b, *forward_arms, *reverse_arms)
    return AcceptanceResult(
        model=model_evidence,
        eligibility=eligibility,
        boundary=boundary,
        uninterrupted=uninterrupted,
        noop_forks=(noop_a, noop_b),
        action_arms=forward_arms,
        reverse_action_arms=reverse_arms,
        gates=tuple(gates),
        timing_seconds={
            "clone_controls": clone_seconds,
            "reference_probe_clone": reference_clone_seconds,
            "reference_probe_forward": reference_forward_seconds,
            "validation_overhead": _validation_overhead_seconds(
                boundary,
                uninterrupted,
                all_arms,
                validation_seconds,
            ),
            "total": time.perf_counter() - started_total,
        },
        memory=_memory_evidence(
            boundary,
            reference_memory_baseline,
            reference_memory_peak,
            (noop_a, *forward_arms),
            (noop_b, *reverse_arms),
        ),
    )


def model_signature(model: Any) -> dict[str, object]:
    """Return cheap checkpoint identity and decoder dimensions."""
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("model must expose a Transformers config")
    parameter = next(model.parameters())
    signature: dict[str, object] = {
        "model_class": type(model).__name__,
        "config_class": type(config).__name__,
        "name_or_path": str(getattr(config, "_name_or_path", "")),
        "checkpoint_revision": getattr(config, "_commit_hash", None),
        "vocab_size": int(config.vocab_size),
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(
            getattr(config, "num_key_value_heads", config.num_attention_heads)
        ),
        "head_dim": int(
            getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
        ),
        "parameter_count": sum(item.numel() for item in model.parameters()),
        "parameter_bytes": sum(
            item.numel() * item.element_size() for item in model.parameters()
        ),
        "buffer_bytes": sum(
            item.numel() * item.element_size() for item in model.buffers()
        ),
        "dtype": str(parameter.dtype),
        "device": str(parameter.device),
        "attention_implementation": str(
            getattr(config, "_attn_implementation", "")
        ),
    }
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    signature["checkpoint_fingerprint_kind"] = (
        "model_metadata_signature_not_weight_hash"
    )
    signature["checkpoint_fingerprint"] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()
    return signature


def cache_lengths(cache: Any) -> tuple[int, ...]:
    """Return physical sequence length at every initialized layer."""
    _require_dynamic_cache(cache)
    lengths: list[int] = []
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        if not isinstance(keys, torch.Tensor):
            raise TypeError("all cache layers must be initialized")
        lengths.append(int(keys.shape[-2]))
    return tuple(lengths)


def cache_nbytes(cache: Any) -> int:
    """Return bytes physically retained by KV tensors."""
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in _cache_tensors(cache)
    )


def cache_fingerprint(cache: Any) -> str:
    """Hash cache shapes, dtypes, and exact tensor contents."""
    digest = hashlib.sha256()
    for tensor in _cache_tensors(cache):
        _hash_tensor(digest, tensor)
    return digest.hexdigest()


def decoder_state_fingerprint(
    prompt_ids: torch.Tensor,
    generated_ids: tuple[int, ...],
    cache: Any,
    logical_position: int,
    *,
    attention_mask: torch.Tensor | None = None,
    pending_token_id: int | None = None,
    rng_fingerprint: str | None = None,
    model_state_fingerprint: str | None = None,
) -> str:
    """Hash the full decision state without serializing it to artifacts."""
    digest = hashlib.sha256()
    metadata = {
        "generated_ids": generated_ids,
        "logical_position": logical_position,
        "pending_token_id": pending_token_id,
        "rng_fingerprint": rng_fingerprint,
        "model_state_fingerprint": model_state_fingerprint,
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    _hash_tensor(digest, prompt_ids)
    if attention_mask is not None:
        _hash_tensor(digest, attention_mask)
    digest.update(cache_fingerprint(cache).encode("ascii"))
    return digest.hexdigest()


def _model_runtime_state_fingerprint(model: Any) -> tuple[str, int]:
    """Fingerprint cheap tensor metadata for mutation gates."""
    digest = hashlib.sha256()
    tensor_count = 0
    for kind, named_tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in named_tensors:
            metadata = (
                kind,
                name,
                tuple(tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                tensor.data_ptr(),
                tensor._version,
            )
            digest.update(repr(metadata).encode("utf-8"))
            tensor_count += 1
    module_modes = tuple(
        (name, module.training) for name, module in model.named_modules()
    )
    digest.update(repr(module_modes).encode("utf-8"))
    return digest.hexdigest(), tensor_count


def _inspect_boundary_state(
    model: Any, boundary: BoundaryState
) -> tuple[str, str, str, int, float]:
    """Return current saved-state, model, and global RNG fingerprints."""
    started = time.perf_counter()
    model_fingerprint, tensor_count = _model_runtime_state_fingerprint(model)
    state_fingerprint = decoder_state_fingerprint(
        boundary.prompt_ids,
        boundary.generated_ids,
        boundary.cache,
        boundary.logical_position,
        attention_mask=boundary.attention_mask,
        pending_token_id=boundary.pending_token_id,
        rng_fingerprint=boundary.rng_state.fingerprint,
        model_state_fingerprint=model_fingerprint,
    )
    current_rng_fingerprint = _capture_rng(_model_device(model)).fingerprint
    elapsed = time.perf_counter() - started
    return (
        state_fingerprint,
        model_fingerprint,
        current_rng_fingerprint,
        tensor_count,
        elapsed,
    )


def _validate_model_and_input(model: Any, input_ids: torch.Tensor) -> None:
    if model.training:
        raise ValueError("acceptance requires model.eval()")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("acceptance requires batch size 1 input_ids")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one prompt token")


def _model_device(model: Any) -> torch.device:
    return torch.device(next(model.parameters()).device)


def _require_dynamic_cache(cache: Any) -> None:
    if not hasattr(cache, "layers") or not hasattr(cache, "get_seq_length"):
        raise TypeError("acceptance requires a Transformers DynamicCache")


def _require_full_attention_cache(cache: Any) -> None:
    for layer in cache.layers:
        get_maximum = getattr(layer, "get_max_cache_shape", None)
        maximum = get_maximum() if callable(get_maximum) else -1
        if maximum not in (-1, None):
            raise TypeError("acceptance requires full-attention cache layers")


def _cache_tensors(cache: Any) -> list[torch.Tensor]:
    _require_dynamic_cache(cache)
    tensors: list[torch.Tensor] = []
    for layer in cache.layers:
        for name in ("keys", "values"):
            value = getattr(layer, name, None)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"initialized cache layer has no tensor {name}"
                )
            tensors.append(value)
    return tensors


def _indices_to_tuple(indices: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    cpu = indices.detach().to(device="cpu", dtype=torch.long)
    flattened = cpu.reshape(-1, cpu.shape[-1]).tolist()
    return tuple(tuple(int(value) for value in row) for row in flattened)


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    detached = tensor.detach().contiguous()
    header = f"{detached.dtype}|{tuple(detached.shape)}|".encode("ascii")
    digest.update(header)
    raw = detached.view(torch.uint8).cpu().numpy().tobytes()
    digest.update(raw)


def _capture_rng(device: torch.device) -> RngSnapshot:
    cuda_states: tuple[torch.Tensor, ...] = ()
    if device.type == "cuda":
        cuda_states = tuple(
            state.clone() for state in torch.cuda.get_rng_state_all()
        )
    return RngSnapshot(
        cpu=torch.random.get_rng_state().clone(), cuda=cuda_states
    )


def _restore_rng(snapshot: RngSnapshot, device: torch.device) -> None:
    torch.random.set_rng_state(snapshot.cpu)
    if device.type == "cuda" and snapshot.cuda:
        torch.cuda.set_rng_state_all(list(snapshot.cuda))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _pending_forward(
    model: Any,
    token: torch.Tensor,
    cache: Any,
    logical_position: int,
) -> tuple[Any, float]:
    device = _model_device(model)
    token = token.to(device)
    position = torch.tensor(
        [logical_position], device=device, dtype=torch.long
    )
    physical_attention_mask = torch.ones(
        (1, cache.get_seq_length() + 1),
        device=device,
        dtype=torch.long,
    )
    _sync_device(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(
            input_ids=token,
            attention_mask=physical_attention_mask,
            position_ids=position.unsqueeze(0),
            cache_position=position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    _sync_device(device)
    return output, time.perf_counter() - started


def _probe_logits(
    model: Any,
    boundary: BoundaryState,
    action: ActionSpec | None,
) -> tuple[torch.Tensor, float, float, int | None, int | None]:
    _restore_rng(boundary.rng_state, _model_device(model))
    device = _model_device(model)
    baseline_memory = _begin_peak_memory_measurement(device)
    _sync_device(device)
    clone_started = time.perf_counter()
    cache = clone_cache(boundary.cache)
    _sync_device(device)
    clone_elapsed = time.perf_counter() - clone_started
    if action is not None:
        compress_knorm(cache, action.removal_fraction)
    pending = torch.tensor(
        [[boundary.pending_token_id]],
        device=_model_device(model),
        dtype=boundary.prompt_ids.dtype,
    )
    output, elapsed = _pending_forward(
        model, pending, cache, boundary.logical_position
    )
    logits = output.logits[:, -1, :].detach().clone()
    peak_memory = _finish_peak_memory_measurement(device)
    return logits, elapsed, clone_elapsed, baseline_memory, peak_memory


def _begin_peak_memory_measurement(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    _sync_device(device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def _finish_peak_memory_measurement(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    _sync_device(device)
    return int(torch.cuda.max_memory_allocated(device))


def _continue_cache(
    model: Any,
    boundary: BoundaryState,
    cache: Any,
    *,
    max_new_tokens: int,
    eos_ids: set[int] | frozenset[int],
    first_logits_observer: Callable[[torch.Tensor], None] | None,
) -> ContinuationResult:
    device = _model_device(model)
    generated = list(boundary.generated_ids)
    pending = torch.tensor(
        [[boundary.pending_token_id]],
        device=device,
        dtype=boundary.prompt_ids.dtype,
    )
    output, first_seconds = _pending_forward(
        model, pending, cache, boundary.logical_position
    )
    forward_seconds = first_seconds
    logits = output.logits[:, -1, :]
    if first_logits_observer is not None:
        first_logits_observer(logits)
    cache = output.past_key_values
    termination = "token_budget"
    while len(generated) < max_new_tokens:
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos_ids:
            termination = "eos"
            break
        if len(generated) == max_new_tokens:
            break
        token_tensor = torch.tensor(
            [[token]], device=device, dtype=boundary.prompt_ids.dtype
        )
        logical_position = (
            int(boundary.prompt_ids.shape[1]) + len(generated) - 1
        )
        output, elapsed = _pending_forward(
            model, token_tensor, cache, logical_position
        )
        forward_seconds += elapsed
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
    validation_started = time.perf_counter()
    final_cache_lengths = cache_lengths(cache)
    final_cache_bytes = cache_nbytes(cache)
    final_cache_fingerprint = cache_fingerprint(cache)
    validation_seconds = time.perf_counter() - validation_started
    return ContinuationResult(
        token_ids=tuple(generated),
        termination_reason=termination,
        forward_seconds=forward_seconds,
        first_forward_seconds=first_seconds,
        final_cache_lengths=final_cache_lengths,
        final_cache_bytes=final_cache_bytes,
        final_cache_fingerprint=final_cache_fingerprint,
        validation_seconds=validation_seconds,
    )


def _greedy_from_prompt(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_ids: set[int] | frozenset[int],
) -> ContinuationResult:
    device = _model_device(model)
    prompt = input_ids.to(device)
    mask = torch.ones_like(prompt)
    _sync_device(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model(
            input_ids=prompt,
            attention_mask=mask,
            use_cache=True,
            return_dict=True,
        )
    _sync_device(device)
    forward_seconds = time.perf_counter() - started
    first_seconds = forward_seconds
    cache = output.past_key_values
    logits = output.logits[:, -1, :]
    generated: list[int] = []
    termination = "token_budget"
    while len(generated) < max_new_tokens:
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos_ids:
            termination = "eos"
            break
        if len(generated) == max_new_tokens:
            break
        token_tensor = torch.tensor(
            [[token]], device=device, dtype=prompt.dtype
        )
        logical_position = int(prompt.shape[1]) + len(generated) - 1
        output, elapsed = _pending_forward(
            model, token_tensor, cache, logical_position
        )
        forward_seconds += elapsed
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
    validation_started = time.perf_counter()
    final_cache_lengths = cache_lengths(cache)
    final_cache_bytes = cache_nbytes(cache)
    final_cache_fingerprint = cache_fingerprint(cache)
    validation_seconds = time.perf_counter() - validation_started
    return ContinuationResult(
        token_ids=tuple(generated),
        termination_reason=termination,
        forward_seconds=forward_seconds,
        first_forward_seconds=first_seconds,
        final_cache_lengths=final_cache_lengths,
        final_cache_bytes=final_cache_bytes,
        final_cache_fingerprint=final_cache_fingerprint,
        validation_seconds=validation_seconds,
    )


def _tensor_token_ids(value: torch.Tensor) -> list[int]:
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("expected batch-1 token tensor")
    return [int(item) for item in value.detach().cpu()[0].tolist()]


def _continuations_equal(
    left: ContinuationResult, right: ContinuationResult
) -> bool:
    return (
        left.token_ids == right.token_ids
        and left.termination_reason == right.termination_reason
        and left.final_cache_lengths == right.final_cache_lengths
        and left.final_cache_bytes == right.final_cache_bytes
        and left.final_cache_fingerprint == right.final_cache_fingerprint
    )


def _validation_overhead_seconds(
    boundary: BoundaryState,
    uninterrupted: ContinuationResult,
    arms: tuple[ArmResult, ...],
    extra_seconds: float,
) -> float:
    return (
        boundary.validation_seconds
        + uninterrupted.validation_seconds
        + sum(
            arm.compression.validation_seconds
            + arm.continuation.validation_seconds
            for arm in arms
        )
        + extra_seconds
    )


def _peak_or_none(values: list[int | None]) -> int | None:
    observed = [value for value in values if value is not None]
    return max(observed) if observed else None


def _memory_evidence(
    boundary: BoundaryState,
    reference_baseline: int | None,
    reference_peak: int | None,
    probe_arms: tuple[ArmResult, ...],
    no_probe_arms: tuple[ArmResult, ...],
) -> dict[str, object]:
    return {
        "reference_probe_baseline_bytes": reference_baseline,
        "reference_probe_peak_allocated_bytes": reference_peak,
        "reference_probe_peak_increment_bytes": (
            reference_peak - reference_baseline
            if reference_peak is not None and reference_baseline is not None
            else None
        ),
        "probe_enabled_peak_allocated_bytes": _peak_or_none(
            [arm.device_peak_allocated_bytes for arm in probe_arms]
        ),
        "probe_disabled_peak_allocated_bytes": _peak_or_none(
            [arm.device_peak_allocated_bytes for arm in no_probe_arms]
        ),
        "reference_state_cache_bytes": boundary.cache_bytes,
        "reference_state_preserved_during_measurement": True,
        "without_reference_state_peak_allocated_bytes": None,
        "without_reference_state_peak_status": (
            "not_measured_requires_separate_run"
        ),
    }
