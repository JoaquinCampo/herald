"""Capped reference-prefix lookahead for one paired cache boundary.

The probe measures the distributions that would be produced after the
decision boundary while keeping both branches on the same synthetic greedy
reference prefix.  It does not continue either branch to an answer and it
never retains logits or generated output.
"""

import gc
import time
from dataclasses import dataclass
from typing import Any

import torch

from herald_v3.engineering import engine

MAX_LOOKAHEAD_STEPS = 8
FIRST_OUTPUT_INDEX = 32


@dataclass(frozen=True)
class LookaheadStep:
    """One compact distribution comparison at a synthetic output index."""

    output_index: int
    input_token_id: int
    input_position: int
    reference_argmax: int
    action_argmax: int
    reference_argmax_is_eos: bool
    reference_forward_seconds: float
    action_forward_seconds: float
    probe: engine.DistributionProbe

    def to_dict(self) -> dict[str, object]:
        return {
            "output_index": self.output_index,
            "input_token_id": self.input_token_id,
            "input_position": self.input_position,
            "reference_argmax": self.reference_argmax,
            "action_argmax": self.action_argmax,
            "reference_argmax_is_eos": self.reference_argmax_is_eos,
            "reference_forward_seconds": self.reference_forward_seconds,
            "action_forward_seconds": self.action_forward_seconds,
            "probe": self.probe.to_dict(),
        }


@dataclass(frozen=True)
class LookaheadResult:
    """Serializable evidence for one reference-prefix lookahead arm."""

    action: engine.ActionSpec
    requested_steps: int
    steps: tuple[LookaheadStep, ...]
    boundary_state_fingerprint_before: str
    boundary_state_fingerprint_after: str
    boundary_rng_fingerprint_before: str
    boundary_rng_fingerprint_observed: str
    boundary_rng_fingerprint_after: str
    model_state_fingerprint_before: str
    model_state_fingerprint_after: str
    source_cache_preserved: bool
    source_rng_preserved: bool
    model_state_preserved: bool
    timing_seconds: dict[str, float]
    memory: dict[str, object]
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        """Return whether all engineering checks passed."""
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "herald_v3.engineering.lookahead.v1",
            "passed": self.passed,
            "action": self.action.to_dict(),
            "requested_steps": self.requested_steps,
            "realized_steps": len(self.steps),
            "output_indices": [step.output_index for step in self.steps],
            "input_positions": [step.input_position for step in self.steps],
            "forced_input_token_ids": [
                step.input_token_id for step in self.steps
            ],
            "reference_argmax_token_ids": [
                step.reference_argmax for step in self.steps
            ],
            "steps": [step.to_dict() for step in self.steps],
            "source": {
                "boundary_state_fingerprint_before": (
                    self.boundary_state_fingerprint_before
                ),
                "boundary_state_fingerprint_after": (
                    self.boundary_state_fingerprint_after
                ),
                "boundary_rng_fingerprint_before": (
                    self.boundary_rng_fingerprint_before
                ),
                "boundary_rng_fingerprint_observed": (
                    self.boundary_rng_fingerprint_observed
                ),
                "boundary_rng_fingerprint_after": (
                    self.boundary_rng_fingerprint_after
                ),
                "model_state_fingerprint_before": (
                    self.model_state_fingerprint_before
                ),
                "model_state_fingerprint_after": (
                    self.model_state_fingerprint_after
                ),
                "source_cache_preserved": self.source_cache_preserved,
                "source_rng_preserved": self.source_rng_preserved,
                "model_state_preserved": self.model_state_preserved,
            },
            "timing_seconds": dict(self.timing_seconds),
            "memory": engine.to_builtin(self.memory),
            "checks": dict(self.checks),
        }


def run_lookahead(
    model: Any,
    boundary: engine.BoundaryState,
    action: engine.ActionSpec,
    *,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    max_steps: int = MAX_LOOKAHEAD_STEPS,
    noop_tolerance: float = 1e-6,
) -> LookaheadResult:
    """Measure up to eight paired distributions from ``boundary``.

    The first forward in each arm consumes the saved pending token at
    ``boundary.logical_position`` and predicts output index 32.  Each later
    forward consumes the previous reference argmax at the next absolute
    position.  The action branch receives those same reference tokens even
    when its own argmax differs or is EOS.
    """
    if model.training:
        raise ValueError("lookahead requires model.eval()")
    if not 1 <= max_steps <= MAX_LOOKAHEAD_STEPS:
        raise ValueError(f"max_steps must be in [1, {MAX_LOOKAHEAD_STEPS}]")
    if noop_tolerance < 0.0:
        raise ValueError("noop_tolerance must be non-negative")
    _validate_boundary(boundary)
    normalized_eos = _validate_eos_ids(eos_ids)
    device = engine._model_device(model)
    source_rng = engine._capture_rng(device)
    state_before = boundary.state_fingerprint
    model_before = boundary.model_state_fingerprint
    started_total = time.perf_counter()
    reference_cache: Any | None = None
    action_cache: Any | None = None
    reference_forward_seconds = 0.0
    action_forward_seconds = 0.0
    reduction_seconds = 0.0
    compression_seconds = 0.0
    clone_seconds = 0.0
    validation_seconds = 0.0
    reference_peak: int | None = None
    action_peak: int | None = None
    compression: engine.CompressionEvidence | None = None
    independent_clones = False
    steps: list[LookaheadStep] = []
    rng_observed = ""

    baseline = engine._begin_peak_memory_measurement(device)
    try:
        engine._sync_device(device)
        clone_started = time.perf_counter()
        reference_cache = engine.clone_cache(boundary.cache)
        engine._sync_device(device)
        clone_seconds += time.perf_counter() - clone_started

        clone_started = time.perf_counter()
        action_cache = engine.clone_cache(boundary.cache)
        engine._sync_device(device)
        clone_seconds += time.perf_counter() - clone_started
        independent_clones = (
            engine.cache_storage_independent(boundary.cache, reference_cache)
            and engine.cache_storage_independent(boundary.cache, action_cache)
            and engine.cache_tensors_equal(boundary.cache, reference_cache)
            and engine.cache_tensors_equal(boundary.cache, action_cache)
        )

        compression = engine.compress_knorm(
            action_cache, action.removal_fraction
        )
        compression_seconds = compression.compression_seconds

        engine._restore_rng(source_rng, device)
        forced_token = boundary.pending_token_id
        for step_index in range(max_steps):
            output_index = FIRST_OUTPUT_INDEX + step_index
            input_position = boundary.logical_position + step_index
            step_rng = engine._capture_rng(device)
            reference_token = torch.tensor(
                [[forced_token]],
                device=device,
                dtype=boundary.prompt_ids.dtype,
            )
            reference_output, elapsed = engine._pending_forward(
                model,
                reference_token,
                reference_cache,
                input_position,
            )
            reference_elapsed = elapsed
            reference_forward_seconds += reference_elapsed
            reference_logits = (
                reference_output.logits[:, -1, :].detach().clone()
            )
            reference_cache = reference_output.past_key_values

            # Give the action branch the same stochastic state at every
            # matched decision, while retaining only scalar probe evidence.
            engine._restore_rng(step_rng, device)
            action_token = torch.tensor(
                [[forced_token]],
                device=device,
                dtype=boundary.prompt_ids.dtype,
            )
            action_output, elapsed = engine._pending_forward(
                model,
                action_token,
                action_cache,
                input_position,
            )
            action_elapsed = elapsed
            action_forward_seconds += action_elapsed
            action_logits = action_output.logits[:, -1, :].detach().clone()
            action_cache = action_output.past_key_values

            reduction_started = time.perf_counter()
            probe = engine.full_vocabulary_js(
                reference_logits, action_logits, action
            )
            reduction_seconds += time.perf_counter() - reduction_started
            reference_argmax = probe.reference_argmax
            action_argmax = probe.action_argmax
            reference_eos = reference_argmax in normalized_eos
            steps.append(
                LookaheadStep(
                    output_index=output_index,
                    input_token_id=forced_token,
                    input_position=input_position,
                    reference_argmax=reference_argmax,
                    action_argmax=action_argmax,
                    reference_argmax_is_eos=reference_eos,
                    reference_forward_seconds=reference_elapsed,
                    action_forward_seconds=action_elapsed,
                    probe=probe,
                )
            )
            del reference_logits, action_logits
            if reference_eos:
                break
            forced_token = reference_argmax
        engine._sync_device(device)
        peak = engine._finish_peak_memory_measurement(device)
        reference_peak = peak
        action_peak = peak
    finally:
        # The probe must leave both the saved decoder state and caller RNG
        # exactly as they were before measurement, even when a forward fails.
        rng_observed = engine._capture_rng(device).fingerprint
        engine._restore_rng(source_rng, device)
        source_state_after, model_after, rng_after, _, validation_seconds = (
            engine._inspect_boundary_state(model, boundary)
        )
        del reference_cache, action_cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        engine._sync_device(device)

    total_seconds = time.perf_counter() - started_total
    source_cache_preserved = source_state_after == state_before
    source_rng_preserved = rng_observed == source_rng.fingerprint
    model_state_preserved = model_after == model_before
    forward_counts_ok = len(steps) <= max_steps <= MAX_LOOKAHEAD_STEPS
    finite_ok = all(step.probe.finite for step in steps)
    prefix_ok = all(
        step.input_position == boundary.logical_position + index
        and (
            step.input_token_id == boundary.pending_token_id
            if index == 0
            else step.input_token_id == steps[index - 1].reference_argmax
        )
        for index, step in enumerate(steps)
    )
    eos_ok = (
        not steps
        or steps[-1].reference_argmax_is_eos
        or len(steps) == max_steps
    )
    if action.removal_fraction == 0.0:
        noop_ok = all(
            step.probe.argmax_match
            and step.probe.max_probability_difference <= noop_tolerance
            and step.probe.max_logit_difference <= noop_tolerance
            and step.probe.js_divergence <= noop_tolerance
            for step in steps
        )
    else:
        noop_ok = True
    if compression is None:
        compression_ok = False
        before_bytes = after_bytes = None
    else:
        before_bytes = compression.before_bytes
        after_bytes = compression.after_bytes
        compression_ok = action.removal_fraction == 0.0 or (
            compression.after_bytes < compression.before_bytes
            and all(
                after < before
                for before, after in zip(
                    compression.before_lengths,
                    compression.after_lengths,
                    strict=True,
                )
            )
        )
    checks = {
        "independent_clones": independent_clones,
        "forward_budget": forward_counts_ok,
        "reference_prefix_alignment": prefix_ok,
        "reference_eos_stop": eos_ok,
        "finite_distributions": finite_ok,
        "noop_distribution_parity": noop_ok,
        "physical_eviction": compression_ok,
        "source_cache_preserved": source_cache_preserved,
        "source_rng_preserved": source_rng_preserved,
        "model_state_preserved": model_state_preserved,
    }
    return LookaheadResult(
        action=action,
        requested_steps=max_steps,
        steps=tuple(steps),
        boundary_state_fingerprint_before=state_before,
        boundary_state_fingerprint_after=source_state_after,
        boundary_rng_fingerprint_before=source_rng.fingerprint,
        boundary_rng_fingerprint_observed=rng_observed,
        boundary_rng_fingerprint_after=rng_after,
        model_state_fingerprint_before=model_before,
        model_state_fingerprint_after=model_after,
        source_cache_preserved=source_cache_preserved,
        source_rng_preserved=source_rng_preserved,
        model_state_preserved=model_state_preserved,
        timing_seconds={
            "clone": clone_seconds,
            "eviction": compression_seconds,
            "forward_reference": reference_forward_seconds,
            "forward_action": action_forward_seconds,
            "forward": reference_forward_seconds + action_forward_seconds,
            "distribution_reduction": reduction_seconds,
            "source_validation": validation_seconds,
            "total": total_seconds,
        },
        memory={
            "baseline_allocated_bytes": baseline,
            "reference_peak_allocated_bytes": reference_peak,
            "action_peak_allocated_bytes": action_peak,
            "peak_allocated_bytes": _maximum(reference_peak, action_peak),
            "peak_increment_bytes": _difference(
                _maximum(reference_peak, action_peak), baseline
            ),
            "reference_cache_bytes": boundary.cache_bytes,
            "action_cache_before_bytes": before_bytes,
            "action_cache_after_bytes": after_bytes,
            "cuda_available": device.type == "cuda",
            "peak_scope": (
                "boundary clones, Knorm eviction, paired forwards and "
                "distribution reduction; source validation is timed in "
                "total but may occur after the peak sample"
            ),
            "cost_includes_source_preservation_and_validation": True,
        },
        checks=checks,
    )


def measure_lookahead(
    model: Any,
    boundary: engine.BoundaryState,
    action: engine.ActionSpec,
    *,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    max_steps: int = MAX_LOOKAHEAD_STEPS,
    noop_tolerance: float = 1e-6,
) -> LookaheadResult:
    """Descriptive alias for :func:`run_lookahead`."""
    return run_lookahead(
        model,
        boundary,
        action,
        eos_ids=eos_ids,
        max_steps=max_steps,
        noop_tolerance=noop_tolerance,
    )


def _validate_boundary(boundary: engine.BoundaryState) -> None:
    if len(boundary.generated_ids) != FIRST_OUTPUT_INDEX:
        raise ValueError("lookahead requires the exact 32-token boundary")
    expected_length = int(boundary.prompt_ids.shape[1]) + 31
    if boundary.logical_position != expected_length:
        raise ValueError("boundary logical position is not token index 31")
    if boundary.pending_token_id != boundary.generated_ids[31]:
        raise ValueError("boundary pending token must be generated_ids[31]")
    if any(length != expected_length for length in boundary.cache_lengths):
        raise ValueError(
            "boundary cache does not contain prompt plus indices 0..30"
        )


def _validate_eos_ids(values: set[int] | frozenset[int]) -> frozenset[int]:
    if not isinstance(values, (set, frozenset)):
        raise TypeError("eos_ids must be a set or frozenset of integers")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise TypeError("eos_ids must contain only integers")
    return frozenset(values)


def _difference(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left - right


def _maximum(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None
