"""Deterministic intervention parity checks for the approved compressors.

The harness compares token IDs first.  Task quality is optional secondary
 evidence and never turns a token mismatch into a parity pass.
"""

import hashlib
import inspect
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from herald.config import MODELS
from herald.features import FeatureCollector
from herald.generate import (
    LoadedModel,
    generate_reference,
    prefill_continuation,
)
from herald.live_controller import (
    _extend_reference,
    _fork_score_cache,
    cache_storage_independent,
    cache_tensors_equal,
    check_cache_mutation_isolation,
    clone_cache,
    continue_from_cache,
    split_switch_boundary,
)
from herald.presses import get_press
from herald.tasks import PromptRecord

APPROVED_COMPRESSORS = ("streaming_llm", "knorm", "expected_attention")
CACHE_NATIVE_COMPRESSORS = ("streaming_llm", "knorm")

Status = Literal["passed", "failed", "not_applicable"]


@dataclass(frozen=True)
class ParityCheck:
    """One required parity condition and its evidence."""

    status: Status
    exact_match: bool | None
    first_divergence_index: int | None
    left_token_ids: list[int] | None
    right_token_ids: list[int] | None
    left_token_sha256: str | None
    right_token_sha256: str | None
    reason: str | None = None


@dataclass(frozen=True)
class CacheIsolation:
    """Evidence that cache forks do not alias or mutate their source."""

    source_cache_available: bool
    clone_storage_independent: bool | None
    source_unchanged_after_probe_mutation: bool | None
    compressed_fork_storage_independent: bool | None
    source_unchanged_after_compression: bool | None
    mutation_checked: bool
    reason: str | None = None


@dataclass(frozen=True)
class ParityResult:
    """JSON-serializable report for one model/prompt/intervention case."""

    schema_version: str
    model_key: str
    model_id: str | None
    task: str
    prompt_id: str
    compressor: str
    intervention_classification: str
    ratio: float
    s: int
    continuation_budget: int
    reference_token_ids: list[int]
    reference_token_sha256: str
    checks: dict[str, ParityCheck]
    quality_scores: dict[str, float] | None
    quality_deltas: dict[str, float] | None
    cache_isolation: CacheIsolation
    complete: bool
    passed: bool
    failure_reasons: list[str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible mapping."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the report with stable key ordering."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )


def _token_sha256(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(
        zip(left, right, strict=False)
    ):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _check(left: list[int], right: list[int]) -> ParityCheck:
    exact = left == right
    return ParityCheck(
        status="passed" if exact else "failed",
        exact_match=exact,
        first_divergence_index=_first_divergence(left, right),
        left_token_ids=list(left),
        right_token_ids=list(right),
        left_token_sha256=_token_sha256(left),
        right_token_sha256=_token_sha256(right),
        reason=None if exact else "token ID sequences differ",
    )


def _not_applicable(reason: str) -> ParityCheck:
    return ParityCheck(
        status="not_applicable",
        exact_match=None,
        first_divergence_index=None,
        left_token_ids=None,
        right_token_ids=None,
        left_token_sha256=None,
        right_token_sha256=None,
        reason=reason,
    )


def _failed(reason: str) -> ParityCheck:
    return ParityCheck(
        status="failed",
        exact_match=None,
        first_divergence_index=None,
        left_token_ids=None,
        right_token_ids=None,
        left_token_sha256=None,
        right_token_sha256=None,
        reason=reason,
    )


def _score_output(
    scorer: Callable[..., float], record: PromptRecord, text: str
) -> float:
    """Accept either ``scorer(text)`` or ``scorer(task, text, gold)``."""
    try:
        parameter_count = len(inspect.signature(scorer).parameters)
    except (TypeError, ValueError):
        parameter_count = 0
    if parameter_count == 1:
        return float(scorer(text))
    if parameter_count >= 3:
        return float(scorer(record.task, text, record.gold))
    return float(scorer(record, text))


def _incomplete(
    *,
    model_key: str,
    model_id: str | None,
    record: PromptRecord,
    compressor: str,
    ratio: float,
    s: int,
    continuation_budget: int,
    reference_ids: list[int],
    reason: str,
) -> ParityResult:
    checks = {
        "fork_clone": _failed(reason),
        "live_reference": _failed(reason),
        "sham_no_press": _failed(reason),
        "compressed_live_fork": (
            _not_applicable("incomplete case")
            if compressor == "expected_attention"
            else _failed(reason)
        ),
    }
    return ParityResult(
        schema_version="herald.parity.v1",
        model_key=model_key,
        model_id=model_id,
        task=record.task,
        prompt_id=record.prompt_id,
        compressor=compressor,
        intervention_classification=(
            "reprefill_only"
            if compressor == "expected_attention"
            else "cache_native_live_fork"
        ),
        ratio=ratio,
        s=s,
        continuation_budget=continuation_budget,
        reference_token_ids=list(reference_ids),
        reference_token_sha256=_token_sha256(reference_ids),
        checks=checks,
        quality_scores=None,
        quality_deltas=None,
        cache_isolation=CacheIsolation(
            source_cache_available=False,
            clone_storage_independent=None,
            source_unchanged_after_probe_mutation=None,
            compressed_fork_storage_independent=None,
            source_unchanged_after_compression=None,
            mutation_checked=False,
            reason=reason,
        ),
        complete=False,
        passed=False,
        failure_reasons=[reason],
        provenance={"token_hash": "sha256(canonical-json-int-list)"},
    )


def run_parity_case(
    lm: LoadedModel,
    record: PromptRecord,
    *,
    compressor: str,
    ratio: float,
    s: int,
    continuation_budget: int,
    scorer: Callable[..., float] | None = None,
    seed: int = 0,
    model_id: str | None = None,
) -> ParityResult:
    """Run all parity checks for one deterministic intervention case."""
    if compressor not in APPROVED_COMPRESSORS:
        raise ValueError(
            f"unsupported parity compressor {compressor!r}; "
            f"choose from {APPROVED_COMPRESSORS}"
        )
    if not math.isfinite(ratio) or ratio <= 0 or ratio >= 1:
        raise ValueError("ratio must satisfy 0 < ratio < 1")
    if continuation_budget <= 0:
        raise ValueError("continuation_budget must be positive")
    if s < 0 or s >= continuation_budget:
        raise ValueError("s must leave at least one post-boundary token")
    model_id = model_id or MODELS.get(lm.key)

    reference = generate_reference(lm, [record], continuation_budget)[0]
    if len(reference.gen_ids) <= s:
        return _incomplete(
            model_key=lm.key,
            model_id=model_id,
            record=record,
            compressor=compressor,
            ratio=ratio,
            s=s,
            continuation_budget=continuation_budget,
            reference_ids=reference.gen_ids,
            reason=(
                f"reference ended before pending token s={s}; "
                f"only {len(reference.gen_ids)} tokens available"
            ),
        )

    prompt_ids = torch.tensor(reference.prompt_input_ids, dtype=torch.long)
    prefix_ids, pending_id = split_switch_boundary(reference.gen_ids, s)
    ref_collector = FeatureCollector()
    boundary_ids, source_cache = _extend_reference(
        lm,
        prompt_ids,
        [],
        None,
        ref_collector,
        s + 1,
    )
    if boundary_ids != reference.gen_ids[: s + 1] or source_cache is None:
        return _incomplete(
            model_key=lm.key,
            model_id=model_id,
            record=record,
            compressor=compressor,
            ratio=ratio,
            s=s,
            continuation_budget=continuation_budget,
            reference_ids=reference.gen_ids,
            reason=(
                "boundary cache generation did not reproduce reference prefix"
            ),
        )

    source_snapshot = clone_cache(source_cache)
    isolation_probe = clone_cache(source_cache)
    clone_a = clone_cache(source_cache)
    clone_b = clone_cache(source_cache)
    clone_storage_ok = (
        cache_storage_independent(source_cache, isolation_probe)
        and cache_storage_independent(source_cache, clone_a)
        and cache_storage_independent(source_cache, clone_b)
        and cache_storage_independent(clone_a, clone_b)
    )
    probe_mutation_ok = check_cache_mutation_isolation(
        source_cache, isolation_probe
    )
    live_no_press, _ = continue_from_cache(
        lm,
        prompt_ids,
        prefix_ids,
        [pending_id],
        clone_a,
        continuation_budget - s,
    )
    second_live_no_press, _ = continue_from_cache(
        lm,
        prompt_ids,
        prefix_ids,
        [pending_id],
        clone_b,
        continuation_budget - s,
    )
    sham = prefill_continuation(
        lm, prompt_ids, prefix_ids, continuation_budget - s, None, seed
    )
    checks: dict[str, ParityCheck] = {
        "fork_clone": _check(live_no_press, second_live_no_press),
        "live_reference": _check(live_no_press, reference.gen_ids[s:]),
        "sham_no_press": _check(live_no_press, sham),
    }

    compressed_full: list[int] | None = None
    compressed_storage_ok: bool | None = None
    source_after_compression_ok: bool | None = None
    if compressor in CACHE_NATIVE_COMPRESSORS:
        compressed_cache = _fork_score_cache(
            lm.model, source_cache, get_press(compressor, ratio)
        )
        compressed_storage_ok = cache_storage_independent(
            source_cache, compressed_cache
        )
        source_after_compression_ok = cache_tensors_equal(
            source_cache, source_snapshot
        )
        compressed_full, _ = continue_from_cache(
            lm,
            prompt_ids,
            prefix_ids,
            [pending_id],
            compressed_cache,
            continuation_budget - s,
        )
        source_after_compression_ok = (
            source_after_compression_ok
            and cache_tensors_equal(source_cache, source_snapshot)
        )
        canonical = prefill_continuation(
            lm,
            prompt_ids,
            prefix_ids,
            continuation_budget - s,
            get_press(compressor, ratio),
            seed,
        )
        checks["compressed_live_fork"] = _check(compressed_full, canonical)
    else:
        checks["compressed_live_fork"] = _not_applicable(
            "plain ExpectedAttention is re-prefill-defined; "
            "live-fork parity is not applicable"
        )
        canonical = prefill_continuation(
            lm,
            prompt_ids,
            prefix_ids,
            continuation_budget - s,
            get_press(compressor, ratio),
            seed,
        )

    isolation_failures = []
    if not clone_storage_ok:
        isolation_failures.append("uncompressed cache clones share storage")
    if not probe_mutation_ok:
        isolation_failures.append("clone mutation changed the source cache")
    if compressed_storage_ok is False:
        isolation_failures.append("compressed fork shares source storage")
    if source_after_compression_ok is False:
        isolation_failures.append("compression changed the source cache")
    isolation = CacheIsolation(
        source_cache_available=True,
        clone_storage_independent=clone_storage_ok,
        source_unchanged_after_probe_mutation=probe_mutation_ok,
        compressed_fork_storage_independent=compressed_storage_ok,
        source_unchanged_after_compression=source_after_compression_ok,
        mutation_checked=True,
        reason="; ".join(isolation_failures) or None,
    )

    quality_scores: dict[str, float] | None = None
    quality_deltas: dict[str, float] | None = None
    if scorer is not None:
        ref_score = _score_output(scorer, record, reference.text)
        live_text = lm.tokenizer.decode(
            prefix_ids + live_no_press, skip_special_tokens=True
        )
        sham_text = lm.tokenizer.decode(
            prefix_ids + sham, skip_special_tokens=True
        )
        pressed_text = lm.tokenizer.decode(
            prefix_ids + canonical, skip_special_tokens=True
        )
        live_score = _score_output(scorer, record, live_text)
        sham_score = _score_output(scorer, record, sham_text)
        pressed_score = _score_output(scorer, record, pressed_text)
        quality_scores = {
            "reference": ref_score,
            "live_full_cache_fork": live_score,
            "sham_no_press": sham_score,
            "pressed_reprefill": pressed_score,
        }
        quality_deltas = {
            "sham_reprefill_vs_reference": ref_score - sham_score,
            "pressed_reprefill": sham_score - pressed_score,
        }
        if compressed_full is not None:
            compressed_score = _score_output(
                scorer,
                record,
                lm.tokenizer.decode(
                    prefix_ids + compressed_full, skip_special_tokens=True
                ),
            )
            quality_scores["compressed_live_fork"] = compressed_score
            quality_deltas["compressed_live_fork"] = (
                live_score - compressed_score
            )

    required_checks = [
        checks["fork_clone"],
        checks["live_reference"],
        checks["sham_no_press"],
        checks["compressed_live_fork"],
    ]
    failures = [
        check.reason or "required parity check failed"
        for check in required_checks
        if check.status == "failed"
    ]
    failures.extend(isolation_failures)
    return ParityResult(
        schema_version="herald.parity.v1",
        model_key=lm.key,
        model_id=model_id,
        task=record.task,
        prompt_id=record.prompt_id,
        compressor=compressor,
        intervention_classification=(
            "reprefill_only"
            if compressor == "expected_attention"
            else "cache_native_live_fork"
        ),
        ratio=ratio,
        s=s,
        continuation_budget=continuation_budget,
        reference_token_ids=list(reference.gen_ids),
        reference_token_sha256=_token_sha256(reference.gen_ids),
        checks=checks,
        quality_scores=quality_scores,
        quality_deltas=quality_deltas,
        cache_isolation=isolation,
        complete=True,
        passed=not failures,
        failure_reasons=failures,
        provenance={
            "generation": "greedy model.generate",
            "pending_boundary_token": pending_id,
            "token_hash": "sha256(canonical-json-int-list)",
            "sham_reprefill_token_sha256": _token_sha256(sham),
            "pressed_reprefill_token_sha256": _token_sha256(canonical),
            "sham_vs_pressed_reprefill_exact": sham == canonical,
        },
    )
