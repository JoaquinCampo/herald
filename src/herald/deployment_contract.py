"""Executable deployment contract for Herald's north-star objective."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeploymentContract:
    """Thresholds that make a Herald configuration deployable."""

    quality_noninferiority_margin: float = 0.01
    major_damage_threshold: float = 0.5
    max_major_damage_rate: float = 0.01
    max_end_to_end_slowdown: float = 0.05
    confidence: float = 0.95
    min_pairs: int = 30
    bootstrap_resamples: int = 2_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.quality_noninferiority_margin < 0:
            raise ValueError(
                "quality_noninferiority_margin must be nonnegative"
            )
        if self.max_end_to_end_slowdown < 0:
            raise ValueError("max_end_to_end_slowdown must be nonnegative")
        if self.major_damage_threshold <= 0:
            raise ValueError("major_damage_threshold must be positive")
        if not 0 <= self.max_major_damage_rate <= 1:
            raise ValueError("max_major_damage_rate must be between 0 and 1")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.min_pairs <= 0:
            raise ValueError("min_pairs must be positive")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")


@dataclass(frozen=True)
class DeploymentMeasurement:
    """One paired uncompressed/candidate inference measurement."""

    prompt_id: str
    quality_reference: float
    quality_candidate: float
    baseline_wall_s: float
    candidate_wall_s: float
    baseline_tokens: int
    candidate_tokens: int
    baseline_peak_kv_bytes: int | None
    candidate_peak_kv_bytes: int | None
    baseline_kv_byte_tokens: float | None
    candidate_kv_byte_tokens: float | None

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("prompt_id must not be empty")
        for name, value in (
            ("quality_reference", self.quality_reference),
            ("quality_candidate", self.quality_candidate),
            ("baseline_wall_s", self.baseline_wall_s),
            ("candidate_wall_s", self.candidate_wall_s),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.baseline_wall_s <= 0:
            raise ValueError("baseline_wall_s must be positive")
        if self.candidate_wall_s <= 0:
            raise ValueError("candidate_wall_s must be positive")
        if self.baseline_tokens <= 0:
            raise ValueError("baseline_tokens must be positive")
        if self.candidate_tokens <= 0:
            raise ValueError("candidate_tokens must be positive")
        self._validate_memory_pair(
            "peak_kv_bytes",
            self.baseline_peak_kv_bytes,
            self.candidate_peak_kv_bytes,
        )
        self._validate_memory_pair(
            "kv_byte_tokens",
            self.baseline_kv_byte_tokens,
            self.candidate_kv_byte_tokens,
        )

    @staticmethod
    def _validate_memory_pair(
        name: str,
        baseline: int | float | None,
        candidate: int | float | None,
    ) -> None:
        if (baseline is None) != (candidate is None):
            raise ValueError(
                f"{name} baseline and candidate must both be set"
            )
        if baseline is None or candidate is None:
            return
        if not isfinite(float(baseline)) or baseline <= 0:
            raise ValueError(f"baseline_{name} must be positive and finite")
        if not isfinite(float(candidate)) or candidate < 0:
            raise ValueError(
                f"candidate_{name} must be nonnegative and finite"
            )


@dataclass(frozen=True)
class MetricEstimate:
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class DeploymentEvaluation:
    contract: DeploymentContract
    n_pairs: int
    n_prompts: int
    quality_damage: MetricEstimate
    major_damage_rate: MetricEstimate
    end_to_end_slowdown: MetricEstimate
    per_token_slowdown: MetricEstimate
    peak_kv_savings: MetricEstimate | None
    kv_byte_token_savings: MetricEstimate | None
    quality_pass: bool
    tail_quality_pass: bool
    speed_pass: bool
    memory_verified: bool
    memory_pass: bool
    feasible: bool
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_deployment(
    measurements: Sequence[DeploymentMeasurement],
    *,
    contract: DeploymentContract | None = None,
) -> DeploymentEvaluation:
    """Evaluate paired measurements without substituting proxy metrics."""
    active = contract or DeploymentContract()
    if not measurements:
        raise ValueError("at least one measurement is required")

    prompt_ids = [row.prompt_id for row in measurements]
    quality_damage = _estimate(
        [
            row.quality_reference - row.quality_candidate
            for row in measurements
        ],
        prompt_ids,
        active,
        seed_offset=0,
    )
    major_damage_rate = _estimate(
        [
            float(
                row.quality_reference - row.quality_candidate
                >= active.major_damage_threshold
            )
            for row in measurements
        ],
        prompt_ids,
        active,
        seed_offset=5,
    )
    end_to_end_slowdown = _estimate(
        [
            row.candidate_wall_s / row.baseline_wall_s - 1.0
            for row in measurements
        ],
        prompt_ids,
        active,
        seed_offset=1,
    )
    per_token_slowdown = _estimate(
        [
            (row.candidate_wall_s / row.candidate_tokens)
            / (row.baseline_wall_s / row.baseline_tokens)
            - 1.0
            for row in measurements
        ],
        prompt_ids,
        active,
        seed_offset=2,
    )

    memory_verified = all(
        row.baseline_peak_kv_bytes is not None
        and row.candidate_peak_kv_bytes is not None
        for row in measurements
    )
    peak_kv_savings = None
    kv_byte_token_savings = None
    if memory_verified:
        peak_kv_savings = _estimate(
            [
                1.0
                - _required_float(row.candidate_peak_kv_bytes)
                / _required_float(row.baseline_peak_kv_bytes)
                for row in measurements
            ],
            prompt_ids,
            active,
            seed_offset=3,
        )
        area_verified = all(
            row.baseline_kv_byte_tokens is not None
            and row.candidate_kv_byte_tokens is not None
            for row in measurements
        )
        if area_verified:
            kv_byte_token_savings = _estimate(
                [
                    1.0
                    - _required_float(row.candidate_kv_byte_tokens)
                    / _required_float(row.baseline_kv_byte_tokens)
                    for row in measurements
                ],
                prompt_ids,
                active,
                seed_offset=4,
            )

    enough_pairs = len(measurements) >= active.min_pairs
    mean_quality_pass = (
        quality_damage.upper <= active.quality_noninferiority_margin
    )
    tail_quality_pass = (
        major_damage_rate.upper <= active.max_major_damage_rate
    )
    quality_pass = mean_quality_pass and tail_quality_pass
    speed_pass = end_to_end_slowdown.upper <= active.max_end_to_end_slowdown
    memory_pass = (
        peak_kv_savings is not None
        and peak_kv_savings.lower > 0
        and (kv_byte_token_savings is None or kv_byte_token_savings.lower > 0)
    )

    failures = []
    if not enough_pairs:
        failures.append("minimum_pairs")
    if not mean_quality_pass:
        failures.append("quality_noninferiority")
    if not tail_quality_pass:
        failures.append("major_damage_rate")
    if not speed_pass:
        failures.append("end_to_end_slowdown")
    if not memory_verified:
        failures.append("isolated_kv_memory")
    elif not memory_pass:
        failures.append("positive_kv_savings")

    return DeploymentEvaluation(
        contract=active,
        n_pairs=len(measurements),
        n_prompts=len(set(prompt_ids)),
        quality_damage=quality_damage,
        major_damage_rate=major_damage_rate,
        end_to_end_slowdown=end_to_end_slowdown,
        per_token_slowdown=per_token_slowdown,
        peak_kv_savings=peak_kv_savings,
        kv_byte_token_savings=kv_byte_token_savings,
        quality_pass=quality_pass,
        tail_quality_pass=tail_quality_pass,
        speed_pass=speed_pass,
        memory_verified=memory_verified,
        memory_pass=memory_pass,
        feasible=not failures,
        failures=tuple(failures),
    )


def measurement_from_live_records(
    episode: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> DeploymentMeasurement:
    """Adapt live JSONL facts without accepting allocator-memory proxies."""
    prompt_id = str(_required(episode, "prompt_id"))
    if prompt_id != str(_required(baseline, "prompt_id")):
        raise ValueError("episode and baseline prompt_id must match")

    commit_s = episode.get("commit_s")
    if commit_s is None:
        candidate_tokens = int(
            episode.get("ref_len_live", _required(baseline, "ref_len"))
        )
    else:
        candidate_tokens = int(commit_s) + int(
            _required(episode, "n_new_ids")
        )

    baseline_peak = baseline.get("peak_kv_cache_bytes")
    candidate_peak = episode.get("peak_kv_cache_bytes")
    if baseline_peak is None or candidate_peak is None:
        baseline_peak_bytes = None
        candidate_peak_bytes = None
    else:
        baseline_peak_bytes = int(baseline_peak)
        candidate_peak_bytes = int(candidate_peak)

    baseline_area = baseline.get("kv_byte_tokens")
    candidate_area = episode.get("kv_byte_tokens")
    if baseline_area is None or candidate_area is None:
        baseline_byte_tokens = None
        candidate_byte_tokens = None
    else:
        baseline_byte_tokens = float(baseline_area)
        candidate_byte_tokens = float(candidate_area)

    return DeploymentMeasurement(
        prompt_id=prompt_id,
        quality_reference=float(_required(baseline, "q_ref_live")),
        quality_candidate=float(_required(episode, "q_live")),
        baseline_wall_s=float(_required(baseline, "wall_s")),
        candidate_wall_s=float(_required(episode, "total_wall_s")),
        baseline_tokens=int(_required(baseline, "ref_len")),
        candidate_tokens=candidate_tokens,
        baseline_peak_kv_bytes=baseline_peak_bytes,
        candidate_peak_kv_bytes=candidate_peak_bytes,
        baseline_kv_byte_tokens=baseline_byte_tokens,
        candidate_kv_byte_tokens=candidate_byte_tokens,
    )


def _estimate(
    values: Sequence[float],
    prompt_ids: Sequence[str],
    contract: DeploymentContract,
    *,
    seed_offset: int,
) -> MetricEstimate:
    if len(values) != len(prompt_ids):
        raise ValueError("values and prompt_ids must have equal length")
    clusters: dict[str, list[float]] = defaultdict(list)
    for prompt_id, value in zip(prompt_ids, values, strict=True):
        if not isfinite(value):
            raise ValueError("metric values must be finite")
        clusters[prompt_id].append(float(value))

    keys = sorted(clusters)
    rng = np.random.default_rng(contract.seed + seed_offset)
    means = np.empty(contract.bootstrap_resamples, dtype=float)
    for i in range(contract.bootstrap_resamples):
        sampled = rng.integers(0, len(keys), size=len(keys))
        draw = [value for index in sampled for value in clusters[keys[index]]]
        means[i] = float(np.mean(draw))

    tail = (1.0 - contract.confidence) / 2.0
    return MetricEstimate(
        mean=float(np.mean(values)),
        lower=float(np.quantile(means, tail)),
        upper=float(np.quantile(means, 1.0 - tail)),
    )


def _required_float(value: int | float | None) -> float:
    if value is None:
        raise AssertionError("required measurement was not verified")
    return float(value)


def _required(record: Mapping[str, Any], key: str) -> Any:
    if key not in record:
        raise ValueError(f"missing required field: {key}")
    return record[key]
