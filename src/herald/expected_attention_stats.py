# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false

"""Frozen, training-only query statistics for ExpectedAttentionStatsPress.

KVPress can lazily fetch query statistics from the Hugging Face Hub. That
would make a deployment evaluation depend on an undeclared artifact and, more
importantly, can silently use statistics calibrated on evaluation prompts.
This
module persists the exact query moments and their prompt provenance locally.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kvpress.presses.expected_attention_with_stats import (
    ExpectedAttentionStatsPress,
    patch_rotary_embedding,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

METADATA_FILE = "metadata.json"
STATISTICS_FILE = "query_statistics.npz"
SCHEMA_VERSION = 1


class StatisticsMetadata(BaseModel):
    """Provenance and compatibility contract for one statistics artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    model_id: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    num_hidden_layers: int = Field(ge=1)
    num_attention_heads: int = Field(ge=1)
    head_dim: int = Field(ge=1)
    n_future_positions: int = Field(ge=1)
    n_sink: int = Field(ge=0)
    use_covariance: bool = True
    calibration_task: str = Field(min_length=1)
    calibration_prompt_ids: list[str] = Field(min_length=1)
    excluded_test_prompt_ids: list[str] = Field(min_length=1)
    max_prompt_tokens: int = Field(ge=1)
    query_token_count: int = Field(ge=2)
    artifact_sha256: str | None = None

    @model_validator(mode="after")
    def _validate_prompt_provenance(self) -> "StatisticsMetadata":
        calibration = set(self.calibration_prompt_ids)
        excluded = set(self.excluded_test_prompt_ids)
        if len(calibration) != len(self.calibration_prompt_ids):
            raise ValueError("calibration_prompt_ids must be unique")
        if len(excluded) != len(self.excluded_test_prompt_ids):
            raise ValueError("excluded_test_prompt_ids must be unique")
        overlap = sorted(calibration & excluded)
        if overlap:
            raise ValueError(
                "calibration and excluded test prompt IDs overlap: "
                f"{overlap[:3]}"
            )
        return self


@dataclass(frozen=True)
class StatisticsArtifact:
    """Validated query moments and the metadata that binds them to a run."""

    metadata: StatisticsMetadata
    mu: torch.Tensor
    cov: torch.Tensor

    def __post_init__(self) -> None:
        expected_mu = (
            self.metadata.num_hidden_layers,
            self.metadata.num_attention_heads,
            self.metadata.head_dim,
        )
        expected_cov = (*expected_mu, self.metadata.head_dim)
        if tuple(self.mu.shape) != expected_mu:
            raise ValueError(
                f"mu shape must be {expected_mu}, got {tuple(self.mu.shape)}"
            )
        if tuple(self.cov.shape) != expected_cov:
            raise ValueError(
                "cov shape must be "
                f"{expected_cov}, got {tuple(self.cov.shape)}"
            )
        if not (self.mu.is_floating_point() and self.cov.is_floating_point()):
            raise ValueError("query moments must use floating-point tensors")
        if not (
            torch.isfinite(self.mu).all() and torch.isfinite(self.cov).all()
        ):
            raise ValueError("query moments must be finite")

    @property
    def digest(self) -> str:
        """Return the content digest used to bind a bundle to statistics."""
        return _artifact_digest(self.metadata, self.mu, self.cov)

    def save(self, directory: Path) -> str:
        """Write a portable non-pickle artifact and return its digest."""
        directory.mkdir(parents=True, exist_ok=True)
        digest = self.digest
        metadata = self.metadata.model_copy(
            update={"artifact_sha256": digest}
        )
        _write_metadata(directory / METADATA_FILE, metadata)
        np.savez_compressed(
            directory / STATISTICS_FILE,
            mu=self.mu.detach().cpu().to(torch.float32).numpy(),
            cov=self.cov.detach().cpu().to(torch.float32).numpy(),
        )
        return digest

    @classmethod
    def load(cls, directory: Path) -> "StatisticsArtifact":
        """Load and integrity-check a statistics artifact."""
        metadata_path = directory / METADATA_FILE
        statistics_path = directory / STATISTICS_FILE
        try:
            metadata = StatisticsMetadata.model_validate_json(
                metadata_path.read_text()
            )
        except OSError as error:
            raise ValueError(
                f"could not read statistics metadata at {metadata_path}"
            ) from error
        if metadata.artifact_sha256 is None:
            raise ValueError("statistics metadata is missing artifact_sha256")
        try:
            with np.load(statistics_path, allow_pickle=False) as data:
                mu = torch.from_numpy(np.array(data["mu"], copy=True))
                cov = torch.from_numpy(np.array(data["cov"], copy=True))
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(
                f"could not read query statistics at {statistics_path}"
            ) from error
        artifact = cls(metadata=metadata, mu=mu, cov=cov)
        if artifact.digest != metadata.artifact_sha256:
            raise ValueError(
                "statistics artifact digest does not match metadata"
            )
        return artifact

    def validate_model(self, model: Any) -> None:
        """Fail before generation if the artifact cannot score this model."""
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("model does not expose a config for statistics")
        expected = {
            "model_type": self.metadata.model_type,
            "num_hidden_layers": self.metadata.num_hidden_layers,
            "num_attention_heads": self.metadata.num_attention_heads,
            "head_dim": self.metadata.head_dim,
        }
        for field, wanted in expected.items():
            observed = getattr(config, field, None)
            if observed != wanted:
                raise ValueError(
                    "statistics artifact is incompatible with model "
                    f"{field}: expected {wanted!r}, got {observed!r}"
                )
        observed_model_id = str(getattr(config, "name_or_path", ""))
        if observed_model_id != self.metadata.model_id:
            raise ValueError(
                "statistics artifact is incompatible with model_id: "
                f"expected {self.metadata.model_id!r}, "
                f"got {observed_model_id!r}"
            )


def validate_statistics_provenance(
    artifact: StatisticsArtifact,
    *,
    task: str,
    test_prompt_ids: Sequence[str],
) -> None:
    """Require exact train-only provenance for one frozen evaluation split."""
    if artifact.metadata.calibration_task != task:
        raise ValueError(
            "statistics artifact was calibrated for "
            f"{artifact.metadata.calibration_task!r}, not {task!r}"
        )
    if sorted(artifact.metadata.excluded_test_prompt_ids) != sorted(
        test_prompt_ids
    ):
        raise ValueError(
            "statistics artifact excluded test prompts do not match "
            "the frozen evaluation split"
        )


def validate_statistics_bundle_binding(
    artifact: StatisticsArtifact,
    bundle_metadata: Mapping[str, object],
) -> None:
    """Require the alarm bundle to identify this exact statistics artifact."""
    observed = bundle_metadata.get("expected_attention_stats_sha256")
    if observed != artifact.digest:
        raise ValueError(
            "alarm bundle does not bind to this expected-attention "
            "statistics artifact"
        )


class QueryMomentAccumulator:
    """Streaming sample moments for per-layer, per-head query vectors."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_attention_heads: int,
        head_dim: int,
        n_sink: int,
    ) -> None:
        if min(num_layers, num_attention_heads, head_dim) < 1:
            raise ValueError("query dimensions must be positive")
        if n_sink < 0:
            raise ValueError("n_sink must be non-negative")
        self._shape = (num_layers, num_attention_heads, head_dim)
        self._n_sink = n_sink
        self._sum = torch.zeros(self._shape, dtype=torch.float64)
        self._sum_outer = torch.zeros(
            (*self._shape, head_dim), dtype=torch.float64
        )
        self._count = 0

    def add(self, queries: torch.Tensor) -> None:
        """Add ``[layers, heads, positions, head_dim]`` pre-RoPE queries."""
        layers, heads, positions, width = queries.shape
        expected_layers, expected_heads, expected_width = self._shape
        if (layers, heads, width) != (
            expected_layers,
            expected_heads,
            expected_width,
        ):
            raise ValueError(
                "query shape is incompatible with accumulator: "
                f"got {tuple(queries.shape)}"
            )
        try:
            usable = queries[:, :, self._n_sink :, :].to(
                device="cpu", dtype=torch.float64
            )
        except RuntimeError as error:
            raise ValueError("could not move query states to CPU") from error
        count = usable.shape[2]
        if count == 0:
            return
        self._sum += usable.sum(dim=2)
        self._sum_outer += torch.einsum("lhpd,lhpe->lhde", usable, usable)
        self._count += count

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return sample mean, unbiased covariance, and pooled token count."""
        if self._count < 2:
            raise ValueError(
                "at least two non-sink query tokens are required"
            )
        mu = self._sum / self._count
        centered_outer = (
            self._sum_outer
            - torch.einsum("lhd,lhe->lhde", self._sum, self._sum)
            / self._count
        )
        cov = centered_outer / (self._count - 1)
        return mu.to(torch.float32), cov.to(torch.float32), self._count


def collect_query_moments(
    model: Any,
    input_ids: Sequence[torch.Tensor],
    *,
    n_sink: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Collect pre-RoPE query moments from training-only token sequences.

    KVPress's own statistics routine captures the inputs to
    ``apply_rotary_pos_emb``. Using the same hook is intentional: the cached
    mean and covariance have exactly the representation that the press later
    rotates for the candidate cache length.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(
            "model does not expose a config for query statistics"
        )
    try:
        dimensions = (
            int(config.num_hidden_layers),
            int(config.num_attention_heads),
            int(config.head_dim),
        )
        parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError, ValueError) as error:
        raise ValueError(
            "model does not expose expected-attention dimensions"
        ) from error
    accumulator = QueryMomentAccumulator(
        num_layers=dimensions[0],
        num_attention_heads=dimensions[1],
        head_dim=dimensions[2],
        n_sink=n_sink,
    )
    for index, sequence in enumerate(input_ids):
        if sequence.ndim != 1:
            raise ValueError(
                f"calibration sequence {index} must be rank 1, "
                f"got {sequence.ndim}"
            )
        if sequence.numel() == 0:
            raise ValueError(f"calibration sequence {index} is empty")
        batch = sequence.unsqueeze(0).to(parameter.device)
        attention_mask = torch.ones_like(batch)
        with (
            torch.inference_mode(),
            patch_rotary_embedding(model) as captured_queries,
        ):
            model(
                input_ids=batch,
                attention_mask=attention_mask,
                use_cache=False,
            )
        if len(captured_queries) != dimensions[0]:
            raise RuntimeError(
                "query capture did not observe every transformer layer: "
                f"expected {dimensions[0]}, got {len(captured_queries)}"
            )
        try:
            queries = torch.cat(captured_queries, dim=0)
        except RuntimeError as error:
            raise RuntimeError(
                "could not concatenate captured queries"
            ) from error
        accumulator.add(queries)
    return accumulator.finalize()


def make_expected_attention_stats_press(
    artifact: StatisticsArtifact,
    model: Any,
    *,
    compression_ratio: float,
) -> ExpectedAttentionStatsPress:
    """Create a press from frozen in-memory moments without Hub fallback."""
    artifact.validate_model(model)
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration) as error:
        raise ValueError(
            "model has no parameters for statistics placement"
        ) from error
    press = ExpectedAttentionStatsPress(
        compression_ratio=compression_ratio,
        n_future_positions=artifact.metadata.n_future_positions,
        n_sink=artifact.metadata.n_sink,
        use_covariance=artifact.metadata.use_covariance,
    )
    press.mu = artifact.mu.to(device=parameter.device, dtype=parameter.dtype)
    press.cov = artifact.cov.to(
        device=parameter.device, dtype=parameter.dtype
    )
    return press


def load_expected_attention_stats_press(
    directory: Path,
    model: Any,
    *,
    compression_ratio: float,
) -> ExpectedAttentionStatsPress:
    """Load local frozen moments and create an ExpectedAttentionStatsPress."""
    return make_expected_attention_stats_press(
        StatisticsArtifact.load(directory),
        model,
        compression_ratio=compression_ratio,
    )


def _artifact_digest(
    metadata: StatisticsMetadata,
    mu: torch.Tensor,
    cov: torch.Tensor,
) -> str:
    metadata_payload = metadata.model_dump(exclude={"artifact_sha256"})
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    for tensor in (mu, cov):
        digest.update(
            tensor.detach()
            .cpu()
            .to(torch.float32)
            .contiguous()
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _write_metadata(path: Path, metadata: StatisticsMetadata) -> None:
    path.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n"
    )
