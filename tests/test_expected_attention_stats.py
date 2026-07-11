# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportPrivateImportUsage=false

"""Training-only ExpectedAttentionStatsPress artifact tests."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from herald.config import Config
from herald.expected_attention_stats import (
    QueryMomentAccumulator,
    StatisticsArtifact,
    StatisticsMetadata,
    load_expected_attention_stats_press,
    validate_statistics_bundle_binding,
    validate_statistics_provenance,
)


class ModelStub:
    """Minimal model surface required by the frozen press factory."""

    def __init__(self, *, num_layers: int) -> None:
        self.config = SimpleNamespace(
            name_or_path="example/llama",
            model_type="llama",
            num_hidden_layers=num_layers,
            num_attention_heads=3,
            head_dim=4,
        )
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        yield self._parameter


def _metadata() -> StatisticsMetadata:
    return StatisticsMetadata(
        model_id="example/llama",
        model_type="llama",
        num_hidden_layers=2,
        num_attention_heads=3,
        head_dim=4,
        n_future_positions=16,
        n_sink=1,
        use_covariance=True,
        calibration_task="ifeval",
        calibration_prompt_ids=["train-1", "train-2"],
        excluded_test_prompt_ids=["test-1"],
        max_prompt_tokens=128,
        query_token_count=10,
    )


def _artifact() -> StatisticsArtifact:
    mu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    cov = torch.eye(4, dtype=torch.float32).repeat(2, 3, 1, 1)
    return StatisticsArtifact(metadata=_metadata(), mu=mu, cov=cov)


def test_artifact_roundtrip_and_press_factory(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "stats"
    digest = _artifact().save(artifact_dir)

    loaded = StatisticsArtifact.load(artifact_dir)
    assert loaded.digest == digest
    assert loaded.metadata.artifact_sha256 == digest
    assert torch.equal(loaded.mu, _artifact().mu)
    assert torch.equal(loaded.cov, _artifact().cov)

    model = ModelStub(num_layers=2)
    press = load_expected_attention_stats_press(
        artifact_dir,
        model,
        compression_ratio=0.5,
    )
    assert press.n_future_positions == 16
    assert press.n_sink == 1
    assert press.use_covariance
    assert press.mu.device == model._parameter.device
    assert press.mu.dtype == model._parameter.dtype
    assert torch.equal(press.mu, loaded.mu)


def test_sweep_config_requires_frozen_statistics_path() -> None:
    with pytest.raises(
        ValueError,
        match="requires expected_attention_stats_path",
    ):
        Config(compressors=["expected_attention_stats"])
    config = Config(
        compressors=["expected_attention_stats"],
        expected_attention_stats_path=Path("results/stats"),
    )
    assert config.expected_attention_stats_path == Path("results/stats")


def test_statistics_provenance_and_bundle_binding_are_exact() -> None:
    artifact = _artifact()

    validate_statistics_provenance(
        artifact,
        task="ifeval",
        test_prompt_ids=["test-1"],
    )
    validate_statistics_bundle_binding(
        artifact,
        {"expected_attention_stats_sha256": artifact.digest},
    )
    with pytest.raises(ValueError, match="do not match"):
        validate_statistics_provenance(
            artifact,
            task="ifeval",
            test_prompt_ids=["other-test"],
        )
    with pytest.raises(ValueError, match="does not bind"):
        validate_statistics_bundle_binding(artifact, {})


def test_artifact_rejects_test_prompt_in_calibration() -> None:
    with pytest.raises(ValueError, match="overlap"):
        StatisticsMetadata(
            **_metadata()
            .model_copy(update={"excluded_test_prompt_ids": ["train-1"]})
            .model_dump()
        )


def test_artifact_rejects_model_shape_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "stats"
    _artifact().save(artifact_dir)
    model = ModelStub(num_layers=1)

    with pytest.raises(ValueError, match="num_hidden_layers"):
        load_expected_attention_stats_press(
            artifact_dir,
            model,
            compression_ratio=0.5,
        )


def test_query_moments_match_sample_covariance() -> None:
    acc = QueryMomentAccumulator(
        num_layers=1,
        num_attention_heads=1,
        head_dim=2,
        n_sink=1,
    )
    queries = torch.tensor([[[[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]]]])
    acc.add(queries)
    mu, cov, count = acc.finalize()

    assert count == 2
    torch.testing.assert_close(mu, torch.tensor([[[3.0, 30.0]]]))
    torch.testing.assert_close(
        cov,
        torch.tensor([[[[2.0, 20.0], [20.0, 200.0]]]]),
    )
