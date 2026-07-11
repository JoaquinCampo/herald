# pyright: reportMissingImports=false

from pathlib import Path

import numpy as np
import pyarrow as pa  # pyright: ignore[reportMissingImports]
import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
import pytest

from herald.config import Config
from herald.features import FEATURE_NAMES
from herald.storage import append_hybrid, save_reference
from herald.sweep_provenance import (
    bind_table_to_sweep_config,
    initialize_sweep_config,
    validate_parquet_sweep_config,
    validate_sweep_completeness,
)


def _write_config(path: Path, *, statistics_sha256: str = "a" * 64) -> None:
    config = Config(
        models=["llama"],
        tasks=["ifeval"],
        compressors=["expected_attention_stats"],
        ratios=[0.25],
        results_dir=path.parent,
        expected_attention_stats_path=Path("stats"),
        expected_attention_stats_sha256=statistics_sha256,
    )
    path.write_text(config.model_dump_json(indent=2))


def _write_bound_parquet(parquet: Path, config: Path) -> None:
    table = pa.table({"prompt_id": ["p0"], "dq": [0.0]})
    pq.write_table(  # type: ignore[no-untyped-call]
        bind_table_to_sweep_config(table, config),
        parquet,
    )


def test_initializes_only_a_fresh_or_matching_sweep_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    config = Config.model_validate_json(config_path.read_text())
    fresh_results = tmp_path / "fresh"

    expected_path = fresh_results / "config.json"
    assert initialize_sweep_config(fresh_results, config) == expected_path
    assert initialize_sweep_config(fresh_results, config) == expected_path

    changed = config.model_copy(update={"ratios": [0.5]})
    with pytest.raises(ValueError, match="does not match"):
        initialize_sweep_config(fresh_results, changed)

    orphaned_results = tmp_path / "orphaned"
    orphaned_results.mkdir()
    (orphaned_results / "reference.json").write_text("{}\n")
    with pytest.raises(ValueError, match="without config.json"):
        initialize_sweep_config(orphaned_results, config)


def test_rejects_incomplete_hybrid_sweep_before_dataset_build(
    tmp_path: Path,
) -> None:
    config = Config(
        models=["llama"],
        tasks=["ifeval"],
        compressors=["random"],
        ratios=[0.25],
        prompts_per_task=1,
        switch_stride=2,
        results_dir=tmp_path,
    )
    features = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    save_reference(
        tmp_path,
        "llama",
        "ifeval",
        prompt_id="p0",
        prompt_input_ids=[1],
        gen_ids=[2, 3, 4],
        text="reference",
        q=1.0,
        features=features,
    )
    append_hybrid(
        tmp_path,
        "llama",
        "ifeval",
        "random",
        0.25,
        prompt_id="p0",
        s=0,
        new_ids=[2, 3, 4],
        text="hybrid",
        q=1.0,
        dq=0.0,
        features=features,
    )

    with pytest.raises(ValueError, match="incomplete hybrid sweep"):
        validate_sweep_completeness(tmp_path, config)

    append_hybrid(
        tmp_path,
        "llama",
        "ifeval",
        "random",
        0.25,
        prompt_id="p0",
        s=2,
        new_ids=[4],
        text="hybrid",
        q=1.0,
        dq=0.0,
        features=features,
    )
    validate_sweep_completeness(tmp_path, config)

    with pytest.raises(ValueError, match="models not in sweep config"):
        validate_sweep_completeness(tmp_path, config, models=["qwen3"])
    with pytest.raises(ValueError, match="tasks not in sweep config"):
        validate_sweep_completeness(tmp_path, config, tasks=["gsm8k"])

    append_hybrid(
        tmp_path,
        "llama",
        "ifeval",
        "random",
        0.25,
        prompt_id="p0",
        s=0,
        new_ids=[2, 3, 4],
        text="hybrid",
        q=1.0,
        dq=0.0,
        features=features,
    )
    with pytest.raises(ValueError, match="duplicate hybrid cells"):
        validate_sweep_completeness(tmp_path, config)

    hybrid_shard = next(
        (tmp_path / "llama" / "ifeval" / "hybrids").glob("*.jsonl")
    )
    hybrid_shard.write_text(hybrid_shard.read_text().splitlines()[0] + "\n")
    append_hybrid(
        tmp_path,
        "llama",
        "ifeval",
        "random",
        0.25,
        prompt_id="p0",
        s=99,
        new_ids=[4],
        text="hybrid",
        q=1.0,
        dq=0.0,
        features=features,
    )
    extra_feature = next(
        (tmp_path / "llama" / "ifeval" / "hybrid_features").rglob("*s99.npy")
    )
    extra_feature.unlink()
    with pytest.raises(ValueError, match="unexpected hybrid cells"):
        validate_sweep_completeness(tmp_path, config)

    stale_shard = (
        tmp_path / "llama" / "ifeval" / "hybrids" / "stale__0.2500.jsonl"
    )
    stale_shard.write_text("")
    with pytest.raises(ValueError, match="unexpected hybrid shards"):
        validate_sweep_completeness(tmp_path, config)

    stale_reference = (
        tmp_path / "llama" / "ifeval" / "references" / "stale.json"
    )
    stale_reference.write_text("{}\n")
    with pytest.raises(ValueError, match="unexpected reference artifacts"):
        validate_sweep_completeness(tmp_path, config)

    (tmp_path / "llama" / "ifeval" / "references" / "p0.npy").unlink()
    with pytest.raises(ValueError, match="missing reference features"):
        validate_sweep_completeness(tmp_path, config)


def test_validates_parquet_against_bound_sweep_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    parquet = tmp_path / "switch.parquet"
    _write_config(config)
    _write_bound_parquet(parquet, config)

    loaded = validate_parquet_sweep_config(
        parquet,
        config,
        required_compressor="expected_attention_stats",
        expected_statistics_sha256="a" * 64,
    )

    assert loaded.expected_attention_stats_sha256 == "a" * 64


def test_rejects_stale_config_or_wrong_statistics_artifact(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    parquet = tmp_path / "switch.parquet"
    _write_config(config)
    _write_bound_parquet(parquet, config)

    with pytest.raises(ValueError, match="statistics digest"):
        validate_parquet_sweep_config(
            parquet,
            config,
            required_compressor="expected_attention_stats",
            expected_statistics_sha256="b" * 64,
        )

    config.write_text(config.read_text() + "\n")
    with pytest.raises(ValueError, match="sha256"):
        validate_parquet_sweep_config(
            parquet,
            config,
            required_compressor="expected_attention_stats",
            expected_statistics_sha256="a" * 64,
        )
