from pathlib import Path

import polars as pl
import pytest

from herald.metrics.io import PerRunPaths, write_run_record
from herald.metrics.repair import validate_manifest


def _seed(root: Path, run_id: str, status: str = "failed") -> None:
    rec = {
        "run_id": run_id,
        "prompt_id": "p",
        "prompt_text": "Q",
        "prompt_hash": "h",
        "model": "x",
        "model_revision": "main",
        "tokenizer_revision": "main",
        "dtype": "float16",
        "device_class": "cpu",
        "task": "gsm8k",
        "press": "none",
        "compression_ratio": 0.0,
        "max_new_tokens": 8,
        "decoding_config": {"do_sample": "False"},
        "seed": 42,
        "baseline_run_id": run_id,
        "generated_text": "a",
        "generated_token_ids": [1],
        "num_tokens_generated": 1,
        "stop_reason": "eos",
        "predicted_answer": "a",
        "ground_truth": "a",
        "correct": True,
        "catastrophes": [],
        "replay_status": status,
        "replay_error": "boom",
        "created_at": "2026-05-01T00:00:00Z",
        "herald_git_sha": "abc",
    }
    p = PerRunPaths(root=root, run_id=run_id)
    write_run_record(rec, p.run)


def test_validate_manifest_accepts_complete_record(tmp_path: Path):
    _seed(tmp_path, "r1")
    row = validate_manifest(tmp_path / "raw" / "runs" / "r1.parquet")
    assert row["run_id"] == "r1"


def test_validate_manifest_rejects_missing_field(tmp_path: Path):
    _seed(tmp_path, "r1")
    df = pl.read_parquet(tmp_path / "raw" / "runs" / "r1.parquet").drop(
        ["seed"]
    )
    df.write_parquet(tmp_path / "raw" / "runs" / "r1.parquet")
    with pytest.raises(ValueError):
        validate_manifest(tmp_path / "raw" / "runs" / "r1.parquet")
