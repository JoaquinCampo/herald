from pathlib import Path

import polars as pl

from herald.metrics.io import (
    PerRunPaths,
    finalize_dataset,
    write_run_record,
)


def _seed_run(root: Path, run_id: str, press: str, ratio: float) -> None:
    rec = {
        "run_id": run_id,
        "prompt_id": "p",
        "prompt_text": "q",
        "prompt_hash": "h",
        "model": "x",
        "model_revision": None,
        "tokenizer_revision": None,
        "dtype": "float16",
        "device_class": "cpu",
        "task": "gsm8k",
        "press": press,
        "compression_ratio": ratio,
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
        "replay_status": "ok",
        "replay_error": None,
        "created_at": "2026-05-01T00:00:00Z",
        "herald_git_sha": "abc",
    }
    p = PerRunPaths(root=root, run_id=run_id)
    write_run_record(rec, p.run)


def test_finalize_dataset_concatenates_runs(tmp_path: Path):
    _seed_run(tmp_path, "r1", "none", 0.0)
    _seed_run(tmp_path, "r2", "snapkv", 0.875)

    finalize_dataset(root=tmp_path)

    runs_path = tmp_path / "final" / "runs.parquet"
    assert runs_path.exists()
    df = pl.read_parquet(runs_path)
    assert df.height == 2
    assert set(df["run_id"]) == {"r1", "r2"}
