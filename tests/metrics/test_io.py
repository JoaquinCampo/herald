from pathlib import Path

import pyarrow.parquet as pq

from herald.metrics.io import (
    PerRunPaths,
    read_runs,
    write_replay_rows,
    write_run_record,
    write_tokens_rows,
)


def test_runs_schema_has_determinism_columns():
    from herald.metrics.io import RUNS_SCHEMA

    cols = {f.name for f in RUNS_SCHEMA}
    required = {
        "run_id",
        "prompt_id",
        "prompt_text",
        "prompt_hash",
        "model",
        "model_revision",
        "tokenizer_revision",
        "dtype",
        "device_class",
        "task",
        "press",
        "compression_ratio",
        "max_new_tokens",
        "decoding_config",
        "seed",
        "baseline_run_id",
        "generated_text",
        "generated_token_ids",
        "replay_status",
        "herald_git_sha",
        "created_at",
    }
    missing = required - cols
    assert not missing, f"missing columns: {missing}"


def test_per_run_paths_layout(tmp_path: Path):
    p = PerRunPaths(root=tmp_path, run_id="abc1234")
    assert p.run.parent.name == "runs"
    assert p.tokens.parent.name == "tokens"
    assert p.replay.parent.name == "replay"
    assert p.run.parent.parent.name == "raw"


def test_write_and_read_run_record_roundtrip(tmp_path: Path):
    rec = {
        "run_id": "r1",
        "prompt_id": "p1",
        "prompt_text": "Q",
        "prompt_hash": "deadbeef",
        "model": "x",
        "model_revision": None,
        "tokenizer_revision": None,
        "dtype": "float16",
        "device_class": "cpu",
        "task": "gsm8k",
        "press": "none",
        "compression_ratio": 0.0,
        "max_new_tokens": 10,
        "decoding_config": {"do_sample": False},
        "seed": 42,
        "baseline_run_id": "r1",
        "generated_text": "A",
        "generated_token_ids": [1, 2],
        "num_tokens_generated": 2,
        "stop_reason": "eos",
        "predicted_answer": "A",
        "ground_truth": "A",
        "correct": True,
        "catastrophes": [],
        "replay_status": "ok",
        "replay_error": None,
        "created_at": "2026-05-01T00:00:00Z",
        "herald_git_sha": "abc",
    }
    paths = PerRunPaths(root=tmp_path, run_id="r1")
    write_run_record(rec, paths.run)
    df = read_runs(paths.run)
    assert df.height == 1
    assert df["run_id"][0] == "r1"
    assert df["baseline_run_id"][0] == "r1"


def test_write_tokens_rows_roundtrip(tmp_path: Path):
    rows = [
        {
            "run_id": "r1",
            "token_pos": 0,
            "token_id": 5,
            "token_str": "A",
            "entropy": 1.2,
            "top1_prob": 0.8,
            "top5_prob": 0.95,
            "top5_logprobs": [-0.1, -0.5, -1.0, -2.0, -3.0],
            "h_alts": 0.3,
            "avg_logp": -5.0,
            "delta_h": float("nan"),
            "delta_h_valid": False,
            "kl_div": float("nan"),
            "top10_jaccard": float("nan"),
            "eff_vocab_size": 3.3,
            "tail_mass": 0.01,
            "logit_range": 12.0,
            "lookback_ratio": float("nan"),
        },
    ]
    paths = PerRunPaths(root=tmp_path, run_id="r1")
    write_tokens_rows(rows, paths.tokens)
    table = pq.read_table(paths.tokens)
    assert table.num_rows == 1
    assert table.column("token_pos").to_pylist() == [0]


def test_write_replay_rows_roundtrip(tmp_path: Path):
    rows = [
        {
            "run_id": "r1",
            "token_pos": 0,
            "realized_token_id": 5,
            "union_top_k_token_ids": [5, 7, 9],
            "logprobs_compressed": [-0.1, -2.0, -3.0],
            "logprobs_uncompressed": [-0.1, -2.0, -3.0],
            "tail_mass_compressed": 0.01,
            "tail_mass_uncompressed": 0.01,
            "realized_logprob_compressed": -0.1,
            "realized_logprob_uncompressed": -0.1,
            "js_full": 0.0,
            "kl_unc_comp_full": 0.0,
            "kl_comp_unc_full": 0.0,
            "top1_match": True,
            "top1_rank_comp_under_unc": 0,
            "top1_rank_unc_under_comp": 0,
        },
    ]
    paths = PerRunPaths(root=tmp_path, run_id="r1")
    write_replay_rows(rows, paths.replay)
    table = pq.read_table(paths.replay)
    assert table.num_rows == 1
    assert table.column("js_full").to_pylist() == [0.0]
