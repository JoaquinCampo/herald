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


def test_metrics_build_runs_all_modules(tmp_path: Path):
    final = tmp_path / "final"
    final.mkdir()
    pl.DataFrame(
        {
            "run_id": ["b", "c"],
            "prompt_id": ["p", "p"],
            "press": ["none", "snapkv"],
            "compression_ratio": [0.0, 0.875],
            "baseline_run_id": ["b", "b"],
            "generated_text": ["x", "y"],
            "correct": [True, False],
            "catastrophes": [[], ["looping"]],
        }
    ).write_parquet(final / "runs.parquet")
    rep = final / "replay" / "press=snapkv" / "ratio=0.8750"
    rep.mkdir(parents=True)
    pl.DataFrame(
        {
            "run_id": ["c"],
            "token_pos": [0],
            "realized_token_id": [1],
            "union_top_k_token_ids": [[1, 2]],
            "logprobs_compressed": [[-0.1, -2.0]],
            "logprobs_uncompressed": [[-0.1, -2.0]],
            "tail_mass_compressed": [0.0],
            "tail_mass_uncompressed": [0.0],
            "realized_logprob_compressed": [-0.5],
            "realized_logprob_uncompressed": [-0.3],
            "js_full": [0.05],
            "kl_unc_comp_full": [0.04],
            "kl_comp_unc_full": [0.04],
            "top1_match": [True],
            "top1_rank_comp_under_unc": [0],
            "top1_rank_unc_under_comp": [0],
        }
    ).write_parquet(rep / "c.parquet")

    from herald.metrics import outcome, tags, token, trajectory

    out = tmp_path / "metrics"
    out.mkdir()
    token.build(final, out)
    trajectory.build(final, out)
    outcome.build(final, out)
    tags.build(final, out)

    for name in (
        "token_metrics",
        "trajectory_metrics",
        "outcome",
        "tags",
    ):
        assert (out / f"{name}.parquet").exists()
