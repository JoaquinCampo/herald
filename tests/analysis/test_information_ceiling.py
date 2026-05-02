"""Tests for src/herald/analysis/information_ceiling.py.

CPU-only, fast. Synthetic fixtures mimic the production parquet
schema. Catastrophic runs have a feature ramp before onset that
should make `online` separate clearly above `position+metadata`.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from herald.analysis.information_ceiling import (
    InformationCeilingConfig,
    build_token_dataset,
    collect_run_onsets,
    compute_group_auroc,
    compute_per_feature_mi,
    run_information_ceiling,
)
from herald.metrics.io import RUNS_SCHEMA, TOKENS_SCHEMA


def _looping_token_ids(window: int = 20, repeats: int = 4) -> list[int]:
    prefix = list(range(100, 150))
    pattern = list(range(window))
    suffix = list(range(50, 70))
    return prefix + pattern * repeats + suffix


def _runs_record(
    run_id: str,
    press: str,
    ratio: float,
    catastrophes: list[str],
    token_ids: list[int],
    max_new_tokens: int = 256,
    correct: bool | None = None,
    task: str = "gsm8k",
) -> dict:
    return {
        "run_id": run_id,
        "prompt_id": run_id + "_p",
        "prompt_text": "x",
        "prompt_hash": "h",
        "model": "fake",
        "model_revision": None,
        "tokenizer_revision": None,
        "dtype": "float16",
        "device_class": "cpu",
        "task": task,
        "press": press,
        "compression_ratio": ratio,
        "max_new_tokens": max_new_tokens,
        "decoding_config": [],
        "seed": 0,
        "baseline_run_id": "",
        "generated_text": "",
        "generated_token_ids": token_ids,
        "num_tokens_generated": len(token_ids),
        "stop_reason": "eos",
        "predicted_answer": None,
        "ground_truth": "",
        "correct": correct,
        "catastrophes": catastrophes,
        "replay_status": "ok",
        "replay_error": None,
        "created_at": "2026-05-02T00:00:00Z",
        "herald_git_sha": "test",
        "wall_clock_per_token": 0.0,
        "peak_memory_mb": 0.0,
        "kv_size_at_end": 0.0,
        "policy_name": "fixed_ratio",
        "replay_wall_clock_seconds": 0.0,
    }


def _tokens_rows(
    run_id: str,
    n: int,
    entropy_fn=lambda t: 1.0,
) -> list[dict]:
    rows: list[dict] = []
    for t in range(n):
        rows.append(
            {
                "run_id": run_id,
                "token_pos": t,
                "token_id": t,
                "token_str": "x",
                "entropy": float(entropy_fn(t)),
                "top1_prob": 0.9,
                "top5_prob": 0.95,
                "top5_logprobs": [-0.1, -1.0, -2.0, -3.0, -4.0],
                "h_alts": 0.5,
                "avg_logp": -1.0,
                "delta_h": 0.0,
                "delta_h_valid": True,
                "kl_div": 0.05,
                "top10_jaccard": 0.8,
                "eff_vocab_size": 5.0,
                "tail_mass": 0.05,
                "logit_range": 10.0,
                "lookback_ratio": float("nan"),
            }
        )
    return rows


def _write_runs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=RUNS_SCHEMA)
    pq.write_table(table, path)


def _write_tokens(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=TOKENS_SCHEMA)
    pq.write_table(table, path)


@pytest.fixture
def synthetic_information_dataset(tmp_path: Path) -> Path:
    """Multi-run dataset where entropy ramps up shortly before onset
    in catastrophic runs but stays flat in healthy runs and well
    before onset. Built so that online features should beat
    position+metadata at small horizons.
    """
    rng = np.random.default_rng(11)
    root = tmp_path / "phase0"
    runs_path = root / "final" / "runs.parquet"
    tokens_root = root / "final" / "tokens"

    runs: list[dict] = []
    onset_target = 70  # detect_looping_onset returns 50+20=70.

    n_cat = 16
    n_ctrl = 16

    for i in range(n_cat):
        rid = f"cat_{i:02d}"
        ids = _looping_token_ids(window=20, repeats=4)
        runs.append(
            _runs_record(
                rid,
                "snapkv",
                0.875,
                ["looping"],
                ids,
                max_new_tokens=256,
            )
        )

        def cat_entropy(t: int, _i: int = i) -> float:
            base = 0.6 + 0.05 * rng.standard_normal()
            ramp = max(0.0, (t - (onset_target - 30))) * 0.05
            return base + ramp

        _write_tokens(
            tokens_root
            / "press=snapkv"
            / "ratio=0.8750"
            / f"cat_{i:02d}.parquet",
            _tokens_rows(rid, len(ids), entropy_fn=cat_entropy),
        )

    for i in range(n_ctrl):
        rid = f"ctrl_{i:02d}"
        ids = list(range(150))
        runs.append(
            _runs_record(
                rid,
                "snapkv",
                0.875,
                [],
                ids,
                max_new_tokens=256,
            )
        )

        def ctrl_entropy(t: int, _i: int = i) -> float:
            return 0.6 + 0.05 * rng.standard_normal()

        _write_tokens(
            tokens_root
            / "press=snapkv"
            / "ratio=0.8750"
            / f"ctrl_{i:02d}.parquet",
            _tokens_rows(rid, len(ids), entropy_fn=ctrl_entropy),
        )

    _write_runs(runs_path, runs)
    return root


def test_collect_run_onsets_assigns_onset_to_catastrophic_only(
    synthetic_information_dataset: Path,
):
    runs = pl.read_parquet(
        synthetic_information_dataset / "final" / "runs.parquet"
    )
    onsets, sources = collect_run_onsets(runs, InformationCeilingConfig())
    cat_ids = [r for r in onsets if r.startswith("cat_")]
    ctrl_ids = [r for r in onsets if r.startswith("ctrl_")]
    assert len(cat_ids) == 16
    assert len(ctrl_ids) == 16
    assert all(onsets[r] == 70 for r in cat_ids)
    assert all(onsets[r] is None for r in ctrl_ids)
    assert all(sources[r] == "looping" for r in cat_ids)
    assert all(sources[r] == "none" for r in ctrl_ids)


def test_build_token_dataset_excludes_post_onset_and_labels_horizons(
    synthetic_information_dataset: Path,
):
    runs = pl.read_parquet(
        synthetic_information_dataset / "final" / "runs.parquet"
    )
    from herald.analysis.event_study import load_tokens_for_runs

    tokens = load_tokens_for_runs(
        synthetic_information_dataset / "final" / "tokens",
        runs["run_id"].to_list(),
    )
    cfg = InformationCeilingConfig(horizons=(5, 10))
    df, used, missing = build_token_dataset(runs, tokens, cfg)
    assert df.height > 0
    assert "entropy" in used
    assert missing == []
    # No catastrophic-run tokens at or past onset (=70).
    cat_post = df.filter(
        (pl.col("run_id").str.starts_with("cat_"))
        & (pl.col("token_pos") >= 70)
    )
    assert cat_post.height == 0
    # Future-damage label exists and is binary.
    assert set(df["future_damage_h5"].unique().to_list()) <= {0, 1}
    # At t=66 (within 5 of onset) catastrophic rows should be 1.
    cat_warn = df.filter(
        (pl.col("run_id").str.starts_with("cat_"))
        & (pl.col("token_pos") == 66)
    )
    assert cat_warn.height > 0 and cat_warn[
        "future_damage_h5"
    ].unique().to_list() == [1]
    # At t=10 (well before onset) the H=5 label should be 0.
    cat_far = df.filter(
        (pl.col("run_id").str.starts_with("cat_"))
        & (pl.col("token_pos") == 10)
    )
    assert cat_far["future_damage_h5"].unique().to_list() == [0]


def test_compute_per_feature_mi_separates_signal_from_noise(
    synthetic_information_dataset: Path,
):
    runs = pl.read_parquet(
        synthetic_information_dataset / "final" / "runs.parquet"
    )
    from herald.analysis.event_study import load_tokens_for_runs

    tokens = load_tokens_for_runs(
        synthetic_information_dataset / "final" / "tokens",
        runs["run_id"].to_list(),
    )
    cfg = InformationCeilingConfig(horizons=(5,), max_samples=5000)
    df, used, _ = build_token_dataset(runs, tokens, cfg)
    by_feature = compute_per_feature_mi(
        df,
        feature_names=["entropy", "top1_prob", "token_pos"],
        discrete_mask=[False, False, False],
        horizons=(5,),
        max_samples=cfg.max_samples,
        seed=0,
    )
    assert by_feature.height == 3
    mi_entropy = float(
        by_feature.filter(pl.col("feature") == "entropy")["mi"][0]
    )
    mi_top1 = float(
        by_feature.filter(pl.col("feature") == "top1_prob")["mi"][0]
    )
    # Entropy carries the signal; top1_prob is constant -> ~0 MI.
    assert mi_entropy > mi_top1


def test_compute_group_auroc_returns_higher_score_for_online_group(
    synthetic_information_dataset: Path,
):
    runs = pl.read_parquet(
        synthetic_information_dataset / "final" / "runs.parquet"
    )
    from herald.analysis.event_study import load_tokens_for_runs

    tokens = load_tokens_for_runs(
        synthetic_information_dataset / "final" / "tokens",
        runs["run_id"].to_list(),
    )
    cfg = InformationCeilingConfig(
        horizons=(5,),
        n_bootstrap=0,
        n_splits=4,
        max_samples=5000,
    )
    df, used, _ = build_token_dataset(runs, tokens, cfg)
    by_group = compute_group_auroc(
        df,
        group_features={
            "position": ["token_pos", "relative_progress"],
            "online": ["entropy"],
        },
        horizons=(5,),
        cfg=cfg,
    )
    pos_auroc = by_group.filter(pl.col("group") == "position")["auroc"][0]
    online_auroc = by_group.filter(pl.col("group") == "online")["auroc"][0]
    assert pos_auroc is not None
    assert online_auroc is not None
    assert float(online_auroc) > float(pos_auroc)


def test_run_information_ceiling_smoke(
    synthetic_information_dataset: Path, tmp_path: Path
):
    out = tmp_path / "ic"
    cfg = InformationCeilingConfig(
        horizons=(5, 10),
        n_bootstrap=10,
        n_splits=4,
        max_samples=5000,
        min_runs_per_press=10,
    )
    result = run_information_ceiling(synthetic_information_dataset, out, cfg)
    summary = result.summary
    assert "blockers" not in summary
    assert summary["n_compressed_runs"] == 32
    assert summary["positive_label_counts_by_horizon"]["5"] > 0
    headline = summary["headline_by_horizon"]
    assert "5" in headline and "10" in headline
    h5 = headline["5"]
    assert h5["auroc_position_only"] is not None
    assert h5["auroc_all"] is not None
    # On this fixture, online should add something at H=5.
    assert h5["incremental_auroc_gain"] is not None
    assert (out / "information_summary.json").exists()
    assert (out / "information_by_feature.parquet").exists()
    assert (out / "information_by_group.parquet").exists()
    assert (out / "information_by_horizon.png").exists()
    # Phase 0 smoke flag should be true (only one task, single press).
    assert summary["phase0_smoke"] is True


def test_run_information_ceiling_blockers_when_input_missing(
    tmp_path: Path,
):
    cfg = InformationCeilingConfig()
    result = run_information_ceiling(tmp_path / "nope", tmp_path / "out", cfg)
    assert "blockers" in result.summary
    assert any("missing" in b for b in result.summary["blockers"])


def test_cli_main_synthetic(
    synthetic_information_dataset: Path, tmp_path: Path
):
    from scripts.build_information_ceiling import main

    out = tmp_path / "cli_out"
    rc = main(
        [
            "--input",
            str(synthetic_information_dataset),
            "--output",
            str(out),
            "--horizons",
            "5,10",
            "--n-bootstrap",
            "5",
            "--n-splits",
            "4",
            "--max-samples",
            "3000",
        ]
    )
    assert rc == 0
    assert (out / "information_summary.json").exists()
    summary = json.loads((out / "information_summary.json").read_text())
    assert "headline_by_horizon" in summary
