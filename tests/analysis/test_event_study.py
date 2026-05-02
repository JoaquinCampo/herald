"""Tests for src/herald/analysis/event_study.py.

CPU-only, fast. Built around tiny synthetic parquet fixtures that
mimic the production schema (runs.parquet plus partitioned tokens
parquets).
"""

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from herald.analysis.event_study import (
    EventStudyConfig,
    OnsetRecord,
    _bootstrap_ci,
    aggregate,
    build_event_dataframe,
    collect_onsets,
    derive_onset,
    run_event_study,
    select_catastrophic_runs,
)
from herald.metrics.io import RUNS_SCHEMA, TOKENS_SCHEMA


def _looping_token_ids(window: int = 20, repeats: int = 4) -> list[int]:
    # Prefix tokens >= 100 ensure the loop pattern (0..window-1) is
    # unique to the looping region, so detect_looping_onset returns
    # exactly len(prefix) + window for the second-occurrence position.
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
        "task": "gsm8k",
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


def _tokens_rows(run_id: str, n: int) -> list[dict]:
    rows: list[dict] = []
    for t in range(n):
        rows.append(
            {
                "run_id": run_id,
                "token_pos": t,
                "token_id": t,
                "token_str": "x",
                "entropy": 1.0 + 0.01 * t,
                "top1_prob": 0.9 - 0.001 * t,
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
def synthetic_phase0(tmp_path: Path) -> Path:
    """A miniature phase0 layout: 3 runs.

    - run_loop: streaming_llm @ 0.875, looping with predictable onset.
    - run_nt: snapkv @ 0.875, non_termination only.
    - run_clean: streaming_llm @ 0.5, no catastrophes (excluded).
    """
    root = tmp_path / "phase0"
    runs_path = root / "final" / "runs.parquet"
    tokens_root = root / "final" / "tokens"

    loop_ids = _looping_token_ids(window=20, repeats=4)
    nt_ids = list(range(200))
    clean_ids = list(range(120))

    runs = [
        _runs_record(
            "run_loop",
            "streaming_llm",
            0.875,
            ["looping"],
            loop_ids,
            max_new_tokens=256,
        ),
        _runs_record(
            "run_nt",
            "snapkv",
            0.875,
            ["non_termination"],
            nt_ids,
            max_new_tokens=200,
        ),
        _runs_record(
            "run_clean",
            "streaming_llm",
            0.5,
            [],
            clean_ids,
        ),
    ]
    _write_runs(runs_path, runs)

    _write_tokens(
        tokens_root / "press=streaming_llm" / "ratio=0.8750" / "p1.parquet",
        _tokens_rows("run_loop", len(loop_ids)),
    )
    _write_tokens(
        tokens_root / "press=snapkv" / "ratio=0.8750" / "p2.parquet",
        _tokens_rows("run_nt", len(nt_ids)),
    )
    _write_tokens(
        tokens_root / "press=streaming_llm" / "ratio=0.5000" / "p3.parquet",
        _tokens_rows("run_clean", len(clean_ids)),
    )
    return root


def test_derive_onset_looping_uses_window_detector():
    ids = _looping_token_ids(window=20, repeats=4)
    onset, src = derive_onset(["looping"], ids, 256, 0.75)
    assert src == "looping"
    # detect_looping_onset returns the start of the SECOND occurrence
    # of the repeating window (the first repeat).
    assert onset == 50 + 20


def test_derive_onset_nt_proxy_clamped_to_seq_length():
    onset, src = derive_onset(["non_termination"], list(range(50)), 256, 0.75)
    assert src == "non_termination_proxy"
    assert onset == 49


def test_derive_onset_returns_none_when_no_trainable_tag():
    onset, src = derive_onset(["wrong_answer"], [1, 2, 3], 256, 0.75)
    assert onset is None
    assert src == "none"


def test_select_catastrophic_runs_filters_compressed_only(
    synthetic_phase0: Path,
):
    runs = pl.read_parquet(synthetic_phase0 / "final" / "runs.parquet")
    out = select_catastrophic_runs(runs)
    assert sorted(out["run_id"].to_list()) == ["run_loop", "run_nt"]


def test_collect_onsets_records_exclusions(synthetic_phase0: Path):
    runs = pl.read_parquet(synthetic_phase0 / "final" / "runs.parquet")
    cat = select_catastrophic_runs(runs)
    cfg = EventStudyConfig()
    onsets, excluded = collect_onsets(cat, cfg)
    assert len(onsets) == 2
    assert excluded == {"no_token_ids": 0, "no_onset_derivable": 0}

    # Now exclude one by stripping its token_ids.
    runs_bad = runs.with_columns(
        pl.when(pl.col("run_id") == "run_loop")
        .then(pl.lit([], dtype=pl.List(pl.Int32)))
        .otherwise(pl.col("generated_token_ids"))
        .alias("generated_token_ids")
    )
    cat_bad = select_catastrophic_runs(runs_bad)
    onsets_bad, excluded_bad = collect_onsets(cat_bad, cfg)
    assert excluded_bad["no_token_ids"] == 1
    assert {o.run_id for o in onsets_bad} == {"run_nt"}


def test_relative_position_alignment(synthetic_phase0: Path):
    """relative_pos = token_pos - onset_token, clipped to window."""
    runs = pl.read_parquet(synthetic_phase0 / "final" / "runs.parquet")
    cat = select_catastrophic_runs(runs)
    cfg = EventStudyConfig(window_before=10, window_after=10)
    onsets, _ = collect_onsets(cat, cfg)
    # Reload tokens
    from herald.analysis.event_study import load_tokens_for_runs

    tokens = load_tokens_for_runs(
        synthetic_phase0 / "final" / "tokens",
        [o.run_id for o in onsets],
    )
    event_df, used, missing = build_event_dataframe(onsets, tokens, cfg)
    assert event_df.height > 0
    # All relative_pos lie inside the window.
    rps = event_df["relative_pos"].to_numpy()
    assert rps.min() >= -10 and rps.max() <= 10
    # The looping run's relative_pos == 0 row should equal onset_token=70.
    loop_zero = event_df.filter(
        (pl.col("run_id") == "run_loop") & (pl.col("relative_pos") == 0)
    )
    assert loop_zero.height > 0
    assert loop_zero["onset_token"].unique().to_list() == [70]
    assert "top1_top2_margin" in used
    assert missing == []


def test_aggregation_shape(synthetic_phase0: Path):
    runs = pl.read_parquet(synthetic_phase0 / "final" / "runs.parquet")
    cat = select_catastrophic_runs(runs)
    cfg = EventStudyConfig(window_before=5, window_after=5, n_bootstrap=10)
    onsets, _ = collect_onsets(cat, cfg)
    from herald.analysis.event_study import load_tokens_for_runs

    tokens = load_tokens_for_runs(
        synthetic_phase0 / "final" / "tokens",
        [o.run_id for o in onsets],
    )
    event_df, used, _ = build_event_dataframe(onsets, tokens, cfg)
    pooled = aggregate(event_df, cfg, group=None)
    by_press = aggregate(event_df, cfg, group="press")

    # Each (feature, relative_pos) cell yields exactly one row.
    n_pos = event_df["relative_pos"].n_unique()
    assert pooled.height <= len(used) * n_pos
    assert {"n", "mean", "median", "ci_lo", "ci_hi"} <= set(pooled.columns)
    assert "press" in by_press.columns


def test_bootstrap_ci_handles_tiny_samples():
    import numpy as np

    rng = np.random.default_rng(0)
    # Single value => CI collapses to (mean, mean).
    lo, hi = _bootstrap_ci(np.array([3.0]), n_boot=100, rng=rng)
    assert lo == hi == 3.0
    # Two values => bootstrap runs and returns finite numbers.
    lo, hi = _bootstrap_ci(np.array([1.0, 5.0]), n_boot=50, rng=rng)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_run_event_study_smoke(synthetic_phase0: Path, tmp_path: Path):
    out = tmp_path / "event_study"
    cfg = EventStudyConfig(window_before=20, window_after=20, n_bootstrap=20)
    result = run_event_study(synthetic_phase0, out, cfg)

    summary = result.summary
    assert summary["n_total_runs"] == 3
    assert summary["n_catastrophic_runs_selected"] == 2
    assert summary["n_runs_with_onset"] == 2
    assert summary["onset_source_counts"]["looping"] == 1
    assert summary["onset_source_counts"]["non_termination_proxy"] == 1
    assert "blockers" not in summary
    # Artifacts on disk.
    for key in (
        "event_study_parquet",
        "agg_pooled_parquet",
        "agg_by_press_parquet",
        "agg_by_onset_source_parquet",
        "headline_png",
    ):
        assert Path(summary["artifacts"][key]).exists()
    # JSON written.
    summary_path = out / "event_study_summary.json"
    on_disk = json.loads(summary_path.read_text())
    assert on_disk["n_runs_with_onset"] == 2
    assert "phase0_caveat" in on_disk


def test_run_event_study_blockers_when_input_missing(tmp_path: Path):
    cfg = EventStudyConfig()
    result = run_event_study(tmp_path / "nope", tmp_path / "out", cfg)
    assert "blockers" in result.summary
    assert any("missing" in b for b in result.summary["blockers"])


def test_cli_main_synthetic(synthetic_phase0: Path, tmp_path: Path):
    """Function-level CLI smoke."""
    from scripts.build_event_study import main

    out = tmp_path / "cli_out"
    rc = main(
        [
            "--input",
            str(synthetic_phase0),
            "--output",
            str(out),
            "--window-before",
            "20",
            "--window-after",
            "20",
            "--n-bootstrap",
            "10",
        ]
    )
    assert rc == 0
    assert (out / "event_study.parquet").exists()
    assert (out / "event_study_summary.json").exists()


def test_collect_onsets_uses_dataclass_fields():
    """Sanity: OnsetRecord retains the source for downstream
    stratification."""
    rec = OnsetRecord(
        run_id="r",
        press="p",
        compression_ratio=0.5,
        onset_token=10,
        onset_source="looping",
        n_tokens=100,
    )
    assert rec.onset_source == "looping"
