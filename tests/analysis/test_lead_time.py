"""Tests for src/herald/analysis/lead_time.py.

CPU-only, fast. Uses tiny synthetic parquet fixtures that mimic the
production schema. Catastrophic and control trajectories are
constructed so that a target feature separates them sharply
sufficiently before the onset to produce a finite lead-time, while
other features stay degenerate.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from herald.analysis.lead_time import (
    LeadTimeConfig,
    assign_virtual_onsets,
    build_aligned_long,
    compute_auroc_with_ci,
    compute_lead_time,
    run_lead_time,
    select_healthy_controls,
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


def test_select_healthy_controls_filters_compressed_only():
    runs = pl.DataFrame(
        {
            "run_id": ["a", "b", "c", "d"],
            "press": ["none", "snapkv", "snapkv", "streaming_llm"],
            "catastrophes": [
                [],
                [],
                ["looping"],
                ["wrong_answer"],
            ],
        }
    )
    out = select_healthy_controls(runs)
    assert sorted(out["run_id"].to_list()) == ["b", "d"]


@pytest.fixture
def synthetic_pair_dataset(tmp_path: Path) -> Path:
    """A dataset with many catastrophic and control runs in one
    stratum (gsm8k, snapkv, 0.875), constructed so that entropy
    separates the two cohorts before catastrophic onset.

    Catastrophic runs have rising entropy starting ~50 tokens before
    the onset. Controls have flat low entropy throughout.
    """
    rng = np.random.default_rng(7)
    root = tmp_path / "phase0"
    runs_path = root / "final" / "runs.parquet"
    tokens_root = root / "final" / "tokens"

    runs: list[dict] = []
    onset_target = 70  # detect_looping_onset returns 50+20=70 here.

    n_cat = 8
    n_ctrl = 8

    # Catastrophic runs: looping pattern with rising entropy near onset.
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
            # Rising entropy starting at ~ onset - 60 tokens.
            base = 0.6 + 0.05 * rng.standard_normal()
            ramp = max(0.0, (t - (onset_target - 60))) * 0.04
            return base + ramp

        _write_tokens(
            tokens_root
            / "press=snapkv"
            / "ratio=0.8750"
            / f"cat_{i:02d}.parquet",
            _tokens_rows(rid, len(ids), entropy_fn=cat_entropy),
        )

    # Control runs: same press/ratio, flat low entropy, no catastrophes.
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


def test_assign_virtual_onsets_pairs_by_stratum(
    synthetic_pair_dataset: Path,
):
    runs = pl.read_parquet(synthetic_pair_dataset / "final" / "runs.parquet")
    from herald.analysis.event_study import (
        EventStudyConfig,
        collect_onsets,
        select_catastrophic_runs,
    )

    cat = select_catastrophic_runs(runs)
    onsets, _ = collect_onsets(cat, EventStudyConfig())
    healthy = select_healthy_controls(runs)
    cfg = LeadTimeConfig()
    matched, info = assign_virtual_onsets(onsets, healthy, runs, cfg)
    assert len(matched) == 8
    # All controls receive the same virtual onset (median = 70 here).
    assert {c.virtual_onset for c in matched} == {70}
    sk = list(info["stratum_counts"].keys())[0]
    assert info["stratum_counts"][sk]["n_cat"] == 8
    assert info["stratum_counts"][sk]["n_ctrl"] == 8


def test_build_aligned_long_includes_label(
    synthetic_pair_dataset: Path,
):
    runs = pl.read_parquet(synthetic_pair_dataset / "final" / "runs.parquet")
    from herald.analysis.event_study import (
        EventStudyConfig,
        collect_onsets,
        load_tokens_for_runs,
        select_catastrophic_runs,
    )

    cat = select_catastrophic_runs(runs)
    onsets, _ = collect_onsets(cat, EventStudyConfig())
    healthy = select_healthy_controls(runs)
    cfg = LeadTimeConfig(window_before=80, window_after=20, n_bootstrap=0)
    matched, _ = assign_virtual_onsets(onsets, healthy, runs, cfg)
    tokens = load_tokens_for_runs(
        synthetic_pair_dataset / "final" / "tokens",
        [o.run_id for o in onsets] + [c.run_id for c in matched],
    )
    long, used, missing = build_aligned_long(
        onsets, matched, tokens, cfg, runs
    )
    assert long.height > 0
    labels = set(long["label"].unique().to_list())
    assert labels == {0, 1}
    rps = long["relative_pos"].to_numpy()
    assert rps.min() >= -80 and rps.max() <= 20
    assert "entropy" in used
    assert missing == []


def test_compute_auroc_separates_when_signal_is_present(
    synthetic_pair_dataset: Path,
):
    runs = pl.read_parquet(synthetic_pair_dataset / "final" / "runs.parquet")
    from herald.analysis.event_study import (
        EventStudyConfig,
        collect_onsets,
        load_tokens_for_runs,
        select_catastrophic_runs,
    )

    cat = select_catastrophic_runs(runs)
    onsets, _ = collect_onsets(cat, EventStudyConfig())
    healthy = select_healthy_controls(runs)
    cfg = LeadTimeConfig(
        window_before=70,
        window_after=20,
        n_bootstrap=20,
    )
    matched, _ = assign_virtual_onsets(onsets, healthy, runs, cfg)
    tokens = load_tokens_for_runs(
        synthetic_pair_dataset / "final" / "tokens",
        [o.run_id for o in onsets] + [c.run_id for c in matched],
    )
    long, _, _ = build_aligned_long(onsets, matched, tokens, cfg, runs)
    by_feat = compute_auroc_with_ci(long, cfg)
    assert by_feat.height > 0
    # Entropy AUROC should be near 0.5 well before the ramp and near
    # 1 by onset.
    ent = by_feat.filter(pl.col("feature") == "entropy")
    near_onset = ent.filter(pl.col("relative_pos") == 0)
    assert near_onset.height == 1
    auc_at_onset = float(near_onset["auroc"][0])
    assert auc_at_onset > 0.85


def test_lead_time_returns_finite_value_when_separation_persists(
    synthetic_pair_dataset: Path,
):
    runs = pl.read_parquet(synthetic_pair_dataset / "final" / "runs.parquet")
    from herald.analysis.event_study import (
        EventStudyConfig,
        collect_onsets,
        load_tokens_for_runs,
        select_catastrophic_runs,
    )

    cat = select_catastrophic_runs(runs)
    onsets, _ = collect_onsets(cat, EventStudyConfig())
    healthy = select_healthy_controls(runs)
    cfg = LeadTimeConfig(
        window_before=70,
        window_after=20,
        n_bootstrap=40,
        persistence=5,
    )
    matched, _ = assign_virtual_onsets(onsets, healthy, runs, cfg)
    tokens = load_tokens_for_runs(
        synthetic_pair_dataset / "final" / "tokens",
        [o.run_id for o in onsets] + [c.run_id for c in matched],
    )
    long, _, _ = build_aligned_long(onsets, matched, tokens, cfg, runs)
    by_feat = compute_auroc_with_ci(long, cfg)
    leads = compute_lead_time(by_feat, cfg)
    # Entropy should produce a finite, positive lead time.
    assert leads.get("entropy") is not None
    assert leads["entropy"] > 0
    # A flat feature like top10_jaccard (constant 0.8 here) should
    # never satisfy the AUROC threshold.
    assert leads.get("top10_jaccard") is None


def test_lead_time_none_when_no_signal():
    # Build a degenerate "by_feature" frame where AUROC never crosses
    # the threshold.
    df = pl.DataFrame(
        {
            "feature": ["entropy"] * 10,
            "relative_pos": list(range(-9, 1)),
            "auroc": [0.55] * 10,
            "ci_lo": [0.40] * 10,
            "ci_hi": [0.65] * 10,
            "n_pos": [5] * 10,
            "n_neg": [5] * 10,
        }
    )
    leads = compute_lead_time(df, LeadTimeConfig(persistence=5))
    assert leads["entropy"] is None


def test_lead_time_picks_earliest_qualifying_streak():
    # Construct a curve that only qualifies for r in [-9, -5], and
    # check the earliest-qualifying convention returns 9.
    rps = list(range(-12, 1))
    auroc = [
        0.55,
        0.55,
        0.55,
        0.7,
        0.7,
        0.7,
        0.7,
        0.7,
        0.55,
        0.55,
        0.7,
        0.7,
        0.7,
    ]
    ci_lo = [
        0.45,
        0.45,
        0.45,
        0.55,
        0.55,
        0.55,
        0.55,
        0.55,
        0.45,
        0.45,
        0.55,
        0.55,
        0.55,
    ]
    df = pl.DataFrame(
        {
            "feature": ["x"] * len(rps),
            "relative_pos": rps,
            "auroc": auroc,
            "ci_lo": ci_lo,
            "ci_hi": [c + 0.1 for c in ci_lo],
            "n_pos": [5] * len(rps),
            "n_neg": [5] * len(rps),
        }
    )
    leads = compute_lead_time(df, LeadTimeConfig(persistence=5))
    # The first 5-streak of qualifying positions is rps[3:8] = -9..-5,
    # so lead-time is 9.
    assert leads["x"] == 9


def test_run_lead_time_smoke_pair(
    synthetic_pair_dataset: Path, tmp_path: Path
):
    out = tmp_path / "lead_time"
    cfg = LeadTimeConfig(
        window_before=70,
        window_after=20,
        n_bootstrap=20,
        persistence=5,
    )
    result = run_lead_time(synthetic_pair_dataset, out, cfg)
    summary = result.summary
    assert "blockers" not in summary
    assert summary["n_total_runs"] == 16
    assert summary["matched_control_count"] == 8
    assert summary["lead_time_tokens"]["entropy"] is not None
    # Artifacts on disk.
    assert (out / "lead_time_by_feature.parquet").exists()
    assert (out / "lead_time_summary.json").exists()
    assert (out / "lead_time_curves.png").exists()


def test_run_lead_time_blockers_when_input_missing(tmp_path: Path):
    cfg = LeadTimeConfig()
    result = run_lead_time(tmp_path / "nope", tmp_path / "out", cfg)
    assert "blockers" in result.summary
    assert any("missing" in b for b in result.summary["blockers"])


def test_run_lead_time_insufficient_pairs_when_no_controls(
    tmp_path: Path,
):
    """Catastrophic runs only, no healthy compressed runs -> the
    pipeline should still write a summary with status flag set."""
    root = tmp_path / "phase0"
    runs_path = root / "final" / "runs.parquet"
    tokens_root = root / "final" / "tokens"

    ids = _looping_token_ids(window=20, repeats=4)
    runs = [
        _runs_record(
            "cat_only",
            "snapkv",
            0.875,
            ["looping"],
            ids,
        )
    ]
    _write_runs(runs_path, runs)
    _write_tokens(
        tokens_root / "press=snapkv" / "ratio=0.8750" / "cat_only.parquet",
        _tokens_rows("cat_only", len(ids)),
    )

    out = tmp_path / "lt"
    cfg = LeadTimeConfig(window_before=20, window_after=10, n_bootstrap=5)
    result = run_lead_time(root, out, cfg)
    assert result.summary["status"] == "insufficient_pairs"
    assert result.summary["control_limited"] is True
    assert result.summary["matched_control_count"] == 0
    assert (out / "lead_time_summary.json").exists()


def test_cli_main_synthetic(synthetic_pair_dataset: Path, tmp_path: Path):
    from scripts.build_lead_time_curves import main

    out = tmp_path / "cli_out"
    rc = main(
        [
            "--input",
            str(synthetic_pair_dataset),
            "--output",
            str(out),
            "--window-before",
            "70",
            "--window-after",
            "20",
            "--n-bootstrap",
            "10",
            "--persistence",
            "5",
        ]
    )
    assert rc == 0
    assert (out / "lead_time_summary.json").exists()
    summary = json.loads((out / "lead_time_summary.json").read_text())
    assert "lead_time_tokens" in summary
