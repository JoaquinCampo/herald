"""Tests for src/herald/predictor_per_run.py.

Behaviors under test:
- per-run aggregation produces max / mean / p95 / mean-top-k / count-above
- mean_top_k handles short runs gracefully (k > run length)
- correlation against run_damage drops null validators
- correlation against gross_harm_final returns AUROC; drops nulls
"""

import polars as pl
import pytest

from herald.predictor_per_run import (
    aggregate_per_run,
    correlate_with_run_damage,
)


def test_aggregate_per_run_emits_required_columns() -> None:
    df = pl.DataFrame(
        {
            "run_id": ["A", "A", "A", "B", "B"],
            "score": [0.1, 0.5, 0.9, 0.2, 0.4],
        }
    )
    out = aggregate_per_run(df, score_col="score", top_k=2, threshold=0.3)
    out = out.sort("run_id")
    assert out.shape == (2, 6)
    assert out.columns == [
        "run_id",
        "max_score",
        "mean_score",
        "p95_score",
        "mean_top_2_score",
        "n_above_0.3",
    ]
    rA = out.filter(pl.col("run_id") == "A").to_dicts()[0]
    assert rA["max_score"] == 0.9
    assert rA["mean_score"] == pytest.approx(0.5)
    # mean_top_2 of [0.1, 0.5, 0.9] = mean(0.5, 0.9) = 0.7
    assert rA["mean_top_2_score"] == pytest.approx(0.7)
    assert rA["n_above_0.3"] == 2  # 0.5 and 0.9


def test_aggregate_per_run_handles_short_runs() -> None:
    df = pl.DataFrame({"run_id": ["A"], "score": [0.42]})
    out = aggregate_per_run(df, top_k=10, threshold=0.5)
    rA = out.to_dicts()[0]
    assert rA["max_score"] == 0.42
    assert rA["mean_top_10_score"] == 0.42
    assert rA["n_above_0.5"] == 0


def test_correlate_with_run_damage_returns_spearman_and_auroc() -> None:
    per_run = pl.DataFrame(
        {
            "run_id": ["A", "B", "C", "D", "E"],
            "max_score": [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    )
    rd = pl.DataFrame(
        {
            "run_id": ["A", "B", "C", "D", "E"],
            "quality_delta": [0.0, 0.1, 0.2, 0.3, 0.4],
            "gross_harm_final": [False, False, True, True, True],
            "has_looping": [False, True, False, True, True],
            "has_non_termination": [False, False, False, False, False],
            "rouge_l_drop": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    out = correlate_with_run_damage(per_run, rd, score_col="max_score")
    # Spearman: max_score and quality_delta are perfectly rank-aligned
    assert out["spearman_quality_delta"] == pytest.approx(1.0)
    # AUROC vs gross_harm_final: 0.1, 0.3 are negative; 0.5, 0.7, 0.9 positive
    # Perfectly separable -> AUROC = 1.0
    assert out["auroc_gross_harm_final"] == pytest.approx(1.0)
    # has_non_termination is all False -> AUROC undefined
    assert out["auroc_has_non_termination"] is None


def test_correlate_with_run_damage_drops_null_validators() -> None:
    per_run = pl.DataFrame(
        {
            "run_id": ["A", "B", "C", "D"],
            "max_score": [0.1, 0.3, 0.5, 0.7],
        }
    )
    rd = pl.DataFrame(
        {
            "run_id": ["A", "B", "C", "D"],
            "quality_delta": [0.0, None, 0.2, None],
            "gross_harm_final": [False, None, True, None],
            "has_looping": [False, False, True, True],
            "has_non_termination": [False] * 4,
            "rouge_l_drop": [0.1, 0.2, 0.3, 0.4],
        }
    )
    out = correlate_with_run_damage(per_run, rd, score_col="max_score")
    # n_used_quality_delta = 2 (only A and C have it)
    assert out["n_used_quality_delta"] == 2
    assert out["n_used_gross_harm_final"] == 2
