"""CPU tests for quality-risk feature-audit helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from materialize_current_state_damage_dev import (  # noqa: E402
    add_causal_history,
)

from herald.quality_risk_features import (  # noqa: E402
    EXPECTED_COLUMNS,
    V1_DIV_COLUMNS,
    check_columns,
    verify_causal_sample,
    verify_divergence_sample,
    verify_engineered_sample,
)


def test_check_columns_accepts_expected() -> None:
    check_columns(list(EXPECTED_COLUMNS))


def test_check_columns_rejects_forbidden() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        check_columns([*EXPECTED_COLUMNS, "quality_delta"])
    with pytest.raises(ValueError, match="forbidden"):
        check_columns([*EXPECTED_COLUMNS, "future_sum_js_5"])
    with pytest.raises(ValueError, match="forbidden"):
        check_columns([*EXPECTED_COLUMNS, "has_looping"])


def test_check_columns_rejects_missing() -> None:
    with pytest.raises(ValueError, match="missing"):
        check_columns([c for c in EXPECTED_COLUMNS if c != "damage"])


def test_verify_causal_sample_matches_recomputation() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=40)
    frame = pd.DataFrame(
        {
            "run_id": ["r"] * 40,
            "token_pos": list(range(40)),
            "entropy": values,
            "top1_prob": rng.uniform(size=40),
            "h_alts": rng.uniform(size=40),
            "delta_h": rng.normal(size=40),
            "kl_div": rng.uniform(size=40),
            "top10_jaccard": rng.uniform(size=40),
        }
    )
    materialized = frame.copy()
    add_causal_history(materialized)
    error = verify_causal_sample(materialized, frame)
    assert error < 1e-5


def test_verify_engineered_sample_matches_recomputation() -> None:
    from herald.quality_risk_engineering import add_engineered_features

    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "run_id": ["r"] * 40,
            "token_pos": list(range(40)),
            "entropy": rng.normal(size=40),
            "top1_prob": rng.uniform(size=40),
            "top5_prob": rng.uniform(0.5, 1.0, size=40),
            "h_alts": rng.uniform(size=40),
            "avg_logp": rng.normal(size=40),
            "delta_h": rng.normal(size=40),
            "delta_h_valid": True,
            "kl_div": rng.uniform(size=40),
            "top10_jaccard": rng.uniform(size=40),
            "eff_vocab_size": rng.uniform(50, 500, size=40),
            "tail_mass": rng.uniform(size=40),
            "logit_range": rng.uniform(1, 10, size=40),
        }
    )
    materialized = add_engineered_features(frame.copy())
    error = verify_engineered_sample(materialized, frame)
    assert error == 0.0


def test_verify_divergence_sample_join_and_fill() -> None:
    materialized = pd.DataFrame(
        {
            "token_pos": [0, 1, 2],
            "div_band_rate_0_5": [0.4, 0.0, 0.5],
            "div_band_rate_5_10": [0.3, 0.0, 0.4],
            "div_band_rate_10_25": [0.2, 0.0, 0.3],
            "div_band_rate_25_50": [0.1, 0.0, 0.2],
        }
    )
    v1 = pd.DataFrame(
        {
            "token_pos": [0, 1, 2],
            "pred_causal_xgb_band_rate_0_5": [0.4, np.nan, 0.5],
            "pred_causal_xgb_band_rate_5_10": [0.3, np.nan, 0.4],
            "pred_causal_xgb_band_rate_10_25": [0.2, np.nan, 0.3],
            "pred_causal_xgb_band_rate_25_50": [0.1, np.nan, 0.2],
        }
    )
    error = verify_divergence_sample(materialized, v1, [1.0, 1.0, 1.0, 1.0])
    assert error == 0.0
    assert set(V1_DIV_COLUMNS) == {
        "div_band_rate_0_5",
        "div_band_rate_5_10",
        "div_band_rate_10_25",
        "div_band_rate_25_50",
    }
