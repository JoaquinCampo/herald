"""CPU tests for free causal feature engineering (protocol section 5)."""

import numpy as np
import pandas as pd

from herald.quality_risk_engineering import (
    ENGINEERED_NAMES,
    add_engineered_features,
    trailing_slope,
)


def _run(values: dict, n: int = 20) -> pd.DataFrame:
    frame = pd.DataFrame({"run_id": ["r"] * n, "token_pos": list(range(n))})
    defaults = {
        "entropy": 1.0,
        "top1_prob": 0.4,
        "top5_prob": 0.8,
        "h_alts": 0.5,
        "avg_logp": -1.0,
        "delta_h": 0.0,
        "delta_h_valid": True,
        "kl_div": 0.1,
        "top10_jaccard": 0.9,
        "eff_vocab_size": 100.0,
        "tail_mass": 0.1,
        "logit_range": 5.0,
    }
    for key, default in defaults.items():
        frame[key] = values.get(key, np.full(n, default))
    for key, series in values.items():
        if key not in defaults:
            frame[key] = np.asarray(series, dtype=np.float64)
    return frame


def test_trailing_slope_ramp_and_flat() -> None:
    ramp = np.arange(10, dtype=np.float64)
    slope = trailing_slope(ramp, window=5)
    assert slope[-1] == 1.0
    assert slope[0] != slope[0]  # NaN with a single point
    flat = np.full(10, 3.0)
    assert trailing_slope(flat, window=5)[-1] == 0.0


def test_startdiff_step() -> None:
    frame = _run({"entropy": [1.0] * 10 + [3.0] * 10})
    out = add_engineered_features(frame)
    assert out["entropy_startdiff_8"].iloc[9] == 0.0
    assert out["entropy_startdiff_8"].iloc[10] == 2.0


def test_causality_prefix_stable() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=30)
    full = _run({"entropy": values}, n=30)
    prefix = _run({"entropy": values[:12]}, n=12)
    full_out = add_engineered_features(full)
    prefix_out = add_engineered_features(prefix)
    for name in ENGINEERED_NAMES:
        left = prefix_out[name].to_numpy(dtype=np.float64)
        right = full_out[name].to_numpy(dtype=np.float64)[:12]
        assert np.allclose(left, right, equal_nan=True), name


def test_focus_ratio_and_windows_present() -> None:
    frame = _run(
        {
            "entropy": np.linspace(1, 2, 20),
            "top1_prob": np.full(20, 0.4),
            "top5_prob": np.full(20, 0.8),
            "tail_mass": np.full(20, 0.1),
            "eff_vocab_size": np.full(20, 100.0),
        }
    )
    out = add_engineered_features(frame)
    assert set(ENGINEERED_NAMES) <= set(out.columns)
    assert np.allclose(out["focus_ratio"].to_numpy(), 0.5)
    assert np.isfinite(
        out["entropy_causal_mean_64"].to_numpy(dtype=np.float64)
    ).all()
