"""CPU tests for quality-risk materialization helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from augment_quality_risk_rows_v1 import (  # noqa: E402
    V1_BAND_COLUMNS,
    join_v1_divergence,
)

from herald.quality_risk_labels import (  # noqa: E402
    apply_run_labels,
    checkpoint_eligible_lengths,
)


def _runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["a", "b", "c", "d"],
            "prompt_id": ["p1", "p1", "p2", "p3"],
            "baseline_quality_score": [1.0, 0.0, 1.0, np.nan],
            "compressed_quality_score": [0.0, 0.0, np.nan, 0.5],
        }
    )


def test_apply_run_labels_normal_imputed_and_lift() -> None:
    labeled = apply_run_labels(
        _runs(), gap_prompts=["p9"], lift_prompts=["p3"]
    )
    assert set(labeled["run_id"]) == {"a", "b", "c", "d"}
    damage = dict(zip(labeled["run_id"], labeled["damage"], strict=True))
    assert damage == {"a": 1, "b": 0, "c": 1, "d": 0}
    imputed = dict(
        zip(
            labeled["run_id"],
            labeled["imputed_compressed_zero"],
            strict=True,
        )
    )
    assert imputed == {"a": 0, "b": 0, "c": 1, "d": 0}
    lift = dict(
        zip(labeled["run_id"], labeled["lift_reference_zero"], strict=True)
    )
    assert lift == {"a": 0, "b": 0, "c": 0, "d": 1}


def test_apply_run_labels_drops_gap_prompts() -> None:
    labeled = apply_run_labels(
        _runs(), gap_prompts=["p1"], lift_prompts=["p3"]
    )
    assert set(labeled["run_id"]) == {"c", "d"}


def test_checkpoint_eligibility_is_strict() -> None:
    eligible = checkpoint_eligible_lengths(
        pd.Series([5, 8, 9, 200], index=["a", "b", "c", "d"]), checkpoint=8
    )
    assert eligible.to_dict() == {
        "a": False,
        "b": False,
        "c": True,
        "d": True,
    }


def test_join_v1_divergence_fills_missing_with_zero() -> None:
    rows = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "token_pos": [0, 1, 0],
            "damage": [1, 1, 0],
        }
    )
    v1 = pd.DataFrame(
        {
            "run_id": ["r1"],
            "token_pos": [0],
            "pred_causal_xgb_band_rate_0_5": [0.4],
            "pred_causal_xgb_band_rate_5_10": [0.3],
            "pred_causal_xgb_band_rate_10_25": [0.2],
            "pred_causal_xgb_band_rate_25_50": [0.1],
        }
    )
    out = join_v1_divergence(rows, v1, scales=[1.0, 1.0, 1.0, 1.0])
    assert set(V1_BAND_COLUMNS) <= set(out.columns)
    assert np.allclose(
        out["div_band_rate_0_5"].to_numpy(), [0.4, 0.0, 0.0], atol=1e-6
    )
    assert (out["div_band_rate_25_50"].to_numpy() == 0.0).sum() == 2
    assert "div_band_missing" not in out.columns
