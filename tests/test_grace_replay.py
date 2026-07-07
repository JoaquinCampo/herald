"""Tests for the grace-window replay machinery.

Semantics are the funded AIMD feasibility spec (scratchpad
aimd_feasibility.py), ported verbatim: sequential attempts at every
recorded grid point in ascending s, commit at the first alarm score
<= theta, overhead = k * reverts / ref_len.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from herald.grace_replay import (
    ReplayMatrices,
    bootstrap_group_ci,
    calibrate_theta,
    gated_replay,
    oracle_replay,
    replay,
    replay_matrices,
)
from herald.switch_baselines import leave_one_compressor_splits

PARQUET = Path("results/predictor/switch_dataset_attn.parquet")
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")


def _load_real_train_rows() -> list[dict[str, object]]:
    df = pd.read_parquet(PARQUET)
    df = df[df["task"] == "ifeval"].reset_index(drop=True)
    mask = df["compressor"].isin(COMPRESSORS)
    rows = df[mask].to_dict("records")
    splits = leave_one_compressor_splits(
        rows,
        compressors=("expected_attention", "knorm", "streaming_llm"),
        seed=0,
        test_group_fraction=0.25,
    )
    split = splits[0]
    for candidate in splits:
        if candidate.heldout_compressor == "expected_attention":
            split = candidate
            break
    test_prompt_ids = {
        str(row["prompt_id"])
        for row in split.test
        if str(row["compressor"]) == split.heldout_compressor
    }
    return [
        row
        for row in rows
        if str(row["compressor"]) == split.heldout_compressor
        and str(row["prompt_id"]) not in test_prompt_ids
    ]


def make_rows() -> tuple[list[dict[str, object]], list[float]]:
    """Two groups: prompt a (3 grid points), prompt b (2 grid points).

    Rows arrive deliberately out of s-order to exercise sorting.
    """
    rows = [
        {"prompt_id": "a", "ratio": 0.5, "s": 16, "dq": 0.0, "ref_len": 64},
        {"prompt_id": "a", "ratio": 0.5, "s": 0, "dq": 1.0, "ref_len": 64},
        {"prompt_id": "a", "ratio": 0.5, "s": 32, "dq": 0.5, "ref_len": 64},
        {"prompt_id": "b", "ratio": 0.5, "s": 0, "dq": 1.0, "ref_len": 32},
        {"prompt_id": "b", "ratio": 0.5, "s": 16, "dq": 1.0, "ref_len": 32},
    ]
    # scores aligned to rows: group a: s0 -> 0.9, s16 -> 0.2, s32 -> 0.1
    # group b: s0 -> 0.8, s16 -> 0.7
    scores = [0.2, 0.9, 0.1, 0.8, 0.7]
    return rows, scores


class TestReplayMatrices:
    def test_shapes_and_ordering(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        assert isinstance(mats, ReplayMatrices)
        assert mats.pred.shape == (2, 3)
        i_a = mats.keys.index(("a", 0.5))
        i_b = mats.keys.index(("b", 0.5))
        assert mats.valid[i_a] == 3
        assert mats.valid[i_b] == 2
        # ascending s within the group
        np.testing.assert_allclose(mats.pred[i_a], [0.9, 0.2, 0.1])
        np.testing.assert_allclose(mats.dq[i_a], [1.0, 0.0, 0.5])
        # padding is inf for pred (never commits) and 0 elsewhere
        assert mats.pred[i_b, 2] == np.inf
        assert mats.dq[i_b, 2] == 0.0
        assert mats.ref_len[i_a] == 64.0
        assert mats.prompt_ids[i_a] == "a"

    def test_savings_formula(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        i_a = mats.keys.index(("a", 0.5))
        np.testing.assert_allclose(
            mats.sav[i_a], [1.0, 1.0 - 16 / 64, 1.0 - 32 / 64]
        )

    def test_equal_s_preserves_row_order(self) -> None:
        rows = [
            {
                "prompt_id": "a",
                "ratio": 0.5,
                "s": 16,
                "dq": 0.3,
                "ref_len": 64,
            },
            {
                "prompt_id": "a",
                "ratio": 0.5,
                "s": 16,
                "dq": 0.1,
                "ref_len": 64,
            },
        ]
        mats = replay_matrices(rows, [0.9, 0.2])
        np.testing.assert_allclose(mats.pred[0], [0.9, 0.2])
        np.testing.assert_allclose(mats.dq[0], [0.3, 0.1])


class TestReplay:
    def test_commit_at_first_hit(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        out = replay(mats, theta=0.2, k=2)
        i_a = mats.keys.index(("a", 0.5))
        i_b = mats.keys.index(("b", 0.5))
        # group a commits at s=16 (score 0.2 <= 0.2), one revert (s=0)
        assert out.savings[i_a] == pytest.approx(1.0 - 16 / 64)
        assert out.cost[i_a] == pytest.approx(0.0)
        assert out.commit_index[i_a] == 1
        assert out.overhead[i_a] == pytest.approx(2 * 1 / 64)
        # group b never commits: all attempts reverted
        assert out.savings[i_b] == 0.0
        assert out.cost[i_b] == 0.0
        assert out.commit_index[i_b] == -1
        assert out.overhead[i_b] == pytest.approx(2 * 2 / 32)

    def test_theta_below_all_never_commits(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        out = replay(mats, theta=-1.0, k=2)
        assert (out.savings == 0.0).all()
        assert (out.commit_index == -1).all()


class TestGatedReplay:
    def test_minus_inf_gate_matches_ungated_replay(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        gate_scores = np.full_like(mats.pred, 0.0)
        ungated = replay(mats, theta=0.2, k=2)
        gated = gated_replay(
            mats,
            scores=mats.pred,
            gate_scores=gate_scores,
            g_tau=float("-inf"),
            theta=0.2,
            k=2,
        )
        np.testing.assert_allclose(gated.savings, ungated.savings)
        np.testing.assert_allclose(gated.cost, ungated.cost)
        np.testing.assert_allclose(gated.overhead, ungated.overhead)
        np.testing.assert_array_equal(
            gated.commit_index, ungated.commit_index
        )
        np.testing.assert_array_equal(gated.n_attempts, [2, 2])

    def test_skipped_hit_delays_commit_and_counts_kept_attempts(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        gate_scores = np.full_like(mats.pred, -1.0)
        i_a = mats.keys.index(("a", 0.5))
        gate_scores[i_a, 0] = 1.0
        gate_scores[i_a, 2] = 1.0
        out = gated_replay(
            mats,
            scores=mats.pred,
            gate_scores=gate_scores,
            g_tau=0.0,
            theta=0.2,
            k=2,
        )
        assert out.commit_index[i_a] == 2
        assert out.savings[i_a] == pytest.approx(1.0 - 32 / 64)
        assert out.cost[i_a] == pytest.approx(0.5)
        assert out.overhead[i_a] == pytest.approx(2 * 1 / 64)
        assert out.n_attempts[i_a] == 2

    def test_no_commit_overhead_counts_only_kept_attempts(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        gate_scores = np.full_like(mats.pred, -1.0)
        i_b = mats.keys.index(("b", 0.5))
        gate_scores[i_b, 1] = 1.0
        out = gated_replay(
            mats,
            scores=mats.pred,
            gate_scores=gate_scores,
            g_tau=0.0,
            theta=0.2,
            k=2,
        )
        assert out.commit_index[i_b] == -1
        assert out.savings[i_b] == 0.0
        assert out.cost[i_b] == 0.0
        assert out.overhead[i_b] == pytest.approx(2 * 1 / 32)
        assert out.n_attempts[i_b] == 1

    def test_nan_gate_scores_are_treated_as_kept(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        gate_scores = np.full_like(mats.pred, np.nan)
        out = gated_replay(
            mats,
            scores=mats.pred,
            gate_scores=gate_scores,
            g_tau=0.0,
            theta=0.2,
            k=2,
        )
        ungated = replay(mats, theta=0.2, k=2)
        np.testing.assert_allclose(out.savings, ungated.savings)
        np.testing.assert_allclose(out.cost, ungated.cost)
        assert np.array_equal(out.commit_index, ungated.commit_index)

    def test_minus_inf_matches_ungated_on_real_train_matrices(self) -> None:
        rows = _load_real_train_rows()
        score_seed = np.arange(len(rows), dtype=np.float64)
        scores = ((score_seed * 0.001) % 1.0).tolist()
        mats = replay_matrices(rows, scores)
        ungated = replay(mats, theta=0.2, k=2)
        gated = gated_replay(
            mats,
            scores=mats.pred,
            gate_scores=np.full_like(mats.pred, 0.0),
            g_tau=float("-inf"),
            theta=0.2,
            k=2,
        )
        np.testing.assert_allclose(gated.savings, ungated.savings)
        np.testing.assert_allclose(gated.cost, ungated.cost)
        np.testing.assert_allclose(gated.overhead, ungated.overhead)
        np.testing.assert_array_equal(
            gated.commit_index,
            ungated.commit_index,
        )
        assert float(gated.n_attempts.sum()) == float(
            ungated.n_attempts.sum()
        )


class TestOracleReplay:
    def test_first_undamaged_point(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        savings, overhead = oracle_replay(mats, k=2)
        i_a = mats.keys.index(("a", 0.5))
        i_b = mats.keys.index(("b", 0.5))
        # a: first dq <= 0 at index 1 (s=16)
        assert savings[i_a] == pytest.approx(1.0 - 16 / 64)
        assert overhead[i_a] == pytest.approx(2 * 1 / 64)
        # b: no undamaged point
        assert savings[i_b] == 0.0
        assert overhead[i_b] == pytest.approx(2 * 2 / 32)


class TestCalibrateTheta:
    def test_point_maximizes_savings_within_budget(self) -> None:
        rows, scores = make_rows()
        mats = replay_matrices(rows, scores)
        # eps = 0.3: committing a@s16 (dq 0) and b@s16 (dq 1) would mean
        # mean cost 0.5 > eps; theta 0.2 commits only a@s16, cost 0.
        theta = calibrate_theta(mats, scores, epsilon=0.3, k=2)
        out = replay(mats, theta=theta, k=2)
        assert float(out.cost.mean()) <= 0.3
        assert out.commit_index[mats.keys.index(("a", 0.5))] >= 0
        assert out.commit_index[mats.keys.index(("b", 0.5))] == -1

    def test_impossible_budget_falls_back_to_never(self) -> None:
        rows = [
            {
                "prompt_id": "a",
                "ratio": 0.5,
                "s": 0,
                "dq": 1.0,
                "ref_len": 64,
            },
            {
                "prompt_id": "b",
                "ratio": 0.5,
                "s": 0,
                "dq": 1.0,
                "ref_len": 64,
            },
        ]
        scores = [0.1, 0.2]
        mats = replay_matrices(rows, scores)
        theta = calibrate_theta(mats, scores, epsilon=0.0, k=2)
        out = replay(mats, theta=theta, k=2)
        assert (out.commit_index == -1).all()


class TestBootstrapGroupCi:
    def test_deterministic_and_contains_mean(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(0.5, 0.1, size=40)
        prompts = [f"p{i % 20}" for i in range(40)]
        lo1, hi1 = bootstrap_group_ci(values, prompts, seed=0)
        lo2, hi2 = bootstrap_group_ci(values, prompts, seed=0)
        assert (lo1, hi1) == (lo2, hi2)
        assert lo1 < float(values.mean()) < hi1

    def test_single_prompt_degenerates_to_mean(self) -> None:
        values = np.array([0.2, 0.4])
        lo, hi = bootstrap_group_ci(values, ["p", "p"], seed=0)
        assert lo == pytest.approx(0.3)
        assert hi == pytest.approx(0.3)

    def test_clusters_resample_prompts_jointly(self) -> None:
        # Two prompts with internally identical values: every resample
        # mean is a mixture of 0 and 1, so the CI spans most of [0, 1]
        # rather than collapsing.
        values = np.array([0.0, 0.0, 1.0, 1.0])
        prompts = ["p0", "p0", "p1", "p1"]
        lo, hi = bootstrap_group_ci(values, prompts, seed=0)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)
