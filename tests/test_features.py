"""Tests for herald.features — feature engineering."""

from herald.config import TokenSignals
from herald.features import (
    ROLLING_TARGETS,
    ROLLING_WINDOW,
    ROLLING_WINDOWS,
    add_rolling_features,
    flatten_signals,
)


def _make_signals(n: int = 5) -> list[TokenSignals]:
    """Create a sequence of n TokenSignals for testing."""
    sigs = []
    for t in range(n):
        sigs.append(
            TokenSignals(
                entropy=1.0 + t * 0.1,
                top1_prob=0.5 - t * 0.02,
                top5_prob=0.9,
                top5_logprobs=[-0.5, -1.0, -1.5, -2.0, -2.5],
                h_alts=0.8 + t * 0.05,
                avg_logp=-5.0,
                delta_h=float("nan") if t == 0 else 0.1,
                delta_h_valid=t > 0,
                kl_div=float("nan") if t == 0 else 0.05,
                top10_jaccard=float("nan") if t == 0 else 0.7,
                eff_vocab_size=2.72,
                tail_mass=0.05,
                logit_range=15.0,
            )
        )
    return sigs


# -------------------------------------------------------------------
# flatten_signals
# -------------------------------------------------------------------


class TestFlattenSignals:
    def test_basic(self):
        sigs = _make_signals(3)
        rows = flatten_signals(sigs)
        assert len(rows) == 3
        # Each row should have same keys
        assert rows[0].keys() == rows[1].keys() == rows[2].keys()

    def test_logprob_expansion(self):
        sigs = _make_signals(1)
        rows = flatten_signals(sigs)
        row = rows[0]
        assert row["logprob_0"] == -0.5
        assert row["logprob_1"] == -1.0
        assert row["logprob_4"] == -2.5

    def test_nan_replacement_first_token(self):
        sigs = _make_signals(2)
        rows = flatten_signals(sigs)
        # First token: structural NaN replaced with 0.0
        assert rows[0]["delta_h"] == 0.0
        assert rows[0]["kl_div"] == 0.0
        assert rows[0]["top10_jaccard"] == 0.0
        # Second token: real values preserved
        assert rows[1]["delta_h"] == 0.1
        assert rows[1]["kl_div"] == 0.05

    def test_token_position_normalized(self):
        sigs = _make_signals(5)
        rows = flatten_signals(sigs, max_new_tokens=100)
        for t in range(5):
            assert rows[t]["token_position"] == float(t) / 100.0

    def test_token_position_default(self):
        sigs = _make_signals(5)
        rows = flatten_signals(sigs)  # default max_new_tokens=512
        assert rows[0]["token_position"] == 0.0
        assert abs(rows[1]["token_position"] - 1.0 / 512) < 1e-9

    def test_delta_h_valid_is_float(self):
        sigs = _make_signals(2)
        rows = flatten_signals(sigs)
        assert rows[0]["delta_h_valid"] == 0.0
        assert rows[1]["delta_h_valid"] == 1.0

    def test_short_logprobs_padded(self):
        sig = TokenSignals(
            entropy=1.0,
            top1_prob=0.5,
            top5_prob=0.9,
            top5_logprobs=[-0.5, -1.0],  # only 2
        )
        rows = flatten_signals([sig])
        assert rows[0]["logprob_0"] == -0.5
        assert rows[0]["logprob_1"] == -1.0
        assert rows[0]["logprob_2"] == 0.0
        assert rows[0]["logprob_3"] == 0.0
        assert rows[0]["logprob_4"] == 0.0

    def test_raw_feature_count(self):
        sigs = _make_signals(1)
        rows = flatten_signals(sigs)
        # 11 scalar + 1 delta_h_valid + 5 logprobs + 1 position
        assert len(rows[0]) == 18


# -------------------------------------------------------------------
# add_rolling_features
# -------------------------------------------------------------------


class TestAddRollingFeatures:
    def test_adds_rolling_columns(self):
        sigs = _make_signals(10)
        rows = flatten_signals(sigs)
        rows = add_rolling_features(rows)
        for name in ROLLING_TARGETS:
            for w in ROLLING_WINDOWS:
                assert f"{name}_mean_{w}" in rows[0]
                assert f"{name}_std_{w}" in rows[0]

    def test_rolling_count(self):
        sigs = _make_signals(10)
        rows = flatten_signals(sigs)
        n_before = len(rows[0])
        rows = add_rolling_features(rows)
        n_after = len(rows[0])
        assert n_after - n_before == len(ROLLING_TARGETS) * 2 * len(
            ROLLING_WINDOWS
        )

    def test_single_token_std_is_zero(self):
        sigs = _make_signals(1)
        rows = flatten_signals(sigs)
        rows = add_rolling_features(rows)
        for name in ROLLING_TARGETS:
            for w in ROLLING_WINDOWS:
                assert rows[0][f"{name}_std_{w}"] == 0.0

    def test_mean_correctness_short_window(self):
        sigs = _make_signals(3)
        rows = flatten_signals(sigs)
        rows = add_rolling_features(rows)
        # entropy values: 1.0, 1.1, 1.2
        # mean at t=2 (window=[1.0, 1.1, 1.2]) = 1.1
        mean_key = f"entropy_mean_{ROLLING_WINDOW}"
        assert abs(rows[2][mean_key] - 1.1) < 1e-6

    def test_long_window_captures_more(self):
        # With 40 tokens and window 32, long window average
        # differs from short window average (drift)
        sigs = _make_signals(40)
        rows = flatten_signals(sigs)
        rows = add_rolling_features(rows)
        short = rows[-1]["entropy_mean_8"]
        long = rows[-1]["entropy_mean_32"]
        assert short != long

    def test_total_feature_count_is_42(self):
        sigs = _make_signals(10)
        rows = flatten_signals(sigs)
        rows = add_rolling_features(rows)
        # 17 raw + 1 delta_h_valid isn't separate, it's raw +
        # 5 logprobs + 1 position = 18 raw; 6 targets x 2 stats
        # x 2 windows = 24 rolling; total = 42
        assert len(rows[0]) == 42
