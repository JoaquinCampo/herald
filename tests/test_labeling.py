"""Tests for herald.labeling — horizon-based hazard labels."""

from herald.labeling import (
    TRAINABLE_CATASTROPHES,
    create_horizon_labels,
    earliest_onset,
)

# -------------------------------------------------------------------
# earliest_onset
# -------------------------------------------------------------------


class TestEarliestOnset:
    def test_empty(self):
        assert earliest_onset({}, []) is None

    def test_looping_only(self):
        assert earliest_onset({"looping": 50}, ["looping"]) == 50

    def test_non_termination_uses_proxy(self):
        # True onset is at last token (511), but proxy should be
        # 0.75 * 512 = 384
        result = earliest_onset(
            {},
            ["non_termination"],
            max_new_tokens=512,
            nt_onset_frac=0.75,
        )
        assert result == 384

    def test_non_termination_proxy_clamped_to_n_tokens(self):
        # proxy = 0.75 * 512 = 384, but n_tokens=100 → clamp to 99
        result = earliest_onset(
            {},
            ["non_termination"],
            max_new_tokens=512,
            nt_onset_frac=0.75,
            n_tokens=100,
        )
        assert result == 99

    def test_filters_wrong_answer(self):
        assert earliest_onset({"wrong_answer": 100}, []) is None

    def test_looping_earlier_than_nt_proxy(self):
        # looping at 50, NT proxy at 384 → picks 50
        result = earliest_onset(
            {"looping": 50},
            ["looping", "non_termination"],
            max_new_tokens=512,
        )
        assert result == 50

    def test_nt_proxy_earlier_than_looping(self):
        # looping at 400, NT proxy at 384 → picks 384
        result = earliest_onset(
            {"looping": 400},
            ["looping", "non_termination"],
            max_new_tokens=512,
        )
        assert result == 384

    def test_wrong_answer_ignored_in_mix(self):
        result = earliest_onset(
            {"wrong_answer": 10, "looping": 50},
            ["wrong_answer", "looping"],
        )
        assert result == 50

    def test_trainable_catastrophes_set(self):
        assert "looping" in TRAINABLE_CATASTROPHES
        assert "non_termination" in TRAINABLE_CATASTROPHES
        assert "wrong_answer" not in TRAINABLE_CATASTROPHES


# -------------------------------------------------------------------
# create_horizon_labels
# -------------------------------------------------------------------


class TestCreateHorizonLabels:
    def test_no_catastrophe_all_zeros(self):
        labels = create_horizon_labels(100, None, horizon=10)
        assert len(labels) == 100
        assert all(v == 0 for v in labels)

    def test_empty_sequence(self):
        assert create_horizon_labels(0, None, horizon=10) == []

    def test_catastrophe_at_end(self):
        labels = create_horizon_labels(100, 99, horizon=10)
        assert labels[99] == 1
        for t in range(89, 99):
            assert labels[t] == 1, f"token {t}"
        assert labels[88] == 0

    def test_onset_at_start(self):
        labels = create_horizon_labels(10, 0, horizon=5)
        assert all(v == 1 for v in labels)

    def test_horizon_1(self):
        labels = create_horizon_labels(50, 30, horizon=1)
        assert labels[28] == 0
        assert labels[29] == 1
        assert labels[30] == 1
        assert labels[31] == 1

    def test_horizon_larger_than_onset(self):
        labels = create_horizon_labels(50, 5, horizon=20)
        for t in range(5):
            assert labels[t] == 1, f"token {t}"
        assert labels[5] == 1

    def test_censored_all_zeros(self):
        labels = create_horizon_labels(200, None, horizon=50)
        assert sum(labels) == 0

    def test_label_count(self):
        labels = create_horizon_labels(100, 50, horizon=10)
        assert sum(labels) == 60
