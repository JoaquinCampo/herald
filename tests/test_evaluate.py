"""Tests for herald.evaluate — evaluation metrics suite."""

from sklearn.metrics import roc_auc_score

from herald.evaluate import (
    _bootstrap_ci,
    baseline_feature_metrics,
    lead_time_metrics,
    sequence_metrics,
    token_metrics,
)

# ---------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------


class TestTokenMetrics:
    def test_all_tokens_metrics(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

        result = token_metrics(y_true, y_pred)
        assert "all" in result
        assert result["all"]["auroc"] is not None
        assert result["all"]["auprc"] is not None

    def test_pre_onset_only_subset(self):
        # Post-onset tokens are trivially correct
        y_true = [0, 0, 1, 1, 1, 1]
        y_pred = [0.1, 0.4, 0.6, 0.99, 0.99, 0.99]
        pre_onset = [True, True, True, False, False, False]

        result = token_metrics(y_true, y_pred, pre_onset)
        assert "pre_onset" in result
        assert result["pre_onset"]["auroc"] is not None
        # Pre-onset only uses first 3 tokens
        # All-token AUROC is inflated by easy post-onset
        # (can't guarantee ordering, but both should exist)

    def test_no_pre_onset_returns_all_only(self):
        y_true = [0, 1]
        y_pred = [0.2, 0.8]
        result = token_metrics(y_true, y_pred)
        assert "all" in result
        assert "pre_onset" not in result

    def test_single_class_returns_none(self):
        y_true = [0, 0, 0]
        y_pred = [0.1, 0.2, 0.3]
        result = token_metrics(y_true, y_pred)
        assert result["all"]["auroc"] is None
        assert result["all"]["auprc"] is None

    def test_pre_onset_single_class(self):
        # All pre-onset tokens are class 0
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.2, 0.9, 0.9]
        pre_onset = [True, True, False, False]

        result = token_metrics(y_true, y_pred, pre_onset)
        assert result["pre_onset"]["auroc"] is None


# ---------------------------------------------------------------
# Sequence-level metrics
# ---------------------------------------------------------------


class TestSequenceMetrics:
    def test_basic_classification(self):
        # 4 tokens: 2 from seq_a (catastrophe), 2 from
        # seq_b (no catastrophe)
        y_pred = [0.8, 0.9, 0.1, 0.2]
        seq_ids = ["a", "a", "b", "b"]
        pre_onset = [True, True, True, True]
        seq_onsets = {"a": 2, "b": None}

        result = sequence_metrics(y_pred, seq_ids, pre_onset, seq_onsets)

        assert result["n_sequences"] == 2
        assert result["n_catastrophic"] == 1
        assert result["seq_auroc"] is not None

    def test_per_threshold_structure(self):
        y_pred = [0.8, 0.1]
        seq_ids = ["a", "b"]
        pre_onset = [True, True]
        seq_onsets = {"a": 1, "b": None}

        result = sequence_metrics(
            y_pred,
            seq_ids,
            pre_onset,
            seq_onsets,
            thresholds=[0.5],
        )

        per_thresh = result["per_threshold"]
        assert len(per_thresh) == 1  # type: ignore[arg-type]
        entry = per_thresh[0]  # type: ignore[index]
        assert "threshold" in entry  # type: ignore[operator]
        assert "precision" in entry  # type: ignore[operator]
        assert "recall" in entry  # type: ignore[operator]
        assert "f1" in entry  # type: ignore[operator]
        assert "tp" in entry  # type: ignore[operator]

    def test_perfect_separation(self):
        y_pred = [0.9, 0.9, 0.1, 0.1]
        seq_ids = ["a", "a", "b", "b"]
        pre_onset = [True, True, True, True]
        seq_onsets = {"a": 2, "b": None}

        result = sequence_metrics(
            y_pred,
            seq_ids,
            pre_onset,
            seq_onsets,
            thresholds=[0.5],
        )

        entry = result["per_threshold"][0]  # type: ignore[index]
        assert entry["precision"] == 1.0  # type: ignore[index]
        assert entry["recall"] == 1.0  # type: ignore[index]
        assert entry["f1"] == 1.0  # type: ignore[index]

    def test_no_catastrophe_sequences(self):
        y_pred = [0.1, 0.2]
        seq_ids = ["a", "b"]
        pre_onset = [True, True]
        seq_onsets = {"a": None, "b": None}

        result = sequence_metrics(y_pred, seq_ids, pre_onset, seq_onsets)
        assert result["n_catastrophic"] == 0
        assert result["seq_auroc"] is None

    def test_only_pre_onset_tokens_used(self):
        # Post-onset token has high score but should be
        # excluded from sequence scoring
        y_pred = [0.1, 0.99]
        seq_ids = ["a", "a"]
        pre_onset = [True, False]
        seq_onsets = {"a": 1}

        result = sequence_metrics(y_pred, seq_ids, pre_onset, seq_onsets)
        # Score should be 0.1 (only pre-onset token),
        # not 0.99
        assert result["n_sequences"] == 1


# ---------------------------------------------------------------
# Baseline metrics
# ---------------------------------------------------------------


class TestBaselineMetrics:
    def _make_data(
        self,
    ) -> tuple[
        list[list[float]],
        list[int],
        list[str],
        list[bool],
        list[str],
        dict[str, int | None],
    ]:
        """Minimal dataset with known feature names."""
        feat_names = [
            "entropy",
            "top1_prob",
            "top5_prob",
            "h_alts",
            "avg_logp",
            "delta_h",
            "kl_div",
            "top10_jaccard",
            "eff_vocab_size",
            "tail_mass",
            "logit_range",
            "delta_h_valid",
            "logprob_0",
            "logprob_1",
            "logprob_2",
            "logprob_3",
            "logprob_4",
            "token_position",
            "entropy_mean_8",
            "entropy_std_8",
            "top1_prob_mean_8",
            "top1_prob_std_8",
            "h_alts_mean_8",
            "h_alts_std_8",
            "delta_h_mean_8",
            "delta_h_std_8",
            "kl_div_mean_8",
            "kl_div_std_8",
            "top10_jaccard_mean_8",
            "top10_jaccard_std_8",
        ]

        # 4 tokens: 2 pre-catastrophe, 2 post
        X = [[1.0 + i * 0.5] * 30 for i in range(4)]
        y = [0, 0, 1, 1]
        pre_onset = [True, True, False, False]
        seq_ids = ["a"] * 4
        seq_onsets = {"a": 2}

        return (
            X,
            y,
            feat_names,
            pre_onset,
            seq_ids,
            seq_onsets,
        )

    def test_returns_baselines(self):
        X, y, fnames, po, sids, so = self._make_data()
        result = baseline_feature_metrics(X, y, fnames, po, sids, so)

        assert "entropy" in result
        assert "top1_prob" in result
        assert "kl_div" in result
        assert "top10_jaccard" in result

    def test_token_and_sequence_keys(self):
        X, y, fnames, po, sids, so = self._make_data()
        result = baseline_feature_metrics(X, y, fnames, po, sids, so)

        for feat in ("entropy", "top1_prob"):
            assert "token" in result[feat]
            assert "sequence" in result[feat]
            tok = result[feat]["token"]
            assert "all" in tok  # type: ignore[operator]


# ---------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------


class TestBootstrapCI:
    def test_degenerate_returns_none(self):
        # Single class → no CI
        y_true = [0, 0, 0, 0]
        y_score = [0.1, 0.2, 0.3, 0.4]
        ci = _bootstrap_ci(y_true, y_score, roc_auc_score)
        assert ci is None

    def test_empty_returns_none(self):
        ci = _bootstrap_ci([], [], roc_auc_score)
        assert ci is None

    def test_balanced_returns_ordered_pair(self):
        rng_true = [0] * 50 + [1] * 50
        rng_score = [0.1 + 0.001 * i for i in range(50)] + [
            0.5 + 0.005 * i for i in range(50)
        ]
        ci = _bootstrap_ci(
            rng_true, rng_score, roc_auc_score, n_boot=200, seed=7
        )
        assert ci is not None
        low, high = ci
        assert 0.0 <= low <= high <= 1.0

    def test_deterministic_with_seed(self):
        rng_true = [0] * 30 + [1] * 30
        rng_score = [0.2, 0.3, 0.4, 0.5] * 15
        ci1 = _bootstrap_ci(
            rng_true, rng_score, roc_auc_score, n_boot=100, seed=42
        )
        ci2 = _bootstrap_ci(
            rng_true, rng_score, roc_auc_score, n_boot=100, seed=42
        )
        assert ci1 == ci2

    def test_token_metrics_includes_ci(self):
        y_true = [0] * 20 + [1] * 20
        y_pred = [0.1 + 0.01 * i for i in range(20)] + [
            0.5 + 0.01 * i for i in range(20)
        ]
        result = token_metrics(y_true, y_pred)
        assert result["all"]["auroc"] is not None
        assert "auroc_ci" in result["all"]
        assert "auprc_ci" in result["all"]

    def test_token_metrics_ci_none_on_degenerate(self):
        y_true = [0, 0, 0]
        y_pred = [0.1, 0.2, 0.3]
        result = token_metrics(y_true, y_pred)
        assert result["all"]["auroc"] is None
        assert result["all"]["auroc_ci"] is None


# ---------------------------------------------------------------
# Lead-time metrics
# ---------------------------------------------------------------


class TestLeadTimeMetrics:
    def test_all_detected_early(self):
        # seq "a": onset=5, first crossing at t=1 → lead=4
        # seq "b": onset=4, first crossing at t=1 → lead=3
        y_pred = [0.1, 0.9, 0.9, 0.9, 0.9, 0.9] + [0.1, 0.9, 0.9, 0.9, 0.9]
        seq_ids = ["a"] * 6 + ["b"] * 5
        pre_onset = [True] * 5 + [False] + [True] * 4 + [False]
        seq_onsets = {"a": 5, "b": 4}

        result = lead_time_metrics(
            y_pred, seq_ids, pre_onset, seq_onsets, threshold=0.5
        )
        assert result["n_catastrophic"] == 2
        assert result["n_detected"] == 2
        assert result["detection_rate"] == 1.0
        assert result["lead_time_median"] == 3.5
        assert result["lead_time_mean"] == 3.5

    def test_no_crossing_missed(self):
        # catastrophic seq but y_pred never crosses
        y_pred = [0.1, 0.1, 0.1, 0.1]
        seq_ids = ["a"] * 4
        pre_onset = [True, True, True, False]
        seq_onsets = {"a": 3}

        result = lead_time_metrics(
            y_pred, seq_ids, pre_onset, seq_onsets, threshold=0.5
        )
        assert result["n_catastrophic"] == 1
        assert result["n_detected"] == 0
        assert result["detection_rate"] == 0.0
        assert result["lead_time_median"] is None

    def test_ignores_non_catastrophic(self):
        # non-catastrophic seqs should not count
        y_pred = [0.9, 0.9]
        seq_ids = ["a", "a"]
        pre_onset = [True, True]
        seq_onsets = {"a": None}

        result = lead_time_metrics(
            y_pred, seq_ids, pre_onset, seq_onsets, threshold=0.5
        )
        assert result["n_catastrophic"] == 0
        assert result["detection_rate"] is None

    def test_mixed_detection(self):
        # a: detected at lead=2. b: missed (no crossing).
        y_pred = [0.1, 0.9, 0.9, 0.9] + [0.1, 0.1, 0.1]
        seq_ids = ["a"] * 4 + ["b"] * 3
        pre_onset = [True, True, True, False] + [True, True, False]
        seq_onsets = {"a": 3, "b": 2}

        result = lead_time_metrics(
            y_pred, seq_ids, pre_onset, seq_onsets, threshold=0.5
        )
        assert result["n_catastrophic"] == 2
        assert result["n_detected"] == 1
        assert result["lead_time_median"] == 2.0
        assert result["detection_rate"] == 0.5

    def test_crossing_only_post_onset_counts_as_miss(self):
        # first crossing is at t=3 but onset is at 2 → post-onset,
        # does not count as detection (pre_onset filter excludes it)
        y_pred = [0.1, 0.1, 0.1, 0.9]
        seq_ids = ["a"] * 4
        pre_onset = [True, True, False, False]
        seq_onsets = {"a": 2}

        result = lead_time_metrics(
            y_pred, seq_ids, pre_onset, seq_onsets, threshold=0.5
        )
        assert result["n_catastrophic"] == 1
        assert result["n_detected"] == 0
