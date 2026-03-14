"""Tests for herald.detectors — catastrophe detection and answer parsing."""

import pytest

from herald.detectors import (
    detect_all,
    detect_answer_failure,
    detect_catastrophe_onsets,
    detect_looping,
    detect_looping_onset,
    detect_non_termination,
    parse_gsm8k_answer,
)


# ---------------------------------------------------------------------------
# detect_non_termination
# ---------------------------------------------------------------------------


class TestDetectNonTermination:
    def test_max_tokens(self):
        assert detect_non_termination("max_tokens") is True

    def test_eos(self):
        assert detect_non_termination("eos") is False

    def test_other_reason(self):
        assert detect_non_termination("timeout") is False


# ---------------------------------------------------------------------------
# detect_looping
# ---------------------------------------------------------------------------


class TestDetectLooping:
    def test_no_looping_short_sequence(self):
        assert detect_looping([1, 2, 3]) is False

    def test_no_looping_unique_tokens(self):
        assert detect_looping(list(range(200))) is False

    def test_looping_detected(self):
        # 20-token window repeated 3 times
        pattern = list(range(20))
        token_ids = pattern * 3
        assert detect_looping(token_ids) is True

    def test_looping_with_prefix(self):
        prefix = list(range(100, 150))
        pattern = list(range(20))
        token_ids = prefix + pattern * 3
        assert detect_looping(token_ids) is True

    def test_barely_not_enough_repeats(self):
        pattern = list(range(20))
        token_ids = pattern * 2  # only 2 repeats, need 3
        assert detect_looping(token_ids) is False

    def test_custom_window_and_repeats(self):
        pattern = [1, 2, 3, 4, 5]
        token_ids = pattern * 4
        assert detect_looping(token_ids, window_size=5, min_repeats=4) is True
        assert detect_looping(token_ids, window_size=5, min_repeats=5) is False

    def test_empty_sequence(self):
        assert detect_looping([]) is False

    def test_single_token_repeated(self):
        # Single token repeated many times — detected with window_size=1
        token_ids = [42] * 10
        assert detect_looping(token_ids, window_size=1, min_repeats=3) is True


# ---------------------------------------------------------------------------
# detect_looping_onset
# ---------------------------------------------------------------------------


class TestDetectLoopingOnset:
    def test_no_looping_returns_none(self):
        assert detect_looping_onset(list(range(100))) is None

    def test_onset_at_second_occurrence(self):
        pattern = list(range(20))
        token_ids = pattern * 3
        onset = detect_looping_onset(token_ids)
        # Onset is the start of the second occurrence of the first window
        # that reaches min_repeats
        assert onset is not None
        assert onset == 20  # second repetition starts here

    def test_onset_with_prefix(self):
        prefix = list(range(100, 130))
        pattern = list(range(20))
        token_ids = prefix + pattern * 3
        onset = detect_looping_onset(token_ids)
        assert onset is not None
        assert onset == len(prefix) + 20

    def test_empty_sequence(self):
        assert detect_looping_onset([]) is None

    def test_short_sequence(self):
        assert detect_looping_onset([1, 2, 3]) is None


# ---------------------------------------------------------------------------
# detect_catastrophe_onsets
# ---------------------------------------------------------------------------


class TestDetectCatastropheOnsets:
    def test_looping_onset(self):
        pattern = list(range(20))
        token_ids = pattern * 3
        catastrophes = ["looping"]
        onsets = detect_catastrophe_onsets(token_ids, "eos", catastrophes)
        assert "looping" in onsets
        assert onsets["looping"] == 20

    def test_non_termination_onset(self):
        token_ids = list(range(100))
        catastrophes = ["non_termination"]
        onsets = detect_catastrophe_onsets(token_ids, "max_tokens", catastrophes)
        assert "non_termination" in onsets
        assert onsets["non_termination"] == 99  # last token index

    def test_wrong_answer_has_no_onset(self):
        token_ids = list(range(50))
        catastrophes = ["wrong_answer"]
        onsets = detect_catastrophe_onsets(token_ids, "eos", catastrophes)
        assert "wrong_answer" not in onsets

    def test_no_catastrophes(self):
        onsets = detect_catastrophe_onsets([1, 2, 3], "eos", [])
        assert onsets == {}

    def test_multiple_catastrophes(self):
        pattern = list(range(20))
        token_ids = pattern * 3
        catastrophes = ["looping", "non_termination", "wrong_answer"]
        onsets = detect_catastrophe_onsets(token_ids, "max_tokens", catastrophes)
        assert "looping" in onsets
        assert "non_termination" in onsets
        assert "wrong_answer" not in onsets


# ---------------------------------------------------------------------------
# parse_gsm8k_answer
# ---------------------------------------------------------------------------


class TestParseGsm8kAnswer:
    def test_standard_format(self):
        assert parse_gsm8k_answer("The answer is #### 42") == "42"

    def test_with_commas(self):
        assert parse_gsm8k_answer("#### 1,234") == "1234"

    def test_negative_number(self):
        assert parse_gsm8k_answer("#### -5") == "-5"

    def test_decimal(self):
        assert parse_gsm8k_answer("#### 3.14") == "3.14"

    def test_boxed_format(self):
        assert parse_gsm8k_answer("The answer is \\boxed{42}") == "42"

    def test_no_answer(self):
        assert parse_gsm8k_answer("I don't know the answer") is None

    def test_last_match_wins(self):
        text = "First #### 10 then #### 20"
        assert parse_gsm8k_answer(text) == "20"

    def test_last_boxed_match_wins(self):
        text = "\\boxed{10} and \\boxed{20}"
        assert parse_gsm8k_answer(text) == "20"

    def test_hash_format_preferred_over_boxed(self):
        text = "\\boxed{10} then #### 20"
        assert parse_gsm8k_answer(text) == "20"

    def test_boxed_fallback_when_no_hash(self):
        text = "The answer is \\boxed{42}"
        assert parse_gsm8k_answer(text) == "42"

    def test_empty_string(self):
        assert parse_gsm8k_answer("") is None


# ---------------------------------------------------------------------------
# detect_answer_failure
# ---------------------------------------------------------------------------


class TestDetectAnswerFailure:
    def test_correct_answer(self):
        assert detect_answer_failure("#### 42", "42") is False

    def test_wrong_answer(self):
        assert detect_answer_failure("#### 41", "42") is True

    def test_no_answer_found(self):
        assert detect_answer_failure("no answer here", "42") is True

    def test_float_tolerance(self):
        assert detect_answer_failure("#### 42.0", "42") is False

    def test_comma_in_answer(self):
        assert detect_answer_failure("#### 1,000", "1000") is False

    def test_invalid_number(self):
        assert detect_answer_failure("#### abc", "42") is True


# ---------------------------------------------------------------------------
# detect_all
# ---------------------------------------------------------------------------


class TestDetectAll:
    def test_no_catastrophes(self):
        cats = detect_all("#### 42", [1, 2, 3], "eos", "42")
        assert cats == []

    def test_non_termination_only(self):
        cats = detect_all("#### 42", [1, 2, 3], "max_tokens", "42")
        assert "non_termination" in cats
        assert "looping" not in cats

    def test_looping_detected(self):
        pattern = list(range(20))
        token_ids = pattern * 3
        cats = detect_all("no answer", token_ids, "eos", "42")
        assert "looping" in cats
        assert "wrong_answer" in cats

    def test_all_catastrophes(self):
        pattern = list(range(20))
        token_ids = pattern * 3
        cats = detect_all("no answer", token_ids, "max_tokens", "42")
        assert "non_termination" in cats
        assert "looping" in cats
        assert "wrong_answer" in cats

    def test_correct_answer_no_wrong_answer_flag(self):
        cats = detect_all("#### 42", [1, 2, 3], "eos", "42")
        assert "wrong_answer" not in cats
