"""Tests for deterministic post-hoc task scorers (Qasper + IFEval).

These cover the public surface of `herald.metrics.task_scoring`:

  - Qasper: SQuAD-style normalization, exact match, token-level F1,
    multiple-gold aggregation, threshold-based binary correctness.
  - IFEval: per-instruction checks for the supported instruction
    families, explicit unsupported reporting (no silent True), and
    threshold-based binary correctness.
"""

import pytest

from herald.metrics import task_scoring as ts

# -----------------------------
# Qasper: normalization and EM
# -----------------------------


class TestQasperNormalize:
    def test_lowercases(self) -> None:
        assert ts.qasper_normalize("HELLO World") == "hello world"

    def test_strips_articles(self) -> None:
        assert ts.qasper_normalize("the cat sat on a mat") == "cat sat on mat"

    def test_strips_punctuation(self) -> None:
        assert ts.qasper_normalize("hello, world!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert (
            ts.qasper_normalize("  many   spaces\there  ")
            == "many spaces here"
        )

    def test_handles_empty(self) -> None:
        assert ts.qasper_normalize("") == ""
        assert ts.qasper_normalize("   ") == ""


class TestQasperEM:
    def test_exact_normalized_match(self) -> None:
        assert ts.qasper_em("The Cat", "the cat") is True

    def test_articles_dont_matter(self) -> None:
        assert ts.qasper_em("a cat", "the cat") is True

    def test_different_strings_not_em(self) -> None:
        assert ts.qasper_em("dog", "cat") is False

    def test_partial_overlap_not_em(self) -> None:
        assert ts.qasper_em("the cat sat", "the cat") is False


class TestQasperF1:
    def test_perfect_match_is_one(self) -> None:
        assert ts.qasper_token_f1("the cat sat", "the cat sat") == 1.0

    def test_no_overlap_is_zero(self) -> None:
        assert ts.qasper_token_f1("dog ran fast", "cat sat slow") == 0.0

    def test_partial_overlap_is_between(self) -> None:
        # pred: "the cat sat" → tokens: cat sat
        # gold: "the dog sat" → tokens: dog sat
        # common: sat (1). p = 1/2, r = 1/2, f1 = 0.5
        f1 = ts.qasper_token_f1("the cat sat", "the dog sat")
        assert 0.49 < f1 < 0.51

    def test_empty_prediction_is_zero(self) -> None:
        assert ts.qasper_token_f1("", "the cat") == 0.0

    def test_empty_gold_is_zero(self) -> None:
        assert ts.qasper_token_f1("the cat", "") == 0.0


# ---------------------------------------
# Qasper: aggregation over multiple golds
# ---------------------------------------


class TestQasperScore:
    def test_takes_max_over_multiple_golds(self) -> None:
        # Bad pred matches one gold perfectly; the other is unrelated.
        out = ts.qasper_score(
            generated_text="apple",
            gold_answers=["apple", "the orange"],
            threshold=0.5,
        )
        assert out["qasper_em"] == 1.0
        assert out["qasper_f1"] == 1.0
        assert out["qasper_correct"] is True

    def test_partial_match_above_threshold(self) -> None:
        out = ts.qasper_score(
            generated_text="the cat sat on the mat",
            gold_answers=["a cat sat on a mat"],
            threshold=0.5,
        )
        # All four content tokens match → F1 = 1.0
        assert out["qasper_f1"] == 1.0
        assert out["qasper_correct"] is True

    def test_partial_match_below_threshold(self) -> None:
        out = ts.qasper_score(
            generated_text="completely unrelated answer here",
            gold_answers=["the actual answer"],
            threshold=0.5,
        )
        assert out["qasper_f1"] < 0.5
        assert out["qasper_correct"] is False

    def test_no_gold_answers_undefined(self) -> None:
        out = ts.qasper_score(
            generated_text="anything",
            gold_answers=[],
            threshold=0.5,
        )
        assert out["qasper_correct"] is None
        assert out["qasper_f1"] is None
        assert out["qasper_em"] is None

    def test_empty_generation_is_zero(self) -> None:
        out = ts.qasper_score(
            generated_text="",
            gold_answers=["the answer"],
            threshold=0.5,
        )
        assert out["qasper_f1"] == 0.0
        assert out["qasper_correct"] is False

    def test_threshold_is_configurable(self) -> None:
        # F1 = 0.5 (computed above). Threshold 0.6 should fail.
        out = ts.qasper_score(
            generated_text="the cat sat",
            gold_answers=["the dog sat"],
            threshold=0.6,
        )
        assert out["qasper_correct"] is False
        out2 = ts.qasper_score(
            generated_text="the cat sat",
            gold_answers=["the dog sat"],
            threshold=0.4,
        )
        assert out2["qasper_correct"] is True


# -----------------------------
# IFEval: supported instructions
# -----------------------------


class TestIFEvalLengthConstraints:
    def test_number_words_at_least_satisfied(self) -> None:
        # 5 words >= 3
        out = ts.ifeval_score(
            generated_text="one two three four five",
            instruction_id_list=["length_constraints:number_words"],
            kwargs_list=[{"num_words": 3, "relation": "at least"}],
        )
        assert out["ifeval_num_constraints"] == 1
        assert out["ifeval_num_satisfied"] == 1
        assert out["ifeval_num_unsupported"] == 0
        assert out["ifeval_correct"] is True
        assert out["ifeval_score"] == 1.0

    def test_number_words_at_least_violated(self) -> None:
        out = ts.ifeval_score(
            generated_text="one two",
            instruction_id_list=["length_constraints:number_words"],
            kwargs_list=[{"num_words": 5, "relation": "at least"}],
        )
        assert out["ifeval_num_satisfied"] == 0
        assert out["ifeval_correct"] is False

    def test_number_words_at_most_satisfied(self) -> None:
        out = ts.ifeval_score(
            generated_text="one two three",
            instruction_id_list=["length_constraints:number_words"],
            kwargs_list=[{"num_words": 5, "relation": "less than"}],
        )
        assert out["ifeval_correct"] is True


class TestIFEvalPunctuation:
    def test_no_comma_satisfied(self) -> None:
        out = ts.ifeval_score(
            generated_text="hello world how are you today",
            instruction_id_list=["punctuation:no_comma"],
            kwargs_list=[{}],
        )
        assert out["ifeval_correct"] is True

    def test_no_comma_violated(self) -> None:
        out = ts.ifeval_score(
            generated_text="hello, world",
            instruction_id_list=["punctuation:no_comma"],
            kwargs_list=[{}],
        )
        assert out["ifeval_correct"] is False


class TestIFEvalKeywords:
    def test_existence_satisfied(self) -> None:
        out = ts.ifeval_score(
            generated_text="The cat ate the apple under the tree.",
            instruction_id_list=["keywords:existence"],
            kwargs_list=[{"keywords": ["cat", "apple"]}],
        )
        assert out["ifeval_correct"] is True

    def test_existence_violated(self) -> None:
        out = ts.ifeval_score(
            generated_text="A dog ran around.",
            instruction_id_list=["keywords:existence"],
            kwargs_list=[{"keywords": ["cat", "apple"]}],
        )
        assert out["ifeval_correct"] is False

    def test_forbidden_satisfied(self) -> None:
        out = ts.ifeval_score(
            generated_text="A neutral statement.",
            instruction_id_list=["keywords:forbidden_words"],
            kwargs_list=[{"forbidden_words": ["cat", "apple"]}],
        )
        assert out["ifeval_correct"] is True

    def test_forbidden_violated(self) -> None:
        out = ts.ifeval_score(
            generated_text="The cat sat there.",
            instruction_id_list=["keywords:forbidden_words"],
            kwargs_list=[{"forbidden_words": ["cat"]}],
        )
        assert out["ifeval_correct"] is False


class TestIFEvalChangeCase:
    def test_lowercase_satisfied(self) -> None:
        out = ts.ifeval_score(
            generated_text="all lowercase here",
            instruction_id_list=["change_case:english_lowercase"],
            kwargs_list=[{}],
        )
        assert out["ifeval_correct"] is True

    def test_lowercase_violated(self) -> None:
        out = ts.ifeval_score(
            generated_text="Has Some Caps",
            instruction_id_list=["change_case:english_lowercase"],
            kwargs_list=[{}],
        )
        assert out["ifeval_correct"] is False


class TestIFEvalUnsupported:
    def test_unsupported_makes_correct_null(self) -> None:
        # 'language:response_language' is real but not implemented in our
        # partial scorer. The scorer must NOT silently mark it satisfied.
        out = ts.ifeval_score(
            generated_text="anything goes",
            instruction_id_list=["language:response_language"],
            kwargs_list=[{"language": "fr"}],
        )
        assert out["ifeval_num_constraints"] == 1
        assert out["ifeval_num_supported"] == 0
        assert out["ifeval_num_unsupported"] == 1
        assert out["ifeval_correct"] is None  # undefined
        assert "language:response_language" in out["ifeval_unsupported_types"]

    def test_mix_of_supported_and_unsupported(self) -> None:
        # Two constraints: one supported and satisfied, one unsupported.
        # Run-level correct must be None because we cannot prove all-pass.
        out = ts.ifeval_score(
            generated_text="all lowercase here",
            instruction_id_list=[
                "change_case:english_lowercase",
                "language:response_language",
            ],
            kwargs_list=[{}, {"language": "fr"}],
        )
        assert out["ifeval_num_constraints"] == 2
        assert out["ifeval_num_supported"] == 1
        assert out["ifeval_num_unsupported"] == 1
        assert out["ifeval_num_satisfied"] == 1
        # Score is over the supported subset.
        assert out["ifeval_score"] == 1.0
        assert out["ifeval_correct"] is None

    def test_empty_instruction_list_is_undefined(self) -> None:
        out = ts.ifeval_score(
            generated_text="text",
            instruction_id_list=[],
            kwargs_list=[],
        )
        assert out["ifeval_correct"] is None
        assert out["ifeval_num_constraints"] == 0


class TestIFEvalThreshold:
    def test_default_requires_all_satisfied(self) -> None:
        # 1 of 2 satisfied → not "all satisfied"
        out = ts.ifeval_score(
            generated_text="HAS CAPS",  # violates lowercase
            instruction_id_list=[
                "change_case:english_lowercase",
                "punctuation:no_comma",
            ],
            kwargs_list=[{}, {}],
        )
        assert out["ifeval_num_satisfied"] == 1
        assert out["ifeval_correct"] is False

    def test_relaxed_threshold(self) -> None:
        # 1/2 = 0.5 supported satisfied; threshold 0.4 lets it pass.
        out = ts.ifeval_score(
            generated_text="HAS CAPS",
            instruction_id_list=[
                "change_case:english_lowercase",
                "punctuation:no_comma",
            ],
            kwargs_list=[{}, {}],
            threshold=0.4,
        )
        assert out["ifeval_correct"] is True


# ----------------------------
# Argument-shape robustness
# ----------------------------


class TestIFEvalArgShape:
    def test_kwargs_list_can_be_none_entries(self) -> None:
        # IFEval kwargs are sometimes [{}, {}] with all-None values.
        out = ts.ifeval_score(
            generated_text="xx",
            instruction_id_list=["punctuation:no_comma"],
            kwargs_list=[None],
        )
        assert out["ifeval_correct"] is True

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            ts.ifeval_score(
                generated_text="x",
                instruction_id_list=[
                    "punctuation:no_comma",
                    "keywords:existence",
                ],
                kwargs_list=[{}],
            )
