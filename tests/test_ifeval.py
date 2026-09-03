"""Model-free tests for src/herald/ifeval.py.

Tests use hand-crafted gold dicts to verify score_ifeval without
requiring a network connection or loading the actual HuggingFace
dataset.

Instruction types chosen for their absence of NLTK corpus deps:
- ``keywords:existence`` uses regex (corpus-free).
- ``change_case:english_capital`` uses str.isupper() + langdetect.
- ``length_constraints:number_words`` uses RegexpTokenizer (corpus-free).

The ``change_case:english_capital`` checker calls langdetect internally;
the seed is set in ifeval.py so behaviour is deterministic.
"""

import pytest

from herald.ifeval import (
    IFEvalScores,
    score_ifeval,
    score_ifeval_robustness,
)

# ---------------------------------------------------------------------------
# Test 1: single keyword:existence -- full credit and zero credit
# ---------------------------------------------------------------------------


def test_keywords_existence_full_credit() -> None:
    """Output containing all required keywords scores 1.0."""
    gold: dict[str, object] = {
        "prompt": "Write something about the ocean.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["ocean", "water"]}],
    }
    output = "The ocean is full of water and amazing creatures."
    assert score_ifeval(output, gold) == pytest.approx(1.0)


def test_keywords_existence_zero_credit() -> None:
    """Output missing all required keywords scores 0.0."""
    gold: dict[str, object] = {
        "prompt": "Write something about the ocean.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["ocean", "water"]}],
    }
    output = "The sky is blue and birds are singing."
    assert score_ifeval(output, gold) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 2: two instructions, partial credit
# ---------------------------------------------------------------------------


def test_two_instructions_partial_credit() -> None:
    """Output satisfying only one of two instructions scores 0.5."""
    # Instruction 1: keyword:existence for "python"
    # Instruction 2: change_case:english_capital (all caps)
    # The output contains "python" but is not fully uppercase.
    gold: dict[str, object] = {
        "prompt": "Write about python.",
        "instruction_id_list": [
            "keywords:existence",
            "change_case:english_capital",
        ],
        "kwargs": [
            {"keywords": ["python"]},
            {},
        ],
    }
    output = "Python is a popular programming language."
    result = score_ifeval(output, gold)
    # keywords:existence: "python" appears (case-insensitive) -> followed
    # change_case:english_capital: not all caps -> not followed
    assert result == pytest.approx(0.5)


def test_two_instructions_full_credit() -> None:
    """Output satisfying both instructions scores 1.0."""
    gold: dict[str, object] = {
        "prompt": "Write about python.",
        "instruction_id_list": [
            "keywords:existence",
            "change_case:english_capital",
        ],
        "kwargs": [
            {"keywords": ["PYTHON"]},
            {},
        ],
    }
    # All-caps output; "PYTHON" appears -> both instructions satisfied
    output = "PYTHON IS A POPULAR PROGRAMMING LANGUAGE."
    result = score_ifeval(output, gold)
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 3: length_constraints:number_words
# ---------------------------------------------------------------------------


def test_number_words_at_least_satisfied() -> None:
    """Output with enough words satisfies the length instruction."""
    gold: dict[str, object] = {
        "prompt": "Write a long answer.",
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"num_words": 5, "relation": "at least"}],
    }
    output = "one two three four five six seven"  # 7 words
    assert score_ifeval(output, gold) == pytest.approx(1.0)


def test_number_words_at_least_not_satisfied() -> None:
    """Output below the word count threshold scores 0.0."""
    gold: dict[str, object] = {
        "prompt": "Write a long answer.",
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"num_words": 50, "relation": "at least"}],
    }
    output = "Short answer."  # 2 words
    assert score_ifeval(output, gold) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 4: empty instruction list edge case
# ---------------------------------------------------------------------------


def test_empty_instruction_list_returns_zero() -> None:
    """Empty instruction list returns 0.0 (no ZeroDivisionError)."""
    gold: dict[str, object] = {
        "prompt": "Any prompt.",
        "instruction_id_list": [],
        "kwargs": [],
    }
    assert score_ifeval("any output", gold) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 5: none-padded kwargs (simulates HuggingFace uniform-schema padding)
# ---------------------------------------------------------------------------


def test_none_padded_kwargs_filtered() -> None:
    """None kwargs values are filtered before calling build_description."""
    gold: dict[str, object] = {
        "prompt": "Write something.",
        "instruction_id_list": ["keywords:existence"],
        # HF may pad with None values for keys unused by this instruction
        "kwargs": [
            {"keywords": ["hello"], "num_words": None, "relation": None}
        ],
    }
    output = "Hello world!"
    # Should not raise TypeError despite extra None-valued keys
    assert score_ifeval(output, gold) == pytest.approx(1.0)


# Strict and loose scoring modes
# ---------------------------------------------------------------------------


def test_strict_is_lower_when_only_loose_transformation_passes() -> None:
    """Strict rejects punctuation discarded with the first line."""
    gold: dict[str, object] = {
        "prompt": "Write without commas.",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }

    scores = score_ifeval_robustness(
        "Discard this,\nValid body", gold, mode="both"
    )

    assert isinstance(scores, IFEvalScores)
    assert scores.strict == pytest.approx(0.0)
    assert scores.loose == pytest.approx(1.0)
    assert scores.strict <= scores.loose


def test_strict_equals_loose_when_original_passes() -> None:
    """Both modes agree when the original output passes."""
    gold: dict[str, object] = {
        "prompt": "Write about the ocean.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["ocean"]}],
    }

    scores = score_ifeval_robustness("The ocean is blue.", gold, mode="both")

    assert isinstance(scores, IFEvalScores)
    assert scores.strict == pytest.approx(1.0)
    assert scores.loose == pytest.approx(scores.strict)


def test_both_mode_preserves_fractional_instruction_scores() -> None:
    """Instructions contribute independently to both scores."""
    gold: dict[str, object] = {
        "prompt": "Mention Python without commas.",
        "instruction_id_list": [
            "keywords:existence",
            "punctuation:no_comma",
        ],
        "kwargs": [{"keywords": ["python"]}, {}],
    }

    scores = score_ifeval_robustness("python,\nValid body", gold, mode="both")

    assert isinstance(scores, IFEvalScores)
    assert scores.strict == pytest.approx(0.5)
    assert scores.loose == pytest.approx(1.0)


def test_robustness_rejects_invalid_mode() -> None:
    """The explicit mode API rejects unsupported scoring variants."""
    gold: dict[str, object] = {
        "prompt": "Write something.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["hello"]}],
    }

    with pytest.raises(ValueError, match="invalid IFEval scoring mode"):
        score_ifeval_robustness(
            "hello",
            gold,
            mode="unsupported",  # type: ignore[arg-type]
        )


def test_public_loose_score_matches_explicit_loose_mode() -> None:
    """The established score_ifeval entry point remains the loose score."""
    gold: dict[str, object] = {
        "prompt": "Write about the ocean.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["ocean"]}],
    }
    output = "*ocean*"

    assert score_ifeval(output, gold) == pytest.approx(
        score_ifeval_robustness(output, gold, mode="loose")
    )


# Network-dependent test (skipped offline)
# ---------------------------------------------------------------------------


def test_load_ifeval_smoke() -> None:
    """Load 3 examples from HuggingFace; skip if network is unavailable."""
    try:
        from herald.ifeval import load_ifeval, score_ifeval
    except ImportError:
        pytest.skip("datasets not available")

    try:
        records = load_ifeval(3)
    except Exception as exc:
        pytest.skip(f"network/dataset unavailable: {exc}")

    # Structure checks (load path)
    assert len(records) == 3
    for rec in records:
        assert rec.task == "ifeval"
        assert rec.prompt_id.startswith("ifeval-")
        assert len(rec.messages) == 1
        assert rec.messages[0]["role"] == "user"
        assert "instruction_id_list" in rec.gold
        assert "kwargs" in rec.gold
        assert "prompt" in rec.gold
        # gold["kwargs"] must be a list of dicts, not column-major
        kw = rec.gold["kwargs"]
        assert isinstance(kw, list), (
            f"expected list-of-dicts for kwargs, got {type(kw)}"
        )

    # Scoring path: score the prompt text against its own gold.
    # The prompt never self-satisfies its instructions, so q is in
    # [0, 1] but not necessarily 0 or 1. The goal is no crash.
    for rec in records:
        q = score_ifeval(str(rec.messages[0]["content"]), rec.gold)
        assert 0.0 <= q <= 1.0, f"score out of range: {q}"
