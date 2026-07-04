"""Tests for src/herald/scoring.py.

All tests are model-free and network-free.
"""

import os
import tempfile

import pytest

from herald.scoring import extract_pred_answer, score

# ---------------------------------------------------------------------------
# extract_pred_answer
# ---------------------------------------------------------------------------


class TestExtractPredAnswer:
    def test_hash_delimiter(self) -> None:
        text = "Step 1: 3+4=7\n#### 7"
        assert extract_pred_answer(text) == "7"

    def test_hash_delimiter_last_match(self) -> None:
        # Should return the LAST #### match, not the first.
        text = "#### 3\nSome more reasoning\n#### 42"
        assert extract_pred_answer(text) == "42"

    def test_hash_with_comma(self) -> None:
        text = "The total is #### 1,234"
        assert extract_pred_answer(text) == "1234"

    def test_boxed_fallback(self) -> None:
        text = r"Therefore the answer is \boxed{99}"
        assert extract_pred_answer(text) == "99"

    def test_boxed_last_match(self) -> None:
        text = r"First \boxed{3} then \boxed{55}"
        assert extract_pred_answer(text) == "55"

    def test_trailing_prose_after_number(self) -> None:
        # "The answer is X." pattern
        text = "The answer is 42."
        assert extract_pred_answer(text) == "42"

    def test_float_normalized_to_int(self) -> None:
        # 42.0 should normalize to "42"
        text = "#### 42.0"
        assert extract_pred_answer(text) == "42"

    def test_last_number_fallback(self) -> None:
        # No structured marker: strict (default) returns None so a
        # damaged output is not awarded credit for a stray number; the
        # loose fallback is opt-in.
        text = "We have 5 apples and 3 oranges making 8 total."
        assert extract_pred_answer(text) is None
        assert extract_pred_answer(text, allow_loose=True) == "8"

    def test_no_answer_returns_none(self) -> None:
        text = "I'm not sure how to solve this problem."
        assert extract_pred_answer(text) is None

    def test_dollar_sign_stripped(self) -> None:
        text = "Answer: $72"
        assert extract_pred_answer(text) == "72"


# ---------------------------------------------------------------------------
# score("gsm8k", ...)
# ---------------------------------------------------------------------------


class TestScoreGSM8K:
    def test_correct_match(self) -> None:
        output = "Some reasoning\n#### 42"
        gold: dict[str, object] = {"answer": "42"}
        assert score("gsm8k", output, gold) == 1.0

    def test_mismatch(self) -> None:
        output = "I think it is #### 7"
        gold: dict[str, object] = {"answer": "42"}
        assert score("gsm8k", output, gold) == 0.0

    def test_comma_in_output_matches(self) -> None:
        output = "#### 1,200"
        gold: dict[str, object] = {"answer": "1200"}
        assert score("gsm8k", output, gold) == 1.0

    def test_float_output_matches_int_gold(self) -> None:
        output = "#### 72.0"
        gold: dict[str, object] = {"answer": "72"}
        assert score("gsm8k", output, gold) == 1.0

    def test_no_answer_in_output(self) -> None:
        output = "I cannot determine the answer."
        gold: dict[str, object] = {"answer": "10"}
        assert score("gsm8k", output, gold) == 0.0

    def test_boxed_matches(self) -> None:
        output = r"Therefore \boxed{15}"
        gold: dict[str, object] = {"answer": "15"}
        assert score("gsm8k", output, gold) == 1.0


# ---------------------------------------------------------------------------
# score("humaneval", ...)
# ---------------------------------------------------------------------------

_ADD_PROMPT = 'def add(a: int, b: int) -> int:\n    """Add two ints."""\n'
_ADD_TEST = (
    "def check(candidate):\n"
    "    assert candidate(2, 3) == 5\n"
    "    assert candidate(0, 0) == 0\n"
)
_ADD_GOLD: dict[str, object] = {
    "prompt": _ADD_PROMPT,
    "test": _ADD_TEST,
    "entry_point": "add",
}


class TestScoreHumanEval:
    def test_correct_fenced_python(self) -> None:
        output = (
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "```"
        )
        assert score("humaneval", output, _ADD_GOLD) == 1.0

    def test_wrong_implementation(self) -> None:
        output = (
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a - b\n"
            "```"
        )
        assert score("humaneval", output, _ADD_GOLD) == 0.0

    def test_raising_implementation(self) -> None:
        output = (
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    raise RuntimeError('nope')\n"
            "```"
        )
        assert score("humaneval", output, _ADD_GOLD) == 0.0

    def test_infinite_loop_times_out(self) -> None:
        output = (
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    while True:\n"
            "        pass\n"
            "```"
        )
        # Use a very short timeout so the test suite stays fast.
        assert score("humaneval", output, _ADD_GOLD, timeout=2) == 0.0

    def test_no_fenced_block_falls_back_to_plain_fence(self) -> None:
        # Plain ``` fence without 'python' tag
        output = "```\ndef add(a: int, b: int) -> int:\n    return a + b\n```"
        assert score("humaneval", output, _ADD_GOLD) == 1.0

    def test_no_fence_falls_back_to_whole_text(self) -> None:
        # No fence at all; entire text should be treated as code.
        output = "def add(a: int, b: int) -> int:\n    return a + b\n"
        assert score("humaneval", output, _ADD_GOLD) == 1.0

    def test_no_leftover_temp_files(self) -> None:
        tmp_dir = tempfile.gettempdir()
        before = set(os.listdir(tmp_dir))
        output = (
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "```"
        )
        score("humaneval", output, _ADD_GOLD)
        after = set(os.listdir(tmp_dir))
        new_files = after - before
        # Any herald-created temp files should be gone.
        herald_files = [f for f in new_files if "herald" in f.lower()]
        assert herald_files == []

    def test_syntax_error_returns_zero(self) -> None:
        output = "```python\ndef add(a b):\n    return a + b\n```"
        assert score("humaneval", output, _ADD_GOLD) == 0.0


# ---------------------------------------------------------------------------
# Unknown task
# ---------------------------------------------------------------------------


def test_unknown_task_raises() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        score("unknown_task", "output", {})
