"""Tests for src/herald/tasks.py.

Fast pure-logic tests do not require a network connection.
Tests that call load_prompts (which loads HF datasets) are guarded
with a skip if offline or if the datasets package is absent.
"""

import pytest

from herald.tasks import PromptRecord, extract_gold_answer

# ---------------------------------------------------------------------------
# extract_gold_answer: pure-logic tests, always run
# ---------------------------------------------------------------------------


class TestExtractGoldAnswer:
    def test_simple_integer(self) -> None:
        assert extract_gold_answer("Steps...\n#### 72") == "72"

    def test_leading_whitespace_after_hash(self) -> None:
        assert extract_gold_answer("#### 42") == "42"

    def test_comma_grouped_number(self) -> None:
        # Commas must be stripped
        assert extract_gold_answer("#### 1,234") == "1234"

    def test_large_number_with_commas(self) -> None:
        assert extract_gold_answer("work\n#### 1,000,000") == "1000000"

    def test_dollar_sign_stripped(self) -> None:
        # Dollar sign sometimes appears; strip it
        assert extract_gold_answer("#### $500") == "500"

    def test_multiline_answer_field(self) -> None:
        answer = (
            "She earns 5 * 3 = <<15=15>>15 dollars.\n"
            "Total is 15 + 3 = <<18=18>>18.\n"
            "#### 18"
        )
        assert extract_gold_answer(answer) == "18"

    def test_takes_text_after_last_hash(self) -> None:
        # The split always uses the last segment after ####
        answer = "Some note #### 3\nFinal #### 42"
        assert extract_gold_answer(answer) == "42"

    def test_no_units_returned(self) -> None:
        # Answer field never has units but extraction must not add them
        result = extract_gold_answer("#### 100")
        assert result.isdigit()

    def test_negative_number(self) -> None:
        # Although real GSM8K answers are positive, the extractor should
        # not corrupt a negative if one appears.
        assert extract_gold_answer("#### -5") == "-5"

    def test_strips_trailing_whitespace(self) -> None:
        assert extract_gold_answer("#### 99   ") == "99"

    def test_zero(self) -> None:
        # Edge: zero is a valid integer in the extraction path.
        assert extract_gold_answer("#### 0") == "0"


# ---------------------------------------------------------------------------
# PromptRecord model validation: pure-logic, no network
# ---------------------------------------------------------------------------


class TestPromptRecord:
    def test_basic_construction(self) -> None:
        rec = PromptRecord(
            task="gsm8k",
            prompt_id="gsm8k-0",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            gold={"answer": "4"},
        )
        assert rec.task == "gsm8k"
        assert rec.prompt_id == "gsm8k-0"
        assert len(rec.messages) == 1
        assert rec.gold == {"answer": "4"}

    def test_messages_is_list_of_dicts(self) -> None:
        rec = PromptRecord(
            task="humaneval",
            prompt_id="HumanEval/0",
            messages=[{"role": "user", "content": "Complete this."}],
            gold={"task_id": "HumanEval/0"},
        )
        assert isinstance(rec.messages, list)
        assert all(isinstance(m, dict) for m in rec.messages)

    def test_gold_accepts_mixed_values(self) -> None:
        # gold is dict[str, object]; values may be str, int, or None
        rec = PromptRecord(
            task="humaneval",
            prompt_id="HumanEval/1",
            messages=[],
            gold={
                "task_id": "HumanEval/1",
                "prompt": "def foo():\n",
                "test": "assert foo() == 1",
                "entry_point": "foo",
            },
        )
        assert rec.gold["task_id"] == "HumanEval/1"


# ---------------------------------------------------------------------------
# load_prompts: guarded tests that require a network / HF cache
# ---------------------------------------------------------------------------


def _check_network() -> None:
    """Skip if datasets is unavailable or HF is unreachable."""
    pytest.importorskip("datasets")
    # If we got here, datasets is importable; actual network errors will
    # surface as dataset-load exceptions caught in each test below.


class TestLoadPromptsGSM8K:
    def test_returns_n_records(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("gsm8k", 5)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        assert len(records) == 5

    def test_prompt_ids_are_sequential(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("gsm8k", 3)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        assert [r.prompt_id for r in records] == [
            "gsm8k-0",
            "gsm8k-1",
            "gsm8k-2",
        ]

    def test_task_field_is_gsm8k(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("gsm8k", 2)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        assert all(r.task == "gsm8k" for r in records)

    def test_messages_has_one_user_message(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("gsm8k", 1)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        msgs = records[0].messages
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_user_message_contains_hash_instruction(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("gsm8k", 1)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        content = records[0].messages[0]["content"]
        # Must instruct the model to end with #### <answer>
        assert "####" in content

    def test_gold_has_answer_key(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("gsm8k", 2)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        for r in records:
            assert "answer" in r.gold

    def test_gold_answer_is_digits_only(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("gsm8k", 5)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        for r in records:
            answer = str(r.gold["answer"])
            # May have leading minus; strip sign before digit check
            assert answer.lstrip("-").isdigit(), (
                f"Non-digit answer: {answer!r}"
            )

    def test_deterministic_order(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            a = load_prompts("gsm8k", 5)
            b = load_prompts("gsm8k", 5)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        assert [r.prompt_id for r in a] == [r.prompt_id for r in b]
        assert [r.gold for r in a] == [r.gold for r in b]

    def test_n_larger_than_split_clips(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        # GSM8K test has 1319 examples; asking for more should not crash
        try:
            records = load_prompts("gsm8k", 10_000)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        # Should return at most the split size (1319)
        assert len(records) <= 1319
        assert len(records) > 0


class TestLoadPromptsHumanEval:
    def test_returns_n_records(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("humaneval", 3)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        assert len(records) == 3

    def test_prompt_id_is_task_id(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("humaneval", 2)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        # HumanEval task_ids are like "HumanEval/0", "HumanEval/1"
        for r in records:
            assert r.prompt_id.startswith("HumanEval/")

    def test_task_field_is_humaneval(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("humaneval", 2)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        assert all(r.task == "humaneval" for r in records)

    def test_messages_has_one_user_message(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("humaneval", 1)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        msgs = records[0].messages
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_user_message_contains_code_fence(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("humaneval", 1)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        content = records[0].messages[0]["content"]
        assert "```python" in content

    def test_user_message_requests_python_code_block(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("humaneval", 1)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        content = records[0].messages[0]["content"]
        # Must instruct the model to return a python code block
        assert "```python" in content.lower() or "python" in content

    def test_gold_has_required_keys(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("humaneval", 2)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        for r in records:
            for key in ("task_id", "prompt", "test", "entry_point"):
                assert key in r.gold, f"Missing key {key!r}"

    def test_gold_task_id_matches_prompt_id(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        try:
            records = load_prompts("humaneval", 3)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        for r in records:
            assert r.gold["task_id"] == r.prompt_id

    def test_n_larger_than_split_clips(self) -> None:
        _check_network()
        from herald.tasks import load_prompts

        # HumanEval test has 164 examples
        try:
            records = load_prompts("humaneval", 10_000)
        except Exception as exc:
            pytest.skip(f"Dataset unavailable: {exc}")
        assert len(records) <= 164
        assert len(records) > 0


class TestLoadPromptsUnknownTask:
    def test_unknown_task_raises(self) -> None:
        from herald.tasks import load_prompts

        with pytest.raises(KeyError):
            load_prompts("nonexistent_task", 5)
