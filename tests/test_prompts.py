"""Tests for herald.prompts — prompt formatting."""

from herald.prompts import SYSTEM_PROMPT, format_chat


class TestFormatChat:
    def test_returns_list_of_dicts(self):
        result = format_chat("What is 2+2?")
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(m, dict) for m in result)

    def test_system_message_first(self):
        result = format_chat("What is 2+2?")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == SYSTEM_PROMPT

    def test_user_message_second(self):
        question = "If Sally has 3 apples and buys 2 more, how many does she have?"
        result = format_chat(question)
        assert result[1]["role"] == "user"
        assert result[1]["content"] == question

    def test_message_keys(self):
        result = format_chat("test")
        for msg in result:
            assert set(msg.keys()) == {"role", "content"}

    def test_system_prompt_mentions_reasoning(self):
        assert "step by step" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_answer_format(self):
        assert "####" in SYSTEM_PROMPT
