"""Tests for herald.tasks — Task abstraction.

The Phase-1 tasks (HumanEval, IFEval, LongBench) cover only the
cheap parse / is_wrong logic here. `load()` hits Hugging Face and is
exercised by the Orion smoke runs, not by unit tests.
"""

from herald.tasks import (
    DEFAULT_TASK,
    LONGBENCH_PROMPT_TOKEN_BUDGET,
    LONGBENCH_SYSTEM_PROMPT,
    PHASE1_TASKS,
    GSM8KTask,
    HumanEvalTask,
    IFEvalTask,
    LongBenchSingleTask,
    Task,
    _extract_python_code,
)


class _StubTokenizer:
    """Whitespace-token tokenizer for CPU tests.

    Encode = split on whitespace; decode = join with single space.
    `apply_chat_template` mimics ChatML overhead by prepending and
    appending fixed control tokens.
    """

    def __init__(self, chat_overhead: int = 16) -> None:
        self._chat_overhead = chat_overhead

    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> list[int]:
        words = text.split()
        # Map each word to a stable id; collisions are fine for tests.
        return [hash(w) & 0xFFFF for w in words]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"t{i}" for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        body = "\n".join(m["content"] for m in messages)
        ctrl = " ".join(["<|c|>"] * self._chat_overhead)
        return f"{ctrl}\n{body}\n{ctrl}"


class TestGSM8KTask:
    def test_implements_task_interface(self):
        task = GSM8KTask()
        assert isinstance(task, Task)
        assert task.name == "gsm8k"
        assert hasattr(task, "load")
        assert hasattr(task, "is_wrong")
        assert hasattr(task, "parse_answer")

    def test_parse_answer_gsm8k_format(self):
        task = GSM8KTask()
        assert task.parse_answer("The answer is #### 42") == "42"
        assert task.parse_answer("\\boxed{7}") == "7"
        assert task.parse_answer("no answer here") is None

    def test_is_wrong_correct_answer(self):
        task = GSM8KTask()
        assert task.is_wrong("#### 42", "42") is False
        assert task.is_wrong("#### 99", "42") is True
        assert task.is_wrong("I don't know", "42") is True

    def test_default_task_is_gsm8k(self):
        assert isinstance(DEFAULT_TASK, GSM8KTask)


class TestExtractPythonCode:
    def test_strips_fenced_block(self) -> None:
        text = "Here you go:\n```python\ndef f():\n    return 1\n```\nthanks"
        assert _extract_python_code(text) == "def f():\n    return 1"

    def test_no_fence_returns_text(self) -> None:
        assert _extract_python_code("    return 1") == "return 1"


class TestHumanEvalTask:
    def setup_method(self) -> None:
        self.task = HumanEvalTask()

    def test_name_and_interface(self) -> None:
        assert self.task.name == "humaneval"
        assert isinstance(self.task, Task)

    def test_is_wrong_empty(self) -> None:
        assert self.task.is_wrong("", "") is True
        assert self.task.is_wrong("   ", "") is True

    def test_is_wrong_no_def_no_indent(self) -> None:
        assert self.task.is_wrong("hello world", "") is True

    def test_is_wrong_valid_def(self) -> None:
        ok = "def f(x):\n    return x + 1"
        assert self.task.is_wrong(ok, "") is False

    def test_is_wrong_indented_body_only(self) -> None:
        # Model returned only the body — wrap-and-parse path catches it.
        body = "    return x + 1"
        assert self.task.is_wrong(body, "") is False

    def test_is_wrong_syntax_error(self) -> None:
        bad = "def f(x):\n    return x +"
        assert self.task.is_wrong(bad, "") is True

    def test_parse_answer_returns_code(self) -> None:
        text = "```python\ndef f():\n    return 1\n```"
        assert self.task.parse_answer(text) == "def f():\n    return 1"

    def test_parse_answer_empty_returns_none(self) -> None:
        assert self.task.parse_answer("   ") is None


class TestIFEvalTask:
    def setup_method(self) -> None:
        self.task = IFEvalTask()

    def test_name_and_interface(self) -> None:
        assert self.task.name == "ifeval"
        assert isinstance(self.task, Task)

    def test_is_wrong_empty_only(self) -> None:
        assert self.task.is_wrong("", "") is True
        assert self.task.is_wrong("any non-empty answer", "") is False

    def test_parse_answer_strips(self) -> None:
        assert self.task.parse_answer("  hi  ") == "hi"
        assert self.task.parse_answer("") is None


class TestLongBenchSingleTask:
    def setup_method(self) -> None:
        self.task = LongBenchSingleTask()

    def test_name_and_interface(self) -> None:
        assert self.task.name == "longbench_single"
        assert isinstance(self.task, Task)

    def test_is_wrong_empty_only(self) -> None:
        assert self.task.is_wrong("", "ground") is True
        assert self.task.is_wrong("an answer", "ground") is False

    def test_parse_answer_strips(self) -> None:
        assert self.task.parse_answer("  hi\n") == "hi"
        assert self.task.parse_answer("") is None


class TestLongBenchFormatPrompt:
    """LongBench deterministic context truncation."""

    def setup_method(self) -> None:
        self.task = LongBenchSingleTask()
        self.tok = _StubTokenizer(chat_overhead=16)

    def _make_prompt(self, ctx_words: int = 50_000) -> dict:
        ctx = " ".join(f"w{i}" for i in range(ctx_words))
        inp = "What happened to the protagonist at the end?"
        return {
            "id": "longbench_test_001",
            "context": ctx,
            "input": inp,
            "question": f"Context:\n{ctx}\n\nQuestion: {inp}\n\nAnswer:",
            "ground_truth": "they survived",
            "system_prompt": LONGBENCH_SYSTEM_PROMPT,
            "all_answers": "they survived",
        }

    def test_truncation_keeps_chat_text_under_budget(self) -> None:
        prompt = self._make_prompt(ctx_words=50_000)
        question, meta = self.task.format_prompt(
            prompt, self.tok, max_new_tokens=512
        )
        assert meta["truncated"] is True
        # Final chat-templated length plus max_new_tokens must respect
        # the budget the format_prompt was given.
        assert (
            meta["chat_templated_tokens"] + meta["max_new_tokens"]
            <= meta["budget"]
        )

    def test_truncation_preserves_question_and_instruction(
        self,
    ) -> None:
        prompt = self._make_prompt(ctx_words=50_000)
        question, _ = self.task.format_prompt(
            prompt, self.tok, max_new_tokens=512
        )
        # Question text and Answer scaffolding must survive truncation.
        assert (
            "Question: What happened to the protagonist at the end?"
            in question
        )
        assert question.endswith("Answer:")
        assert question.startswith("Context:\n")

    def test_truncation_is_deterministic(self) -> None:
        prompt = self._make_prompt(ctx_words=50_000)
        q1, m1 = self.task.format_prompt(prompt, self.tok, max_new_tokens=512)
        q2, m2 = self.task.format_prompt(prompt, self.tok, max_new_tokens=512)
        assert q1 == q2
        assert m1 == m2

    def test_short_context_is_not_truncated(self) -> None:
        prompt = self._make_prompt(ctx_words=200)
        question, meta = self.task.format_prompt(
            prompt, self.tok, max_new_tokens=512
        )
        assert meta["truncated"] is False
        # Question is the un-truncated formatted form.
        assert question == prompt["question"]

    def test_no_tokenizer_falls_back_to_full_question(self) -> None:
        prompt = self._make_prompt(ctx_words=50_000)
        question, meta = self.task.format_prompt(
            prompt, tokenizer=None, max_new_tokens=512
        )
        assert meta["truncated"] is False
        assert meta["reason"] == "no_tokenizer"
        assert question == prompt["question"]

    def test_missing_raw_fields_falls_back(self) -> None:
        # Old-style payload without context/input still works.
        prompt = {
            "id": "longbench_legacy",
            "question": "Context:\nfoo\n\nQuestion: q?\n\nAnswer:",
            "ground_truth": "bar",
            "system_prompt": LONGBENCH_SYSTEM_PROMPT,
        }
        question, meta = self.task.format_prompt(
            prompt, self.tok, max_new_tokens=512
        )
        assert meta["truncated"] is False
        assert meta["reason"] == "missing_raw_fields"
        assert question == prompt["question"]

    def test_failing_prompt_ids_fixture(self) -> None:
        """Documented failing prompt IDs from Phase 1 Block 2 Option B.

        Kept here as a fixture so the diagnosis is grep-able from the
        test suite. These six prompt IDs OOM'd in the matched-prefix
        replay forward on Qwen2.5-7B-Instruct in the Block 2 profile.
        See gold/phase-1-longbench-failure-diagnosis.md.
        """
        failing = {
            "longbench_1842b0ff1882e545a6d41d5caf67bba5312872423fa48e74",
            "longbench_32e116c58a3c59fc170aa5f4e1dde414c8f3881872889826",
            "longbench_7570a52d69ab93c5f54eba4c45d44a3411650c1e4694760a",
            "longbench_b03244c8cc2681df1008d27c974d81415336396dff81f06d",
            "longbench_df6c6350671baab25c635bfa495eea90c69a7d201b5fe460",
            "longbench_fbeb825de92309788269da33aa6bd189c7b1d46b997746f4",
        }
        assert len(failing) == 6


class TestNonLongBenchFormatPromptUnchanged:
    """Hard rule: GSM8K / HumanEval / IFEval behavior is unchanged."""

    def test_gsm8k_format_prompt_is_noop(self) -> None:
        task = GSM8KTask()
        tok = _StubTokenizer()
        prompt = {
            "id": "gsm8k_0",
            "question": "If 2+2=?, what is the answer?",
            "ground_truth": "4",
        }
        question, meta = task.format_prompt(prompt, tok)
        assert question == prompt["question"]
        assert meta == {}

    def test_humaneval_format_prompt_is_noop(self) -> None:
        task = HumanEvalTask()
        tok = _StubTokenizer()
        prompt = {
            "id": "humaneval_HumanEval_0",
            "question": 'def add(a, b):\n    """Sum."""\n',
            "ground_truth": "    return a + b",
        }
        question, meta = task.format_prompt(prompt, tok)
        assert question == prompt["question"]
        assert meta == {}

    def test_ifeval_format_prompt_is_noop(self) -> None:
        task = IFEvalTask()
        tok = _StubTokenizer()
        prompt = {
            "id": "ifeval_42",
            "question": "Write 3 bullet points about cats.",
            "ground_truth": "",
        }
        question, meta = task.format_prompt(prompt, tok)
        assert question == prompt["question"]
        assert meta == {}


class TestLongBenchBudgetConstant:
    """Canary: the chosen budget is in a sane range for Qwen 32k."""

    def test_budget_under_positional_limit(self) -> None:
        # Qwen2.5-7B-Instruct positional ceiling is 32768.
        # Budget plus a 512-token generation must fit inside it with
        # comfortable room for chat-template control overhead.
        assert LONGBENCH_PROMPT_TOKEN_BUDGET + 512 + 1024 <= 32768


def test_phase1_tasks_in_order() -> None:
    names = tuple(t.name for t in PHASE1_TASKS)
    assert names == (
        "gsm8k",
        "humaneval",
        "ifeval",
        "longbench_single",
    )
