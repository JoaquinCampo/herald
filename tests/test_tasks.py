"""Tests for herald.tasks — Task abstraction."""

from herald.tasks import DEFAULT_TASK, GSM8KTask, Task


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
        # Correct: model says 42, gt is 42
        assert task.is_wrong("#### 42", "42") is False
        # Wrong: model says 99, gt is 42
        assert task.is_wrong("#### 99", "42") is True
        # No answer at all: counts as wrong
        assert task.is_wrong("I don't know", "42") is True

    def test_default_task_is_gsm8k(self):
        assert isinstance(DEFAULT_TASK, GSM8KTask)
