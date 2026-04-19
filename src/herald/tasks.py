"""Task abstractions: benchmark loading and answer scoring.

A Task ties together prompt loading, answer parsing, and
wrong-answer detection for one benchmark. Iteration 1 ships
only GSM8KTask; adding LongBench/RULER means a new class here
and no changes to experiment.py or detectors.py.
"""

from abc import ABC, abstractmethod

from herald.detectors import detect_answer_failure, parse_gsm8k_answer
from herald.prompts import load_gsm8k


class Task(ABC):
    """A benchmark with prompt loading + answer scoring."""

    name: str

    @abstractmethod
    def load(self, num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
        """Return ordered list of prompt dicts.

        Each dict must have keys: id, question, ground_truth.
        """

    @abstractmethod
    def is_wrong(self, generated_text: str, ground_truth: str) -> bool:
        """True iff the answer is wrong or not extractable."""

    @abstractmethod
    def parse_answer(self, generated_text: str) -> str | None:
        """Normalized answer string; None if not parseable."""


class GSM8KTask(Task):
    """GSM8K math reasoning benchmark."""

    name = "gsm8k"

    def load(self, num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
        return load_gsm8k(num_prompts, seed)

    def is_wrong(self, generated_text: str, ground_truth: str) -> bool:
        return detect_answer_failure(generated_text, ground_truth)

    def parse_answer(self, generated_text: str) -> str | None:
        return parse_gsm8k_answer(generated_text)


DEFAULT_TASK: Task = GSM8KTask()
