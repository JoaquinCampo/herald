"""Prompt loading for HERALD generation sweeps.

Provides `load_prompts`, which returns a list of `PromptRecord`
objects for a given task name. Each record holds the chat messages
to send to the model and the gold metadata needed for scoring.

Design notes:
- `datasets` is imported lazily inside `load_prompts` so that importing
  this module works even when the `datasets` package is absent (e.g.
  during pure-unit-test runs without heavy deps installed).
- Tokenization and chat-template application are NOT done here; the
  caller is responsible.
- Dataset order is preserved as-is (deterministic, not shuffled).
"""

import re

from pydantic import BaseModel

from herald.config import TASKS, TaskSpec

# Regex matching the official GSM8K #### delimiter.
# The pattern accepts an optional minus sign so negative numbers are
# handled correctly even though real GSM8K answers are always positive.
_GOLD_ANS_RE = re.compile(r"####\s*\$?(-?[0-9,]+)")


def extract_gold_answer(answer_text: str) -> str:
    """Extract the numeric answer from a GSM8K `answer` field.

    Splits on the last occurrence of `#### <number>`, strips commas
    and dollar signs, and returns the bare digit string (with optional
    leading minus). Raises `ValueError` if no match is found.

    Args:
        answer_text: The raw `answer` field from the GSM8K dataset,
            e.g. ``"Step-by-step...\\n#### 72"``.

    Returns:
        The extracted answer as a string of digits (possibly with a
        leading ``-``), e.g. ``"72"`` or ``"-5"``.
    """
    matches: list[str] = _GOLD_ANS_RE.findall(answer_text)
    if not matches:
        raise ValueError(f"No #### answer found in: {answer_text!r}")
    # Take the last match in case the field somehow contains multiple
    raw: str = matches[-1]
    return raw.replace(",", "").strip()


class PromptRecord(BaseModel):
    """One prompt ready to send to the model.

    Attributes:
        task: Task name, e.g. ``"gsm8k"`` or ``"humaneval"``.
        prompt_id: Stable unique identifier for the example.
        messages: Chat messages in the format accepted by
            ``tokenizer.apply_chat_template``, e.g.
            ``[{"role": "user", "content": "..."}]``.
        gold: Task-specific scoring metadata. All values must be
            JSON-serializable. See module docstring for per-task
            contracts.
    """

    task: str
    prompt_id: str
    messages: list[dict[str, str]]
    gold: dict[str, object]


def _build_gsm8k_record(
    index: int,
    example: dict[str, object],
) -> PromptRecord:
    """Build a PromptRecord for one GSM8K example."""
    question = str(example["question"])
    answer_text = str(example["answer"])
    gold_answer = extract_gold_answer(answer_text)

    user_content = (
        "Solve the following math problem step by step. "
        "At the end of your solution, write the final numeric answer "
        "on its own line in the format:\n"
        "#### <answer>\n\n"
        f"{question}"
    )

    return PromptRecord(
        task="gsm8k",
        prompt_id=f"gsm8k-{index}",
        messages=[{"role": "user", "content": user_content}],
        gold={"answer": gold_answer},
    )


def _build_humaneval_record(
    example: dict[str, object],
) -> PromptRecord:
    """Build a PromptRecord for one HumanEval example."""
    task_id = str(example["task_id"])
    prompt = str(example["prompt"])
    test = str(example["test"])
    entry_point = str(example["entry_point"])

    user_content = (
        "Complete the following Python function. "
        "Return the complete function implementation "
        "in a ```python code block.\n\n"
        f"```python\n{prompt}\n```"
    )

    return PromptRecord(
        task="humaneval",
        prompt_id=task_id,
        messages=[{"role": "user", "content": user_content}],
        gold={
            "task_id": task_id,
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
        },
    )


def load_prompts(
    task: str,
    n: int,
    spec: TaskSpec | None = None,
) -> list[PromptRecord]:
    """Load the first `n` examples for `task` and return PromptRecords.

    Examples are returned in the dataset's native order (deterministic).
    If `n` exceeds the split size, all available examples are returned
    without error.

    Args:
        task: Task name; must be a key in ``config.TASKS`` if `spec`
            is ``None``.
        n: Number of examples to load (first `n` in dataset order).
        spec: Optional ``TaskSpec`` override. If ``None``, looks up
            ``config.TASKS[task]``.

    Returns:
        List of ``PromptRecord`` objects, one per example.

    Raises:
        KeyError: If `task` is not in ``config.TASKS`` and `spec` is
            ``None``.
    """
    # ifeval and longbench have bespoke loaders (custom datasets and
    # prompt templates); dispatch before the generic spec-based path.
    if task == "ifeval":
        from herald.ifeval import load_ifeval

        return load_ifeval(n)
    if task == "longbench":
        from herald.longbench import load_longbench

        return load_longbench(n)

    # Lazy import: keeps the module importable when datasets is absent.
    from datasets import load_dataset  # type: ignore[import-untyped]

    if spec is None:
        spec = TASKS[task]

    load_kwargs: dict[str, object] = {"split": spec.split}
    if spec.hf_subset is not None:
        ds = load_dataset(spec.hf_path, spec.hf_subset, **load_kwargs)
    else:
        ds = load_dataset(spec.hf_path, **load_kwargs)

    actual_n = min(n, len(ds))
    subset = ds.select(range(actual_n))

    records: list[PromptRecord] = []
    for i, example in enumerate(subset):
        ex: dict[str, object] = dict(example)
        if task == "gsm8k":
            records.append(_build_gsm8k_record(i, ex))
        elif task == "humaneval":
            records.append(_build_humaneval_record(ex))
        else:
            raise ValueError(f"No builder for task {task!r}")

    return records
