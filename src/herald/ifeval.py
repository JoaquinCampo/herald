"""IFEval (instruction-following eval) task for HERALD.

Implements instruction-level loose accuracy as described in:
    Zhou et al. (2023), "Instruction-Following Evaluation for Large
    Language Models", https://arxiv.org/abs/2311.07911

The scorer exposes:

    - instruction-level loose accuracy (the backward-compatible default)
    - instruction-level strict accuracy, checking only the original response
    - a typed API returning both scores from one checker construction pass

Loose scoring
-------------
For each instruction in a prompt, the instruction is deemed followed if *any*
of eight response transformations passes the official checker. The eight
transforms are:

    1. original response
    2. response with all ``*`` removed
    3. response with first line removed
    4. response with last line removed
    5. response with first and last lines removed
    6. (3) with ``*`` removed
    7. (4) with ``*`` removed
    8. (5) with ``*`` removed

Strict scoring checks only the original response. Both scores are fractions
of individual instructions followed, each ranging in [0, 1].

Offline caveat (Orion)
----------------------
Several instruction types call NLTK tokenizers that require pre-cached
corpora. Two are needed:

- ``punkt_tab``: used by ``nltk.word_tokenize`` (called by
  ``change_case:capital_word_frequency`` and others).
- ``punkt``: used by ``count_sentences``
  (``length_constraints:number_sentences``).

Pre-cache both on Orion before running::

    python -c "import nltk
    nltk.download('punkt_tab')
    nltk.download('punkt')"

The ``keywords:existence``, ``change_case:english_capital/lowercase``,
``length_constraints:number_words``, ``punctuation:no_comma``, and
``startend:*`` checks are corpus-free and safe offline without
any download.

Dependencies
------------
Vendored from google-research/instruction_following_eval under
``src/herald/_ifeval_vendor/`` (Apache 2.0). Runtime deps added to
``pyproject.toml``: ``langdetect``, ``nltk``, ``immutabledict``,
``absl-py``.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, cast, overload

import langdetect

from herald._ifeval_vendor import instructions_registry
from herald.tasks import PromptRecord

# Set a fixed seed so langdetect checks are deterministic.
langdetect.DetectorFactory.seed = 0


def load_ifeval_exact(
    prompt_ids: Sequence[str],
    *,
    dataset: object | None = None,
) -> list[PromptRecord]:
    """Load exactly the requested IFEval prompts, without scoring metadata."""
    requested: list[str] = []
    keys: list[int] = []
    for prompt_id in prompt_ids:
        if not isinstance(prompt_id, str) or not prompt_id.startswith(
            "ifeval-"
        ):
            raise ValueError(f"malformed IFEval prompt ID: {prompt_id!r}")
        raw_key = prompt_id.removeprefix("ifeval-")
        if not raw_key.isdecimal() or (
            len(raw_key) > 1 and raw_key.startswith("0")
        ):
            raise ValueError(f"malformed IFEval prompt ID: {prompt_id!r}")
        key = int(raw_key)
        requested.append(prompt_id)
        keys.append(key)
    if len(requested) != len(set(requested)):
        raise ValueError("requested IFEval prompt IDs contain duplicates")

    injected_dataset = dataset is not None
    if dataset is None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        dataset = load_dataset("google/IFEval", split="train")
    source: Any = dataset
    try:
        source = source.select_columns(["key", "prompt"])
        column = source.data.column("key")
    except (AttributeError, KeyError, TypeError) as error:
        if not injected_dataset:
            raise ValueError(
                "production IFEval dataset must expose an Arrow key column"
            ) from error
        try:
            rows = [
                {"key": row["key"], "prompt": row["prompt"]} for row in source
            ]
        except (KeyError, TypeError) as fallback_error:
            raise ValueError(
                "IFEval dataset key column is unavailable"
            ) from fallback_error
        fallback_keys: list[int] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("IFEval dataset row is malformed") from None
            source_key = row.get("key")
            if isinstance(source_key, bool) or not isinstance(
                source_key, Integral
            ):
                raise ValueError(
                    "IFEval source keys must be integral"
                ) from None
            fallback_keys.append(int(source_key))
        if len(fallback_keys) != len(set(fallback_keys)):
            raise ValueError("IFEval source keys are not unique") from None
        positions = {key: index for index, key in enumerate(fallback_keys)}
        missing = [key for key in keys if key not in positions]
        if missing:
            raise ValueError(
                f"requested IFEval IDs are missing: {missing!r}"
            ) from None
        records: list[PromptRecord] = []
        for expected_key in keys:
            row = rows[positions[expected_key]]
            key = row["key"]
            prompt_text = row["prompt"]
            if (
                isinstance(key, bool)
                or not isinstance(key, Integral)
                or not isinstance(prompt_text, str)
            ):
                raise ValueError("IFEval selected row is malformed") from None
            records.append(
                PromptRecord(
                    task="ifeval",
                    prompt_id=f"ifeval-{int(key)}",
                    messages=[{"role": "user", "content": prompt_text}],
                    gold={},
                )
            )
        return records
    raw_keys: list[int] = []
    for value in column:
        value = value.as_py() if hasattr(value, "as_py") else value
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("IFEval source keys must be integral")
        raw_keys.append(int(value))
    if len(raw_keys) != len(set(raw_keys)):
        raise ValueError("IFEval source keys are not unique")
    positions = {key: index for index, key in enumerate(raw_keys)}
    missing = [key for key in keys if key not in positions]
    if missing:
        raise ValueError(f"requested IFEval IDs are missing: {missing!r}")
    try:
        subset = source.select([positions[key] for key in keys])
        subset = subset.select_columns(["key", "prompt"])
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(
            "IFEval dataset cannot select key and prompt columns"
        ) from error
    records = []
    for expected_key, example in zip(keys, subset, strict=True):
        if not isinstance(example, Mapping):
            raise ValueError("IFEval selected row is malformed")
        selected_key = example.get("key")
        prompt_text = example.get("prompt")
        if (
            isinstance(selected_key, bool)
            or not isinstance(selected_key, Integral)
            or not isinstance(prompt_text, str)
        ):
            raise ValueError("IFEval selected row is malformed")
        if int(selected_key) != expected_key:
            raise ValueError(
                "IFEval selected row does not match requested ID"
            )
        records.append(
            PromptRecord(
                task="ifeval",
                prompt_id=f"ifeval-{int(selected_key)}",
                messages=[{"role": "user", "content": prompt_text}],
                gold={},
            )
        )
    return records


def load_ifeval(n: int) -> list[PromptRecord]:
    """Load the first ``n`` examples from google/IFEval (train split).

    ``datasets`` is imported lazily so the module can be imported
    without the package present (e.g. in pure unit-test runs).

    Args:
        n: Number of examples to load (in dataset order, deterministic).

    Returns:
        A list of ``PromptRecord`` objects, one per example.
        Each record carries:
        - task: ``"ifeval"``
        - prompt_id: ``f"ifeval-{key}"`` (using the dataset's ``key``
          field as a stable identifier)
        - messages: single user turn with the raw prompt text
        - gold: ``{"prompt": ..., "instruction_id_list": [...],
          "kwargs": [...]}`` -- everything ``score_ifeval`` needs
    """
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    actual_n = min(n, len(ds))
    subset = ds.select(range(actual_n))

    records: list[PromptRecord] = []
    for example in subset:
        ex: dict[str, object] = dict(example)
        key = int(cast(int, ex["key"]))
        prompt_text = str(ex["prompt"])
        instruction_id_list = cast(list[str], ex["instruction_id_list"])
        kwargs_list = cast(list[dict[str, object]], ex["kwargs"])
        records.append(
            PromptRecord(
                task="ifeval",
                prompt_id=f"ifeval-{key}",
                messages=[{"role": "user", "content": prompt_text}],
                gold={
                    "prompt": prompt_text,
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs_list,
                },
            )
        )
    return records


def _loose_transforms(response: str) -> list[str]:
    """Return the eight loose-variant transforms of ``response``.

    Replicates ``test_instruction_following_loose`` from the official
    evaluation_lib.py without pulling in the dataclass wrappers.
    """
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    no_star = response.replace("*", "")
    no_star_remove_first = remove_first.replace("*", "")
    no_star_remove_last = remove_last.replace("*", "")
    no_star_remove_both = remove_both.replace("*", "")
    return [
        response,
        no_star,
        remove_first,
        remove_last,
        remove_both,
        no_star_remove_first,
        no_star_remove_last,
        no_star_remove_both,
    ]


IFEvalMode = Literal["loose", "strict", "both"]


@dataclass(frozen=True, slots=True)
class IFEvalScores:
    """Instruction-level loose and strict scores for one output."""

    loose: float
    strict: float


def _build_checker(
    instruction_id: str,
    kwargs_list: list[dict[str, object]],
    index: int,
    prompt_text: str,
) -> Callable[[str], object]:
    """Build one official checker, including its prompt-dependent metadata."""
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    # Vendor classes are untyped; suppress the cascade of no-untyped-call
    # errors from the ignore_errors override.
    instruction = instruction_cls(  # type: ignore[no-untyped-call]
        instruction_id
    )

    # Filter None-padded kwargs that HuggingFace may inject when the dataset
    # schema is uniform across examples.
    raw_kw: dict[str, object] = {}
    if index < len(kwargs_list):
        raw_kw = {
            k: v for k, v in kwargs_list[index].items() if v is not None
        }

    instruction.build_description(  # type: ignore[no-untyped-call]
        **raw_kw
    )

    # Some instructions (e.g. repeat_prompt) need the prompt text passed
    # explicitly after the initial build.
    args = instruction.get_instruction_args()  # type: ignore[no-untyped-call]
    if args and "prompt" in args:
        instruction.build_description(  # type: ignore[no-untyped-call]
            prompt=prompt_text
        )

    return cast(Callable[[str], object], instruction.check_following)


def _score_ifeval_modes(
    output_text: str,
    gold: dict[str, object],
    mode: IFEvalMode,
) -> tuple[float, float]:
    """Compute loose and strict scores with shared instruction checkers."""
    instruction_ids = cast(list[str], gold["instruction_id_list"])
    kwargs_list = cast(list[dict[str, object]], gold["kwargs"])
    prompt_text = str(gold["prompt"])

    if not instruction_ids:
        return 0.0, 0.0

    transforms = _loose_transforms(output_text) if mode != "strict" else []
    loose_followed: list[bool] = []
    strict_followed: list[bool] = []

    for index, instruction_id in enumerate(instruction_ids):
        checker = _build_checker(
            instruction_id,
            kwargs_list,
            index,
            prompt_text,
        )
        strict_passed = bool(output_text.strip() and checker(output_text))
        strict_followed.append(strict_passed)

        loose_passed = strict_passed
        if mode != "strict" and not loose_passed:
            for variant in transforms[1:]:
                if variant.strip() and bool(checker(variant)):
                    loose_passed = True
                    break
        loose_followed.append(loose_passed)

    loose_score = sum(loose_followed) / len(loose_followed)
    strict_score = sum(strict_followed) / len(strict_followed)
    return loose_score, strict_score


@overload
def score_ifeval_robustness(
    output_text: str,
    gold: dict[str, object],
    *,
    mode: Literal["loose", "strict"],
) -> float: ...


@overload
def score_ifeval_robustness(
    output_text: str,
    gold: dict[str, object],
    *,
    mode: Literal["both"] = "both",
) -> IFEvalScores: ...


def score_ifeval_robustness(
    output_text: str,
    gold: dict[str, object],
    *,
    mode: IFEvalMode = "both",
) -> float | IFEvalScores:
    """Return strict, loose, or both IFEval scores for one output/gold pair.

    ``"loose"`` retains the official eight response transformations.
    ``"strict"`` checks only the original response. ``"both"`` computes both
    scores while constructing each instruction checker only once.

    Raises:
        ValueError: If ``mode`` is not one of the three supported modes.
    """
    if mode not in ("loose", "strict", "both"):
        raise ValueError(
            f"invalid IFEval scoring mode {mode!r}; "
            'expected "loose", "strict", or "both"'
        )

    loose_score, strict_score = _score_ifeval_modes(output_text, gold, mode)
    if mode == "loose":
        return loose_score
    if mode == "strict":
        return strict_score
    return IFEvalScores(loose=loose_score, strict=strict_score)


def score_ifeval(
    output_text: str,
    gold: dict[str, object],
) -> float:
    """Return the instruction-level loose accuracy for one example.

    This backward-compatible entry point uses the official eight loose
    response transformations.
    """
    return score_ifeval_robustness(output_text, gold, mode="loose")
