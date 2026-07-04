"""IFEval (instruction-following eval) task for HERALD.

Implements instruction-level loose accuracy as described in:
    Zhou et al. (2023), "Instruction-Following Evaluation for Large
    Language Models", https://arxiv.org/abs/2311.07911

Variant implemented
-------------------
Instruction-level LOOSE accuracy: for each instruction in a prompt,
the instruction is deemed followed if *any* of eight response
transformations passes the official checker. The eight transforms are:

    1. original response
    2. response with all ``*`` removed
    3. response with first line removed
    4. response with last line removed
    5. response with first and last lines removed
    6. (3) with ``*`` removed
    7. (4) with ``*`` removed
    8. (5) with ``*`` removed

The score for one example is the fraction of individual instructions
that are followed (instruction-level), ranging in [0, 1].

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

from typing import cast

import langdetect

from herald._ifeval_vendor import instructions_registry
from herald.tasks import PromptRecord

# Set a fixed seed so langdetect checks are deterministic.
langdetect.DetectorFactory.seed = 0


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
    from datasets import load_dataset  # type: ignore[import-untyped]

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


def score_ifeval(
    output_text: str,
    gold: dict[str, object],
) -> float:
    """Return the instruction-level loose accuracy for one example.

    For each instruction in ``gold["instruction_id_list"]``, the
    instruction is considered followed if at least one of the eight
    loose response transformations passes the official checker.
    The return value is the fraction of instructions followed.

    Args:
        output_text: The model's full generated text.
        gold: The gold dict stored by ``load_ifeval``, containing:
            - ``"prompt"``: the original prompt string
            - ``"instruction_id_list"``: list of instruction-id strings
            - ``"kwargs"``: list of per-instruction kwarg dicts (may
              contain ``None``-padded keys from the HuggingFace dataset)

    Returns:
        Fraction of instructions followed, in [0.0, 1.0]. Returns 0.0
        for an empty instruction list.
    """
    instruction_ids = cast(list[str], gold["instruction_id_list"])
    kwargs_list = cast(list[dict[str, object]], gold["kwargs"])
    prompt_text = str(gold["prompt"])

    if not instruction_ids:
        return 0.0

    transforms = _loose_transforms(output_text)
    followed: list[bool] = []

    for idx, instruction_id in enumerate(instruction_ids):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[
            instruction_id
        ]
        # Vendor classes are untyped; suppress the cascade of
        # no-untyped-call errors from the ignore_errors override.
        instruction = instruction_cls(  # type: ignore[no-untyped-call]
            instruction_id
        )

        # Filter None-padded kwargs that HuggingFace may inject when
        # the dataset schema is uniform across examples.
        raw_kw: dict[str, object] = {}
        if idx < len(kwargs_list):
            raw_kw = {
                k: v for k, v in kwargs_list[idx].items() if v is not None
            }

        instruction.build_description(  # type: ignore[no-untyped-call]
            **raw_kw
        )

        # Some instructions (e.g. repeat_prompt) need the prompt text
        # passed explicitly after the initial build.
        args = instruction.get_instruction_args()  # type: ignore[no-untyped-call]
        if args and "prompt" in args:
            instruction.build_description(  # type: ignore[no-untyped-call]
                prompt=prompt_text
            )

        is_followed = False
        for variant in transforms:
            if variant.strip() and bool(
                instruction.check_following(  # type: ignore[no-untyped-call]
                    variant
                )
            ):
                is_followed = True
                break
        followed.append(is_followed)

    return sum(followed) / len(followed)
