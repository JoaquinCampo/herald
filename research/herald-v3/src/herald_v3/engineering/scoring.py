"""Strict and loose official-checker scoring for the engineering slice."""

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast, overload

from herald_v3.engineering._ifeval_vendor import instructions_registry


class ScoringResourceError(RuntimeError):
    """Raised when an official IFEval dependency or corpus is absent."""


@dataclass(frozen=True, slots=True)
class IFEvalScores:
    """Instruction-level scores and pass vectors for one response."""

    loose: float
    strict: float
    loose_pass: tuple[bool, ...]
    strict_pass: tuple[bool, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "loose": self.loose,
            "strict": self.strict,
            "loose_pass": list(self.loose_pass),
            "strict_pass": list(self.strict_pass),
            "instruction_count": len(self.strict_pass),
        }


@dataclass(frozen=True, slots=True)
class SignedScorePair:
    """Reference/action scores and signed quality differences."""

    reference: IFEvalScores
    action: IFEvalScores

    @property
    def d_loose(self) -> float:
        return self.reference.loose - self.action.loose

    @property
    def d_strict(self) -> float:
        return self.reference.strict - self.action.strict

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference.to_dict(),
            "action": self.action.to_dict(),
            "d_loose": self.d_loose,
            "d_strict": self.d_strict,
        }


IFEvalMode = Literal["loose", "strict", "both"]


def check_ifeval_resources() -> dict[str, object]:
    """Require every dependency/corpus used by the vendored checkers.

    The engineering run must retain failed cases as failures. It therefore
    does not fall back to partial scoring when NLTK or a checker dependency is
    unavailable.
    """
    missing_modules: list[str] = []
    for module_name in ("absl", "immutabledict", "langdetect", "nltk"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_modules.append(module_name)
    if missing_modules:
        raise ScoringResourceError(
            "missing IFEval scorer dependencies: "
            + ", ".join(missing_modules)
        )

    import nltk  # type: ignore[import-untyped]

    missing_resources: list[str] = []
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            missing_resources.append(resource)
    if missing_resources:
        raise ScoringResourceError(
            "missing NLTK resources for official IFEval checkers: "
            + ", ".join(missing_resources)
        )

    import langdetect  # type: ignore[import-untyped]

    langdetect.DetectorFactory.seed = 0
    return {
        "modules": {
            name: _module_version(name)
            for name in ("absl", "immutabledict", "langdetect", "nltk")
        },
        "nltk_resources": ["tokenizers/punkt", "tokenizers/punkt_tab"],
    }


def score_ifeval_output(
    output_text: str,
    *,
    prompt: str,
    instruction_id_list: Sequence[str],
    kwargs: Sequence[Mapping[str, object]],
    check_resources: bool = True,
) -> IFEvalScores:
    """Score one output with vendored official IFEval checkers."""
    if not isinstance(output_text, str):
        raise TypeError("output_text must be a string")
    if check_resources:
        check_ifeval_resources()
    ids = tuple(instruction_id_list)
    checker_kwargs = tuple(dict(item) for item in kwargs)
    if not ids:
        raise ValueError("IFEval prompt has no instructions")
    if len(ids) != len(checker_kwargs):
        raise ValueError(
            "IFEval instruction IDs and kwargs have different lengths"
        )

    checkers = tuple(
        _build_checker(instruction_id, checker_kwargs[index], prompt)
        for index, instruction_id in enumerate(ids)
    )
    strict_pass = tuple(
        bool(output_text.strip() and checker(output_text))
        for checker in checkers
    )
    transforms = _loose_transforms(output_text)
    loose_pass: list[bool] = []
    for checker, strict in zip(checkers, strict_pass, strict=True):
        passed = strict
        if not passed:
            passed = any(
                bool(variant.strip() and checker(variant))
                for variant in transforms[1:]
            )
        loose_pass.append(passed)
    return IFEvalScores(
        loose=sum(loose_pass) / len(loose_pass),
        strict=sum(strict_pass) / len(strict_pass),
        loose_pass=tuple(loose_pass),
        strict_pass=strict_pass,
    )


def score_ifeval_gold(
    output_text: str,
    gold: Mapping[str, object],
    *,
    check_resources: bool = True,
) -> IFEvalScores:
    """Score an output from a JSON-like IFEval gold metadata mapping."""
    prompt = gold.get("prompt")
    ids = gold.get("instruction_id_list")
    kwargs = gold.get("kwargs")
    if not isinstance(prompt, str):
        raise ValueError("IFEval gold metadata is missing string prompt")
    if not isinstance(ids, Sequence) or isinstance(ids, str):
        raise ValueError("IFEval gold metadata is missing instruction IDs")
    if not isinstance(kwargs, Sequence) or isinstance(kwargs, str):
        raise ValueError("IFEval gold metadata is missing instruction kwargs")
    if not all(isinstance(item, str) for item in ids):
        raise ValueError("IFEval instruction IDs must be strings")
    if not all(isinstance(item, Mapping) for item in kwargs):
        raise ValueError("IFEval instruction kwargs must be objects")
    return score_ifeval_output(
        output_text,
        prompt=prompt,
        instruction_id_list=cast(Sequence[str], ids),
        kwargs=cast(Sequence[Mapping[str, object]], kwargs),
        check_resources=check_resources,
    )


def score_pair(
    reference_text: str,
    action_text: str,
    gold: Mapping[str, object],
) -> SignedScorePair:
    """Score reference/action outputs and retain signed ``q0 - qa``."""
    check_ifeval_resources()
    reference = score_ifeval_gold(reference_text, gold, check_resources=False)
    action = score_ifeval_gold(action_text, gold, check_resources=False)
    return SignedScorePair(reference=reference, action=action)


@overload
def score_ifeval_robustness(
    output_text: str,
    gold: Mapping[str, object],
    *,
    mode: Literal["loose", "strict"],
) -> float: ...


@overload
def score_ifeval_robustness(
    output_text: str,
    gold: Mapping[str, object],
    *,
    mode: Literal["both"] = "both",
) -> IFEvalScores: ...


def score_ifeval_robustness(
    output_text: str,
    gold: Mapping[str, object],
    *,
    mode: IFEvalMode = "both",
) -> float | IFEvalScores:
    """Compatibility API returning the requested official score mode."""
    if mode not in ("loose", "strict", "both"):
        raise ValueError(
            f"invalid IFEval scoring mode {mode!r}; "
            "expected loose, strict, or both"
        )
    scores = score_ifeval_gold(output_text, gold)
    if mode == "loose":
        return scores.loose
    if mode == "strict":
        return scores.strict
    return scores


def _build_checker(
    instruction_id: str,
    instruction_kwargs: Mapping[str, object],
    prompt: str,
) -> Callable[[str], object]:
    try:
        instruction_cls = instructions_registry.INSTRUCTION_DICT[
            instruction_id
        ]
    except KeyError as error:
        raise ValueError(
            f"unsupported official IFEval instruction: {instruction_id}"
        ) from error
    instruction = instruction_cls(instruction_id)  # type: ignore[no-untyped-call]
    raw_kwargs = {
        key: value
        for key, value in instruction_kwargs.items()
        if value is not None
    }
    instruction.build_description(**raw_kwargs)  # type: ignore[no-untyped-call]
    args = instruction.get_instruction_args()  # type: ignore[no-untyped-call]
    if isinstance(args, Mapping) and "prompt" in args:
        instruction.build_description(prompt=prompt)  # type: ignore[no-untyped-call]
    return cast(Callable[[str], object], instruction.check_following)


def _loose_transforms(response: str) -> tuple[str, ...]:
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    no_star = response.replace("*", "")
    return (
        response,
        no_star,
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    )


def _module_version(module_name: str) -> str | None:
    module = importlib.import_module(module_name)
    return cast(str | None, getattr(module, "__version__", None))
