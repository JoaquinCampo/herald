"""Prompt selection and provenance for the bounded IFEval acceptance run."""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_INPUTS_PATH = Path(
    "/Users/joaquincamponario/orca/workspaces/herald-v2/cero/results/"
    "recovered/quality-risk-v1/audit-2026-09-04/hidden_state_development/"
    "inputs.json"
)
DEFAULT_IFEVAL_ARROW_PATH = Path(
    "/Users/joaquincamponario/.cache/huggingface/datasets/google___if_eval/"
    "default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84/"
    "if_eval-train.arrow"
)
MANIFEST_VERSION = "herald_v3.engineering_prompt_manifest.v1"
_PROMPT_ID_RE = re.compile(r"^ifeval_(?P<key>[1-9][0-9]*)$")
_MESSAGE_RE = re.compile(
    r"<\|im_start\|>(?P<role>system|user|assistant)\n"
    r"(?P<content>.*?)<\|im_end\|>",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class EngineeringPrompt:
    """One source prompt with the official IFEval checker metadata."""

    prompt_id: str
    key: int
    fold: int
    fold_hash: str
    split_hash: str
    prompt_text: str
    user_prompt: str
    messages: tuple[dict[str, str], ...]
    instruction_id_list: tuple[str, ...]
    kwargs: tuple[dict[str, object], ...]
    prompt_text_sha256: str
    user_prompt_sha256: str

    @property
    def multi_instruction(self) -> bool:
        return len(self.instruction_id_list) > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "key": self.key,
            "fold": self.fold,
            "fold_hash": self.fold_hash,
            "split_hash": self.split_hash,
            "prompt_text": self.prompt_text,
            "prompt_text_utf8_sha256": self.prompt_text_sha256,
            "user_prompt": self.user_prompt,
            "user_prompt_utf8_sha256": self.user_prompt_sha256,
            "messages": [dict(message) for message in self.messages],
            "instruction_id_list": list(self.instruction_id_list),
            "kwargs": [_json_copy(item) for item in self.kwargs],
            "instruction_count": len(self.instruction_id_list),
            "multi_instruction": self.multi_instruction,
        }


@dataclass(frozen=True, slots=True)
class PromptManifest:
    """Selected prompts and hashes needed to reproduce their scoring."""

    source_path: str
    source_sha256: str
    source_bytes: int
    official_dataset_path: str
    official_dataset_sha256: str
    official_dataset_bytes: int
    official_dataset_rows: int
    selection_rule: str
    prompts: tuple[EngineeringPrompt, ...]

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_fingerprint=False),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(
        self, *, include_fingerprint: bool = True
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "manifest_version": MANIFEST_VERSION,
            "source": {
                "path": self.source_path,
                "sha256": self.source_sha256,
                "bytes": self.source_bytes,
            },
            "official_ifeval": {
                "path": self.official_dataset_path,
                "sha256": self.official_dataset_sha256,
                "bytes": self.official_dataset_bytes,
                "rows": self.official_dataset_rows,
            },
            "selection_rule": self.selection_rule,
            "selected_count": len(self.prompts),
            "multi_instruction_count": sum(
                prompt.multi_instruction for prompt in self.prompts
            ),
            "prompts": [prompt.to_dict() for prompt in self.prompts],
        }
        if include_fingerprint:
            result["manifest_sha256"] = self.fingerprint
        return result


def load_prompt_manifest(
    path: str | Path,
    *,
    arrow_path: str | Path | None = None,
    limit: int = 8,
    prompt_ids: Sequence[str] | None = None,
    dataset: object | None = None,
) -> PromptManifest:
    """Load a saved manifest or resolve one from exposed ``inputs.json``.

    A source inputs file is resolved against the pinned local Arrow cache and
    fails if any selected prompt's text or checker metadata differs. A saved
    manifest is self-contained, which lets the manifest created on the Mac be
    transferred to Orion without transferring the exposed dataset.
    """
    source = Path(path)
    raw_bytes = _read_bytes(source)
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"prompt file is not valid JSON: {source}"
        ) from error
    if not isinstance(document, Mapping):
        raise ValueError("prompt file must contain a JSON object")
    if document.get("manifest_version") == MANIFEST_VERSION:
        manifest = _manifest_from_dict(document)
        return _slice_manifest(manifest, limit, prompt_ids)
    return _build_manifest_from_inputs(
        document,
        raw_bytes,
        source=source,
        arrow_path=Path(arrow_path) if arrow_path is not None else None,
        limit=limit,
        prompt_ids=prompt_ids,
        dataset=dataset,
    )


def write_prompt_manifest(manifest: PromptManifest, path: str | Path) -> None:
    """Write a stable, portable JSON manifest for an acceptance launch."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            manifest.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_manifest_from_inputs(
    document: Mapping[str, object],
    raw_bytes: bytes,
    *,
    source: Path | None = None,
    arrow_path: Path | None,
    limit: int,
    prompt_ids: Sequence[str] | None,
    dataset: object | None,
) -> PromptManifest:
    if limit <= 0:
        raise ValueError("limit must be positive")
    raw_prompts = document.get("prompts")
    if not isinstance(raw_prompts, list):
        raise ValueError("inputs JSON must contain a prompts list")
    exposed = [
        _source_prompt(row, index)
        for index, row in enumerate(raw_prompts)
        if isinstance(row, Mapping) and row.get("task") == "ifeval"
    ]
    if not exposed:
        raise ValueError("inputs JSON contains no IFEval prompts")
    resolved_arrow = arrow_path or DEFAULT_IFEVAL_ARROW_PATH
    arrow_bytes = _read_bytes(resolved_arrow)
    arrow_rows = _load_arrow_rows(dataset, resolved_arrow)
    by_prompt = {str(row["prompt"]): row for row in arrow_rows}
    resolved_prompts: list[EngineeringPrompt] = []
    for row in exposed:
        if row["user_prompt"] not in by_prompt:
            raise ValueError(
                f"official IFEval row is missing for {row['prompt_id']}"
            )
        official = by_prompt[row["user_prompt"]]
        resolved_prompts.append(_merge_prompt(row, official))
    if prompt_ids is None:
        prompts = _select_prompts(resolved_prompts, limit)
    else:
        wanted = list(prompt_ids)
        if len(wanted) != len(set(wanted)):
            raise ValueError("prompt_ids contain duplicates")
        by_id = {prompt.prompt_id: prompt for prompt in resolved_prompts}
        missing = [
            prompt_id for prompt_id in wanted if prompt_id not in by_id
        ]
        if missing:
            raise ValueError(f"requested prompt IDs are missing: {missing}")
        prompts = [by_id[prompt_id] for prompt_id in wanted[:limit]]
    return PromptManifest(
        source_path=str(source or "<injected>"),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        source_bytes=len(raw_bytes),
        official_dataset_path=str(resolved_arrow),
        official_dataset_sha256=hashlib.sha256(arrow_bytes).hexdigest(),
        official_dataset_bytes=len(arrow_bytes),
        official_dataset_rows=len(arrow_rows),
        selection_rule=(
            "one multi-instruction prompt per exposed fold, then fill by "
            "sha256(prompt_id + NUL + exact prompt_text); preserve fold order"
        ),
        prompts=tuple(prompts),
    )


def _manifest_from_dict(document: Mapping[str, object]) -> PromptManifest:
    source = _mapping(document, "source")
    official = _mapping(document, "official_ifeval")
    raw_prompts = document.get("prompts")
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("saved prompt manifest must contain prompts")
    prompts = tuple(_prompt_from_dict(item) for item in raw_prompts)
    expected = document.get("manifest_sha256")
    manifest = PromptManifest(
        source_path=_string(source, "path"),
        source_sha256=_string(source, "sha256"),
        source_bytes=_integer(source, "bytes"),
        official_dataset_path=_string(official, "path"),
        official_dataset_sha256=_string(official, "sha256"),
        official_dataset_bytes=_integer(official, "bytes"),
        official_dataset_rows=_integer(official, "rows"),
        selection_rule=_string(document, "selection_rule"),
        prompts=prompts,
    )
    if expected is not None and expected != manifest.fingerprint:
        raise ValueError("saved prompt manifest fingerprint does not match")
    return manifest


def _slice_manifest(
    manifest: PromptManifest,
    limit: int,
    prompt_ids: Sequence[str] | None,
) -> PromptManifest:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if prompt_ids is None:
        selected = manifest.prompts[:limit]
    else:
        wanted = list(prompt_ids)
        if len(wanted) != len(set(wanted)):
            raise ValueError("prompt_ids contain duplicates")
        by_id = {prompt.prompt_id: prompt for prompt in manifest.prompts}
        missing = [
            prompt_id for prompt_id in wanted if prompt_id not in by_id
        ]
        if missing:
            raise ValueError(f"requested prompt IDs are missing: {missing}")
        selected = tuple(by_id[prompt_id] for prompt_id in wanted[:limit])
    return PromptManifest(
        source_path=manifest.source_path,
        source_sha256=manifest.source_sha256,
        source_bytes=manifest.source_bytes,
        official_dataset_path=manifest.official_dataset_path,
        official_dataset_sha256=manifest.official_dataset_sha256,
        official_dataset_bytes=manifest.official_dataset_bytes,
        official_dataset_rows=manifest.official_dataset_rows,
        selection_rule=manifest.selection_rule,
        prompts=tuple(selected),
    )


def _source_prompt(
    row: Mapping[str, object], index: int
) -> dict[str, object]:
    prompt_id = row.get("prompt_id")
    prompt_text = row.get("prompt_text")
    if not isinstance(prompt_id, str) or not isinstance(prompt_text, str):
        raise ValueError(f"source prompt row {index} is malformed")
    match = _PROMPT_ID_RE.fullmatch(prompt_id)
    if match is None:
        raise ValueError(f"malformed source IFEval ID: {prompt_id!r}")
    user_prompt, messages = _parse_chat_prompt(prompt_text)
    return {
        "prompt_id": prompt_id,
        "key": int(match.group("key")),
        "fold": _required_int(row, "fold", index),
        "fold_hash": _required_string(row, "fold_hash", index),
        "split_hash": _required_string(row, "split_hash", index),
        "prompt_text": prompt_text,
        "user_prompt": user_prompt,
        "messages": messages,
    }


def _merge_prompt(
    source: Mapping[str, object], official: Mapping[str, object]
) -> EngineeringPrompt:
    source_key = source.get("key")
    source_fold = source.get("fold")
    source_fold_hash = source.get("fold_hash")
    source_split_hash = source.get("split_hash")
    source_prompt_id = source.get("prompt_id")
    source_prompt_text = source.get("prompt_text")
    source_user_prompt = source.get("user_prompt")
    source_messages = source.get("messages")
    if (
        isinstance(source_key, bool)
        or not isinstance(source_key, int)
        or isinstance(source_fold, bool)
        or not isinstance(source_fold, int)
        or not isinstance(source_fold_hash, str)
        or not isinstance(source_split_hash, str)
        or not isinstance(source_prompt_id, str)
        or not isinstance(source_prompt_text, str)
        or not isinstance(source_user_prompt, str)
        or not isinstance(source_messages, tuple)
    ):
        raise ValueError("source IFEval prompt is malformed")
    if not all(isinstance(item, Mapping) for item in source_messages):
        raise ValueError("source IFEval prompt messages are malformed")
    key = official.get("key")
    prompt = official.get("prompt")
    ids = official.get("instruction_id_list")
    kwargs = official.get("kwargs")
    if isinstance(key, bool) or not isinstance(key, int):
        raise ValueError("official IFEval key is not integral")
    if not isinstance(prompt, str) or not isinstance(ids, list):
        raise ValueError("official IFEval row is malformed")
    if not isinstance(kwargs, list) or len(ids) != len(kwargs):
        raise ValueError(
            "official IFEval IDs and kwargs have different lengths"
        )
    if key != source_key or prompt != source_user_prompt:
        raise ValueError(
            f"official IFEval row does not match {source_prompt_id}"
        )
    normalized_ids: list[str] = []
    normalized_kwargs: list[dict[str, object]] = []
    for instruction_id, instruction_kwargs in zip(ids, kwargs, strict=True):
        if not isinstance(instruction_id, str):
            raise ValueError("official instruction ID is not a string")
        if not isinstance(instruction_kwargs, Mapping):
            raise ValueError("official instruction kwargs are not an object")
        normalized_ids.append(instruction_id)
        normalized_kwargs.append(dict(_json_copy(instruction_kwargs)))
    prompt_text = source_prompt_text
    user_prompt = source_user_prompt
    return EngineeringPrompt(
        prompt_id=source_prompt_id,
        key=source_key,
        fold=source_fold,
        fold_hash=source_fold_hash,
        split_hash=source_split_hash,
        prompt_text=prompt_text,
        user_prompt=user_prompt,
        messages=tuple(dict(item) for item in source_messages),
        instruction_id_list=tuple(normalized_ids),
        kwargs=tuple(normalized_kwargs),
        prompt_text_sha256=_sha256_text(prompt_text),
        user_prompt_sha256=_sha256_text(user_prompt),
    )


def _prompt_from_dict(item: object) -> EngineeringPrompt:
    if not isinstance(item, Mapping):
        raise ValueError("saved prompt entry is not an object")
    messages = item.get("messages")
    ids = item.get("instruction_id_list")
    kwargs = item.get("kwargs")
    if not isinstance(messages, list) or not isinstance(ids, list):
        raise ValueError("saved prompt entry is missing messages or IDs")
    if not isinstance(kwargs, list) or len(ids) != len(kwargs):
        raise ValueError("saved prompt IDs and kwargs have different lengths")
    prompt_text = _string(item, "prompt_text")
    user_prompt = _string(item, "user_prompt")
    if _sha256_text(prompt_text) != _string(item, "prompt_text_utf8_sha256"):
        raise ValueError("saved prompt text hash does not match")
    if _sha256_text(user_prompt) != _string(item, "user_prompt_utf8_sha256"):
        raise ValueError("saved user prompt hash does not match")
    normalized_messages = tuple(
        {
            "role": _string(message, "role"),
            "content": _string(message, "content"),
        }
        for message in messages
        if isinstance(message, Mapping)
    )
    if len(normalized_messages) != len(messages):
        raise ValueError("saved prompt messages are malformed")
    normalized_kwargs = tuple(
        dict(_json_copy(value))
        for value in kwargs
        if isinstance(value, Mapping)
    )
    if len(normalized_kwargs) != len(kwargs):
        raise ValueError("saved prompt kwargs are malformed")
    return EngineeringPrompt(
        prompt_id=_string(item, "prompt_id"),
        key=_integer(item, "key"),
        fold=_integer(item, "fold"),
        fold_hash=_string(item, "fold_hash"),
        split_hash=_string(item, "split_hash"),
        prompt_text=prompt_text,
        user_prompt=user_prompt,
        messages=normalized_messages,
        instruction_id_list=tuple(
            _string({"value": value}, "value") for value in ids
        ),
        kwargs=normalized_kwargs,
        prompt_text_sha256=_string(item, "prompt_text_utf8_sha256"),
        user_prompt_sha256=_string(item, "user_prompt_utf8_sha256"),
    )


def _select_prompts(
    rows: list[EngineeringPrompt], limit: int
) -> list[EngineeringPrompt]:
    ranked = sorted(rows, key=_rank_key)
    selected: list[EngineeringPrompt] = []
    for fold in sorted({row.fold for row in rows}):
        candidates = [row for row in ranked if row.fold == fold]
        multi = [row for row in candidates if row.multi_instruction]
        selected.append((multi or candidates)[0])
        if len(selected) == limit:
            return selected
    for row in ranked:
        if row not in selected:
            selected.append(row)
            if len(selected) == limit:
                break
    return selected


def _rank_key(row: EngineeringPrompt) -> tuple[bool, str, str]:
    return (
        not row.multi_instruction,
        _sha256_text(row.prompt_id + "\0" + row.prompt_text),
        row.prompt_id,
    )


def _parse_chat_prompt(
    prompt_text: str,
) -> tuple[str, tuple[dict[str, str], ...]]:
    matches = list(_MESSAGE_RE.finditer(prompt_text))
    if not matches:
        raise ValueError("source prompt is missing chat-template messages")
    messages = tuple(
        {"role": match.group("role"), "content": match.group("content")}
        for match in matches
    )
    user_messages = [
        message for message in messages if message["role"] == "user"
    ]
    if len(user_messages) != 1:
        raise ValueError(
            "source prompt must contain exactly one user message"
        )
    assistant_suffix = "<|im_start|>assistant\n"
    if not prompt_text.endswith(assistant_suffix):
        raise ValueError(
            "source prompt is missing the assistant generation marker"
        )
    return user_messages[0]["content"], messages


def _load_arrow_rows(
    dataset: object | None, path: Path
) -> list[Mapping[str, object]]:
    if dataset is None:
        try:
            from datasets import Dataset  # type: ignore[import-untyped]
        except ImportError as error:
            raise RuntimeError(
                "datasets is required to resolve the official IFEval "
                "Arrow cache"
            ) from error
        try:
            dataset = Dataset.from_file(str(path))
        except Exception as error:
            raise RuntimeError(
                f"cannot load official IFEval Arrow cache: {path}"
            ) from error
    rows: list[Mapping[str, object]] = []
    try:
        iterator: Iterable[object] = dataset  # type: ignore[assignment]
    except TypeError as error:
        raise ValueError("official IFEval dataset is not iterable") from error
    for row in iterator:
        if not isinstance(row, Mapping):
            raise ValueError("official IFEval dataset row is malformed")
        rows.append(row)
    if not rows:
        raise ValueError("official IFEval dataset is empty")
    return rows


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise FileNotFoundError(
            f"required prompt resource is missing: {path}"
        ) from error


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_copy(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_copy(item) for item in value]
    return value


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"manifest field {key!r} must be an object")
    return result


def _string(value: Mapping[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise ValueError(f"manifest field {key!r} must be a string")
    return result


def _integer(value: Mapping[str, object], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"manifest field {key!r} must be an integer")
    return result


def _required_string(row: Mapping[str, object], key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"source prompt row {index} has invalid {key}")
    return value


def _required_int(row: Mapping[str, object], key: str, index: int) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"source prompt row {index} has invalid {key}")
    return value
