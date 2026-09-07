"""Synthetic retrieval-head discovery for the bounded Phase A assay.

The implementation keeps the released needle, question, answer, and
haystack source data immutable while making one explicit token-boundary
adaptation: a newline is inserted immediately before the released answer
inside each needle.  Every case then derives its authoritative answer token
sequence from offsets in the fully rendered chat prompt.
"""

import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

REPOSITORY = "https://github.com/nightdessert/Retrieval_Head"
PINNED_COMMIT = "3ac171a6f71ce7ef1cda57d4215c390fb6ab51f2"
SOURCE_SCHEMA_VERSION = "herald_v3.retrieval_discovery_source.v1"
RESULT_SCHEMA_VERSION = "herald_v3.retrieval_discovery.v1"
DEFAULT_SOURCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "retrieval-discovery-v1"
    / "source"
)
MAX_SOURCE_BYTES = 20_000_000
CONTEXT_LENGTHS = (1024, 4096)
PANELS = {
    "discovery": (20.0, 80.0),
    "validation": (35.0, 65.0),
}
TRIPLE_COUNT = 3
MAX_GENERATION_TOKENS = 50
SUCCESS_THRESHOLD = 0.1
MIN_SUCCESSFUL_CASES = 8
MAX_SELECTED_CELLS = 8
SEED = 0
DISCOVERY_DEADLINE_SECONDS = 1800.0
EXPECTED_LAYERS = 28
EXPECTED_QUERY_HEADS = 28
EXPECTED_KV_HEADS = 4
EXPECTED_GQA_GROUPS = 7
EXPECTED_HIDDEN_SIZE = 3584
EXPECTED_HEAD_DIM = 128
EXPECTED_QUERY_PROJECTION = 3584
EXPECTED_KV_PROJECTION = 512


class RetrievalDiscoveryError(RuntimeError):
    """Raised when a synthetic discovery input or measurement is invalid."""


class DiscoveryDeadlineExceeded(RetrievalDiscoveryError):
    """Raised when the bounded assay reaches its operational time cap."""


class DiscoveryRuntime:
    """Runtime boundary for CUDA execution and CPU test injection."""

    def __init__(
        self,
        *,
        require_cuda: bool = True,
        deadline_seconds: float = DISCOVERY_DEADLINE_SECONDS,
        clock: Any = time.perf_counter,
    ) -> None:
        self.require_cuda = require_cuda
        self.deadline_seconds = deadline_seconds
        self.clock = clock
        self._deadline: float | None = None

    def seed(self) -> None:
        random.seed(SEED)
        try:
            import numpy as np
        except ImportError as error:
            raise RetrievalDiscoveryError(
                "NumPy is required for deterministic discovery"
            ) from error
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def validate_model(self, model: Any) -> torch.device:
        try:
            device = cast(torch.device, next(model.parameters()).device)
        except (AttributeError, StopIteration) as error:
            raise RetrievalDiscoveryError(
                "model has no parameters"
            ) from error
        if self.require_cuda and device.type != "cuda":
            raise RetrievalDiscoveryError(
                "production discovery requires a CUDA model"
            )
        if device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RetrievalDiscoveryError("CUDA BF16 is not supported")
            dtypes = {parameter.dtype for parameter in model.parameters()}
            if dtypes != {torch.bfloat16}:
                raise RetrievalDiscoveryError(
                    "production discovery requires BF16 model parameters"
                )
        return device

    def start(self) -> None:
        self._deadline = self.clock() + self.deadline_seconds

    def check_deadline(self) -> None:
        if self._deadline is not None and self.clock() >= self._deadline:
            raise DiscoveryDeadlineExceeded(
                "discovery deadline reached before the next model step"
            )

    def synchronize(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def reset_peak_memory(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def peak_memory(
        self, device: torch.device
    ) -> tuple[int | None, int | None]:
        if device.type != "cuda":
            return None, None
        return (
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        )


@dataclass(frozen=True, slots=True)
class RetrievalTriple:
    """One released needle, question, and source answer triple."""

    triple_index: int
    needle: str
    question: str
    answer: str
    source_part: str


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One deterministic synthetic discovery or validation prompt."""

    case_id: str
    panel: str
    triple_index: int
    source_part: str
    original_needle: str
    transformed_needle: str
    answer: str
    context_length: int
    depth_percent: float
    prompt_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]
    answer_start: int
    answer_end: int
    rendered_prompt_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "panel": self.panel,
            "triple_index": self.triple_index,
            "source_part": self.source_part,
            "original_needle": self.original_needle,
            "transformed_needle": self.transformed_needle,
            "answer": self.answer,
            "input_adaptation": (
                "newline_before_answer_with_unique_casefold_span"
            ),
            "context_length": self.context_length,
            "depth_percent": self.depth_percent,
            "prompt_token_count": len(self.prompt_ids),
            "prompt_ids": list(self.prompt_ids),
            "answer_token_ids": list(self.answer_token_ids),
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
            "answer_token_count": len(self.answer_token_ids),
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
        }


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Compact output and per-query-head evidence for one case."""

    generated_token_ids: tuple[int, ...]
    stop_reason: str
    success: bool
    answer_match_count: int
    head_copy_fraction: dict[str, float]
    elapsed_seconds: float
    prefill_seconds: float
    decode_forward_seconds: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_token_ids": list(self.generated_token_ids),
            "stop_reason": self.stop_reason,
            "success": self.success,
            "answer_match_count": self.answer_match_count,
            "head_copy_fraction": dict(self.head_copy_fraction),
            "elapsed_seconds": self.elapsed_seconds,
            "prefill_seconds": self.prefill_seconds,
            "decode_forward_seconds": self.decode_forward_seconds,
            "forward_seconds": self.prefill_seconds
            + self.decode_forward_seconds,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }


def verify_source_manifest(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
) -> dict[str, object]:
    """Verify the pinned public source files and return their manifest."""
    root = Path(source_root).resolve()
    manifest_path = root.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RetrievalDiscoveryError(
            f"cannot read source manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise RetrievalDiscoveryError("source manifest must be an object")
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise RetrievalDiscoveryError("source manifest schema is invalid")
    if manifest.get("repository") != REPOSITORY:
        raise RetrievalDiscoveryError("source repository is not pinned")
    if manifest.get("commit") != PINNED_COMMIT:
        raise RetrievalDiscoveryError("source commit is not pinned")
    selected = manifest.get("selected_files")
    if not isinstance(selected, list) or not selected:
        raise RetrievalDiscoveryError("source file manifest is empty")
    total_bytes = 0
    seen: set[str] = set()
    selected_by_path: dict[str, Mapping[str, object]] = {}
    for item in selected:
        if not isinstance(item, Mapping):
            raise RetrievalDiscoveryError(
                "source file manifest entry malformed"
            )
        relative = item.get("path")
        expected_size = item.get("bytes")
        expected_hash = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not _safe_relative_path(relative)
            or relative in seen
            or not isinstance(expected_size, int)
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise RetrievalDiscoveryError(
                "source file manifest entry invalid"
            )
        seen.add(relative)
        selected_by_path[relative] = item
        path = root / relative
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise RetrievalDiscoveryError(
                f"source file is missing: {relative}"
            ) from error
        actual_hash = _sha256(raw)
        if len(raw) != expected_size or actual_hash != expected_hash:
            raise RetrievalDiscoveryError(
                f"source file hash mismatch: {relative}"
            )
        total_bytes += len(raw)
    if total_bytes != manifest.get("total_bytes"):
        raise RetrievalDiscoveryError("source total byte count mismatch")
    if total_bytes > MAX_SOURCE_BYTES:
        raise RetrievalDiscoveryError("source preflight exceeds 20 MB")

    parts = manifest.get("part_concatenation")
    if not isinstance(parts, Mapping):
        raise RetrievalDiscoveryError(
            "part concatenation manifest is missing"
        )
    for part in ("part1", "part2", "part3"):
        part_items = [
            (name, item)
            for name, item in selected_by_path.items()
            if name.startswith(f"haystack_for_detect/{part}/")
        ]
        if not part_items:
            raise RetrievalDiscoveryError(f"source part is empty: {part}")
        concatenated = b"".join(
            (root / name).read_bytes()
            for name, _ in sorted(part_items, key=lambda value: value[0])
        )
        expected_part = parts.get(part)
        if not isinstance(expected_part, Mapping):
            raise RetrievalDiscoveryError(f"part hash is missing: {part}")
        if expected_part.get("bytes") != len(
            concatenated
        ) or expected_part.get("sha256") != _sha256(concatenated):
            raise RetrievalDiscoveryError(
                f"part concatenation mismatch: {part}"
            )
    return manifest


def load_triples(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
) -> tuple[RetrievalTriple, ...]:
    """Load the three released triples after source verification."""
    root = Path(source_root).resolve()
    manifest = verify_source_manifest(root)
    path = root / "haystack_for_detect" / "needles.jsonl"
    triples: list[RetrievalTriple] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RetrievalDiscoveryError(
            "cannot read released needles"
        ) from error
    for _index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise RetrievalDiscoveryError(
                "released needle JSON is invalid"
            ) from error
        if not isinstance(item, Mapping):
            raise RetrievalDiscoveryError("released needle entry is invalid")
        values = (
            item.get("needle"),
            item.get("question"),
            item.get("real_needle"),
        )
        if not all(isinstance(value, str) and value for value in values):
            raise RetrievalDiscoveryError(
                "released needle fields are incomplete"
            )
        triples.append(
            RetrievalTriple(
                triple_index=len(triples),
                needle=cast(str, item["needle"]),
                question=cast(str, item["question"]),
                answer=cast(str, item["real_needle"]),
                source_part=f"part{len(triples) + 1}",
            )
        )
    if len(triples) != TRIPLE_COUNT:
        raise RetrievalDiscoveryError(
            "the pinned source must contain three triples"
        )
    adaptation = manifest.get("input_adaptation")
    if not isinstance(adaptation, Mapping):
        raise RetrievalDiscoveryError("input adaptation manifest is missing")
    if adaptation.get("kind") != (
        "newline_before_answer_with_unique_casefold_span"
    ):
        raise RetrievalDiscoveryError("input adaptation manifest is invalid")
    entries = adaptation.get("triples")
    if not isinstance(entries, list) or len(entries) != TRIPLE_COUNT:
        raise RetrievalDiscoveryError("input adaptation triples are invalid")
    for triple, entry in zip(triples, entries, strict=True):
        if not isinstance(entry, Mapping):
            raise RetrievalDiscoveryError("input adaptation entry is invalid")
        if (
            entry.get("triple_index") != triple.triple_index
            or entry.get("original_needle") != triple.needle
            or entry.get("answer") != triple.answer
            or entry.get("transformed_needle") != _adapted_needle(triple)
        ):
            raise RetrievalDiscoveryError(
                "input adaptation does not match released triple"
            )
    return tuple(triples)


def build_cases(
    tokenizer: Any,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
) -> tuple[RetrievalCase, ...]:
    """Build the frozen 12-case discovery and 12-case validation panels."""
    DiscoveryRuntime(require_cuda=False).seed()
    root = Path(source_root).resolve()
    manifest = verify_source_manifest(root)
    triples = load_triples(root)
    part_texts = {
        part: _read_part_text(root, part, manifest)
        for part in ("part1", "part2", "part3")
    }
    cases: list[RetrievalCase] = []
    for panel, depths in PANELS.items():
        for triple in triples:
            for context_length in CONTEXT_LENGTHS:
                for depth in depths:
                    context = _build_context(
                        tokenizer,
                        part_texts[triple.source_part],
                        triple,
                        context_length,
                        depth,
                    )
                    cases.append(
                        _render_case(
                            tokenizer,
                            triple,
                            panel,
                            context_length,
                            depth,
                            context,
                        )
                    )
    if len(cases) != 24:
        raise RetrievalDiscoveryError("synthetic case grid is not 24 cases")
    return tuple(cases)


def _read_part_text(
    root: Path, part: str, manifest: Mapping[str, object]
) -> str:
    selected = manifest.get("selected_files")
    if not isinstance(selected, list):
        raise RetrievalDiscoveryError("source file manifest is malformed")
    names = sorted(
        cast(str, item["path"])
        for item in selected
        if isinstance(item, Mapping)
        and isinstance(item.get("path"), str)
        and cast(str, item["path"]).startswith(f"haystack_for_detect/{part}/")
    )
    try:
        return b"".join((root / name).read_bytes() for name in names).decode(
            "utf-8"
        )
    except (OSError, UnicodeDecodeError) as error:
        raise RetrievalDiscoveryError(
            f"source part is not valid UTF-8: {part}"
        ) from error


def _build_context(
    tokenizer: Any,
    part_text: str,
    triple: RetrievalTriple,
    context_length: int,
    depth_percent: float,
) -> str:
    adapted_needle = _adapted_needle(triple)
    needle_ids = _encode(tokenizer, adapted_needle)
    source_ids = _encode(tokenizer, part_text)
    if len(needle_ids) >= context_length:
        raise RetrievalDiscoveryError(
            "needle is longer than the context target"
        )
    for extra in range(-8, 9):
        background_count = context_length - len(needle_ids) + extra
        if background_count <= 0 or background_count > len(source_ids):
            continue
        background = source_ids[:background_count]
        desired = int(background_count * depth_percent / 100.0)
        for delta in sorted(
            range(-64, 65), key=lambda value: (abs(value), value)
        ):
            insertion = desired + delta
            if not 0 <= insertion <= background_count:
                continue
            before = _decode(tokenizer, background[:insertion])
            after = _decode(tokenizer, background[insertion:])
            candidate = before + "\n" + adapted_needle + " \n" + after
            if len(_encode(tokenizer, candidate)) == context_length:
                return candidate
    raise RetrievalDiscoveryError(
        f"cannot realize exact context length {context_length} for triple "
        f"{triple.triple_index} at depth {depth_percent}"
    )


def _adapted_needle(triple: RetrievalTriple) -> str:
    answer_at = triple.needle.find(triple.answer)
    answer_length = len(triple.answer)
    if (
        answer_at >= 0
        and triple.needle.find(triple.answer, answer_at + answer_length) >= 0
    ):
        raise RetrievalDiscoveryError(
            "released answer occurrence is not unique for triple "
            f"{triple.triple_index}"
        )
    if answer_at < 0:
        folded_needle = triple.needle.casefold()
        folded_answer = triple.answer.casefold()
        answer_at = folded_needle.find(folded_answer)
        if (
            answer_at < 0
            or folded_needle.find(folded_answer, answer_at + answer_length)
            >= 0
        ):
            raise RetrievalDiscoveryError(
                "released answer occurrence is not unique for triple "
                f"{triple.triple_index}"
            )
    return (
        triple.needle[:answer_at]
        + "\n"
        + triple.answer
        + triple.needle[answer_at + answer_length :]
    )


def _render_case(
    tokenizer: Any,
    triple: RetrievalTriple,
    panel: str,
    context_length: int,
    depth_percent: float,
    context: str,
) -> RetrievalCase:
    content = f"{context}\nQuestion: {triple.question}\nAnswer:"
    messages = [{"role": "user", "content": content}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_ids = _field_list(encoded, "input_ids")
        offsets = _field_offsets(encoded)
    except (AttributeError, TypeError, ValueError, KeyError) as error:
        raise RetrievalDiscoveryError(
            "tokenizer cannot provide prompt offsets"
        ) from error
    try:
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        templated_ids = _field_list(templated, "input_ids")
    except (AttributeError, TypeError, ValueError, KeyError) as error:
        raise RetrievalDiscoveryError(
            "chat template tokenization failed"
        ) from error
    if templated_ids != prompt_ids:
        raise RetrievalDiscoveryError(
            "chat template IDs differ from full prompt IDs"
        )
    answer_start_char = rendered.find(triple.answer)
    if (
        answer_start_char < 0
        or rendered.find(triple.answer, answer_start_char + 1) >= 0
    ):
        raise RetrievalDiscoveryError(
            "rendered answer span is missing or ambiguous"
        )
    answer_end_char = answer_start_char + len(triple.answer)
    answer_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if start >= answer_start_char and end <= answer_end_char
    ]
    if not answer_indices or answer_indices != list(
        range(answer_indices[0], answer_indices[-1] + 1)
    ):
        raise RetrievalDiscoveryError(
            "rendered answer token span is not contiguous"
        )
    first, last = answer_indices[0], answer_indices[-1]
    if (
        offsets[first][0] != answer_start_char
        or offsets[last][1] != answer_end_char
    ):
        raise RetrievalDiscoveryError(
            "rendered answer token offsets do not round trip"
        )
    answer_ids = tuple(prompt_ids[first : last + 1])
    if _decode(tokenizer, answer_ids) != triple.answer:
        raise RetrievalDiscoveryError(
            "rendered answer token IDs do not round trip"
        )
    case_id = (
        f"{panel}_t{triple.triple_index + 1}_l{context_length}"
        f"_d{depth_percent:g}"
    )
    return RetrievalCase(
        case_id=case_id,
        panel=panel,
        triple_index=triple.triple_index,
        source_part=triple.source_part,
        original_needle=triple.needle,
        transformed_needle=_adapted_needle(triple),
        answer=triple.answer,
        context_length=context_length,
        depth_percent=depth_percent,
        prompt_ids=tuple(prompt_ids),
        answer_token_ids=answer_ids,
        answer_start=first,
        answer_end=last + 1,
        rendered_prompt_sha256=_sha256_text(rendered),
    )


def query_to_kv_mapping(query_heads: int, kv_heads: int) -> tuple[int, ...]:
    """Verify contiguous repeat-kv ordering and return q-head mappings."""
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise RetrievalDiscoveryError("GQA head counts are incompatible")
    groups = query_heads // kv_heads
    source = torch.arange(kv_heads, dtype=torch.long).view(1, kv_heads, 1, 1)
    try:
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    except ImportError as error:
        raise RetrievalDiscoveryError(
            "Qwen2 repeat_kv is unavailable"
        ) from error
    observed = repeat_kv(source, groups)
    expected = torch.repeat_interleave(source, groups, dim=1)
    if not torch.equal(observed, expected):
        raise RetrievalDiscoveryError(
            "Qwen2 repeat_kv ordering is not contiguous"
        )
    return tuple(query // groups for query in range(query_heads))


def verify_model_gqa_mapping(model: Any) -> tuple[int, ...]:
    """Verify the loaded model's Qwen2 GQA dimensions and repeat factor."""
    config = getattr(model, "config", None)
    if config is None:
        raise RetrievalDiscoveryError("model config is missing")
    if getattr(config, "model_type", None) != "qwen2":
        raise RetrievalDiscoveryError("model must use the Qwen2 architecture")
    if int(config.num_hidden_layers) != EXPECTED_LAYERS:
        raise RetrievalDiscoveryError("model must have exactly 28 layers")
    if int(getattr(config, "hidden_size", -1)) != EXPECTED_HIDDEN_SIZE:
        raise RetrievalDiscoveryError("model must have hidden size 3584")
    query_heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", query_heads))
    if query_heads != EXPECTED_QUERY_HEADS or kv_heads != EXPECTED_KV_HEADS:
        raise RetrievalDiscoveryError(
            "model must have exactly 28 query heads and four KV heads"
        )
    configured_head_dim = getattr(config, "head_dim", None)
    if (
        configured_head_dim is not None
        and int(configured_head_dim) != EXPECTED_HEAD_DIM
    ):
        raise RetrievalDiscoveryError("model must have head dimension 128")
    if EXPECTED_HIDDEN_SIZE // query_heads != EXPECTED_HEAD_DIM:
        raise RetrievalDiscoveryError(
            "model config has an incompatible head dimension"
        )
    mapping = query_to_kv_mapping(query_heads, kv_heads)
    layers = getattr(getattr(model, "model", None), "layers", ())
    if len(layers) != EXPECTED_LAYERS:
        raise RetrievalDiscoveryError("model layer list is not exactly 28")
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RetrievalDiscoveryError(
                "every Qwen2 attention layer must use 28/4/7 GQA"
            )
        try:
            groups = int(attention.num_key_value_groups)
        except (AttributeError, TypeError, ValueError) as error:
            raise RetrievalDiscoveryError(
                "every Qwen2 attention layer must expose seven KV groups"
            ) from error
        if groups != EXPECTED_GQA_GROUPS:
            raise RetrievalDiscoveryError(
                "every Qwen2 attention layer must use 28/4/7 GQA"
            )
        layer_head_dim = getattr(attention, "head_dim", None)
        if (
            layer_head_dim is not None
            and int(layer_head_dim) != EXPECTED_HEAD_DIM
        ):
            raise RetrievalDiscoveryError(
                "every Qwen2 attention layer must use head dimension 128"
            )
        expected_projection_sizes = {
            "q_proj": EXPECTED_QUERY_PROJECTION,
            "k_proj": EXPECTED_KV_PROJECTION,
            "v_proj": EXPECTED_KV_PROJECTION,
            "o_proj": EXPECTED_QUERY_PROJECTION,
        }
        for (
            projection_name,
            expected_size,
        ) in expected_projection_sizes.items():
            projection = getattr(attention, projection_name, None)
            projection_size = getattr(projection, "out_features", None)
            if (
                projection_size is not None
                and int(projection_size) != expected_size
            ):
                raise RetrievalDiscoveryError(
                    "Qwen2 attention projection dimensions are incompatible"
                )
    return mapping


def attention_copy_hits(
    attentions: Sequence[torch.Tensor],
    prompt_ids: Sequence[int],
    answer_start: int,
    answer_end: int,
    generated_token_id: int,
) -> dict[str, bool]:
    """Score max-attended prompt positions for one generated token."""
    hits: dict[str, bool] = {}
    prompt_length = len(prompt_ids)
    for layer_index, attention in enumerate(attentions):
        if (
            attention.ndim != 4
            or attention.shape[0] != 1
            or attention.shape[2] != 1
        ):
            raise RetrievalDiscoveryError("eager attention shape is invalid")
        query_count = int(attention.shape[1])
        key_length = min(prompt_length, int(attention.shape[-1]))
        if key_length <= 0:
            raise RetrievalDiscoveryError(
                "eager attention has no prompt keys"
            )
        maxima = attention[0, :, -1, :key_length].argmax(dim=-1)
        for query_index, maximum in enumerate(maxima.tolist()):
            position = int(maximum)
            hits[f"{layer_index}:{query_index}"] = (
                answer_start <= position < answer_end
                and prompt_ids[position] == generated_token_id
            )
        if query_count != len(maxima):
            raise RetrievalDiscoveryError(
                "attention query head count is unstable"
            )
    return hits


def exact_answer_match(
    generated_token_ids: Sequence[int], answer_token_ids: Sequence[int]
) -> tuple[bool, int]:
    """Apply the frozen exact contiguous source-answer success rule."""
    target = tuple(answer_token_ids)
    if not target:
        raise RetrievalDiscoveryError("answer token sequence is empty")
    observed = tuple(generated_token_ids)
    count = sum(
        observed[index : index + len(target)] == target
        for index in range(len(observed) - len(target) + 1)
    )
    return count > 0, count


def validate_sentinel_ids(
    eager_ids: Sequence[int], sdpa_ids: Sequence[int]
) -> None:
    """Require eager and SDPA outputs to agree through the stop point."""
    if tuple(eager_ids) != tuple(sdpa_ids):
        raise RetrievalDiscoveryError(
            "eager and SDPA sentinel token IDs differ"
        )


def set_attention_backend(model: Any, backend: str) -> None:
    """Select the installed Transformers attention implementation."""
    if backend not in {"eager", "sdpa"}:
        raise ValueError("backend must be eager or sdpa")
    setter = getattr(model, "set_attn_implementation", None)
    if not callable(setter):
        raise RetrievalDiscoveryError("model cannot select attention backend")
    setter(backend)
    config = getattr(model, "config", None)
    if getattr(config, "_attn_implementation", backend) != backend:
        raise RetrievalDiscoveryError(
            "attention backend selection was not applied"
        )


def decode_case(
    model: Any,
    tokenizer: Any,
    case: RetrievalCase,
    *,
    collect_attention: bool,
    runtime: DiscoveryRuntime | None = None,
) -> DecodeResult:
    """Decode one case and optionally retain eager attention scores."""
    runtime = runtime or DiscoveryRuntime()
    device = runtime.validate_model(model)
    runtime.seed()
    if runtime._deadline is None:
        runtime.start()
    runtime.check_deadline()
    runtime.reset_peak_memory(device)
    started = runtime.clock()
    input_ids = torch.tensor(
        [case.prompt_ids], dtype=torch.long, device=device
    )
    if input_ids.shape[1] < 2:
        raise RetrievalDiscoveryError(
            "prompt must contain at least two tokens"
        )
    prefill_started = runtime.clock()
    runtime.synchronize(device)
    with torch.no_grad():
        prefill = model(
            input_ids=input_ids[:, :-1],
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            logits_to_keep=1,
        )
    runtime.synchronize(device)
    prefill_seconds = runtime.clock() - prefill_started
    cache = prefill.past_key_values
    del prefill
    current = input_ids[:, -1:]
    generated: list[int] = []
    hits: dict[str, int] = {}
    stop_reason = "max_tokens"
    seen_non_whitespace = False
    eos_ids = _eos_ids(model, tokenizer)
    decode_forward_seconds = 0.0
    for _ in range(MAX_GENERATION_TOKENS):
        runtime.check_deadline()
        cache_length = int(cache.get_seq_length())
        attention_mask = torch.ones(
            (1, cache_length + 1), dtype=torch.long, device=device
        )
        position = torch.tensor(
            [[cache_length]], dtype=torch.long, device=device
        )
        forward_started = runtime.clock()
        runtime.synchronize(device)
        with torch.no_grad():
            output = model(
                input_ids=current,
                attention_mask=attention_mask,
                position_ids=position,
                cache_position=position[0],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
                output_attentions=collect_attention,
                logits_to_keep=1,
            )
        runtime.synchronize(device)
        decode_forward_seconds += runtime.clock() - forward_started
        next_cache = output.past_key_values
        token_id = int(output.logits[:, -1, :].argmax(dim=-1).item())
        if collect_attention:
            attentions = output.attentions
            if attentions is None:
                raise RetrievalDiscoveryError(
                    "eager decode returned no attentions"
                )
            step_hits = attention_copy_hits(
                attentions,
                case.prompt_ids,
                case.answer_start,
                case.answer_end,
                int(output.logits[:, -1, :].argmax(dim=-1).item()),
            )
            for name, hit in step_hits.items():
                hits[name] = hits.get(name, 0) + int(hit)
            del attentions, step_hits
        generated.append(token_id)
        del output
        cache = next_cache
        if token_id in eos_ids:
            stop_reason = "eos"
            break
        piece = _decode(tokenizer, (token_id,))
        prior_text = _decode(tokenizer, generated)
        if piece.strip():
            seen_non_whitespace = True
        newline = prior_text.find("\n")
        if (
            seen_non_whitespace
            and newline >= 0
            and prior_text[:newline].strip()
        ):
            stop_reason = "line_break"
            break
        current = torch.tensor([[token_id]], dtype=torch.long, device=device)
    success, match_count = exact_answer_match(
        generated, case.answer_token_ids
    )
    denominator = len(case.answer_token_ids)
    peak_allocated, peak_reserved = runtime.peak_memory(device)
    return DecodeResult(
        generated_token_ids=tuple(generated),
        stop_reason=stop_reason,
        success=success,
        answer_match_count=match_count,
        head_copy_fraction={
            name: count / denominator for name, count in sorted(hits.items())
        },
        elapsed_seconds=runtime.clock() - started,
        prefill_seconds=prefill_seconds,
        decode_forward_seconds=decode_forward_seconds,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )


def run_discovery(
    model: Any,
    tokenizer: Any,
    cases: Sequence[RetrievalCase],
    *,
    source_manifest: Mapping[str, object] | None = None,
    runtime: DiscoveryRuntime | None = None,
) -> dict[str, object]:
    """Run the bounded 24-case assay after all preflight inputs are frozen."""
    if len(cases) != 24:
        raise RetrievalDiscoveryError("discovery requires exactly 24 cases")
    runtime = runtime or DiscoveryRuntime()
    runtime.seed()
    runtime.validate_model(model)
    runtime.start()
    started = runtime.clock()
    mapping = verify_model_gqa_mapping(model)
    by_panel = {
        panel: sorted(
            (case for case in cases if case.panel == panel),
            key=lambda case: case.case_id,
        )
        for panel in PANELS
    }
    if any(len(by_panel[panel]) != 12 for panel in PANELS):
        raise RetrievalDiscoveryError(
            "each panel must contain exactly 12 cases"
        )
    sentinels = tuple(
        case
        for panel in PANELS
        for case in (by_panel[panel][0], by_panel[panel][-1])
    )
    if len({case.case_id for case in sentinels}) != 4:
        raise RetrievalDiscoveryError("sentinel cases must be unique")
    sentinel_records: list[dict[str, object]] = []
    eager_sentinel_results: dict[str, DecodeResult] = {}
    sdpa_results: list[DecodeResult] = []
    operational_error: str | None = None
    for case in sentinels:
        try:
            runtime.check_deadline()
            set_attention_backend(model, "eager")
            eager = decode_case(
                model,
                tokenizer,
                case,
                collect_attention=True,
                runtime=runtime,
            )
            eager_sentinel_results[case.case_id] = eager
            runtime.check_deadline()
            set_attention_backend(model, "sdpa")
            sdpa = decode_case(
                model,
                tokenizer,
                case,
                collect_attention=False,
                runtime=runtime,
            )
            sdpa_results.append(sdpa)
            validate_sentinel_ids(
                eager.generated_token_ids, sdpa.generated_token_ids
            )
            sentinel_records.append(
                {
                    "case_id": case.case_id,
                    "generated_token_ids": list(eager.generated_token_ids),
                    "sdpa_generated_token_ids": list(
                        sdpa.generated_token_ids
                    ),
                    "stop_reason": eager.stop_reason,
                    "passed": True,
                }
            )
        except DiscoveryDeadlineExceeded as error:
            operational_error = str(error)
            break
    set_attention_backend(model, "eager")
    case_records: list[dict[str, object]] = []
    completed_case_results: list[DecodeResult] = []
    if operational_error is None:
        for case in cases:
            try:
                runtime.check_deadline()
                result = eager_sentinel_results.get(case.case_id)
                if result is None:
                    result = decode_case(
                        model,
                        tokenizer,
                        case,
                        collect_attention=True,
                        runtime=runtime,
                    )
                case_records.append(
                    {"case": case.to_dict(), "result": result.to_dict()}
                )
                if case.case_id not in eager_sentinel_results:
                    completed_case_results.append(result)
            except DiscoveryDeadlineExceeded as error:
                operational_error = str(error)
                break
    timing = _timing_summary(
        started,
        runtime,
        list(eager_sentinel_results.values())
        + sdpa_results
        + completed_case_results,
        len(eager_sentinel_results)
        + sum(
            cast(Mapping[str, object], record["case"])["case_id"]
            not in eager_sentinel_results
            for record in case_records
        ),
        len(sdpa_results),
    )
    if operational_error is not None:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "operational_stop",
            "passed": False,
            "reason": operational_error,
            "repository": REPOSITORY,
            "commit": PINNED_COMMIT,
            "seed": SEED,
            "case_count": len(cases),
            "panel_case_counts": {
                panel: len(items) for panel, items in by_panel.items()
            },
            "sentinels": sentinel_records,
            "completed_case_records": case_records,
            "completed_eager_sentinel_ids": sorted(eager_sentinel_results),
            "gqa_query_to_kv": list(mapping),
            "selected_cells": [],
            "source_manifest": dict(source_manifest or {}),
            "timing": timing,
        }
    success_by_panel = {
        panel: sum(
            int(bool(cast(Mapping[str, object], item["result"])["success"]))
            for item in case_records
            if cast(Mapping[str, object], item["case"])["panel"] == panel
        )
        for panel in PANELS
    }
    panel_means: dict[str, dict[str, float | None]] = {}
    for panel in PANELS:
        successful = [
            cast(dict[str, object], item["result"])
            for item in case_records
            if cast(Mapping[str, object], item["case"])["panel"] == panel
            and cast(Mapping[str, object], item["result"])["success"]
        ]
        names = sorted(
            {
                name
                for result in successful
                for name in cast(
                    dict[str, float], result["head_copy_fraction"]
                )
            }
        )
        panel_means[panel] = {
            name: (
                sum(
                    cast(dict[str, float], result["head_copy_fraction"])[name]
                    for result in successful
                )
                / len(successful)
                if successful
                else None
            )
            for name in names
        }
    floor_passed = all(
        success_by_panel[panel] >= MIN_SUCCESSFUL_CASES
        and all(
            any(
                cast(Mapping[str, object], item["result"])["success"]
                and cast(Mapping[str, object], item["case"])["context_length"]
                == length
                for item in case_records
                if cast(Mapping[str, object], item["case"])["panel"] == panel
            )
            for length in CONTEXT_LENGTHS
        )
        for panel in PANELS
    )
    selected = _select_cells(panel_means, mapping) if floor_passed else []
    status = "completed" if floor_passed and selected else "scientific_stop"
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "passed": status == "completed",
        "repository": REPOSITORY,
        "commit": PINNED_COMMIT,
        "seed": SEED,
        "case_count": len(cases),
        "panel_case_counts": {
            panel: len(items) for panel, items in by_panel.items()
        },
        "success_by_panel": success_by_panel,
        "success_floor": {
            "minimum_per_panel": MIN_SUCCESSFUL_CASES,
            "passed": floor_passed,
        },
        "sentinels": sentinel_records,
        "panel_head_means": panel_means,
        "gqa_query_to_kv": list(mapping),
        "selected_cells": selected,
        "source_manifest": dict(source_manifest or {}),
        "cases": case_records,
        "timing": timing,
    }


def _timing_summary(
    started: float,
    runtime: DiscoveryRuntime,
    results: Sequence[DecodeResult],
    eager_count: int,
    sdpa_count: int,
) -> dict[str, object]:
    allocated = [
        value
        for value in (result.peak_allocated_bytes for result in results)
        if value is not None
    ]
    reserved = [
        value
        for value in (result.peak_reserved_bytes for result in results)
        if value is not None
    ]
    total_seconds = runtime.clock() - started
    return {
        "wall_seconds": total_seconds,
        "total_seconds": total_seconds,
        "forward_seconds": sum(
            result.prefill_seconds + result.decode_forward_seconds
            for result in results
        ),
        "prefill_seconds": sum(result.prefill_seconds for result in results),
        "decode_forward_seconds": sum(
            result.decode_forward_seconds for result in results
        ),
        "peak_allocated_bytes": max(allocated, default=None),
        "peak_reserved_bytes": max(reserved, default=None),
        "eager_continuations": eager_count,
        "sdpa_continuations": sdpa_count,
        "completed_continuations": eager_count + sdpa_count,
        "deadline_seconds": runtime.deadline_seconds,
    }


def _select_cells(
    panel_means: Mapping[str, Mapping[str, float | None]],
    mapping: Sequence[int],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    kv_heads = max(mapping) + 1
    names = set(panel_means["discovery"]) | set(panel_means["validation"])
    layers = sorted({int(name.split(":", 1)[0]) for name in names})
    for kv_head in range(kv_heads):
        query_indices = [
            index for index, value in enumerate(mapping) if value == kv_head
        ]
        for layer in layers:
            score = max(
                (
                    min(
                        float(
                            panel_means["discovery"].get(f"{layer}:{query}")
                            or 0.0
                        ),
                        float(
                            panel_means["validation"].get(f"{layer}:{query}")
                            or 0.0
                        ),
                    )
                    for query in query_indices
                ),
                default=0.0,
            )
            if score >= SUCCESS_THRESHOLD:
                cells.append(
                    {
                        "layer": layer,
                        "kv_head": kv_head,
                        "stability_score": score,
                    }
                )
    return sorted(
        cells,
        key=lambda item: (
            -float(cast(float, item["stability_score"])),
            int(cast(int, item["layer"])),
            int(cast(int, item["kv_head"])),
        ),
    )[:MAX_SELECTED_CELLS]


def _field_list(value: Any, name: str) -> list[int]:
    if isinstance(value, Mapping):
        field = value[name]
    elif hasattr(value, name):
        field = getattr(value, name)
    elif name == "input_ids":
        field = value
    else:
        raise RetrievalDiscoveryError(f"tokenizer field is missing: {name}")
    return _flatten_ids(field)


def _field_offsets(value: Any) -> list[tuple[int, int]]:
    field = (
        value["offset_mapping"]
        if isinstance(value, Mapping)
        else value.offset_mapping
    )
    if hasattr(field, "tolist"):
        field = field.tolist()
    if field and isinstance(field[0][0], list):
        field = field[0]
    return [(int(start), int(end)) for start, end in field]


def _flatten_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _encode(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer(text, add_special_tokens=False)
    return _field_list(value, "input_ids")


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _eos_ids(model: Any, tokenizer: Any) -> frozenset[int]:
    values: set[int] = set()
    for owner in (tokenizer, getattr(model, "generation_config", None)):
        value = getattr(owner, "eos_token_id", None)
        if isinstance(value, int):
            values.add(value)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            values.update(int(item) for item in value)
    return frozenset(values)


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256(value.encode("utf-8"))
