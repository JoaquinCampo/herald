"""Greedy generation mechanism: reference runs and hybrid runs.

One code path (`model.generate`) for both, so the paired counterfactual
differs only in compression, never in the decoding path or its numerics.

- Reference: full cache, no press, features captured inline.
- Hybrid at switch position s: prefill `[prompt + first s reference
  tokens]` (in token-id space, never detokenized) inside `press(model)`
  so the press fires once on that full-attention cache and evicts to the
  ratio; decoding then continues on the fixed compressed cache.

Greedy is forced through a clean GenerationConfig (no sampling, no
repetition penalty, no n-gram blocking) so that `scores == logits` and
the collected features are the raw model distribution. The generation
budget is the task cap M for the reference and `M - s` for a hybrid, so
both sides get the same total budget.
"""

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from kvpress.presses.base_press import BasePress
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from herald.attention_features import AttentionTap
from herald.config import MODELS
from herald.features import FEATURE_NAMES, FeatureCollector
from herald.tasks import PromptRecord


@dataclass
class ReferenceRun:
    prompt_id: str
    prompt_input_ids: list[int]
    gen_ids: list[int]
    text: str
    # (len(gen_ids), n_features); row t produced gen token t.
    features: np.ndarray
    # Column names of `features`; None means the legacy logit-only
    # superset (herald.features.FEATURE_NAMES order).
    feature_names: list[str] | None = None


@dataclass
class HybridRun:
    prompt_id: str
    compressor: str
    ratio: float
    s: int
    new_ids: list[int]
    # (len(new_ids), n_features); row t produced compressed token t.
    features: np.ndarray
    # Full hybrid output = reference[:s] + new tokens, decoded.
    text: str


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    gen_config: GenerationConfig
    eos_ids: set[int]
    pad_id: int
    key: str


def load_model(
    model_key: str,
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
    attn_implementation: str = "sdpa",
    model_id: str | None = None,
) -> LoadedModel:
    """Load a base model with a clean greedy generation config."""
    hf_id = model_id if model_id is not None else MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)  # type: ignore[no-untyped-call]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,
        ),
    )
    model = model.to(device)  # type: ignore[arg-type]
    model.eval()

    gc = copy.deepcopy(model.generation_config)
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    gc.num_beams = 1
    # Neutralize every score-altering processor so scores == logits.
    gc.repetition_penalty = 1.0
    gc.no_repeat_ngram_size = 0
    gc.pad_token_id = tokenizer.pad_token_id

    eos = gc.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError(f"no eos_token_id configured for {hf_id}")
    eos_ids = {int(e) for e in (eos if isinstance(eos, list) else [eos])}

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        gen_config=gc,
        eos_ids=eos_ids,
        pad_id=int(tokenizer.pad_token_id),
        key=model_key,
    )


def build_input_ids(lm: LoadedModel, record: PromptRecord) -> torch.Tensor:
    """Chat-template a record to a 1D tensor of input ids (no padding)."""
    # Qwen3 runs in non-thinking mode to match the direct-answer regime.
    if lm.key.startswith("qwen"):
        ids = lm.tokenizer.apply_chat_template(
            record.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    else:
        ids = lm.tokenizer.apply_chat_template(
            record.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    return cast(torch.Tensor, ids)[0]


def _left_pad(
    seqs: list[torch.Tensor], pad_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    maxlen = max(int(s.shape[0]) for s in seqs)
    n = len(seqs)
    input_ids = torch.full(
        (n, maxlen), pad_id, dtype=torch.long, device=device
    )
    attn = torch.zeros((n, maxlen), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        length = int(s.shape[0])
        input_ids[i, maxlen - length :] = s.to(device)
        attn[i, maxlen - length :] = 1
    return input_ids, attn


def _trim_at_eos(ids: list[int], eos_ids: set[int]) -> list[int]:
    out: list[int] = []
    for tok in ids:
        out.append(tok)
        if tok in eos_ids:
            break
    return out


def _generate(
    lm: LoadedModel,
    seqs: list[torch.Tensor],
    max_new_tokens: int,
    *,
    press: BasePress | None = None,
    collector: FeatureCollector | None = None,
    seed: int = 0,
) -> tuple[list[list[int]], int]:
    """Left-pad, greedy-generate, return per-row generated ids + prompt
    pad length. Optionally inside a press context, optionally with a
    feature collector. Generated ids are trimmed at the first EOS.

    The RNG is seeded before generation so RandomPress evictions are
    reproducible (the run can be regenerated on resume); greedy decoding
    is otherwise deterministic and the seed is inert for it.
    """
    device = next(lm.model.parameters()).device.type
    input_ids, attn = _left_pad(seqs, lm.pad_id, device)
    prompt_len = int(input_ids.shape[1])
    processors = [collector] if collector is not None else None

    torch.manual_seed(seed)
    press_ctx = press(lm.model) if press is not None else nullcontext()
    with torch.no_grad(), press_ctx:
        out = lm.model.generate(  # type: ignore[operator]
            input_ids=input_ids,
            attention_mask=attn,
            generation_config=lm.gen_config,
            max_new_tokens=max_new_tokens,
            logits_processor=processors,
            return_dict_in_generate=True,
        )
    seqs_out = out.sequences[:, prompt_len:].tolist()
    trimmed = [_trim_at_eos(row, lm.eos_ids) for row in seqs_out]
    return trimmed, prompt_len


def generate_reference(
    lm: LoadedModel,
    records: list[PromptRecord],
    max_new_tokens: int,
    *,
    tap: "AttentionTap | None" = None,
) -> list[ReferenceRun]:
    """Generate references for a batch of prompts with inline features.

    With an ``AttentionTap`` attached, its per-step cross-layer
    moments are appended as extra feature columns and the run carries
    explicit ``feature_names`` describing the widened matrix.
    """
    prompts = [build_input_ids(lm, r) for r in records]
    collector = FeatureCollector()
    if tap is not None:
        tap.begin(prompt_lens=[int(p.shape[-1]) for p in prompts])
    gen_ids, _ = _generate(lm, prompts, max_new_tokens, collector=collector)
    feats = collector.stacked()  # (steps, batch, n_feat)
    argmax = collector.argmax_tokens()  # (steps, batch)
    names: list[str] | None = None
    if tap is not None:
        tap_matrix, tap_names = tap.matrix()  # (batch, steps, n_tap)
        if tap_matrix.shape[1] != feats.shape[0]:
            raise RuntimeError(
                "attention tap saw a different step count than the "
                f"feature collector: {tap_matrix.shape[1]} != "
                f"{feats.shape[0]}"
            )
        feats = np.concatenate([feats, tap_matrix.transpose(1, 0, 2)], axis=2)
        names = list(FEATURE_NAMES) + tap_names

    runs: list[ReferenceRun] = []
    for b, record in enumerate(records):
        ids = gen_ids[b]
        length = len(ids)
        # Self-check: greedy tokens the collector saw must match the
        # tokens generate actually emitted, else a processor altered the
        # decisive scores and scores != logits.
        seen = argmax[:length, b].tolist()
        if seen != ids:
            raise RuntimeError(
                f"feature/token mismatch for {record.prompt_id}: "
                "a logits processor altered scores"
            )
        text = lm.tokenizer.decode(ids, skip_special_tokens=True)
        runs.append(
            ReferenceRun(
                prompt_id=record.prompt_id,
                prompt_input_ids=prompts[b].tolist(),
                gen_ids=ids,
                text=text,
                features=feats[:length, b, :].copy(),
                feature_names=names,
            )
        )
    return runs


def generate_hybrids(
    lm: LoadedModel,
    items: list[tuple[ReferenceRun, int]],
    compressor: str,
    ratio: float,
    press: BasePress,
    max_new_tokens_cap: int,
    *,
    seed: int = 0,
) -> list[HybridRun]:
    """Generate hybrids for a batch sharing one (compressor, ratio, s).

    Each item is `(reference, s)`. The injected prefix is the first s
    reference token ids, concatenated in id space. Budget is M - s.
    """
    s_values = {s for _, s in items}
    if len(s_values) != 1:
        raise ValueError("batch must share a single switch position s")
    s = s_values.pop()

    seqs: list[torch.Tensor] = []
    for ref, _ in items:
        prefix = ref.gen_ids[:s]
        seq = torch.tensor(ref.prompt_input_ids + prefix, dtype=torch.long)
        seqs.append(seq)

    collector = FeatureCollector()
    new_ids_list, _ = _generate(
        lm,
        seqs,
        max_new_tokens_cap - s,
        press=press,
        collector=collector,
        seed=seed,
    )
    feats = collector.stacked()
    argmax = collector.argmax_tokens()

    runs: list[HybridRun] = []
    for b, ((ref, _), new_ids) in enumerate(
        zip(items, new_ids_list, strict=True)
    ):
        length = len(new_ids)
        seen = argmax[:length, b].tolist()
        if seen != new_ids:
            raise RuntimeError(
                f"hybrid feature/token mismatch for {ref.prompt_id}: "
                "a logits processor altered scores"
            )
        full_ids = ref.gen_ids[:s] + new_ids
        text = lm.tokenizer.decode(full_ids, skip_special_tokens=True)
        runs.append(
            HybridRun(
                prompt_id=ref.prompt_id,
                compressor=compressor,
                ratio=ratio,
                s=s,
                new_ids=new_ids,
                features=feats[:length, b, :].copy(),
                text=text,
            )
        )
    return runs


def switch_positions(run_length: int, stride: int) -> list[int]:
    """Switch positions {0, k, 2k, ...} up to (and including) length."""
    return list(range(0, run_length + 1, stride))
