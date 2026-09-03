"""Label-free compressor-action sensors for the locked M2 protocol.

This module is independent of outcomes and future continuation tokens. It
consumes only tensors available immediately before a press mutates a layer's
cache: layer-input hidden states, rotary keys/values, press scores, and the
eviction mask. Missing or nonfinite capture is an error, never a zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

COMPRESSORS: tuple[str, ...] = (
    "expected_attention",
    "knorm",
    "streaming_llm",
)
RATIOS: tuple[float, ...] = (0.25, 0.5, 0.75, 0.875)
SINK_TOKENS = 4
RECENT_WINDOW = 64

SENSOR_NAMES: tuple[str, ...] = (
    "removed_k_norm_mass_fraction",
    "removed_v_norm_mass_fraction",
    "prompt_region_eviction_share",
    "recent_window_eviction_share",
    "sink_eviction_share",
    "normalized_score_entropy",
    "cutoff_margin",
    "bounded_window_reliance_removed",
    "score_reliance_pearson_alignment",
)
SENSOR_STATS: tuple[str, ...] = ("lmean", "lstd", "lp90", "lmax")


class SensorCaptureError(RuntimeError):
    """Raised when exact pre-mutation sensor state is absent or nonfinite."""


def sensor_feature_names() -> list[str]:
    """Return the locked, ordered cross-layer feature columns."""
    return [
        f"{name}_{stat}" for name in SENSOR_NAMES for stat in SENSOR_STATS
    ]


def _finite_tensor(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.numel() == 0:
        raise SensorCaptureError(f"missing sensor tensor: {name}")
    if not bool(torch.isfinite(value).all()):
        raise SensorCaptureError(f"nonfinite sensor tensor: {name}")
    return value


def eviction_mask(
    scores: Tensor,
    ratio: float,
) -> Tensor:
    """Return a boolean ``(B,H,L)`` mask, true for entries removed.

    This exactly mirrors ``ScorerPress.compress``: floor the kept count and
    retain ``scores.topk(kept_count)`` on the active PyTorch backend.
    """
    scores = _finite_tensor(torch.as_tensor(scores), "scores")
    if scores.ndim != 3:
        raise ValueError("scores must have shape (batch, kv_heads, length)")
    if not 0 <= ratio < 1:
        raise ValueError("ratio must satisfy 0 <= ratio < 1")
    kept_count = int(scores.shape[-1] * (1 - ratio))
    if kept_count == 0:
        return torch.ones_like(scores, dtype=torch.bool)
    kept = torch.zeros_like(scores, dtype=torch.bool)
    kept.scatter_(
        -1,
        scores.topk(kept_count, dim=-1, sorted=False).indices,
        True,
    )
    return ~kept


def _head_mean(numerator: Tensor, denominator: Tensor) -> Tensor:
    return (
        numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
    ).mean(dim=0)


def _scalar(value: Tensor, name: str) -> float:
    value = _finite_tensor(value, name)
    return float(value.detach().float().cpu().item())


def _pearson(scores: Tensor, reliance: Tensor) -> Tensor:
    """Per-head Pearson alignment, retaining only well-defined heads."""
    x = scores.float()
    y = reliance.float().expand_as(x)
    xc = x - x.mean(dim=-1, keepdim=True)
    yc = y - y.mean(dim=-1, keepdim=True)
    denom = xc.norm(dim=-1) * yc.norm(dim=-1)
    if not bool(torch.isfinite(denom).all()):
        raise SensorCaptureError("nonfinite Pearson denominator")
    valid = denom > 1e-12
    if not bool(valid.any()):
        return torch.zeros(scores.shape[0], device=scores.device)
    return cast(Tensor, (xc * yc).sum(dim=-1)[valid] / denom[valid])


def layer_sensor_values(
    *,
    keys: Tensor,
    values: Tensor,
    scores: Tensor,
    evicted: Tensor,
    prompt_len: int,
    reliance: Tensor,
    sink_tokens: int = SINK_TOKENS,
    recent_window: int = RECENT_WINDOW,
) -> dict[str, float]:
    """Compute the nine locked scalars for one layer and compressor/ratio.

    ``reliance`` is a normalized per-KV-entry vector or ``(Hkv,L)`` matrix
    computed by :func:`causal_recent_reliance`. It cannot include future keys.
    """
    keys = _finite_tensor(keys, "keys")
    values = _finite_tensor(values, "values")
    scores = _finite_tensor(scores, "scores")
    evicted = evicted.to(dtype=torch.bool)
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("keys and values must both have shape (B,H,L,D)")
    if scores.shape != keys.shape[:3] or evicted.shape != scores.shape:
        raise ValueError("scores and evicted must have shape (B,H,L)")
    reliance = _finite_tensor(reliance, "reliance").float()
    if reliance.ndim == 1:
        reliance = reliance.unsqueeze(0).expand(scores.shape[1], -1)
    elif reliance.ndim == 2 and reliance.shape[0] == 1:
        reliance = reliance.expand(scores.shape[1], -1)
    if reliance.shape != scores.shape[1:]:
        raise ValueError("reliance must have shape (L,) or (H,L)")
    reliance = reliance / reliance.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if not bool(torch.isfinite(reliance).all()):
        raise SensorCaptureError("nonfinite normalized reliance")

    k_norm = keys.float().norm(dim=-1)
    v_norm = values.float().norm(dim=-1)
    removed = evicted.float()
    total_removed = removed.sum(dim=-1)
    prompt_end = min(max(int(prompt_len), 0), keys.shape[-2])
    recent_start = max(0, keys.shape[-2] - int(recent_window))
    sink_end = min(max(int(sink_tokens), 0), keys.shape[-2])
    probs = torch.softmax(scores.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(dim=-1)
    entropy = entropy / max(log(max(2, keys.shape[-2])), 1e-12)

    kept = ~evicted
    kept_scores = scores.float().masked_fill(~kept, float("inf"))
    removed_scores = scores.float().masked_fill(~evicted, float("-inf"))
    cutoff = kept_scores.amin(dim=-1) - removed_scores.amax(dim=-1)
    cutoff = torch.where(total_removed > 0, cutoff, torch.zeros_like(cutoff))

    values_by_head = {
        "removed_k_norm_mass_fraction": _head_mean(
            (k_norm * removed).sum(-1), k_norm.sum(-1)
        ),
        "removed_v_norm_mass_fraction": _head_mean(
            (v_norm * removed).sum(-1), v_norm.sum(-1)
        ),
        "prompt_region_eviction_share": _head_mean(
            removed[..., :prompt_end].sum(-1), total_removed
        ),
        "recent_window_eviction_share": _head_mean(
            removed[..., recent_start:].sum(-1), total_removed
        ),
        "sink_eviction_share": _head_mean(
            removed[..., :sink_end].sum(-1), total_removed
        ),
        "normalized_score_entropy": entropy.mean(dim=-1),
        "cutoff_margin": cutoff,
        "bounded_window_reliance_removed": (reliance * removed[0]).sum(
            dim=-1
        ),
    }
    values_by_head["score_reliance_pearson_alignment"] = _pearson(
        scores[0], reliance
    )
    result: dict[str, float] = {}
    for name in SENSOR_NAMES:
        result[name] = _scalar(values_by_head[name].mean(), name)
    return result


def causal_recent_reliance(
    module: Any,
    hidden_states: Tensor,
    keys: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    *,
    recent_window: int = RECENT_WINDOW,
) -> Tensor:
    """Compute normalized causal attention mass for recent queries.

    The ``(Hkv,L)`` profile has no attention to positions after each query.
    Queries and keys come from the supplied pre-activation state and are never
    written back to the model or cache.
    """
    hidden_states = _finite_tensor(hidden_states, "hidden_states")
    keys = _finite_tensor(keys, "keys")
    if hidden_states.ndim != 3 or keys.ndim != 4:
        raise ValueError("invalid hidden/key shapes")
    q = (
        module.q_proj(hidden_states)
        .view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            module.config.num_attention_heads,
            module.head_dim,
        )
        .transpose(1, 2)
    )
    q, _ = _rotary(
        q, keys[:, : module.config.num_key_value_heads], position_embeddings
    )
    q_start = max(0, q.shape[-2] - int(recent_window))
    q = q[..., q_start:, :]
    q_positions = torch.arange(
        q_start, q_start + q.shape[-2], device=q.device
    )
    key_positions = torch.arange(keys.shape[-2], device=keys.device)
    from transformers.models.llama.modeling_llama import repeat_kv

    expanded_keys = repeat_kv(
        keys, module.config.num_attention_heads // keys.shape[1]
    )
    logits = (
        torch.matmul(q.float(), expanded_keys.float().transpose(-1, -2))
        * float(module.head_dim) ** -0.5
    )
    logits = logits.masked_fill(
        key_positions.view(1, 1, 1, -1) > q_positions.view(1, 1, -1, 1),
        float("-inf"),
    )
    attn = torch.softmax(logits, dim=-1)
    groups = int(module.config.num_attention_heads // keys.shape[1])
    attn = attn.view(
        attn.shape[0],
        keys.shape[1],
        groups,
        attn.shape[-2],
        attn.shape[-1],
    ).mean(dim=2)
    profile = attn.mean(dim=2)[0]
    profile = profile / profile.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return _finite_tensor(profile, "causal reliance")


def _rotary(
    q: Tensor, k: Tensor, embeddings: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    try:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
        )

        cos, sin = embeddings
        return cast(
            tuple[Tensor, Tensor],
            apply_rotary_pos_emb(  # type: ignore[no-untyped-call]
                q,
                k,
                cos,
                sin,
            ),
        )
    except (ImportError, AttributeError, RuntimeError, ValueError) as error:
        raise SensorCaptureError(
            "unable to apply exact rotary position embeddings"
        ) from error


def aggregate_layer_sensors(
    layer_values: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """Aggregate all scalars across layers, failing on any omission."""
    if not layer_values:
        raise SensorCaptureError("no layers captured")
    try:
        matrix = np.asarray(
            [
                [float(row[name]) for name in SENSOR_NAMES]
                for row in layer_values
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SensorCaptureError("missing or invalid layer sensor") from error
    if (
        matrix.shape != (len(layer_values), len(SENSOR_NAMES))
        or not np.isfinite(matrix).all()
    ):
        raise SensorCaptureError("missing or nonfinite layer sensor")
    result: dict[str, float] = {}
    for index, name in enumerate(SENSOR_NAMES):
        column = matrix[:, index]
        stats = (
            column.mean(),
            column.std(),
            np.percentile(column, 90),
            column.max(),
        )
        for stat, value in zip(SENSOR_STATS, stats, strict=True):
            if not np.isfinite(value):
                raise SensorCaptureError(
                    f"nonfinite aggregate: {name}_{stat}"
                )
            result[f"{name}_{stat}"] = float(value)
    return result
