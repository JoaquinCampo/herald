"""Decode-time attention-reliance features (SPOT-adapted).

Reconstructs attention statistics without materializing any N x N
matrix, so FlashAttention/sdpa stays in place: at every forward, a
side matvec against the live KV cache yields the LAST query
position's attention row per tapped layer. Rows stream per decode
step; column (incoming-attention) statistics accumulate across
steps. Per-step scalars are aggregated across the tapped layers as
mean and variance (SPOT's cross-layer moments M and Sigma).

All reductions run in torch on the model's device; scalars are
buffered there and moved to the host once, in ``matrix()``, so the
tap adds no per-step device synchronization.

Design: `docs/implementation/feature_extension.md`, rationale
`docs/_why/5_attention_features.md`.
"""

import math
from typing import Any, cast

import numpy as np
import torch

ATTENTION_SINK = 4
LOCAL_WINDOW = 64
TOP_K = 8
BASE_SCALARS: tuple[str, ...] = (
    "attn_entropy",
    "attn_prompt_mass",
    "attn_sink_mass",
    "attn_local_mass",
    "attn_top8_mass",
    "attn_top8_mass_maxhead",
    "attn_reliance_prompt_share",
    "attn_reliance_drift",
)


def tap_feature_names() -> list[str]:
    """Persisted per-step column names (cross-layer moments)."""
    names: list[str] = []
    for base in BASE_SCALARS:
        names.append(f"{base}_lmean")
        names.append(f"{base}_lvar")
    return names


def default_layer_indices(num_layers: int) -> list[int]:
    """Quarter-depth tap layers (SPOT convention)."""
    quarters = (0.25, 0.5, 0.75)
    return sorted({int(num_layers * q) for q in quarters})


class AttentionTap:
    """Streams attention-row statistics from selected layers.

    Register once per model, call ``begin`` before each generation
    (batch), read ``matrix()`` afterwards. Assumes the cache only
    grows (reference stream: no press attached).
    """

    def __init__(
        self,
        model: Any,
        layer_indices: list[int] | None = None,
    ) -> None:
        layers = model.model.layers
        if layer_indices is None:
            layer_indices = default_layer_indices(len(layers))
        self.layer_indices = list(layer_indices)
        self._handles = [
            layers[i].self_attn.register_forward_hook(
                self._hook, with_kwargs=True
            )
            for i in self.layer_indices
        ]
        self._scalars: dict[int, list[torch.Tensor]] = {}
        self._acc: dict[int, torch.Tensor] = {}
        self._prev_profile: dict[int, torch.Tensor | None] = {}
        self._last_rows: dict[int, torch.Tensor] = {}
        self._prompt_lens: list[int] = []
        self._padded_len: int = 0

    def begin(self, prompt_lens: list[int]) -> None:
        """Reset state for a new (batch) generation."""
        self._prompt_lens = list(prompt_lens)
        self._padded_len = 0
        self._scalars = {i: [] for i in self.layer_indices}
        self._acc = {}
        self._prev_profile = {i: None for i in self.layer_indices}
        self._last_rows = {}

    def remove(self) -> None:
        """Detach all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def matrix(self) -> tuple[np.ndarray, list[str]]:
        """(batch, steps, features) cross-layer moment matrix."""
        names = tap_feature_names()
        per_layer = [
            torch.stack(self._scalars[i], dim=0)
            for i in self.layer_indices
            if self._scalars[i]
        ]
        if not per_layer:
            return np.zeros((0, 0, len(names))), names
        stacked = torch.stack(per_layer, dim=0)
        # (layers, steps, batch, scalars) -> moments over layers
        lmean = stacked.mean(dim=0)
        lvar = stacked.var(dim=0, unbiased=False)
        steps, batch, n_scalars = lmean.shape
        both = torch.stack([lmean, lvar], dim=-1)
        out = (
            both.reshape(steps, batch, 2 * n_scalars)
            .permute(1, 0, 2)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return out, names

    def debug_last_rows(self) -> list[np.ndarray]:
        """Head-mean rows of the most recent forward, batch 0."""
        return [
            self._last_rows[i][0].cpu().numpy() for i in self.layer_indices
        ]

    def debug_accumulator(self, layer: int) -> np.ndarray:
        """Accumulated incoming attention for batch element 0."""
        return self._acc[layer][0].cpu().numpy()

    def _hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        if not self._prompt_lens:
            return output
        rows, per_head = _last_position_rows(module, kwargs)
        layer = int(module.layer_idx)
        batch, n = rows.shape
        if self._padded_len == 0:
            self._padded_len = n
        self._last_rows[layer] = rows
        acc = self._acc.get(layer)
        if acc is None or acc.shape[1] < n:
            grown = torch.zeros(
                (batch, n), dtype=rows.dtype, device=rows.device
            )
            if acc is not None:
                grown[:, : acc.shape[1]] = acc
            acc = grown
        acc = acc + rows
        self._acc[layer] = acc

        profile = acc / acc.sum(dim=1, keepdim=True).clamp_min(1e-12)
        prev = self._prev_profile[layer]
        if prev is None:
            drift = torch.zeros(batch, dtype=rows.dtype, device=rows.device)
        else:
            width = prev.shape[1]
            drift = (profile[:, :width] - prev).abs().sum(dim=1) + profile[
                :, width:
            ].sum(dim=1)
        self._prev_profile[layer] = profile

        self._scalars[layer].append(
            self._reduce(rows, per_head, profile, drift, n)
        )
        return output

    def _reduce(
        self,
        rows: torch.Tensor,
        per_head: torch.Tensor,
        profile: torch.Tensor,
        drift: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        padded = self._padded_len
        device = rows.device
        pad = torch.tensor(
            [padded - length for length in self._prompt_lens],
            device=device,
        ).unsqueeze(1)
        positions = torch.arange(n, device=device).unsqueeze(0)

        entropy = -(rows * rows.clamp_min(1e-12).log()).sum(dim=1)
        prompt_mass = rows[:, : min(padded, n)].sum(dim=1)
        sink = (positions >= pad) & (positions < pad + ATTENTION_SINK)
        sink_mass = (rows * sink).sum(dim=1)
        local_mass = rows[:, max(0, n - LOCAL_WINDOW) :].sum(dim=1)
        k = min(TOP_K, n)
        top8 = rows.topk(k, dim=1).values.sum(dim=1)
        top8_max = (
            per_head.topk(k, dim=-1).values.sum(dim=-1).max(dim=1)
        ).values
        reliance_prompt = profile[:, : min(padded, profile.shape[1])].sum(
            dim=1
        )
        return torch.stack(
            [
                entropy,
                prompt_mass,
                sink_mass,
                local_mass,
                top8,
                top8_max,
                reliance_prompt,
                drift,
            ],
            dim=1,
        )


def _last_position_rows(
    module: Any, kwargs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Head-mean and per-head attention rows of the last position.

    Recomputes softmax(q_t K^T * scaling) against the live cache;
    the attention kernel itself is untouched. Keys in the cache are
    already rotary-embedded, so only q_t needs RoPE.
    """
    from kvpress.utils import (
        extract_keys_and_values,
        get_prerope_query_states,
    )
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
    )

    hidden_states = kwargs["hidden_states"]
    cache = kwargs["past_key_values"]
    cos, sin = kwargs["position_embeddings"]
    query = get_prerope_query_states(module, hidden_states[:, -1:, :])
    rotate = cast(Any, apply_rotary_pos_emb)
    query, _ = rotate(query, query, cos[:, -1:], sin[:, -1:])
    keys, _ = extract_keys_and_values(cache, int(module.layer_idx))
    batch, n_kv, n, head_dim = keys.shape
    n_heads = query.shape[1]
    if n_heads != n_kv:
        keys = keys.repeat_interleave(n_heads // n_kv, dim=1)
    scaling = getattr(module, "scaling", 1.0 / math.sqrt(head_dim))
    logits = (
        torch.einsum("bhqd,bhnd->bhqn", query.float(), keys.float()) * scaling
    )[:, :, 0, :]
    mask = kwargs.get("attention_mask")
    if mask is not None:
        row_mask = mask[:, :, -1, :n]
        if row_mask.dtype == torch.bool:
            logits = logits.masked_fill(
                ~row_mask, torch.finfo(logits.dtype).min
            )
        else:
            logits = logits + row_mask.float()
    per_head = torch.softmax(logits, dim=-1)
    rows = per_head.mean(dim=1)
    return rows.detach(), per_head.detach()
