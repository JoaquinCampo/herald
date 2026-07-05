"""Press-score features captured at prefill compression.

Every kvpress ``ScorerPress`` computes per-entry importance scores to
decide evictions; HERALD previously discarded them. This module wraps
a press instance's ``score`` method (leaving eviction byte-identical)
and reduces, per layer:

- ``press_evicted_reliance``: share of the observation-window
  attention mass (SnapKV's window recompute, causal, sdpa-safe) that
  sits on entries the press evicts. "How much of what the generation
  currently leans on dies."
- ``press_evicted_share_prompt``: fraction of evicted entries inside
  the prompt region.
- ``press_score_reliance_align``: Pearson correlation between the
  press's scores and the observed window reliance. A press whose
  importance notion mismatches this prompt's reliance is dangerous.

Scalars are aggregated across layers as mean and variance. All
quantities are computable at decision time in deployment: the press
scoring pass is cheap and runs before committing to eviction.

Design: `docs/implementation/feature_extension.md`.
"""

from typing import Any

import numpy as np
import torch

WINDOW = 64
BASE_SCALARS: tuple[str, ...] = (
    "press_evicted_reliance",
    "press_evicted_share_prompt",
    "press_score_reliance_align",
)


def press_feature_names() -> list[str]:
    """Column names of the per-row press features."""
    names: list[str] = []
    for base in BASE_SCALARS:
        names.append(f"{base}_lmean")
        names.append(f"{base}_lvar")
    return names


class PressScoreRecorder:
    """Wraps a ScorerPress to capture score-derived features.

    The wrapper intercepts ``press.score`` via an instance attribute,
    computes features from the scores and a window-attention reliance
    profile, and returns the scores unchanged, so compression behaves
    byte-identically. Call ``begin`` before each item (hybrids run
    batch 1) and ``features`` after its prefill.
    """

    def __init__(self, press: Any, window: int = WINDOW) -> None:
        self.press = press
        self.window = window
        self._prompt_len = 0
        self._per_layer: list[list[float]] = []
        self._original_score = press.score
        press.score = self._wrapped_score

    def begin(self, prompt_len: int) -> None:
        """Reset for the next hybrid item."""
        self._prompt_len = int(prompt_len)
        self._per_layer = []

    def layers_seen(self) -> int:
        """Number of layers captured since the last ``begin``."""
        return len(self._per_layer)

    def unwrap(self) -> None:
        """Restore the press's original score method."""
        self.press.score = self._original_score

    def features(self) -> dict[str, float]:
        """Cross-layer mean/variance of the per-layer scalars."""
        stacked = np.asarray(self._per_layer, dtype=np.float64)
        out: dict[str, float] = {}
        for j, base in enumerate(BASE_SCALARS):
            column = stacked[:, j] if stacked.size else np.array([])
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                out[f"{base}_lmean"] = 0.0
                out[f"{base}_lvar"] = 0.0
            else:
                out[f"{base}_lmean"] = float(finite.mean())
                out[f"{base}_lvar"] = float(finite.var())
        return out

    def _wrapped_score(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Any,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        scores = torch.as_tensor(
            self._original_score(
                module, hidden_states, keys, values, attentions, kwargs
            )
        )
        try:
            self._per_layer.append(
                self._reduce(module, hidden_states, keys, scores, kwargs)
            )
        except Exception:  # noqa: BLE001
            # Feature capture must never break the sweep's compression.
            self._per_layer.append([np.nan] * len(BASE_SCALARS))
        return scores

    def _reduce(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        scores: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> list[float]:
        from kvpress.presses.snapkv_press import SnapKVPress

        k_len = keys.shape[2]
        window = min(self.window, max(4, k_len // 4))
        reliance_len = k_len - window
        if reliance_len < 8:
            return [np.nan] * len(BASE_SCALARS)

        window_attn = SnapKVPress.compute_window_attention(
            module,
            hidden_states,
            keys,
            window,
            kwargs["position_embeddings"],
        )
        # (batch, heads, window, reliance_len) -> per-entry reliance
        reliance = window_attn.float().mean(dim=(1, 2))[0]
        reliance = reliance / reliance.sum().clamp_min(1e-12)

        head_scores = scores[0].float()  # (kv_heads, k_len)
        n_kept = int(k_len * (1 - self.press.compression_ratio))
        kept = torch.zeros_like(head_scores, dtype=torch.bool)
        kept.scatter_(1, head_scores.topk(n_kept, dim=-1).indices, True)
        evicted = ~kept  # (kv_heads, k_len)

        ev_front = evicted[:, :reliance_len].float()
        evicted_reliance = float(
            (ev_front * reliance.unsqueeze(0)).sum(dim=1).mean()
        )
        prompt_end = min(self._prompt_len, k_len)
        share_prompt = float(
            evicted[:, :prompt_end].float().sum(dim=1).mean()
            / max(1.0, float(evicted[0].float().sum()))
        )
        align = _mean_pearson(head_scores[:, :reliance_len], reliance)
        return [evicted_reliance, share_prompt, align]


def _mean_pearson(head_scores: torch.Tensor, reliance: torch.Tensor) -> float:
    """Mean per-head Pearson correlation of scores vs reliance."""
    x = head_scores - head_scores.mean(dim=1, keepdim=True)
    y = reliance - reliance.mean()
    denom = x.norm(dim=1) * y.norm()
    valid = denom > 1e-12
    if not bool(valid.any()):
        return 0.0
    corr = (x @ y) / denom.clamp_min(1e-12)
    return float(corr[valid].mean())
