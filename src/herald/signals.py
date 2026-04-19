"""Per-token signal extraction from logits."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from herald.config import TokenSignals


@dataclass(slots=True)
class StepState:
    """Carry-forward state between consecutive extract_signals calls."""

    entropy: float
    log_probs: torch.Tensor
    top10_ids: frozenset[int]


def extract_signals(
    logits: torch.Tensor,
    prev: StepState | None = None,
) -> tuple[TokenSignals, StepState]:
    """Extract per-token signal vector from logits at a single decoding step.

    Returns the signals and a StepState to pass into the next call.

    Features extracted:
    - top5_logprobs: top-5 log-probabilities
    - entropy (H_overall): total entropy
    - h_alts: entropy excluding top-1 (competitor disagreement)
    - avg_logp: mean log-probability (distribution sharpness)
    - delta_h: H(t) - H(t-1) (confidence shocks)
    - kl_div: KL(p_t || p_{t-1}) (full distributional shift)
    - top10_jaccard: Jaccard similarity of top-10 token IDs vs previous step
    - eff_vocab_size: exp(entropy) (effective vocabulary size)
    - tail_mass: 1 - sum(top20_probs) (probability leaking beyond top-20)
    - logit_range: max_logit - mean_logit (pre-softmax confidence spread)

    Args:
        logits: shape (vocab_size,) raw logits for one position
        prev: state from the previous decoding step (None for first token)
    """
    logits_f = logits.float()
    log_probs = F.log_softmax(logits_f, dim=-1)
    probs = log_probs.exp()

    # Top-20 in one shot — reuse for top-5 logprobs,
    # top-1/5 prob, tail mass, top-10 ids
    k20 = min(20, log_probs.shape[-1])
    top20_lp, top20_idx = torch.topk(log_probs, k=k20)
    top20_probs = probs[top20_idx]

    # Gather all scalar values into a single tensor to minimize GPU→CPU syncs
    entropy_t = -(probs * log_probs).sum()
    scalars = torch.stack(
        [
            entropy_t,
            top20_probs[0],  # top1_prob
            top20_probs[:5].sum(),  # top5_prob
            top20_probs.sum(),  # top20_prob_sum (for tail_mass)
            log_probs.mean(),  # avg_logp
            logits_f.max() - logits_f.mean(),  # logit_range
        ]
    )
    s = scalars.tolist()
    entropy, top1_prob, top5_prob, top20_sum, avg_logp, logit_range = s

    top5_logprobs = top20_lp[:5].tolist()

    # H_alts: entropy excluding top-1 (analytical, no clone needed)
    if top1_prob < 1.0 - 1e-10:
        top1_logp = top20_lp[0].item()
        # H_total = -p1*log(p1) + (1-p1) * H_alts
        # => H_alts = (H_total + p1*log(p1)) / (1-p1)
        h_alts = (entropy + top1_prob * top1_logp) / (1.0 - top1_prob)
    else:
        h_alts = 0.0

    # Temporal features (NaN for first token — XGBoost native)
    nan = float("nan")
    delta_h = nan
    kl_div = nan
    top10_ids = frozenset(top20_idx[:10].tolist())
    top10_jaccard = nan

    if prev is not None:
        delta_h = round(entropy - prev.entropy, 4)
        kl = (probs * (log_probs - prev.log_probs)).sum().item()
        kl_div = round(max(kl, 0.0), 4)
        intersection = len(top10_ids & prev.top10_ids)
        union = len(top10_ids | prev.top10_ids)
        top10_jaccard = round(intersection / union, 4) if union > 0 else 1.0

    signals = TokenSignals(
        entropy=round(entropy, 4),
        top1_prob=round(top1_prob, 4),
        top5_prob=round(top5_prob, 4),
        top5_logprobs=[round(x, 3) for x in top5_logprobs],
        h_alts=round(h_alts, 4),
        avg_logp=round(avg_logp, 4),
        delta_h=delta_h,
        delta_h_valid=prev is not None,
        kl_div=kl_div,
        top10_jaccard=top10_jaccard,
        eff_vocab_size=round(math.exp(entropy), 2),
        tail_mass=round(1.0 - top20_sum, 4),
        logit_range=round(logit_range, 4),
    )

    state = StepState(
        entropy=entropy, log_probs=log_probs.detach(), top10_ids=top10_ids
    )
    return signals, state


def compute_lookback_ratios(
    attentions: tuple[tuple[torch.Tensor, ...], ...],
    input_len: int,
) -> list[float]:
    """Per-token lookback ratio from generation attentions.

    Each entry of `attentions` is one decode step: a tuple of
    layer attention tensors of shape (batch, heads, q, k).
    For step t, the new token attends to positions [0, k);
    lookback ratio = mean attention to context [0, input_len)
    divided by total attention. Averaged across layers and
    heads. Returns one float per generated token.

    Reference: Chuang et al., "Lookback Lens" (2024).
    """
    out: list[float] = []
    for step_attns in attentions:
        per_layer: list[float] = []
        for layer_attn in step_attns:
            # layer_attn: (batch=1, heads, q, k)
            # The newly generated token is the last query row
            row = layer_attn[0, :, -1, :]  # (heads, k)
            k = row.shape[-1]
            ctx_end = min(input_len, k)
            attn_ctx = row[:, :ctx_end].sum(dim=-1)  # (heads,)
            attn_total = row.sum(dim=-1).clamp_min(1e-12)
            ratio = (attn_ctx / attn_total).mean().item()
            per_layer.append(ratio)
        out.append(round(sum(per_layer) / len(per_layer), 4))
    return out
