"""Per-token logit features for HERALD.

The thesis is that compression damage has a signature in the model's own
next-token distribution, readable from cheap per-token logit statistics
alone. This module extracts those statistics in two layers:

`FeatureCollector` is a `LogitsProcessor` that, during greedy decoding,
reads the logits the forward pass already produced and stores a per-step
scalar superset. There is no extra forward pass and the full-vocabulary
logits are discarded each step, so the cost is negligible relative to
generation and memory stays low enough to batch. This per-step superset
is the persisted artifact.

`derive_features` is a downstream transform that expands the stored
per-step series with dynamics (deltas, slope, EWMA) and rolling
statistics. It is recomputable from the stored superset (never persisted)
and every column is causal and O(1)-online: it depends only on past and
current steps over a fixed window, so a streaming predictor reproduces it
at deploy time from the logit stream alone, with no new forward pass.
"""

from collections.abc import Sequence

import numpy as np
import torch
from transformers import LogitsProcessor

# k values for cumulative top-k probability mass (k=1 is max_prob).
TOPK_VALUES: tuple[int, ...] = (2, 5, 10, 20, 50, 100)

# Fixed order of the persisted per-step superset. All are derived from a
# single step's logits (single-token shape), except kl_prev, which is the
# one inline dynamic (it needs the previous step's distribution).
FEATURE_NAMES: tuple[str, ...] = (
    "entropy",  # H of the next-token distribution (nats)
    "varentropy",  # variance of surprisal under p
    "h_alts",  # entropy of the non-top-1 mass (competitor spread)
    "avg_logp",  # mean log-prob over the vocab (sharpness)
    "max_prob",  # top-1 probability
    "margin_prob",  # top-1 minus top-2 probability
    "top1_logit",  # top-1 raw logit
    "margin_logit",  # top-1 minus top-2 logit
    "logit_std",  # spread of logits across the vocab
    "logit_range",  # max minus min logit
    "logit_skew",  # skewness of the logit distribution
    "logit_kurtosis",  # excess kurtosis of the logit distribution
    "chosen_logprob",  # log-prob of the greedy token (= log max_prob)
    "kl_prev",  # KL(p_t || p_{t-1}); NaN at t=0
    *(f"topk_mass_{k}" for k in TOPK_VALUES),
)

# Default base signals the downstream transform builds dynamics on.
DYNAMIC_BASES: tuple[str, ...] = (
    "entropy",
    "max_prob",
    "margin_prob",
    "kl_prev",
    "h_alts",
)


class FeatureCollector(LogitsProcessor):
    """Collects the per-step logit superset during generation.

    One instance per `generate()` call. `stacked()` returns an array of
    shape `(steps, batch, n_features)` aligned so row `t` describes the
    distribution that produced generated token `t`. `argmax_tokens()`
    returns the greedy tokens the collector itself computed, for a
    cross-check that it saw the decisive (unwarped) scores.
    """

    def __init__(self) -> None:
        self._steps: list[np.ndarray] = []
        self._argmax: list[np.ndarray] = []
        self._prev_logp: torch.Tensor | None = None
        self._maxk = max(TOPK_VALUES)

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        logits = scores.float()
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()

        surprisal = -logp
        entropy = (p * surprisal).sum(dim=-1)
        # Stable one-pass variance of surprisal: the two-pass form
        # E[s^2] - E[s]^2 cancels catastrophically at high entropy.
        centered = surprisal - entropy.unsqueeze(-1)
        varentropy = (p * centered * centered).sum(dim=-1)
        avg_logp = logp.mean(dim=-1)

        kk = min(self._maxk, logits.shape[-1])
        top_p, _ = p.topk(kk, dim=-1)
        max_prob = top_p[:, 0]
        margin_prob = top_p[:, 0] - top_p[:, 1]
        masses = [top_p[:, :k].sum(dim=-1) for k in TOPK_VALUES]
        chosen_logprob = max_prob.clamp_min(1e-30).log()

        top_logits, _ = logits.topk(2, dim=-1)
        top1_logit = top_logits[:, 0]
        margin_logit = top_logits[:, 0] - top_logits[:, 1]

        # h_alts: entropy of the distribution with the top-1 removed.
        # H = -p1*log(p1) + (1-p1)*H_alts, solved for H_alts; defined to
        # 0 when the top token already holds all the mass.
        rest = (1.0 - max_prob).clamp_min(1e-6)
        h_alts = ((entropy + max_prob * chosen_logprob) / rest).clamp_min(0.0)

        # Logit-distribution moments across the vocab.
        mean = logits.mean(dim=-1, keepdim=True)
        dev = logits - mean
        var = (dev * dev).mean(dim=-1)
        std = var.sqrt()
        logit_range = logits.max(dim=-1).values - logits.min(dim=-1).values
        denom = var.clamp_min(1e-12)
        logit_skew = (dev**3).mean(dim=-1) / denom**1.5
        logit_kurtosis = (dev**4).mean(dim=-1) / denom**2 - 3.0

        if self._prev_logp is None:
            kl_prev = torch.full_like(entropy, float("nan"))
        else:
            # KL(p_t || p_{t-1}) >= 0; clamp away fp noise.
            kl_prev = (
                (p * (logp - self._prev_logp)).sum(dim=-1).clamp_min(0.0)
            )
        self._prev_logp = logp.detach()

        cols = [
            entropy,
            varentropy,
            h_alts,
            avg_logp,
            max_prob,
            margin_prob,
            top1_logit,
            margin_logit,
            std,
            logit_range,
            logit_skew,
            logit_kurtosis,
            chosen_logprob,
            kl_prev,
            *masses,
        ]
        step = torch.stack(cols, dim=-1)  # (batch, n_features)
        self._steps.append(step.detach().cpu().numpy().astype(np.float32))
        self._argmax.append(logits.argmax(dim=-1).detach().cpu().numpy())
        return scores

    def stacked(self) -> np.ndarray:
        """`(steps, batch, n_features)` float32, empty if no steps."""
        if not self._steps:
            return np.empty((0, 0, len(FEATURE_NAMES)), dtype=np.float32)
        return np.stack(self._steps, axis=0)

    def argmax_tokens(self) -> np.ndarray:
        """`(steps, batch)` int greedy tokens the collector computed."""
        if not self._argmax:
            return np.empty((0, 0), dtype=np.int64)
        return np.stack(self._argmax, axis=0)


def derive_features(
    per_step: np.ndarray,
    *,
    names: Sequence[str] | None = None,
    bases: Sequence[str] = DYNAMIC_BASES,
    short_window: int = 8,
    long_window: int = 32,
) -> tuple[np.ndarray, list[str]]:
    """Expand a single run's stored features with causal dynamics.

    `per_step` is `(steps, n_stored)`, trimmed to the run's real
    generated length; `names` gives its column names (default: the
    legacy FEATURE_NAMES order). Extra stored columns (e.g. attention
    tap moments) pass through raw. Returns the augmented array and
    column names.

    Adds, for each base signal: first difference (delta), second
    difference (acceleration), short/long EWMA, linear slope over each
    window, and rolling mean/std/min/max/median/IQR over each window.
    Also appends the position index. Every column is backward-looking
    only (position t uses steps <= t), so the same values are computable
    online with O(1) per-token state. This is a downstream transform; it
    is not part of the persisted artifact.
    """
    stored = list(FEATURE_NAMES) if names is None else list(names)
    if per_step.shape[1] != len(stored):
        raise ValueError(
            f"feature matrix width {per_step.shape[1]} does not "
            f"match its {len(stored)} column names"
        )
    steps = per_step.shape[0]
    cols: list[np.ndarray] = [
        per_step[:, i] for i in range(per_step.shape[1])
    ]
    names = list(stored)

    cols.append(np.arange(steps, dtype=np.float32))
    names.append("position")

    idx = {name: i for i, name in enumerate(stored)}
    for base in bases:
        x = per_step[:, idx[base]].astype(np.float64)
        cols.append(_diff(x))
        names.append(f"{base}_delta")
        cols.append(_diff(_diff(x)))
        names.append(f"{base}_accel")
        for w in (short_window, long_window):
            cols.append(_ewma(x, w))
            names.append(f"{base}_ewma_{w}")
            stats = _rolling(x, w)
            for stat_name, col in stats.items():
                cols.append(col)
                names.append(f"{base}_{stat_name}_{w}")

    augmented = np.stack(cols, axis=-1).astype(np.float32)
    return augmented, names


def _diff(x: np.ndarray) -> np.ndarray:
    """Backward first difference; NaN at the first (undefined) step."""
    out = np.empty_like(x)
    out[0] = np.nan
    out[1:] = x[1:] - x[:-1]
    return out


def _ewma(x: np.ndarray, window: int) -> np.ndarray:
    """Causal exponential moving average with span `window`."""
    alpha = 2.0 / (window + 1.0)
    out = np.empty_like(x)
    acc = float(x[0])
    out[0] = acc
    for t in range(1, x.shape[0]):
        acc = alpha * float(x[t]) + (1.0 - alpha) * acc
        out[t] = acc
    return out


def _rolling(x: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Causal rolling stats over each backward window of size <= window.

    Returns mean, std, min, max, median, IQR, and least-squares slope,
    each as an array aligned to `x`. NaN entries in the window are
    dropped (kl_prev is NaN at t=0); an all-NaN window yields NaN.
    """
    n = x.shape[0]
    keys = ("rmean", "rstd", "rmin", "rmax", "rmedian", "riqr", "slope")
    out = {k: np.full(n, np.nan, dtype=np.float64) for k in keys}
    for t in range(n):
        chunk = x[max(0, t - window + 1) : t + 1]
        finite = chunk[~np.isnan(chunk)]
        if finite.size == 0:
            continue
        out["rmean"][t] = finite.mean()
        out["rstd"][t] = finite.std()
        out["rmin"][t] = finite.min()
        out["rmax"][t] = finite.max()
        out["rmedian"][t] = np.median(finite)
        q75, q25 = np.percentile(finite, [75, 25])
        out["riqr"][t] = q75 - q25
        if finite.size >= 2:
            pos = np.arange(chunk.size, dtype=np.float64)[~np.isnan(chunk)]
            out["slope"][t] = np.polyfit(pos, finite, 1)[0]
        else:
            out["slope"][t] = 0.0
    return out
