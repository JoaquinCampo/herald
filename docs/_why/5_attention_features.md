# Why attention-reliance, press-score, and probe features

> **Status: historical feature investigation.** This document records the
> earlier controller workstream. Post-switch probes and controller features
> are not inputs to the current pre-switch magnitude forecaster; see
> `docs/_why/2_predictor.md`.

The controller exhaustion report established, quantitatively, that
per-token logit statistics of the reference stream cannot identify
which prompts are fragile: models extract only 0.07-0.09 budget-safe
savings against their own training label on unseen prompts, and the
worst-case deployable ceiling is ~0.057 with an oracle threshold,
against a rung of 0.10. The information is missing, not the model
capacity. Meanwhile the same analysis showed the rung IS reachable
with perfect prompt-level knowledge of damage (consensus-label
ceiling 0.115-0.738 per split). So the fix is new measurements, not
new models.

Three sources, chosen for what they measure and what they cost:

1. **Attention-reliance features** measure how much the generation
   currently depends on the cache, and how stable that dependence
   is. SPOT (arXiv 2511.10488) shows that exactly this family
   (row/column attention moments aggregated across layers) carries
   fine-grained, input-specific token relevance in ViTs at ~4
   percent overhead, and that the moments alone nearly match full
   feature sets. Our adaptation exploits autoregressive decoding:
   the full attention matrix is never needed because each step
   yields one row, and incoming (column) statistics accumulate over
   steps. This keeps FlashAttention/sdpa intact; the row is
   recomputed by a side matvec against the KV cache (the same
   pattern SnapKV uses internally). Compressor-agnostic, near-zero
   deployment cost.

2. **Press-score features** measure what the specific compressor
   would destroy, and whether that overlaps what the generation
   relies on (evicted-reliance mass). Presses already compute these
   scores to evict; capturing them is free. Compressor-specific,
   which is a feature not a bug: it converts the untractable
   "predict an unseen compressor's damage from invariant inputs"
   into "measure this compressor's overlap with observed reliance".

3. **Probe divergence** measures the consequence directly: one
   extra forward pass with the compressed cache, compared against
   the reference distribution at the same position. Highest cost
   (~1 forward per decision point, ~6 percent at stride 16) and
   the most direct signal. Phase 0 of the plan reconstructs a
   1-token probe from artifacts the sweep already stored, so the
   hypothesis is testable before any new GPU time.

Decision rule embedded in the plan: validate the cheap
reconstruction (Phase 0) before regenerating data; regenerate only
what moves the measured ceiling.

## Update (2026-07-05, evening)

The probe family is excluded from the deployed controller (its ~6
percent forward-pass cost sets a savings bar the controller would
have to clear on top of the baseline); it stays in the dataset as a
free diagnostic. The deployed design is: attention-reliance
internals as the primary (compressor-agnostic) fragility signal,
press-score features as a measured ablation of compressor-awareness.
Grounding: the consensus label clears the savings rung in label
space, so compressor-blind prediction is not information-starved at
the label level; logit internals were the wrong sensor, attention
internals are the candidate right one.
