# Introduction

## Purpose of the introduction

The intro has one job: convince a reader who never opens any other section to remember the paper. It does that by answering four questions in order:

1. **Why should I care?** (the problem)
2. **What is the gap?** (why existing tools or papers don't solve it)
3. **What did you do?** (the contribution, in one sentence)
4. **What did you find?** (the headline result, in one sentence)

Then it usually closes with an explicit bulleted contributions list, so a skimming reviewer can extract them in five seconds.

## What this specific intro should mention

### The setup (why care)
- KV-cache compression is a load-bearing optimization for long-context LLM inference: memory and latency scale linearly in cache size, so compressors like StreamingLLM, SnapKV, ExpectedAttention, etc. are now standard.
- At heavy ratios these methods do not just degrade accuracy gracefully. They cause discrete, operational failures: looping and non-termination. These are user-facing and hard to roll back from.

### The gap
- The KV-press literature evaluates compressors with task accuracy averages, which hide catastrophic tails.
- Existing output-level quality monitors detect failure only after the model has already produced bad text.
- No prior work asks whether the model's own next-token distributions reveal an impending catastrophe before it manifests.

### Our angle / what we do
- We treat catastrophic failure as a hazard prediction problem on a token-level time series of logit-derived signals.
- We introduce HERALD: a lightweight XGBoost trained on zero-cost per-token features (entropy, top-k mass, ranks, etc).
- We evaluate on a large GSM8K sweep across six compressors with Qwen2.5-7B-Instruct.

### Headline result
- AUROC 0.97 at one-token horizon, 0.88 even pre-onset, generalizes to held-out compressors. (Reuse abstract numbers, do not introduce new ones.)

### Why it matters (the "so what")
- The signal is structural across compressors, suggesting compression-induced catastrophes share a latent failure trajectory.
- The predictor is cheap enough to run inside the decode loop, opening a path to runtime mitigation (early stop, switch decoding strategy, fall back to uncompressed cache).

## Contributions list (4 to 5 bullets is the sweet spot)

1. We characterize catastrophic failure modes induced by heavy KV-cache compression (looping and non-termination) and operationalize them as token-level hazard labels.
2. We release a large per-token logit dataset across six compressors and a wide compression-ratio sweep on GSM8K with Qwen2.5-7B-Instruct.
3. We introduce HERALD, a lightweight predictor that forecasts catastrophes from zero-cost logit features.
4. We show the signal generalizes to unseen compressors (LOCO-CV) and is informative pre-onset.
5. We position HERALD as a runtime intervention substrate.

## Decisions

- **Lead framing**: scientific. We open with the observation that LLMs broadcast their impending failure through their own logits. It is more memorable than the systems-cost framing and the pre-onset result is the strongest hook we have. The runtime-intervention angle still appears, but lands in the "so what" near the end of the intro.
- **Teaser figure**: yes, ending page 1. A logit signal trace (entropy or top-k mass) overlaid with the HERALD hazard score on a single sequence, with the catastrophe onset marked, so the reader sees the warning fire before the failure. We leave a placeholder for it when we move to LaTeX.
- **Background on KV-cache compression**: one sentence in the intro, full treatment deferred to the related-work or background section. The intro should not turn into a literature review.
