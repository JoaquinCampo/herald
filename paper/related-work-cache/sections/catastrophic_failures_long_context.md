---
section: catastrophic_failures_long_context
papers_used: [loopllm, geometryreason, thoughtanchors, preplananchor]
papers_cited_in_passing: []
---

## Catastrophic Failure Modes in Long-Context and Reasoning LLMs

HERALD's central premise is that the failures it forecasts (looping, non-termination, instruction drift) are not artifacts of compression alone but well-attested structural failure modes of long-context, autoregressive generation. The literature documents these phenomena along three axes: surface-level degeneration of the output stream, geometric collapse of the reasoning trajectory, and drift of the planning or instruction-following signal across long horizons.

Looping and non-termination are the most overt of these failures. LoopLLM demonstrates that autoregressive LLMs are structurally vulnerable to repetitive generation, with adversarially induced runs reaching over 90% of the maximum output length compared to roughly 20% under benign decoding across 12 open-source models [loopllm]. The same work shows that low-entropy, plausible-looking repetition evades simple entropy-based filtering, indicating that surface fluency is preserved even as the generation collapses into a degenerate cycle [loopllm]. This establishes looping and length-saturation as bona fide failure modes in modern decoders, not edge-case artifacts: if adversarial prompts can drive a model into this regime, sufficiently aggressive perturbations of the inference path (such as KV-cache compression) can plausibly do the same without an attacker.

A second strand of work shows that reasoning trajectories themselves carry structure that can degrade. Spectral analysis of attention graphs distinguishes valid from invalid mathematical reasoning with effect sizes |d| >= 2.09 across seven transformer families, reaching 85-95.6% classification accuracy and up to 95.6% with calibrated thresholds [geometryreason]. The fact that validity has a measurable internal signature implies, conversely, that invalidity is a structured regime that internal signals can flag, consistent with HERALD's premise that catastrophes leave per-token traces.

A third strand localises where these trajectories break. Thought-anchor analysis shows that a small subset of sentences, typically those involving planning or uncertainty management, exerts outsized counterfactual influence on the rest of a chain of thought, with specialised attention heads consistently routing focus toward them [thoughtanchors]. The preplan-and-anchor work refines this picture: LLMs exhibit a recurring rhythm in which preplan tokens carry +51.97% higher entropy and anchor tokens receive concentrated future attention influence, and disrupting this rhythm visibly degrades reasoning [preplananchor]. Plan or instruction drift is therefore a documented consequence of perturbing these structural tokens, not a hypothetical mode.

HERALD predicts a subset of these phenomena (specifically those triggered by KV-cache compression) ahead of onset, complementing this descriptive literature with a forecasting capability.
