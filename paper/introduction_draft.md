# Introduction (prose draft)

Large language models broadcast their impending failure through their own logits. Under heavy KV-cache compression, generations collapse into looping or run past the stop token without ever producing one; long before any of this becomes visible in the output, the next-token distribution begins to deform in characteristic ways. This paper turns that observation into a runnable predictor.

KV-cache compression has become a load-bearing optimization for long-context LLM inference, since memory and decode latency both scale linearly in cache size. A growing family of compressors (StreamingLLM, SnapKV, ExpectedAttention, KNorm, TOVA, and others) trade cache fidelity for throughput, and at moderate ratios they preserve task accuracy well. At heavy ratios the picture changes. Errors stop being graceful: the model loops on a phrase or fails to terminate. These failures are user-facing, hard to roll back, and concentrated in the tails of the metric distributions that the literature tends to report.

Existing tools do not catch this in time. Benchmarks for KV-cache compression report task-level accuracy averaged across prompts, which smooths over catastrophic generations and obscures their structure. Output-level quality monitors and post-hoc detectors only fire once the model has already produced bad text. To our knowledge, no prior work asks whether the next-token distribution reveals a compression-induced catastrophe before it manifests, even though the distribution is computed at every step and is therefore free.

We treat the problem as token-level hazard prediction. At each decoding step we extract a small set of zero-cost features from the logit vector (entropy, top-$k$ probability mass, rank-based statistics, and related quantities) and label the step by whether a catastrophe begins within the next $H$ tokens. We then train HERALD, a lightweight XGBoost classifier, on this stream. We evaluate on a large GSM8K sweep across six KV-cache compressors applied to Qwen2.5-7B-Instruct, with a wide compression-ratio range and a held-out set of compressors used only at test time.

HERALD forecasts catastrophic failures with AUROC 0.97 at the one-token horizon, generalizes to held-out compressors via leave-one-compressor-out evaluation, and retains AUROC 0.88 in the pre-onset regime, when no failure is yet visible, while a rolling-entropy baseline barely beats chance. The signal is therefore not an artifact of any single compressor and not a delayed reflection of bad output already on the page; it is a structural precursor that the model itself produces.

This matters for two reasons. Scientifically, the fact that a single cheap predictor transfers across compressors suggests that compression-induced catastrophes share a latent failure trajectory in logit space, rather than being method-specific quirks. Practically, the predictor runs inside the decode loop with negligible compute overhead, opening a path to runtime mitigation: early stopping, switching decoding strategy, or falling back to a larger cache budget at the moment the warning fires.

**Contributions.**

1. We characterize the catastrophic failure modes induced by heavy KV-cache compression (looping and non-termination) and operationalize them as token-level hazard labels.
2. We release a large per-token logit dataset spanning six KV-cache compressors and a wide compression-ratio sweep on GSM8K with Qwen2.5-7B-Instruct.
3. We introduce HERALD, a lightweight predictor that forecasts these catastrophes from zero-cost logit features.
4. We show the signal generalizes to unseen compressors under leave-one-compressor-out evaluation and remains informative in the pre-onset regime, where no failure is yet visible.
5. We position HERALD as a runtime intervention substrate: a hazard score that any downstream system can act on while generation is still in flight.
