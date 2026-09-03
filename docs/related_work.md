# Related work

## FlashMemory-DeepSeek-V4 (LSA), Wang et al., 2026

Tencent AI Lab, HKUST-GZ, Tsinghua. arXiv:2606.09079v2. A suspended-project
technical report, preliminary by its own admission (no ablations on its two
key hyperparameters).

**What it does.** Lookahead Sparse Attention (LSA) attacks the same cost
problem HERALD studies (KV-cache memory in long-context inference) from the
opposite end. A small Neural Memory Indexer, built on DeepSeek-V4's native
Lightning Indexer, runs every 64 decode steps and *predicts which compressed
KV chunks the next window of tokens will need*, fetching only those from a CPU
cold pool into GPU memory. Reported: matches or beats the full-cache baseline
(+0.6 avg accuracy) at 13.5% of the memory footprint.

**Contrast with HERALD.** LSA *selects* which KV to keep, by predicting future
relevance. HERALD *predicts the damage* compression causes, from cheap
per-token logit statistics, without choosing what to evict. Their indexer is a
selection policy; HERALD's predictor is a damage forecaster. The two are
complementary, not competing.

**Why their failure modes motivate HERALD.** LSA breaks exactly where a damage
forecaster would help, because an indexer is blind to damage its own selection
score cannot see:

- *MRCR collapse* (§3.3.2): accuracy drops 76.0 to 48.0 on a
  dense-global-memory benchmark. Even fed 50% of the true golden chunks it
  loses accuracy. The selection score does not register that compression has
  damaged the output.
- *Context-independent leak* (§3.3.1): a fixed 0.5 selection threshold tuned
  at 125K context leaks false positives at 500K. A threshold that is safe at
  one scale mis-fires at another.

A system like LSA could consult an online damage predictor to decide when its
own selection is unsafe and fall back. That is the HERALD-shaped gap.

**Transferable techniques.** Their label-denoising pipeline (Cross-Layer
Majority Voting: a single layer's indexer score is noise, agreement across
layers is signal), their zero-imbalance training (focal loss, fixed negative
sampling), and their measured 2x context-length generalization ceiling are
all relevant to HERALD's open design questions.

## SPOT, Schlesinger et al., CVPR 2026 (Findings)

Duke, Princeton, Apple. Token sparsification for Vision Transformers: a
lightweight plug-in MLP predicts which image patch tokens are redundant and
prunes them, up to 40% FLOPs saved at equal or better accuracy. Same class as
LSA (a selection policy that reuses already-computed model internals), in a
different domain. Like LSA it *selects what to drop*, not what HERALD does
(forecast damage).

Its value to HERALD is direct evidence for cross-layer features: SPOT feeds
its predictor the moments (mean, variance) of each token's attention
distribution aggregated *across layers*, and reports that this aggregation
cuts the relevance estimator's variance by roughly a factor of L (the number
of layers), mitigating single-layer noise. That is published support for
(a) cross-layer statistics as cheap predictor features and (b) aggregating
heterogeneous per-layer signals to denoise an estimate.

## Scope note

LSA is a *learned* compressor that needs the backbone in the loop, a
different class from HERALD's weight-free eviction compressors. It is not in
the current sweep. The current paper trains one model per known compressor and
makes no cross-compressor transfer claim.
