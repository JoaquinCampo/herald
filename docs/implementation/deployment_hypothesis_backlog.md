# Deployment hypothesis backlog

Status: active; rerank after every meaningful result.

The current bottleneck is not merely compressor quality. Both closed direct-cache-fork branches retained the full reference cache concurrently with a materialized candidate cache, and decode regrowth then erased one-shot savings. ExpectedAttentionStats sustained interval 32 preserved quality on all five triage prompts but failed speed and retained-KV gates, making runtime representation the highest-information next lever.

## Ranked hypotheses — 2026-07-11

| Rank | Technically distinct hypothesis | Expected information | Practical cost | Contract relevance |
| --- | --- | --- | --- | --- |
| 1 | **In-place compression with exact host rollback.** Copy the uncompressed cache to host memory, compress the GPU cache in place, and restore exactly from host only after alarm rejection. | Directly tests whether concurrent GPU rollback storage is the binding memory mechanism while preserving the frozen compressor and alarm semantics. It also measures the PCIe latency tax rather than estimating it. | Medium: one cache traversal, explicit restore path, no new dependency. | Removes reference-plus-candidate GPU retention; paired live quality, speed, and isolated GPU-KV evidence remain measurable. Host bytes must be disclosed separately. |
| 2 | **Irreversible pre-compression selector.** Decide from uncompressed reference features before compression, with no grace fork or rollback cache. | Tests whether HERALD can avoid rollback entirely; a negative result isolates calibration transfer from runtime storage. | Medium-high: requires prompt-disjoint selector training and a new frozen bundle. | Best possible memory/runtime shape, but false commits directly threaten quality and major-damage gates. |
| 3 | **Quantized KV cache with mixed precision, then optional token eviction.** | Tests a different compression family that may preserve token coverage and quality while yielding deterministic bytes. | High: validate Transformers cache backend compatibility and actual kernels; may require an approval-gated dependency. | Potential positive savings without rollback, but speed and dequantization overhead are unknown. |
| 4 | **Model-native segmented/indexed retained-token cache.** Attend over retained segments without materializing a contiguous candidate copy. | Separates representation cost from eviction policy and directly answers the materialization failure found in the StreamingLLM branch. | Very high: model-specific attention/runtime work and kernel-sensitive validation. | Could combine exact rollback metadata with real savings, but implementation risk is high. |
| 5 | **Adaptive layer/head token budgets using a non-fake-key representation.** | Tests whether quality failures come from uniform eviction rather than compression itself. | High: needs new prompt-disjoint calibration and physical variable-length storage. | Could improve the quality/savings frontier; fake-key AdaKV is excluded because it does not save memory. |

## Selection

Select rank 1 first. It changes the rollback representation while holding compressor, ratio, alarm, prompts, and sustained policy fixed. Therefore any memory or speed movement has a narrow causal interpretation. If host transfer makes the speed gate impossible or the measured GPU peak still includes both caches, record the result and advance to rank 2; do not tune on triage prompts.

## Retreat triggers

Return to Understand after any measurement disagreement, provenance mismatch, unexpected output divergence, or after two to three runtime-representation hypotheses stall. Reinspect allocator lifetime, cache ownership, transfer synchronization, and whether host rollback merely relocates an unacceptable deployment cost.
