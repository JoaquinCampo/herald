# Segmented attention preflight preregistration

Date: 2026-07-11

## Hypothesis

Llama decode attention can consume two physically separate retained-KV segments
through native CUDA flash-attention calls and combine their outputs exactly via
per-query log-sum-exp metadata. This would avoid concatenating sink and recent
KV tensors into a candidate cache while preserving the same attention result.

This is technically distinct from the closed cache-fork, host-rollback,
quantization, and always-on press branches: it changes the model attention
representation and kernel dispatch rather than the eviction policy or
controller.

## Preflight protocol

No task prompts or quality outcomes are used. Synthetic bfloat16 tensors match
Llama-3.1-8B-Instruct decode geometry: batch 1, 32 query heads, 8 KV heads,
head dimension 128, one query token, and representative retained lengths. The
preflight will:

1. compare two-segment output against one contiguous native flash-attention
   call using identical KV values;
2. verify generalized-query-attention head geometry without repeating KV
   tensors;
3. measure synchronized CUDA latency after warmup, with alternating order and
   enough repetitions to report stable medians;
4. inspect retained and transient tensor bytes, excluding allocator-wide peak;
5. preserve the existing keepalive and run only after Orion is otherwise clean.

## Frozen preflight decisions

- Two segments represent attention sinks and recent tokens.
- Combination uses each native kernel's returned log-sum-exp; no attention
  logits or contiguous KV candidate may be materialized.
- Numerical acceptance: maximum absolute output error at most `0.02` and mean
  absolute error at most `0.002` in bfloat16 relative to contiguous native
  flash attention.
- Runtime interpretation: this microbenchmark cannot pass the deployment
  contract. It may only reject the representation if segmented decode is more
  than 25% slower at every representative length or requires repeated/contiguous
  KV storage. Otherwise it advances to a real model-path implementation and
  frozen five-prompt paired triage.
- No thresholds, segment sizes, compressor ratios, or policies will be selected
  from the five frozen triage prompts.

## Advancement and claim limits

A passing preflight authorizes only a test-first Llama model-path prototype.
That prototype must use real retained-byte accounting and the frozen paired
triage before any N>=30 expansion. A failed preflight closes only the tested
native two-call flash-attention representation, not all segmented or paged
attention kernels.
