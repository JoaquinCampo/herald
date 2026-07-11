# Bfloat16-scale int8 cache preregistration

Date: 2026-07-11

## Hypothesis

The dependency-free int8 cache failed only end-to-end speed while passing all
quality and real-memory gates. Its implementation converts every KV element to
float32 for quantization and dequantization. Computing and storing symmetric
per-vector scales in bfloat16, matching the model dtype, will remove those
full-tensor conversions and reduce scale bytes enough to satisfy the frozen
speed contract without changing token coverage or residual policy.

This is a runtime/storage variant of the closed float32-scale int8 branch, not
a threshold or target-prompt tune. It is selected from direct implementation
profiling after the required strategic retreat.

## Frozen design and protocol

- Signed symmetric int8 values, one scale per final-dimension KV vector.
- Bfloat16 maxima, scales, division, and dequantized multiplication; clamp to
  the smallest positive normal bfloat16 scale.
- Residual length `128`, unchanged from the prior branch.
- Llama-3.1-8B-Instruct, IFEval, greedy bfloat16 generation.
- Same immutable target split and first five sorted evaluation-only triage IDs.
- Fresh paired baselines and 2,000 prompt-cluster bootstrap resamples.
- No residual, grouping, bit width, scale dtype, or prompt may be changed from
  triage outcomes.

A failing numerical/storage test must precede implementation. The existing
real tiny-Llama cache test must run through the parameterized shipped cache.
Peak accounting includes int8 values, bfloat16 scales, residual tensors, full
returned attention tensors, and layer-replacement transients.

## Decision rule

Advance to N>=30 only if five-prompt live evidence has quality upper <=1%,
major-damage upper <=1%, slowdown upper <=5%, and real retained peak-KV
savings lower >0. Any quality, tail, or memory failure rejects immediately. A
speed-only miss may expand only if the executable evaluator identifies it as a
sample-size-only uncertainty; otherwise no expansion or dtype/residual tuning
is allowed.
