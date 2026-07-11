# Preregistration: dependency-free int8 KV cache

Date: 2026-07-11. Status: frozen before implementation and live execution.

## Hypothesis

A model-native cache that stores completed KV blocks as symmetric int8 values with explicit scales and keeps only the latest 128 tokens in bfloat16 will reduce retained KV bytes on every prompt without rollback, token eviction, or an online controller. For Llama-3.1-8B-Instruct, this may preserve paired quality while keeping end-to-end slowdown within 5% if dequantization is small relative to attention and decoding.

This is a distinct compression family from all prior token-eviction branches. It preserves every token and changes numerical precision rather than cache length or switch timing.

## Frozen mechanism

- Model/task: cached Llama-3.1-8B-Instruct, IFEval, greedy generation, existing output limits.
- Quantization: signed int8, symmetric zero point, one float32 scale per batch/head/token vector, clamped to `[-127, 127]`.
- Residual: the Transformers quantized-cache default of 128 newest tokens in model dtype. No residual-length, scale-granularity, bit-width, or mixed-precision tuning is permitted on target prompts.
- Runtime: quantize completed residual blocks, dequantize the old block for attention, and append the full-precision residual. No token eviction, rollback cache, selector, alarm, or recorded-reference substitution.
- Accounting: isolated retained KV bytes include int8 keys/values, all scales, and full-precision residual tensors. Dequantized attention temporaries and allocator peaks are reported diagnostically but are not mislabeled as retained KV.

## Data protocol

No training or calibration is required. The mechanism and all parameters are fixed before target execution. The same five evaluation-only rejection prompts remain frozen: `ifeval-1069`, `ifeval-1075`, `ifeval-1087`, `ifeval-1107`, and `ifeval-1128`. If every non-sample-size gate passes, expand to the existing 46-prompt target split with at least 30 unique live pairs. No target outcome may change quantization parameters.

## Test-first implementation boundary

First add a failing shipped-behavior test requiring a real tiny-Llama generation through the quantized cache. The test must prove that generated cache storage actually contains int8 tensors and explicit scales, retained-byte accounting includes both, sequence length remains exact, every token remains represented, and the real generation entry point emits output and a measured peak. Add numerical unit tests for quantize/dequantize error bounds and storage identity.

Implement only the custom Transformers cache layer, cache construction, retained-byte accounting, paired live runner, immutable run manifest, and tests. Do not add dependencies or alter the deployment contract.

## Validation and decision

1. Run targeted and full pytest, Ruff format/check, MyPy, and a CPU tiny-model generation.
2. Verify flat sync, Orion GPU/process state, and a clean >=10-second launch smoke.
3. Run one immutable five-prompt paired live triage and the executable evaluator with 2,000 prompt-cluster bootstrap resamples.
4. Reject without expansion after any quality, major-damage, slowdown, or positive-savings gate failure. If all non-sample-size gates pass, expand to N>=30 and then broader context/output/model/task validation.
5. Audit hidden scales, residual tensors, temporary dequantization, timing order, warmup, prompt identity, and run provenance. A failure closes only this exact int8/residual-128 implementation and advances to segmented cache or another distinct representation.
