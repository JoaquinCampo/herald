# Preregistration: in-place GPU compression with exact host rollback

Date: 2026-07-11. Status: frozen before implementation and live execution.

## Hypothesis

For the frozen ExpectedAttentionStats ratio-0.25 sustained-interval-32 candidate, replacing the concurrent GPU reference cache plus compressed fork with an exact host rollback image will make the lower 95% confidence bound on isolated retained GPU-KV savings positive without changing generated outputs or the frozen alarm decision. The principal risk is that synchronous device/host transfer pushes the slowdown upper bound above 5%.

This is technically distinct from both closed branches: candidate tensors reuse the original GPU cache object and no full rollback cache remains on GPU during grace evaluation. Host rollback bytes and transfer time are recorded rather than hidden.

## Frozen inputs and identities

- Model/task: cached `Llama-3.1-8B-Instruct`, IFEval, greedy generation, existing maximum output length and stride.
- Compressor: existing provenance-bound ExpectedAttentionStats calibration artifact `results/calibration/expected_attention_stats_ifeval_s0_v2`, ratio `0.25` only.
- Controller: existing frozen alarm bundle `results/expected_stats_alarm_bundle_full_s0_v2`; no model, feature, threshold, ratio, stride, or interval refit.
- Train/calibration population: exactly the prompt identities and source hashes already frozen in that alarm bundle and calibration metadata. The implementation experiment introduces no training.
- Five-prompt rejection triage, evaluation-only: `ifeval-1069`, `ifeval-1075`, `ifeval-1087`, `ifeval-1107`, `ifeval-1128`.
- Held-out expansion if and only if every non-sample-size triage gate passes: the remaining prompt-disjoint frozen target identities in the existing 46-prompt bundle target split, yielding at least 30 unique paired live prompts in one immutable run manifest. No triage identity may be used for tuning.
- Broader validation is forbidden until N≥30 passes all four gates.

## Test-first implementation boundary

Before implementation, add a shipped-behavior test that fails because host rollback mode is unavailable. It must drive the real live-controller entry point and prove:

1. the grace candidate reuses the original cache object rather than a GPU fork;
2. the host image is byte-exact and detached from candidate mutation;
3. rejection restores keys and values exactly and reference generation remains identical;
4. commitment releases the host image before continuation; and
5. peak accounting includes candidate GPU KV, separately records host rollback bytes, and never counts host bytes as GPU savings.

Implement only cache host-copy, in-place compression, exact restoration, accounting, CLI selection, manifest provenance, and tests. Do not alter the compressor, alarm, quality metric, or contract.

## Validation sequence

1. Run targeted pytest and observe the new test fail before implementation.
2. Implement the minimal path; run targeted tests, full pytest, Ruff format/check, MyPy, and a CPU actual-flow smoke.
3. Verify Orion has only the intended keepalive and clean experimental VRAM; flat-sync only permitted content with no deletion and verify checksums.
4. Run one non-triage mechanism smoke only to confirm exact rejection/commit behavior, accounting, logs, and ≥10-second process health. Do not select parameters from it.
5. Run the five frozen triage prompts once with paired live baselines, 2,000 prompt-cluster bootstrap resamples, and the executable evaluator.
6. Reject without expansion if quality upper >1%, major-damage upper >1%, slowdown upper >5%, or peak-KV savings lower ≤0%. A sample-size-only inconclusive gate may justify the preregistered expansion; no other failure may.
7. If all non-sample-size gates pass, run the frozen held-out expansion to N≥30 and then broader validation.

## Decision and audit

Only paired live evidence from the in-place runtime path decides. Replay savings, analytical transfer estimates, allocator peak, or the earlier direct-fork results are diagnostic only. Audit timing order, warmup, synchronization, host-memory disclosure, exact restoration, prompt identity, bundle hashes, and run-manifest completeness. A rejection closes only this exact host-rollback implementation and advances the ranked backlog.
