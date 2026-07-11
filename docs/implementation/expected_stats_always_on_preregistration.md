# Always-on ExpectedAttentionStats preregistration

Date: 2026-07-11

## Hypothesis

Applying the already frozen ExpectedAttentionStats ratio-0.25 compressor
directly at prefill and every 32 decode tokens, without an alarm, gate, grace
window, cache fork, host copy, or rollback, will retain the quality observed in
the closed sustained cache-fork branch while removing enough controller and
recovery overhead to satisfy the speed and real retained-KV gates.

This isolates runtime architecture from compressor quality. It is distinct
from the irreversible learned selector because every prompt is compressed and
no feature scorer decides whether to act.

## Frozen inputs

- Model/task: Llama-3.1-8B-Instruct on IFEval.
- Statistics artifact:
  `results/calibration/expected_attention_stats_ifeval_s0_v2`.
- Ratio: `0.25`; sustained interval: `32`.
- Target IDs: the existing immutable `expected_attention_stats` test split in
  `results/expected_stats_alarm_bundle_full_s0_v2/fidelity_targets.json`.
- Triage: the same first five sorted frozen target IDs, evaluation only.
- Greedy decoding, bfloat16, existing task token cap, paired fresh live
  baselines, 2,000 prompt-cluster bootstrap resamples.

No artifact, ratio, interval, window, prompt, threshold, or implementation
choice may be changed from triage outcomes.

## Implementation and measurement

A generic always-on sustained generation entry point will accept the frozen
in-memory press. A prefill observer, sustained hook, and final cache accounting
will include the full pre-compression cache, pre-prune decode growth, layer
replacement, and final retained cache in isolated peak-KV bytes. The runner
will hash-bind the statistics artifact and record no controller or rollback.

A failing real-generation test must precede implementation. Local pytest,
Ruff, formatting, and MyPy must pass before the flat Orion sync. Orion must be
keepalive-only before launch, and the background run must survive a ten-second
process/log/GPU smoke check.

## Decision rule

The five-prompt candidate advances only if all executable gates pass:

1. paired quality-damage upper 95% CI <= 1%;
2. major-damage-rate upper 95% CI <= 1%;
3. end-to-end slowdown upper 95% CI <= 5%;
4. isolated retained peak-KV savings lower 95% CI > 0.

Any non-sample-size failure rejects this exact branch and forbids N>=30
expansion. Passing triage authorizes N>=30 only; it is not a deployment claim.
