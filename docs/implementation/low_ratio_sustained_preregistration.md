# Preregistration: always-on low-ratio sustained cache

Date: 2026-07-11. Status: frozen before implementation and live execution.

## Hypothesis

Applying StreamingLLM directly from prefill at a 0.05 removal ratio and maintaining that ratio every 32 decode tokens, with no reference fork, rollback, grace window, gate, alarm, or online scorer, can preserve paired quality and remain within 5% end-to-end slowdown while producing a positive lower confidence bound on real retained KV savings.

This exact configuration is outside the closed StreamingLLM direct-cache-fork branch, whose lowest ratio was 0.10 and whose runtime always retained or evaluated a rollback candidate. It tests both a lower continuous ratio and a no-controller runtime.

## Frozen mechanism and protocol

- Model/task: cached Llama-3.1-8B-Instruct, IFEval, greedy generation, existing output limits.
- Compressor: StreamingLLM, removal ratio exactly 0.05.
- Runtime: standard KVPress prefill compression plus the existing `SustainedRatioPress` every 32 decode tokens, active for the entire generation call.
- No train or calibration data and no learned decision. Ratio and interval are fixed before target execution and may not change based on target outcomes.
- Frozen five-prompt rejection triage: `ifeval-1069`, `ifeval-1075`, `ifeval-1087`, `ifeval-1107`, `ifeval-1128`.
- If every non-sample-size gate passes, expand to at least 30 unique pairs from the existing 46-prompt target split, then broader context/output/task/model validation.

## Test-first boundary

First add a failing real tiny-model test requiring an always-on generation entry point. It must prove that the shipped path reproduces ordinary StreamingLLM prefill semantics, activates sustained pruning during decode, reports the maximum retained KV storage including pre-prune growth, avoids rollback/recomputation state, and emits actual model output.

Implement only the generation entry point, paired runner, immutable manifest, and direct tests. Reuse existing compressor and sustained-pruning code; do not add dependencies or change contract metrics.

## Validation and decision

1. Run targeted and full pytest, Ruff format/check, MyPy, and real tiny-model generation.
2. Flat-sync only permitted content, verify Orion GPU/process state, and smoke-check the background run for at least ten seconds.
3. Run one immutable five-prompt paired triage and the executable evaluator with 2,000 prompt-cluster bootstrap resamples.
4. Reject without N>=30 expansion after any quality, major-damage, slowdown, or positive-savings gate failure. Do not test another ratio or interval using these prompts.
5. Audit timing order, warmup, pre-prune peaks, physical storage aliasing, prompt identity, and manifest completeness. A failure closes only this exact no-controller ratio-0.05 interval-32 branch and advances the ranked backlog.
