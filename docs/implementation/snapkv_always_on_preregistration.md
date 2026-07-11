# Preregistration: always-on SnapKV one-shot cache

Date: 2026-07-11. Status: frozen before implementation and live execution.

## Hypothesis

A one-shot, always-on SnapKV ratio-0.25 prefill compression can avoid the controller, rollback, host-transfer, and sustained-pruning costs that bound prior branches, while its recent-query attention policy preserves output quality better than positional StreamingLLM. Removing 25% of prompt KV may retain a positive end-to-end peak savings bound despite decode regrowth and remain within 5% slowdown because scoring runs only once.

This is a new compressor family and runtime mode. SnapKV has not been tested in either closed direct-cache-fork branch.

## Frozen mechanism and protocol

- Model/task: cached Llama-3.1-8B-Instruct, IFEval, greedy generation, existing output limits.
- Compressor: installed KVPress `SnapKVPress`, removal ratio exactly 0.25, default window size.
- Runtime: compressor context around the full generation call; prefill compression only, no sustained pruning.
- No controller, gate, alarm, rollback, recomputation, training, or calibration.
- Frozen five-prompt rejection triage: `ifeval-1069`, `ifeval-1075`, `ifeval-1087`, `ifeval-1107`, `ifeval-1128`.
- Parameters are fixed before target execution. If every non-sample-size gate passes, expand to at least 30 of the existing 46 target prompts and then broader validation.

## Test-first boundary

First add a failing real tiny-model test requiring the generic always-on press entry point. It must drive actual SnapKV generation, return output, and account for the uncompressed per-layer pre-hook transient, compressed retained cache, and final decode-regrown cache without using allocator peak as the deployment metric.

Implement only a generic prefill peak observer, always-on press generation entry point, paired-runner mechanism option, and direct tests. Reuse the installed press and existing evidence machinery.

## Validation and decision

Run full local quality checks, flat-sync permitted content, verify Orion state, and smoke-check the launch for at least ten seconds. Run exactly one five-prompt paired triage with 2,000 bootstrap resamples. Any non-sample-size quality, major-damage, slowdown, or positive-savings failure rejects the exact branch without expansion or parameter changes. Audit hook ordering, transient retained storage, prompt identity, timing order, and provenance.
