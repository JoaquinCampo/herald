# Always-on Knorm preregistration

Date: 2026-07-11

## Hypothesis

A one-shot, always-on Knorm press at ratio 0.25 can provide positive real
retained-KV savings with negligible decode overhead because it scores key
vectors only once at prefill and uses no controller, rollback, sustained
compression, attention-score calculation, or learned artifact.

This is a compressor family not covered by the closed live branches. It tests
whether cheap one-time geometric selection can meet speed before investing in
a new merge/reconstruction operator.

## Frozen protocol

- Llama-3.1-8B-Instruct, IFEval, bfloat16, greedy decoding, existing output cap.
- Knorm compression ratio `0.25`, one-shot prefill only.
- The ratio is a canonical existing sweep-grid value, not selected from live
  triage outcomes.
- Target IDs come from the immutable ExpectedAttentionStats target split only
  to preserve the same evaluation population; Knorm uses no target labels,
  statistics, alarm, or bundle.
- Triage uses the first five sorted frozen target IDs, paired fresh baselines,
  and 2,000 prompt-cluster bootstrap resamples.
- No ratio, prompt, scorer, or scope may be changed from triage outcomes.

## Implementation and evidence

The existing real `generate_always_on_press` path and retained-peak observer
will be reused. Only immutable CLI mechanism wiring is added, preceded by a
failing CLI test; the existing tiny-Llama test already drives this generation
entry point. Peak accounting includes the full pre-compression prefill cache,
replacement transient as observed by the prefill hooks, decode regrowth, and
final retained cache. No allocator peak or analytical estimate can pass.

Local formatting, Ruff, MyPy, pytest, flat sync verification, Orion process and
VRAM checks, and a ten-second background smoke check are mandatory.

## Decision rule

Advance to N>=30 only if five-prompt live evidence simultaneously has quality
upper <=1%, major-damage upper <=1%, slowdown upper <=5%, and isolated
retained peak-KV savings lower >0. Any non-sample-size failure rejects this
exact branch without ratio tuning or expansion.
