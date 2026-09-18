# Strategic retreat after the future-query oracle

## Controlling evidence

The fixed future-query oracle failed both gates on all 64 exposed study045
evaluation prompts. Within-prompt concordance was 0.670635 versus the required
0.81, and prompt-level Spearman was 0.276713 versus 0.70. The pending-query
control Spearman was 0.032601. Subject to the ongoing independent batch audit,
the operational rule closes B0 mask and attention-risk work for this NIAH
task/action family. This is not a mathematical impossibility claim.

Moving from B0 to B16 is not the next test. Study029/030 retained the exact same
eight-loss/four-zero label vector as B0, and its fixed observation missed the MSE
and prompt-win gates. It does not establish that all later boundaries fail, but
it gives no evidence that another timing move is the highest-learning choice.

The earlier common-word task is also a poor retreat target. Its exposed reference
mean was 0.825 and Knorm .05 effects were mostly zero. Reusing it would revisit a
known weak assay rather than test a fresh assumption.

## Competing hypotheses

1. **Numeric NIAH is the limiting population.** Exact seven-digit copying and
   discrete string matching turn modest decoder changes into unstable failures.
   A nonnumeric multi-item reasoning task should yield usable partial signed loss.
2. **B0 Knorm damage is generally hard to assay.** Under a different task, the
   same action will still produce mostly total failures or no effect, so changing
   the task alone will not create an informative target.
3. **Reference task competence is the bottleneck.** Qwen2.5-7B may not solve
   long variable chains reliably enough for paired compression loss to be clean.
4. **Action variation, rather than task mechanism, is missing.** Fixed .05 may
   yield little outcome range even when the reference is strong; this pilot will
   reject the population without starting a severity search.
5. **NIAH was unusually favorable.** Its known schema and perfect references may
   have reduced noise; a relational task may make final loss less reproducible.

## One smallest acceptance slice

Run one no-predictor population pilot on 16 fresh official RULER `vt` prompts:
`variable_tracking`, noise haystack, one chain, four hops, length 4096, and fixed seed2026090750. Use the pinned Qwen2.5-7B revision, original B0 split-prefill
boundary, native Knorm .05, greedy decoding, and the official 30-token horizon.
This changes only the task family and its required horizon. Do not vary action,
boundary, length, chain count, hops, template, seed, or generation cap.

Each prompt has five variable names assigned through one chain. Apply RULER's
unchanged postprocessing and `string_match_all`; each official answer contributes
one fifth of final score. Reference is the independent full B0 continuation,
action is the native compressed clone, and signed loss remains reference score
minus action score. Preserve improvements, imperfect references, failures, raw
outputs, token IDs, termination, source state, native mask hashes, and physical
cache effects. No feature collection or fitted model.

Before interpretation require exact reference/noop identity, independent and
immutable boundary states, exact task provenance, all 16 completed records, and
independent recomputation of every official score. Any technical failure blocks
the assay and must be reproduced on its original case before continuing.

Population viability requires reference mean at least 0.90, at least 12 of 16
perfect references, at least four strictly partial signed losses with
`0 < loss < 1`, at least three distinct signed-loss values, at least three zero
and three positive losses, and sample standard deviation at least 0.10. These
are assay gates, not prediction success.

If all gates pass, the project gains a viable nonnumeric, multi-hop population
for a later prospective predictor decision. It establishes only that task and
action outcomes are measurable. If any gate fails, stop this fixed VT slice
without changing severity or task parameters, and return to whether the research
objective needs action diversity or a different benchmark class. Do not add a
query proxy, attention aggregation, functional probe, nonlinear learner, head
selection, or post-hoc predictor to these 16 outcomes. No confirmation data is
generated or opened.
