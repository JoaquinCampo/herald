# Action assumption review after the VT pilot

## Why pause collection

The corrected fixed VT slice had 16 perfect references and 16 perfect Knorm .05
actions, so every signed loss was zero. The recent fixed-action populations have
alternated between no effect, near-total loss, and partial but poorly predicted
loss. Choosing another task, rate, boundary, or feature now would mostly test a
new operating point without resolving whether prompt susceptibility is stable
across candidate actions.

Fixed-action studies were valid tests of conditional loss for their exact action.
They did not artificially invalidate the estimand. Repeating them has become
low-value because each study reveals only one vertical slice of the
prompt-by-action response surface, and the chosen slice has repeatedly had weak
variation or weak prompt-level predictability.

Study007/010 already contains one independently audited action-response matrix:
12 NIAH and 8 CWE prompts, each evaluated at B0 native Knorm fractions .05, .10,
and .20 against one shared reference. Its fitted EA predictor failed and remains
closed, but the raw signed outcomes can still answer a different exposed-data
assumption question without new GPU work or a new observable.

## Competing hypotheses

1. **Stable prompt susceptibility exists.** Prompts that are unusually harmed at
   one Knorm fraction remain unusually harmed at the other fractions. A later
   prospective study should model prompt and action jointly rather than search
   for a favorable fixed rate.
2. **Action rate dominates.** Task-by-rate means explain nearly all repeatable
   variation, leaving little prompt-specific headroom for HERALD under Knorm.
3. **Prompt and rate interact discontinuously.** Damage rankings cross as the
   fraction changes, so a scalar notion of prompt fragility is not stable enough
   to support a simple action-response predictor.
4. **Task mechanics dominate.** NIAH and CWE have different reference quality,
   score granularity, and damage regimes; an apparent pooled response would be a
   task mixture rather than transferable susceptibility.
5. **This Knorm/RULER line is the wrong substrate.** If even privileged
   same-prompt cross-rate outcomes do not transfer, more rate selection within
   these tasks is parameter chasing. A later restart should change the compressor
   family or benchmark class from a literature-grounded design.

## One bounded recommendation

Run one analysis-only action-identifiability audit on the unchanged 60 compressed
outcomes from study007/010. Do not collect data, fit a feature predictor, alter
scores, or pool tasks. Preserve signed improvements. NIAH is the primary stratum
because all 12 references were perfect; report CWE separately because its mean
reference score was 0.825.

For each task and prompt, let `D[p,r]` be signed final loss at
`r in {.05,.10,.20}`. Report the full matrix and exactly three diagnostics:

1. The fraction of prompts satisfying `D[p,.05] <= D[p,.10] <= D[p,.20]`.
2. Kendall tau-b across prompts for `.05` versus `.10` and for `.10` versus
   `.20`, retaining ties. An undefined correlation is a failed diagnostic.
3. A privileged leave-one-rate-out susceptibility check. For prompt `p` and
   rate `r`, center `D[p,r]` by the mean at that task-rate over all other prompts,
   excluding `p`. Predict the centered held-out-rate loss by the mean of `p`'s
   centered losses at the other two rates, with their means also excluding `p`.
   Compare squared error with the zero prediction, which is the leave-one-prompt
   task-rate-mean baseline. This uses other action outcomes from the same prompt
   and is only an identifiability diagnostic, never an admissible B0 predictor.

The action-response assumption is worth one prospective joint prompt/action
study only if NIAH has at least 8 of 12 monotone prompts, both adjacent-rate
Kendall tau-b values are finite and at least 0.40, and the privileged
leave-one-rate-out prediction reduces squared error by at least 20 percent over
the task-rate mean. All three requirements must pass. CWE cannot rescue a failed
NIAH decision and receives no subgroup promotion.

If the gate passes, the next design may use a genuinely different benchmark
class and a fully predeclared action schedule, retaining every rate as part of
the estimand rather than selecting one after outcomes. It still needs matched
task-by-action baselines and locked evaluation. This review does not authorize
that collection or choose its benchmark.

If any requirement fails, stop new Knorm/Qwen/RULER prediction experiments in
this project. Do not search milder fractions, timing boundaries, query proxies,
aggregations, nonlinear learners, or task subgroups. The only justified restart
would begin with a literature and benchmark review that changes the compressor
family or task-quality setting and states why the new substrate addresses the
observed failure. Keep all existing confirmation populations unopened.
