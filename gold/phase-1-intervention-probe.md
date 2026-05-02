# Phase 1 Intervention Probe Pre-Registration

**Status**: design locked before Phase 1 sweep launch.

**Purpose**: measure recoverability under mid-generation intervention.
This is not a controller dataset and not a Phase 4 result. It answers
the design question that gates Phase 2 and Phase 4:

> Is reactive control feasible after damage is visible, or must the
> controller predict damage early enough to intervene before derailment?

Phase 0 showed that matched-prefix per-token JS saturates in compressed
GSM8K cells. That finding does **not** prove recovery is impossible; it
only shows that the compressed trajectory prefix becomes very different
from the uncompressed model's local continuation. This probe tests
recoverability directly.

## Scope

- Same model as Phase 1 measurement core.
- All four Phase 1 tasks: GSM8K, HumanEval, IFEval, LongBench-Single.
- Approximately 900 additional runs, targeted to keep overhead near 3%
  of the fixed-ratio sweep.
- Greedy decoding.
- Prompts sampled after the fixed-ratio sweep identifies strata, using
  the pre-registered rules below.

## Strata

For each `(task, prompt, press, ratio)` candidate, assign one stratum
using paired uncompressed baseline and fixed-ratio compressed outcomes:

- **Clear success**: baseline and compressed run are both correct or
  high-quality by the task's primary metric.
- **Boundary-unstable**: correctness or quality flips across adjacent
  ratios, or the sequence metric sits near the fixed-ratio decision
  boundary.
- **Clear compression failure**: baseline is correct or high-quality
  and compressed fixed-ratio run fails or degrades sharply.

Sampling should cover all three strata where available, with priority
on boundary-unstable and clear compression failure because those strata
identify the reactive-control window.

## Budget Allocation

Of the ~900 runs, allocate roughly:

- 60% to boundary-unstable (the load-bearing stratum for the decision
  rule).
- 25% to clear compression failure.
- 15% to clear success (sanity baseline).

Within boundary-unstable, prioritize the deployable lift-pressure arm
over reprefill (reprefill is capped at the 20% oracle subset).

If a power calculation at this allocation fails to support the "paired
bootstrap CI excludes zero" rule on the boundary-unstable stratum,
relax the rule to "point estimate plus sensitivity analysis at this
budget" rather than growing the probe. The probe is feasibility, not a
power-calibrated trial.

## Stratum Availability Contingency

If Phase 1 reveals fewer than 30 boundary-unstable candidates for a
given (task, press) pair, the probe reports on the strata that are
populated and flags the boundary regime as unmeasurable for that pair.
Phase 2 then treats that pair as anticipatory-only by default. Do not
reclassify clear-failure prompts as boundary-unstable to fill the
budget.

## Press-Specific Intervention Vocabulary

The action space is press-specific. Do not compare interventions as if
they were the same action across different press mechanisms.

| Press family | Deployable intervention | Oracle / diagnostic intervention |
| --- | --- | --- |
| StreamingLLM / mask-based | Disable or relax the mask for the next segment. | Refill/recompute from the original prompt for comparison only. |
| Continuous eviction (Knorm, TOVA where supported) | Stop further eviction for the next segment. | Refill/recompute from the original prompt. |
| Prompt-time eviction (SnapKV, ExpectedAttention) | No universal lift-pressure action after prefill; ratio reduction only if implementable without changing realized history. | Refill/recompute with full KV or lower compression. |

If a deployable intervention is not well-defined for a press, record it
as not applicable rather than forcing a no-op into the comparison.

## Offset Arms

- **Deployable offsets**: fixed token positions, fixed budget fractions,
  or fixed online thresholds. These are defined before seeing probe
  outcomes.
- **Offline-diagnostic offsets**: generated-length fractions. These are
  used only to characterize recoverability and must not be described as
  deployable.

Segment sizes: evaluate K in `{8, 16, 32}` where the intervention
mechanism supports segment-level operation.

## Primary Estimand

For each valid intervention:

`switch-at-offset-T` versus `continue-fixed`

paired on the same `(task, prompt, press, ratio, history)` wherever the
same realized history can be maintained. If the intervention requires
reprefill and therefore changes history, label the result as oracle /
diagnostic rather than deployable.

### History Preservation by Press and Intervention

| Press family | Intervention | History preserved through T? | Comparison type |
| --- | --- | --- | --- |
| Mask-based (StreamingLLM) | Disable / relax mask | Yes | Deployable, paired |
| Continuous eviction (Knorm, TOVA) | Stop further eviction | Yes | Deployable, paired |
| Prompt-time eviction (SnapKV, ExpectedAttention) | Ratio reduction post-prefill | Implementation-dependent; usually no | Oracle if no |
| Any | Reprefill / recompute with full KV | No (history changes by definition) | Oracle, unpaired |

Only "history preserved = Yes" rows produce paired estimands. The
others are reported as oracle bounds and never plotted on the same
curve as deployable interventions.

Primary endpoints:

- Quality recovered relative to fixed compression:
  `recovery = (quality_switch - quality_fixed) / (quality_baseline - quality_fixed)`
- Compute cost relative to always-uncompressed.
- Recovery latency in tokens or segments.
- Whether the intervention beats random matched-compute switching.

## Decision Rules

- **Reactive control feasible** if deployable intervention recovers
  `>= 50%` of the fixed-compression quality gap on the boundary-unstable
  stratum, with paired bootstrap CI excluding zero, on at least 2 of 4
  tasks.
- **Reactive control likely infeasible** if neither deployable
  interventions nor oracle reprefill recover `> 20%` of the quality gap
  on any task.
- **Mixed outcome** if feasibility depends on task or press. Phase 2
  then uses task/press-specific horizons: reactive targets where recovery
  works, anticipatory targets where recovery does not.

These rules are not paper-acceptance criteria. They decide how to
design the predictor and the later controller.

## Reporting Constraints

- Keep the probe separate from the Phase 1 measurement headline.
- Report the probe as an intervention feasibility study, not as a
  learned control policy.
- Do not tune the stratum rules, offsets, or endpoints after observing
  outcomes.
- Treat random matched-compute switching as the load-bearing control.

