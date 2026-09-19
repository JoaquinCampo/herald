# GPU follow-up synthesis: demand, reserve, and context-dependent failure

> **Status: research record (2026-09-18).** Characterization only. No predictor,
> controller, or compressor is built here. This record synthesizes the completed
> 406-comparison GPU follow-up and corrected coordinate interventions. Counts are
> development-set descriptions of 8 prompts / 4 paired families on
> Qwen2.5-7B-Instruct; they are not held-out performance estimates.

## What the follow-up changes

Two strong earlier interpretations do not survive.

1. **Tail-attention loss is not equivalent to quality loss.** At answer onset,
   preserved Knorm arms can lose about 94% of L23 last-two-digit attention and
   still complete correctly. Conversely, the 02-planted Knorm .1 swap retains
   about 90.5% of that tail attention.
2. **Compressed source values are not generally inert.** Correct packed-cache
   targeting at 02-planted/Knorm .1 decision 20 changes the truth-vs-distractor
   margin from -4.75 to +15.875 when distractor values are mean-replaced and
   changes the selected token from 7 to 2. Replacing both source spans selects
   period. Legacy targeting changed none of seven compressed choices; corrected
   targeting changes four.

The useful replacement is not "compression stops reading context." Compression
changes which context-dependent computation is available and how it is used.
Damage occurs when those changes consume enough task-relevant decision reserve
at a demand where the answer actually distinguishes alternatives.

## Three clocks

Keep three events separate:

1. **cache/access perturbation** -- immediately after compression;
2. **demand-conditioned margin erosion** -- when a later query requires the
   affected information;
3. **visible answer error** -- when the eroded margin crosses a decision boundary.

A fourth, post-error clock is autoregressive self-conditioning. Repetition and
fabricated justification belong there and should not be used to explain the
first error without separate evidence.

This resolves the apparent paradox that a first digit can remain essentially
unchanged while later completion fails: the current demand may not use the
damaged part of the source strongly enough for the perturbation to matter yet.

## Fixed-query deletion dominates the L23 tail-mass change

For the 32 compressed Knorm rows at each available offset, the ordered bridge
RR -> RS -> RC -> CC separates:

- RR -> RS: restrict reference K/V to compressed columns at fixed reference query;
- RS -> RC: replace surviving reference K/V by actual compressed K/V;
- RC -> CC: replace the reference query by the compressed query.

Mean ratios (unweighted means over arms; descriptive only):

| Offset | Outcome | RS/RR | RC/RS | CC/RC | CC/RR |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | damaged (22) | .109 | .982 | .980 | .107 |
| 0 | preserved (10) | .455 | .976 | 1.078 | .450 |
| 4 | damaged (22) | .049 | .995 | 1.141 | .055 |
| 4 | preserved (10) | .448 | .997 | 1.078 | .456 |
| 6 | damaged (20) | .062 | 1.014 | 1.592 | .077 |
| 6 | preserved (10) | .548 | .999 | 1.048 | .542 |

Thus, for this scalar at L23, most of the mass reduction is already produced by
**which columns are absent under the same query**. Changed surviving K/V adds
little on average to this particular mass. Query movement can compensate or
amplify later, especially after/near failure. These are ordered contrasts, not
causal percentages.

The damaged/preserved difference is heavily confounded by family and dose.
Within-family counterexamples are decisive: 03-planted/Knorm .1 is preserved
with onset CC/RR=.0566, while .25 is damaged with .0280; 02-planted changes from
preserved at .05 (onset .9108) to damaged at .1 (.9051). Therefore no universal
tail-mass threshold is supported.

## The missing variable is decision reserve

At offset 0, all 32 Knorm arms still select the correct first digit. Oracle
minimum candidate margins are actually slightly *larger on average* in damaged
arms (29.47) than preserved arms (27.79). The onset tail perturbation therefore
does not imply current decision fragility.

By offset 4, mean oracle minimum margin is 3.44 in damaged arms and 10.36 in
preserved arms; by offset 6 it is -7.60 versus +12.84. But there is substantial
overlap. Preserved arms can be strongly depleted: e.g.
02-base/Knorm .5 has margin 4.75 at offset 4 and still scores correctly;
03-base/Knorm .25 has 8.75; 03-planted/Knorm .1 has 9.25.

Reference-to-compressed margin loss alone is also nonspecific. At offset 4,
all 10 preserved Knorm arms lose >2 logits of oracle reserve, and several lose
20+ logits while remaining correct.

The supported interpretation is therefore **reserve consumption, not drift
magnitude**: a computation can be heavily perturbed and still succeed while its
remaining task-relevant margin stays on the safe side of the relevant boundary.

This is a mechanistic description, not an operational warning, because the
"correct token" and candidate set are oracle-defined here.

## Source identity and continuation are separable competitions

The corrected 02-planted intervention gives a particularly useful local
factorization.

- Native compressed state: distractor digit 7 wins; truth-minus-distractor=-4.75.
- Mean-replace needle values: 7 still wins; margin=-7.125.
- Mean-replace distractor values: truth digit 2 wins; margin=+15.875.
- Mean-replace both source spans: period wins.

A plausible local account is that source values participate in at least two
distinct aspects of the decision: **which content wins** and **whether numeric
continuation is supported at all**. Removing the wrong source can expose the
truth; removing both can make closure win. This is not proof of two dedicated
modules, and mean replacement is not literal information deletion.

The preserved 03-base/Knorm .1 state is equally important: corrected needle
replacement changes its selected token from digit 6 to digit 1. Strong causal
sensitivity to source values therefore exists in a run whose final answer
survives. Value sensitivity itself is not a damage signal.

## Signal inventory from the completed Knorm matched-prefix table

These are descriptive thresholds over the 32 compressed Knorm arms (22 damaged,
10 preserved), not fitted predictors.

### L23 onset tail ratio CC/RR

| threshold | damaged flagged | preserved not flagged |
| ---: | ---: | ---: |
| < .03 | 2/22 | 10/10 |
| < .05 | 11/22 | 9/10 |
| < .10 | 15/22 | 7/10 |
| < .20 | 21/22 | 5/10 |

At <.20 the sole damaged miss is 02-planted/Knorm .1; preserved false alarms
include 02-base .25/.5 and the depleted 03 survivors. Tail loss is an effect
marker with substantial outcome overlap.

### Oracle minimum margin at offset 4

| threshold | damaged flagged | preserved not flagged |
| ---: | ---: | ---: |
| < 0 | 11/22 | 10/10 |
| < 5 | 13/22 | 8/10 |
| < 10 | 16/22 | 4/10 |
| < 15 | 17/22 | 2/10 |
| < 20 | 21/22 | 1/10 |

Among damaged offset-4 compressed-owner comparisons, only 11/22 are actually
ahead of the native first error (positive lead): 9 have lead 2 and 2 have lead
1. A margin<10 catches 5/11 of those; margin<20 catches 10/11 but also flags
9/10 preserved Knorm arms. So even the oracle quantity is not a clean advance
warning at this coarse sampling.

## Revised mechanism account

A useful abstraction is:

**compression perturbation -> future-demand exposure -> directed margin
consumption -> boundary crossing -> autoregressive propagation.**

This is more general than the old three output families.

- **Tail-readout failures** are cases where later continuation/position demands
  expose a perturbation strongly enough that closure or a wrong digit crosses
  the boundary. Tail attention can be severely reduced without crossing it.
- **Competitor swaps** are cases where context-dependent source competition
  changes sign. The corrected value intervention shows the wrong decision can
  remain context-driven rather than prior-only.
- **Cascade** is best reserved for temporal propagation after or around a
  boundary crossing, not treated as a separate root cause.

The decisive question is not "how much did compression change the model?" but
**"how much did it change the computation relevant to the next critical demand,
relative to the reserve protecting the correct decision?"**

## What an early signal would need to observe

Current top-token confidence is insufficient because the first-digit margin can
remain enormous while later-demand access is already altered. Aggregate source
attention is insufficient because preserved and failed arms overlap strongly.
Raw margin erosion is insufficient because heavily depleted survivors exist.

The most promising missing object is **demand-conditioned directional
vulnerability**: how a compressed state would respond to the queries that will
be needed before the task is complete, projected onto the continuation/content
competitions that matter.

The existing bridge measurements provide ingredients but not yet an operational
version. A practical signal cannot use the gold future digit or a full reference
decode. Candidate approaches must therefore be evaluated by what information
they require:

1. own-stream current logits: cheap, but mostly immediate stability;
2. compression-side internal geometry (per-head source odds, attention-output
   contributions, query sensitivity): potentially earlier, but not yet
   characterized for specificity;
3. paired reference/compressed probes: mechanistically informative but costly
   and not necessarily deployable;
4. oracle target margins: diagnostic only.

## Next evidence with highest information value

Do **not** start by adding more ratios. The current uncertainty is mechanistic.

1. **Per-head source-competition decomposition at matched pre-error states.**
   Determine whether aggregate competitor surges come from within-head odds
   reversals, differential head renormalization, or downstream value/readout
   changes. Include preserved depleted controls.
2. **Continuation-vs-content margin interventions.** At selected failure and
   survivor states, record full digit + period margins after needle, distractor,
   and both-span interventions, then continue generation only as a secondary
   outcome. This tests the local two-axis account.
3. **Future-demand sensitivity.** At onset, evaluate the same compressed state
   under controlled later digit-position queries/prefixes while keeping source
   state fixed where possible. The prediction is that failures and depleted
   survivors differ more in directional response/reserve than in current
   attention mass.
4. **Within-family disagreement-position geometry.** Still required to decide
   whether later discriminative positions are genuinely more robust or the
   original four-family law was confounded.
5. **Cross-task/cross-model replication only after the above quantities are
   defined.** Otherwise scale will multiply ambiguous measurements.

## Limits

The 406 comparisons are repeated measurements from 8 prompts and 4 paired
families, not 406 independent examples. Reference-owned and compressed-owned
prefixes answer different counterfactual questions. Offsets after native error
or closure are oracle probes, not advance warnings. The residual projection
exports use per-run RMS normalization; fixed-readout bridge projections do not,
so they must not be naively summed or compared as causal layer contributions.
The bridge is an ordered computational decomposition, not a Shapley or causal
decomposition.

No claim here establishes a deployable predictor, a universal compressor law,
or a continuous quality-severity measure.
