# Exhaustion report: deployable controller, rung 1 (worst-case 0.10)

Date: 2026-07-05. Mission: `docs/implementation/mission.md`. Verdict
sought: the rung (budget respected on all 3 held-out compressors AND
worst-case savings-at-budget >= 0.10, via the unchanged locked
evaluator on canonical splits) is NOT reachable with the current
dataset's compressor-invariant reference-stream features, for any
model, score shaping, or train-only tau rule in the families below.
Only the user can accept this report and close or redirect the goal.

Verified best deployable result: worst-case savings 0.0345 at budget
on all three splits (candidate B run 4). Best per-split honest
results across all candidates: ea 0.0719, knorm 0.0552, sllm 0.0197
(never simultaneously). The rung requires 0.10 on every split at
once.

## 1. Hypothesis families with canonical results (all falsified)

All results below are from the unchanged `herald.controller_metrics`
evaluator on canonical leave-one-compressor-out splits (seed 0,
fingerprint `2a2a1b51...` matching the lock), knobs selected
train-only. Full history: `results/predictor/experiments/
experiment_log.md`.

1. **Cell position gates** (candidate A, 3 runs). Per-(task, ratio)
   earliest-safe-position thresholds fit on the 4 train compressors.
   Falsified: tau creep busts the budget on harsher held-outs (knorm
   cost 0.086/0.041); the flat-block fix respects budgets but caps
   at 0.015-0.021 savings. Cell-level thresholds cannot express
   prompt-level differences, which is where the headroom lives.
2. **Per-row worst-case risk** (candidate B, 4 runs). XGBoost on the
   max-dq-over-train-compressors label, shaped score with a
   position-0 wall. Best honest result: worst-case 0.0345 (all
   budgets respected). Falsified as a path to 0.10 by the label
   itself: thresholding the TRUE worst-of-others label with an
   oracle tau yields at most 0.0034 savings on held-out knorm. The
   worst-case label is structurally near-binary on knorm; no model
   of it can do better.
3. **Severity multitask** (candidate C, 3 runs). dq =
   f(features, compressor-severity scalar), unseen compressor scored
   at a hypothetical severity knob. Protects knorm with margin
   (severity extrapolation works: knorm cost -0.0032 in run 1), but
   two runs each busted one split by knob-selection variance, and
   the variance-hardened run 3 (seed ensembles, one-SE rule) passed
   all budgets at only 0.0072 worst-case. Safety and savings trade
   off inside a frontier that tops out near 0.03-0.07.
4. **Consensus risk** (candidate D, 1 run). P(consensus dq > 0)
   classifier + magnitude head on the mean-over-others label, the
   label family with the highest transfer ceiling (see below).
   Honest prompt-disjoint knob selection collapsed to never-switch
   on all three splits: every probability cut that admits switches
   busts at least one internal fold, exactly as the model-space
   ceiling predicts.

Session-scale context: 15 canonical controller runs across the 4
families, plus the earlier MAE-target program (linear ablations,
MLP, XGBoost, median-mean mix) on the same dataset.

## 2. Ceiling argument

Two-step bound, computed on the canonical test sets
(`scratchpad: ceiling_transfer.py, model_reach*.py`, reproducible
from the dataset alone).

**Label space (what perfect knowledge of the other compressors
buys).** Score each held-out test row with the TRUE dq of the other
4 compressors at the same grid point and give it an oracle tau (max
savings s.t. cost <= 0.01, chosen on the held-out test set itself,
so it dominates every deployable tau rule for that score):

| held-out | max_others | mean_others (best) |
| --- | ---: | ---: |
| expected_attention | 0.452 | 0.738 |
| knorm | 0.003 | 0.115 |
| streaming_llm | 0.427 | 0.449 |

The consensus (mean-others) label clears the rung on all three
splits, so the rung is information-theoretically reachable WITH
PERFECT PROMPT-LEVEL KNOWLEDGE of donor damage. This is the
knowledge a deployable model must reconstruct from features.

**Model space (what the features actually support).** Distilling
the consensus label with XGBoost from ALL 126 invariant features
(incl. cumulative trajectory stats), canonical fit pools, oracle
tau again:

| held-out | best model score | vs own label (no comp shift) |
| --- | ---: | ---: |
| expected_attention | 0.527 | 0.071 |
| knorm | 0.093 | 0.092 |
| streaming_llm | 0.057 | 0.087 |

Worst-case over splits: **0.057 with an oracle tau**. Any deployable
controller (shaping + locked train-only tau) picks points on the
same per-score (savings, cost) frontier, so 0.057 upper-bounds every
variant of the strongest family found. The rung needs 0.10.

The decisive column is the right one: scored against its own
training label (consensus), with no compressor shift at all, the
model extracts only 0.07-0.09 budget-safe savings on unseen
prompts. The binding constraint is prompt-level generalization: the
reference-stream logit features do not identify which prompts are
fragile. This is independently corroborated by the MAE-era program
on the same dataset, where linear, MLP, and GBT residuals over the
(task, ratio, position) structure all shrank to zero under
train-only prompt-disjoint CV. It is not a capacity problem
(three model classes, ensembles, early stopping, cumulative
features all tried); it is missing information.

## 3. What would unblock the rung

In order of expected value:

1. **Hybrid-stream features** (new data generation on Orion, needs
   approval). Per-token logit statistics of the COMPRESSED
   generation (e.g. divergence of the hybrid stream from its own
   recent window, or from the reference stream while it is still
   available). These directly witness compression damage per prompt
   instead of guessing it from the reference stream, and are not on
   the mission's forbidden-input list. This is also HERALD's core
   thesis, so the dataset extension serves the paper regardless.
2. **More training compressors** (new data generation). 4 donors
   spanning a wider severity/mechanism range would let severity
   extrapolation and consensus labels interpolate rather than
   extrapolate; the knorm-random Spearman 0.943 suggests mechanism
   diversity, not volume, is the lever.
3. **Rung/epsilon renegotiation** (user decision, no new data). The
   frontier at epsilon 0.01 tops out at ~0.06 worst-case even with
   oracle tau; at a looser budget (e.g. epsilon 0.02-0.03) the same
   machinery would clear 0.10, if a looser budget is defensible for
   the paper.

## 4. Status of the stop rules

- 4 consecutive canonical experiments without improving the verified
  best worst-case (severity runs 1-3, consensus run 1; rule triggers
  at 5).
- This report claims unreachability under the mission's exhaustion
  clause: 4 hypothesis families with canonical falsifying results, a
  quantitative ceiling argument, and named unblocking resources. It
  is presented to the user for a decision; the goal is not cleared
  until the user accepts it or redirects.
