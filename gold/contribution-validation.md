# HERALD Contribution Validation Checklist

This document records the narrowed novelty bar imposed by related
work. Use it before writing claims, launching Phase 2/3/4, or deciding
whether the result is a predictor/control paper versus a measurement
paper.

## Claim HERALD Is Allowed To Make

HERALD is not novel because "logits contain quality information."
That premise is already present in entropy-based degradation signals,
fine-grained uncertainty scoring, and lightweight per-token
predictors.

The defensible claim is narrower:

> HERALD is a black-box, online risk model for
> KV-cache-compression-attributable damage. It predicts future damage
> at useful horizons across compressors, ratios, tasks, and models, and
> the signal improves the compute-quality frontier when used for
> intervention.

Every experiment should support one of the words in that claim:
black-box, online, compression-attributable, future, transferable, or
control-useful.

## Target Discipline

The predictor contribution is invalid if the headline model is merely
a loop / non-termination detector. Diagnostic tags are useful because
they provide interpretable onsets and failure taxonomy, but they are
secondary.

Primary HERALD labels must be paired counterfactual damage labels:

- `baseline_correct AND compressed_wrong` where task correctness has a
  usable base rate.
- Sequence degradation versus the paired uncompressed output.
- Trajectory degradation versus the paired uncompressed trajectory or
  matched-prefix replay metrics.

If Phase 1 correctness is saturated, the correct move is to train on
continuous sequence/trajectory severity and report correctness as an
extrinsic validator. Do not silently replace the main target with
`looping OR non_termination`.

## Novelty Risks From Related Work

- Entropy / uncertainty methods already establish that output
  distributions carry quality signals.
- Lightweight logit/state predictors already establish that cheap
  predictors can detect some generation failures.
- HALT-style work makes cross-model transfer risk explicit.
- Looping and instruction-failure detectors can look ad hoc unless
  tied to prior failure-mode definitions.
- Controller papers already use internal proxies; HERALD must show an
  external risk signal that improves intervention, not merely describe
  one.
- LimitsLearned-style negative results create the burden to show that
  online post-token features contain more actionable information than
  position or static pre-generation proxies.

## Required Baselines

These are not optional. If HERALD does not beat them by a meaningful
margin, the predictor contribution is weak.

- Random predictor.
- Position-only predictor.
- Compression-ratio-only predictor, reported carefully because ratio is
  deployment metadata, not an online damage signal.
- Entropy threshold.
- EWMA / rolling entropy threshold.
- Online change-point detection on entropy or perplexity.
- Single-feature thresholds for each cheap feature family.
- Logistic regression on Tier 0 features.

The primary predictor must beat the best cheap baseline, not just the
weakest baseline. The Phase 2 minimum bar remains AUROC improvement
>= 0.05 over the best entropy/EWMA-style baseline with paired bootstrap
CI excluding zero; if position-only or ratio-only is stronger, report
that explicitly and treat it as a serious limitation.

## Required Transfer Experiments

The black-box compressor claim requires transfer, not just within-grid
accuracy.

- Held-out ratio: tests interpolation/extrapolation along the
  compression cliff.
- Held-out press: tests cross-compressor transfer and is required for
  the black-box claim.
- Held-out task: tests generation-regime transfer.
- Held-out model family: tests whether the predictor is model-specific
  or broadly useful.

Cross-press transfer should be a headline result, not an appendix-only
analysis. If train-on-one-press / test-on-another collapses, the paper
must say HERALD is a per-compressor calibration framework rather than a
universal compressor-agnostic predictor.

## Required Failure-Mode Hardening

Diagnostic tags are useful only if they are defensible.

- Looping and repetition thresholds should be tied to prior looping /
  repetition-ratio definitions where possible.
- Non-termination should be tied to max-length or >90%-max-length style
  criteria, not only a local heuristic.
- Instruction amnesia, leakage, and format break should be mapped to
  prior instruction-following failure categories where possible.
- Tags remain descriptive and secondary; the primary labels are paired
  counterfactual outcome and sequence/trajectory damage.

If a threshold is heuristic, label it as such and include sensitivity
analysis rather than pretending it is canonical.

## Required Feature-Information Check

Before making a strong predictor claim, quantify whether cheap online
features contain information beyond trivial proxies.

Minimum checks:

- Mutual information or equivalent feature-importance sanity check for
  Tier 0 / Tier 0.5 / Tier 1 features versus the chosen future-damage
  label.
- Compare against position-only and ratio-only features.
- Report whether feature information survives held-out ratio and
  held-out press splits.

This check directly addresses the concern that online logits may have
a low ceiling analogous to pre-generation token-importance predictors.

## Required Closed-Loop Evidence

The control contribution requires at least one real intervention demo.
The Phase 1 intervention probe bounds recoverability, but it is not a
substitute for Phase 4.

Minimum Phase 4 evidence:

- One mask-based controller demo, preferably StreamingLLM toggle /
  relax.
- One eviction-based or recompute-fallback demo if feasible; if not,
  explicitly state that the v1 controller result is mask-based only.
- Pareto curve against fixed compression, no compression, and random
  gating at matched compute.
- Predictor overhead included in the cost accounting.

Without this, the paper should frame control as future work and lean on
measurement + predictability instead.

## Decision Rules For Paper Framing

- Strong paper: HERALD beats cheap baselines, transfers across held-out
  press/ratio/task, and improves at least one closed-loop Pareto curve.
- Solid measurement + predictor paper: HERALD beats baselines within
  some regimes, but transfer or control is limited and honestly scoped.
- Measurement-methodology paper: predictor fails to beat cheap
  baselines or transfer collapses, but matched-prefix replay and the
  alignment study produce a clear, reusable evaluation framework.
- Negative-methodology paper: even damage metrics fail to align
  reliably, and the contribution is showing why compression damage is
  harder to measure than the field assumes.

Do not upgrade the claim beyond what the checklist supports.
