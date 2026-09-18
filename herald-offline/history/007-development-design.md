# Development design: first EA predictive-value test

## Decision

After the one-case transparency and cost check in `005`, run one fresh
development namespace with 20 prompts: 12 NIAH and 8 CWE. Use three prospective
nonzero actions, removal fractions `0.05`, `0.10`, and `0.20`, plus the shared
split-boundary reference and no-op. This is deliberately milder than `0.25`,
where six of eight NIAH cases already saturated at full loss; it should preserve
more target variation while CWE supplies mixed signed outcomes.

Every prompt is one group. All four arms for a prompt stay in the same grouped
fold. Balance prompt groups within task, approximate context length, and NIAH
needle position before assigning four grouped folds. The task label is used only
for fold balance and reporting, never as a predictor.

## Competing hypotheses

1. **H1, EA adds signal.** Non-sink EA removed-mass excess has a positive
   relationship with signed loss after severity, length, position, Knorm, and
   Vnorm controls.
2. **H2, geometry explains it.** EA adds no out-of-fold improvement once the
   matched structural and action-mechanism baselines are included.
3. **H3, saturation hides signal.** The milder schedule restores NIAH variation;
   the previous `.25/.5/.75` ceiling was an identifiability failure, not evidence
   against the feature.
4. **H4, task dependence dominates.** EA helps one task family and fails or
   reverses on the other, so a pooled coefficient would conceal scope.

## Locked measurements

For every prompt/action, preserve the signed continuous target
`q_reference - q_action`, including negative improvements, zeros, failures, and
unscorable rows. Record the `005` non-sink value-weighted mean-query EA excess,
sink-removal fraction, and raw per-layer/head audit values.

The locked baseline contains removal fraction, prompt/cache length, removed
position and recency summaries, Knorm removed key-norm mass, and removed value-norm
mass. The EA model adds only the predeclared EA excess. No answer, continuation
token, final score, task name, or future observed attention may enter either model.

Fit only two deliberately simple models inside each training fold: baseline and
baseline plus EA excess. Fit any centering or scaling on training groups only,
use one fixed ridge penalty of `lambda=1.0` after train-fold scaling, and evaluate
grouped out-of-fold MAE as the primary metric, with signed error and Spearman
correlation as secondary diagnostics. Fit only complete scored rows, report every
excluded failure or unscorable row, and do not tune the feature, action schedule,
or penalty against these outcomes.

## Proceed gate

Before opening any confirmation population, require: exact shared-boundary
reference/no-op parity; independent cache copies; complete per-arm scoring; no
silent exclusions; and successful provenance/hash checks for the new namespace.
Treat the EA family as worth a larger study only if baseline-plus-EA improves
grouped out-of-fold MAE by at least 10% versus baseline, improves in at least
three of four folds, and has a positive Spearman correlation between out-of-fold
predictions and observed signed loss within each task stratum. Otherwise retain
the result as bounded negative or inconclusive evidence
and revise the hypothesis before fitting anything larger.

This experiment is exploratory development only. It must use new seeds and new
raw generator artifacts, never the cached 12-row pilot, and it does not open or
score a confirmation population.
