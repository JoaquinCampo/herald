# Boundary JS diagnostic and fixed follow-up design

The fixed prefix observation can disagree for different reasons. In the
existing paired continuations, `000/.05` eventually loses one answer digit,
while `006/.10` differs in the continuation but still preserves the provided
answer. A first-pending-token full-vocabulary Jensen-Shannon measurement gives
one earlier, model-state-local description of the action effect without using
future continuation tokens or the final score.

## Locked boundary measurement

Use the existing Qwen checkpoint, tokenizer, rendered chat prompts, shared
last-prompt boundary, seed, and official scorer. For every development row,
probe the reference and each Knorm action `.05`, `.10`, and `.20` from
independent clones of the same boundary. The reference and no-op probes must
have exactly equal logits and normalized probabilities. Each action records the
full-vocabulary Jensen-Shannon divergence, both entropy values, top-two
margins, argmax IDs and agreement, probability sums, maximum probability and
logit differences, and the Knorm kept-index hash and physical cache effect.

The measurement is accepted only when the source and boundary caches remain
unchanged and disjoint, reference/no-op logits are exact, all probabilities
are finite and normalized, and every action has the expected per-layer/head
length and physical byte reduction. If prior paired records are supplied,
first argmax, mask hash, and retained lengths must also match them. Raw logits,
probabilities, and kept indices are retained for audit. This is an exposed-data
diagnostic and does not fit a predictor or open confirmation data.

The first-pending-token JS is an offline observation cost. It measures the
action's immediate distributional change, not final task-quality loss, and it
does not establish that a production system can obtain the feature at zero
latency.

## Fixed offline comparison if the diagnostic supports it

If the diagnostic warrants one follow-up, compare exactly two models on the
existing 20-prompt development set and four fixed prompt folds:

1. `B0`, the locked task-by-action baseline used by the prior MSE audit.
2. `B0 + JS`, adding two task-specific centered JS residuals, one for each
   task, with centering, standardization, and Ridge(alpha=1) fit inside each
   training fold only.

The residuals are fixed before fitting as action JS minus its training-fold
task/action mean. No horizon, action, entropy, margin, argmax, transform,
penalty, subgroup, or feature search is allowed. Report pooled and per-fold
MSE, MAE, signed bias, and within-task rank correlations. The proceed rule is
at least a 10% lower pooled MSE and improvement in at least 3 of 4 folds. A
failure closes this fixed JS branch; a pass only justifies a separately
designed confirmation with the feature extraction cost measured.

The boundary probe is implemented in
`scripts/measure_boundary_js.py`. The tiny real-HF CPU proof uses the isolated
`tiny-js` manifest with prior checks skipped; production evidence requires the
locked model and all selected development rows. Do not infer deployment or
causal claims from this diagnostic alone.
