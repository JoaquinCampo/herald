# Next analysis: does saved structure explain cross-rate repeatability?

2026-09-14. Defined after068, before inspecting the saved OOF predictions.
This is a secondary exploratory analysis on the same exposed development data,
not another validation gate or a feature search. No model is fitted.

Competing explanations: the privileged repeatability is already explained by the
frozen structural predictor; it persists in structural residuals; or the older
pooled task-blind fit is too weak to make that comparison informative.

Use existing `results/ea-dev-v1-analysis/predictions.jsonl`, its summary, and its
independent audit. Verify 60 unique prompt/action rows, complete .05/.10/.20
actions, unchanged raw labels against068, one fold per prompt and four folds.
Record file hashes and recompute the original pooled baseline MAE as a provenance
check. Preserve all predictions without clipping. Do not refit Ridge or tune it.

Report task-stratified MSE for the saved structural baseline and a task-by-rate
mean computed only from other folds. The old pooled baseline lacks task identity;
it is not automatically the strongest matched comparator. Do not pool NIAH/CWE.
These four-fold numbers must not be compared as if they used052's leave-one-prompt
split, so label the evaluation schemes explicitly.

For each saved baseline, compute residual e[p,r] = y[p,r] - prediction[p,r].
The privileged correction for rate r is the mean residual at the other two rates
of the same prompt. Report the corrected error and per-prompt deletion sensitivity.
This checks repeatability after the baseline, using information unavailable to a
prospective predictor. No p-value, success threshold or confirmation claim is
attached to this secondary comparison.

If structure removes the advantage, first understand those already available
variables. If residual repeatability persists, it supports searching for a better
decision-time measurement without identifying one. If the structural fit is worse
than the matched task-rate mean, report that limitation and do not treat it as
evidence that all structural predictors fail. Missing predictions or failed
provenance checks return the task to recovery, not automatic refitting.

Status: not executed. After068 completed, the SSH gateway began closing new
connections before Orion was reached. The source names and schema are known
from the preserved analyzer; existence and integrity still require live checks.

Execution update2026-09-15: access recovered and the defined no-fit comparison
completed. See070 for results and the fixed-prediction deletion interpretation.
The original pre-execution design and gateway failure above are preserved.
