# Structural residual audit: the old baseline is weak

2026-09-15. Executed the no-fit development comparison defined in 069 after SSH
access recovered. No new generations, feature selection, refitting or confirmation.

All 60 saved prediction rows match the previous raw signed losses and frozen
prompt folds. All 22 raw-source hashes match 068. The original pooled baseline
MAE reproduces at 0.34947354565 and matches the preserved independent audit.
The scored population remains 12 NIAH and 8 CWE prompts, each with three actions.

## Main result, matched four-fold evaluation

| Task | Frozen structural MSE | Other-fold task/rate mean MSE | Structural error increase |
| --- | --- | --- | --- |
| NIAH | 0.3397402693 | 0.2427983539 | 39.93% |
| CWE | 0.0289721227 | 0.0116666667 | 148.33% |

The frozen pooled Ridge uses action/geometry features but no task identity.
The new arithmetic comparator uses the same four prompt folds, averaging only
training-fold outcomes for that task and rate. It has nine training prompts per
NIAH estimate and six per CWE estimate. Raw predictions are never clipped.
The larger model is not a strong matched baseline in either task.

## What privileged residual correction establishes

The correction adds the average prediction error at the other two rates of the
same held-out prompt. No target-rate outcome enters that prediction, but both
other-rate outcomes remain privileged and unavailable to a prospective predictor.

| Task | Correction of structural MSE | Correction of task/rate-mean MSE |
| --- | --- | --- |
| NIAH | 0.3397402693 to 0.1126625727 (66.84%) | 0.2427983539 to 0.1250000000 (48.52%) |
| CWE | 0.0289721227 to 0.0115192392 (60.24%) | 0.0116666667 to 0.0060069444 (48.51%) |

The two tasks are reported separately. These four-fold values do not replace
study 052's leave-one-prompt-out values. No new success threshold or p-value is
applied. Fixed-prediction deletion ranges in the JSON omit a scored prompt's
contribution while retaining every prediction, including the fitted Ridge and
task/rate averages. They are neither full training-data deletion nor intervals.

## Reflection

Repeatable within-prompt residuals survive the frozen structural predictor.
That does not establish that structural information generally fails: this fit
already loses to a simple comparator. Its larger proportional oracle gain is
partly a consequence of its worse starting error. Some gain against the simple
task/rate mean remains, but no admissible variable has been shown to capture it.

Of 069's explanations, both residual repeatability and a weak task-blind fit are
supported. The claim that the saved structural model already explains the
cross-rate association is not supported. The next strategy must separate model
misspecification from missing predictive information and use a task-conditioned
matched baseline. Do not refit a new structural learner on these closed outcomes
or select a new GPU collection from this diagnostic alone.

## Evidence and reproduction

Script: `scripts/audit_structural_residuals.py`. The actual command completed
with exit 0 using Orion's unchanged v2 uv runtime and `--data-root` pointing to v4,
with the script piped through stdin. Raw source data and saved predictions were
not modified. Inputs and output are in `results/restart-20260915/`; `SHA256SUMS`
pins the script and both JSON artifacts. No lint pass or model refit is claimed.

Independent reviewer recomputation from the input snapshot matched every task's
baseline MSE, corrected MSE and gain. The reviewer accepted the fixed-prediction
sensitivity interpretation and verified all 60 unique prompt/action identities.
