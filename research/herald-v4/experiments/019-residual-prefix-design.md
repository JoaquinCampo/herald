# One parsimonious residual-prefix test

The task/action mean beat the metadata-heavy models in018. Test one fixed
residual augmentation to distinguish metadata overfitting from lack of useful
prefix information. Keep the existing20 exposed prompts, four folds, prefix16
observation, signed target and benchmark-conditioned information contract.

Within each training fold, estimate mean loss and mean prefix bit separately
for every task/action cell. Regress loss minus that cell's training mean on two
task-specific columns of (prefix bit minus training-cell mean bit), using
Ridge(alpha=1, fit_intercept=False). Predict the training-cell loss mean plus
that residual correction. If training bit variance is zero within a cell,
use zero correction for that cell. Missing task/action cells are an error.
No metadata, new observation, horizon or penalty search enters this model.

All centering and outcomes used in fitting come from other prompt folds.
Primary score is equally prompt-weighted MSE. Require at least10% lower OOF MSE
than the task/action mean and wins in at least3/4 folds to warrant a larger,
genuinely unseen study. Report MAE, bias and per-task errors descriptively.
Apply once; this is exposed-data development and cannot validate the predictor.
