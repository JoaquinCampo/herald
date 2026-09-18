# Baseline and estimand audit, locked before execution

Scope is now explicitly benchmark-conditioned: task/quality-metric identity is
known before generation. It belongs in every comparator. This changes the earlier
restricted information contract, not the old results or their recorded verdicts.
No general task-identity inference capability is claimed.

Expected signed loss is a conditional mean. Use prompt-weighted squared error
as the primary predictive score; absolute error, signed bias and per-task error
are descriptive secondary measures. MAE alone targets a median, so the earlier
MAE thresholds cannot validate this mean estimand by themselves.

Use existing20 exposed development prompts and their fixed four grouped folds.
Compare once: (B0) task-by-action training-fold outcome means; (B1) the existing
seven metadata features plus task indicator with train-fold StandardScaler and
Ridge(alpha=1); (P) B1 plus only the fixed prefix16 disagreement bit.
No clipping, new observations, horizon/feature/penalty search or test-fold fitting.
All three actions for each prompt remain together and carry equal prompt weight.

Only consider a larger prespecified study if P reduces grouped OOF MSE by at
least10% against EACH baseline and improves in at least3/4 folds against EACH.
This development gate is applied once and is not confirmation. A future claim
still requires frozen-model genuinely unseen evaluation, uncertainty and measured
observation cost. No GPU work is needed for this audit.
