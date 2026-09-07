# Exploratory fixed prefix16 predictor

The fixed-horizon diagnostic found a directional association in both tasks,
with many exceptions. Test whether that single observation adds predictive value
to the same matched baseline rather than treating it as a classifier.

Freeze before fitting: the seven baseline features and four prompt folds from007,
train-fold StandardScaler plus Ridge(alpha=1), no clipping. The augmented model
adds only the binary first16-token disagreement observation defined in015.
Use the same signed targets and preserve every prompt/action, with no new data.

Reuse the prior exploratory proceed rule: at least10% lower grouped OOF MAE,
improvement in at least3/4 folds, and positive within-task OOF Spearman. Report
all outcomes regardless. Passing warrants genuinely unseen confirmation with
frozen models and measured observation cost; it does not itself validate a
predictor. Failing closes this fixed observation/model for the current slice.
No alternative horizon, metric, penalty, feature transform or task-specific fit.
