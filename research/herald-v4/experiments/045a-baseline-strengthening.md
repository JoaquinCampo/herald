# Prospective baseline correction before outcome collection

Independent preflight accepted the signed-target and evaluation mechanics but
identified a comparability risk: applying alpha1000 to six aggregate features
and to118 correlated features need not produce equivalent shrinkage. A gain
against only heavily penalized low-dimensional models could exaggerate the
information added by head-level measurements.

Before any study045 discovery or evaluation outcomes, add a fourth baseline:
StandardScaler plus ordinary LinearRegression on the first six structural and
aggregate fields, with the same .25 row weights. No tuning or feature search.
Keep the candidate and all original fixed-Ridge baselines unchanged. Require
the candidate to beat this additional baseline by the same10%MSE and43/64wins
criteria as every original baseline. The within-prompt gate remains unchanged.
Record the unpenalized fit coefficients and source hash. This amendment only
strengthens acceptance, and no outcomes motivated it.

Also emit reference-quality and task-loss-distribution reports that were
required by045 but omitted from the initial model script. Neither changes
features, fitting, predictions, losses or acceptance thresholds.
