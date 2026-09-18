# Baseline adequacy and residual-prefix result

The metadata-heavy comparisons used a weak baseline for the now-explicit
benchmark-conditioned setting. A train-fold task/action mean achieved MSE 0.15035,
versus 0.20501 for metadata plus task and 0.18616 after adding prefix16. Thus the
prefix model was 23.8% worse than the simple prior, despite beating its paired
Ridge baseline. This is a baseline problem, not predictive success.

One prespecified two-coefficient residual test then isolated the same prefix bit
without metadata overfitting. It achieved MSE 0.14009, a 6.82% gain over 0.15035,
with 3/4 fold wins. It still missed the 10% development gate. CWE MSE improved
from 0.01167 to 0.00925 and NIAH from 0.24280 to 0.22732. No task subgroup is promoted
based on this post-outcome breakdown, and no confirmation population was opened.

All residual centering, task/action priors and coefficients used training folds
only. Zero-variance bit cells received zero correction. No horizon, penalty or
feature search followed this result. Independent closed-form ridge arithmetic
matched all 60 predictions to 1.2e-16 in owner-arithmetic-check.json.

Evidence: 018 and 019 locked designs, results/task-aware-mse-audit/ and
results/residual-prefix16/. Scripts audit_task_aware_mse.py and
audit_residual_prefix16.py reproduce the calculations from existing data.
The earlier query and prefix extraction/model audit also completed successfully
in results/query-window-v1-audit.json, with no remaining issues.

## Decision

Keep all original results and their original information/metric scopes. For the
conditional-mean objective, use MSE as primary and MAE as secondary from now on;
include known task/quality-metric identity in benchmark-conditioned baselines.
This is an explicit research-contract correction, not retroactive confirmation.

Close this fixed prefix16 observation/model branch under its gate. Before any
new observation or data collection, inspect why short-prefix agreement misses
quality changes and disagreement sometimes preserves quality. Do not lengthen
the horizon by searching these outcomes. No more attention-window variants.
