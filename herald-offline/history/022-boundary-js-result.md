# Boundary JS result

The fixed raw immediate full-vocabulary JS residual model failed the021
exposed-development gate. MSE rose from0.150345679 for task/action train-fold
means to0.175806672, a16.93% deterioration. It won0/4 folds. MAE also rose
from0.294074074 to0.313496046. No confirmation prompts were opened.

All20 measurement cases completed, all720 persisted controls passed, and
owner recomputation of60 JS values from saved logits matched within4.4e-15.
This includes unchanged independent source/boundary caches, exact ref/noop
logits and prior first-token, compression mask and length identity. Evidence:
results/boundary-js-v1-launch/owner-raw-check.json, boundary-js-v1-first/rest,
and results/boundary-js-model/summary.json and predictions.json.

The feature/model was fixed before measurement outcomes. Its only predictors
were task/action training-cell mean loss and two task-specific centered,
training-fold standardized raw JS columns with Ridge1 and no intercept.
Signed losses and prompt folds were retained. Independent closed-form prediction arithmetic and raw-label/fold audit PASSED,
results/boundary-js-model/independent-check.json. Mean synchronized observation
cost was14.87ms shared reference plus40.90ms candidate probe and JS reduction,
55.76ms combined for one candidate. These recorded diagnostic costs exclude
audit controls and do not establish production latency.

Close this fixed immediate-JS branch. This result does not rule out delayed
answer-bearing damage: the observation is at the pending final prompt token,
while examples show first16generatedtokens can be boilerplate. Return to
UNDERSTAND before selecting another feature, horizon or observation. A read-only
primary-source strategy review is underway; no new GPU experiment is authorized.
