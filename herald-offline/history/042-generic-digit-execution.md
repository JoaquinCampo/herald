# Generic numeric prediction execution

Design041 SHA1231e2c804705071e7af7aa78012496cb792b1e615342e2385595e282c5373c2.
Fresh official data independently regenerated,47data checks passed. No overlap
with earlier v4 prompts/contexts. Model source preflight passed. Artifacts:
results/generic-digit-data-audit.json and generic-digit-model-preflight-audit.json.

The first real CPU run preserved in results/generic-digit-cpu-owner failed
only a tuple-versus-JSON-list equality check. Its actual reference/action token
arrays already matched the original fixture. Normalizing both representations
fixed the check; the exact fixture reran successfully without regenerating the
prior. CPU explicit causal batch versus sequential logits passed at1e-5.

Collector sourceb0a7efb... passed first real Orion replay. Worker then changed
only observation timer placement and aggregate cost fields after reporting
completion. Owner preserved the exact first GPU source in
results/generic-digit-v1-replay-source.py, inspected the delta, and required
both CPU and original GPU replay on the final source before discovery.
Final source4fb16fdc2a0ff291b0f6656bcec5e5425d7f1bc743a95564d9756989bec329a2.
Final replay results/generic-digit-v1-replay-final reproduced prior reference,
action, termination and native masks exactly. Original pending argmax matches.
No unrelated processes touched; Orion GPU had only preserved keepalivePID2106.

Discovery48 completed with41positive7nonpositive signed losses, no failed cases.
Fixed Ridge1 models fit on OrionCPU sklearn1.7.2 with no parameter search.
Model825fb4db82330bd76fe57ecdfa6421362427abf4cd5bfd7f794542f2d72f6bc9.
All fitted scalers see discovery only. Evaluation started after model freeze,
with per-case features/predictions persisted before paired outcomes.
No numerical claim yet. See results/generic-digit-discovery andgeneric-digit-fit.

## Locked result and decision

All 48 evaluation cases completed: 37 positive losses and 11 zero losses.
Candidate MSE was 0.180130 versus 0.183594 for the discovery-mean baseline,
a 1.887% improvement with 25/48 strict prompt wins. The frozen requirements
were 10% and 32 wins. Raw positive-direction AUC was 0.4103, below 0.80.
All five baseline comparisons failed at least one required condition. There
is no acceptable predictor here, and no sign inversion or model revision.

The paired-prompt bootstrap 95% interval for relative MSE gain versus the mean
was [-3.43%, 6.56%]. It conditions on the frozen fit and omits model-refitting
uncertainty. Candidate MAE was 0.31033 versus mean baseline 0.30816.
Summary and complete comparisons: results/generic-digit-evaluation-summary.

Median measured observation wall cost was 0.499 seconds from prepared B0,
including the helper's compression validation. Paired probe forward time was
0.0403 seconds; prediction was 0.00052 seconds. Prefill was separately measured
at 0.365 seconds. These are experimental costs, not deployment measurements;
additional correctness replay costs are outside the observation wall timer.
See results/generic-digit-evaluation-cost.json.

Close this generic numeric probe and stop functional-probe variants in the
single-answer Knorm .10 setting. Study043 now checks a graded task/action
population before proposing another predictor. Independent outcome audit is
still pending; the data, model-source and collector/CPU audits have passed.
No genuinely unseen confirmation population was opened.

Independent discovery audit subsequently passed all 144 official scores,
features, provenance and frozen fit arithmetic. The independent reviewer hit
its usage limit before the evaluation audit. The owner completed a separate
recomputation of evaluation's 144 official scores, 48 feature vectors and 288
frozen predictions, confirming the failed gates. That check is explicitly
owner verification, not independent review. Artifact:
results/generic-digit-evaluation-owner-audit.json.
