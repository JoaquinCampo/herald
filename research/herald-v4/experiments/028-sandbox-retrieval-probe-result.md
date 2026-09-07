# Fixed sandbox retrieval probe result

Both fixed features failed027development gates on12exposedNIAH .10 cases.
LOO mean baseline MSE0.264462810. Unforced evicted-mass model MSE0.270686022
(-2.35%gain),7/12promptwins,AUC0.71875. Cued model MSE0.244372012 (+7.60%gain),
4/12wins,AUC0.53125. Required10%gain,8wins,AUC.80 were not met by either.
The cue beat its unforced control in MSE but did not establish useful prediction.

Exact one-sided raw-AUC permutation p:unforced70/495=.1414,cue231/495=.4667.
Fixed-OOF-pair bootstrap95%MSEgain interval:unforced[-.3367,.2468],
cue[-.1806,.3213]. These are descriptive intervals on exposed examples and do
not include model-refitting uncertainty or establish validation.

All12cases168recordedcontrols passed, same18cueIDs and frozenmeasurement
7f060ffa57c3a8186a174f41133c31d4800b503bc4f79db0564aa8243c5d139b.
Owner closed-form single-variable Ridge reconstruction matched24OOFpredictions
within1e-15; baseline matched exactly. Evidence:
results/retrieval-probe-model/owner-arithmetic-check.json andsummary.json.
Root independent rawprobability/Vnorm/mask feature audit PASSED for24features,
maximumscalar difference3.33e-9, results/retrieval-probe-v1-audit.json. RawK/V
were not persisted, so no independent QKdotproduct claim is made. Measurements inretrieval-probe-v1-first/rest.
CPUfeaturetransfer/reduction costs are recorded separately fromprobe/model
memory; these fields omit cloning and do not establish production overhead.

Close the fixed B0query-attention variants without trying more phrases, windows,
heads or formulas. Following027, consider compression at a fixed later generation
point. That changes the candidate action state and therefore the estimand, even
if the unchanged reference trajectory is retained. It cannot be presented as a
solution to the original B0prediction problem. Confirmation remains unopened.
