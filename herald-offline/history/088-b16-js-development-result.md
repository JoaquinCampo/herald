# B16 JS development screen: failed

Frozen087 executed on the32 exposed071 discovery cases. One-case replay and
full32 completed successfully on Orion, sessions16353 and78265, followed by
ordinary-Python analyzer session9311. No new outcome continuations or evaluation
prompts were opened. Raw logits, per-case provenance and controls remain under
results/b16-js-development-one and results/b16-js-development-full on Orion.

All32 state, native-mask, source-cache, no-op and historical token17 argmax
controls passed. Independent SciPy Jensen-Shannon computation from raw logits
agreed within1.81e-16. Reference/no-op raw tensors matched exactly.

The frozen candidate failed every baseline comparison and the raw-AUC gate:
AUC .54545 versus required .80. LOO MSE candidate .449667; mean .240375;
observed digit progress .258707; strong structural+z+progress .321224.
Relative gains were -87.07%, -73.81%, -39.99%; strict wins18,19,17 of32,
versus22 required. All simultaneous95% gain intervals crossed zero.
Full numerical results and per-prompt predictions: results/b16-js-development-screen.json.
Bootstrap conditions on the fixed out-of-fold predictions, not refitted samples.

Median reference+action+JS observation cost .25905 seconds. Ordinary shared
prefix advancement .29843 seconds; extra mask/no-op/clone/compression audit
.50172 seconds and repeat-prefix audit .29816 seconds are separate.
See results/b16-js-development-costs.json. No deployment predictor was selected,
so no prediction-latency claim is made. Observation cost excludes old structural
feature acquisition and must not be described as total nine-feature cost.

Before GPU execution, owner inspection fixed an unequal-length strict zip in
clone-pair checking by testing every pair with combinations, and separated
observation cost from audit-only work. Neither changed frozen scientific inputs
or gates. First real replay passed after these changes; no failed GPU replay.

Decision: close raw B16 JS with this fixed model. Do not search transformations,
other severity levels or horizons on these exposed cases. This rejects the tested
observable/model combination, not a universal impossibility of damage prediction.
Return to UNDERSTAND and reconsider the decision-point/observable relationship
before selecting another experiment. No validated predictor; goal remains active.
