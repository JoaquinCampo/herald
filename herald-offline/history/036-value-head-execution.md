# Design035 execution provenance

Generated48discovery and48evaluation official four-key NIAH rows with frozen
seeds. Metadata and canonical JSONL hashes live in data/value-head-v1/metadata.json.
No generated outcomes have been used in designing the adapter or model.

Runner accepts JSON arrays, so the canonical JSONL files were converted without
row changes into discovery.json (SHAe1fb557e19e9225e3dee78049261374424b2c4e9804fcdc0e5c466219dc03b54)
and evaluation.json (SHA626a27f875fb1684b092d945a5b965384c6b321852ebc888c68725bd23f2f11c).
Exact parsed-array equality was asserted. source_index is the official character
offset, not a filtering/selection index. All48rows from each generation retained.

Collector original tiny CPU reproduction passed all checks and exact reference,
noop and action token IDs, termination and official scores against the existing
needle-rescue-v1-cpu artifacts. Results/value-head-cpu-proof is the new record.

First GPU launch used a mistyped exposed case ID and failed before model load.
Original error preserved remotely as value-head-old-replay-invalid-id.log.
Correct case ID ruler-ea-dev-v1-niah_single_2-000 then completed all collector
controls. Exact original GPU reference/noop/action token IDs, termination and scores
matched; collector native-mask and physical-effect checks passed. Result in
results/value-head-old-replay-comparison.json. Remote exit0 and GPU idle
verified before discovery48launch.

Fit/evaluation scripts implement the frozen035columns and algorithms. Training
will occur on unchanged Orion sklearn1.7.2 to match runtime serialization.
They export scaler/coefficients for independent plain-array prediction checks.
Evaluation additionally checks stored feature hashes and exact agreement with
exported coefficient predictions before computing signed metrics and gates.
Bootstrap is descriptive paired prompt MSE improvement,10000resamples,
seed2026090635. No adjustments to features or model after discovery/evaluation.

Independent CPU preflight results/value-head-preflight-audit.json passed all
96data/adapter rows, overlap/grouping, source timing and frozen model checks.
It establishes instrumentation integrity only, no outcome claim.

Discovery completed48/48with all controls; reference48correct, signed losses
41positive and7zero, zero negative, no adaptermiss. Frozen model SHA
6a817b8a8d759219f677d6d848172c6ecad6bba3ba1bf2580a94b5cc4f24571e
Selected alphas metadata10, mean-retention10, heads1000. Fit on OrionCPU
using discovery only; source and all48record hashes in results/value-head-fit/fit.json.
The model is frozen before the evaluation launch. No feature/alpha/model revisions.

Locked evaluation completed48/48with all runtime controls.41positive7zero,
no negative losses. HeadsMSE0.1243750854 versus mean0.1245659722, gain0.1532%,
24promptwins and AUC0.5087108. Gains versus metadata4.9932%(19wins),
mean-retention4.6118%(21wins). Every predeclared performancegate failed.
Paired95%bootstrap mean-baseline MSEimprovement[-0.0048055,0.0050110].
This fixed head-feature Ridge branch is closed. No confirmation opened.

Evaluator initially failed solely while serializing a NumPy boolean comparison
gate. Preserved source/error under results/value-head-evaluator-serialization-failure.*.
A built-in bool cast fixes serialization; exact original full evaluation rerun
passed unchanged model, metrics and threshold computations.

Post-failure exposed96diagnostic:43heads always retain all value tokens;
69vary,96uniquevectors, centeredrank68. Positive-loss mean eviction0.08857
versus0.07644for zero-loss; rawmeanAUC0.59669. This is exploratory explanation
of failure, not a new predictor. Return to mechanism before choosing next study.
