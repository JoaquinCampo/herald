# Fixed038implementation and real proof

Source measure_digit_copy.py SHA9d6abc71e47a007796cec75ccb0f9380f783608ad2ba8e9bd4ece0cefff6f53f.
Thin Qwen forward with explicit four-dimensional causal mask in physical cache
coordinates and logical RoPE positions. CPU full-versus-sequential probabilities
match at1e-5absolute/relative tolerance for both uncompressed and compressed arms.
The naive two-dimensional-mask hazard is reproduced separately in
results/digit-copy-mask-hazard-reproduction.json; it is not used for measurement.

Owner fixture /tmp/digit-copy-owner-fixture uses a small random Qwen2 model,
real cached tokenizer, four numeric facts and8token unforced continuations.
Prior results/digit-copy-owner-prior. Original first probe run failed at the
EOS-margin helper because an integer was passed where a list was required.
Preserved source and failure in results/digit-copy-owner-cpu-first*.
Changed caller to pass sorted EOS IDs, reran original real fixture immediately,
then added paired action/reference IDs, termination and batch-first-token checks.
Final results/digit-copy-owner-cpu-final passes all controls, including causal
batch/sequential equality, native masks, exact prior continuations and immutable
source state. Tokenizer copy warning preserved; tokenizer not changed between
prior and probe and exact token reconstruction is asserted.

FirstGPUcase000 launched only after Orion ownership/capacity verified, source
hash matched. Full result and exact old replay required before other11cases.
No new outcome population or confirmation accessed. No fitted predictor here.

FirstGPU000failed before measurement with invalid biasdtype: GPU SDPA required
additive maskdtype to match querybfloat16, CPUfloat32proof hadnotexercisedthat.
Failure/source preserved results/digit-copy-v1-first and correspondingfailedsource.
Onlychangedmaskdtype/finfo to model.dtype; exactoriginalGPU000reproduction
passed in results/digit-copy-v1-first-dtypefix, then originalCPUproof reran
successfully in results/digit-copy-owner-cpu-dtypefix. Source now frozen
eabea1a63cb2ab994533828296d86851b5ffb9c8ee1dd917160d7ff32722590d.
All reference/action IDs and termination exact, native masks exact, source
immutable, control masks equal, causalbatch firsttokens exact. No feature
definition changes. Remaining11 launched with this same source after idleGPUcheck.

All12completed. PrimaryzAUC0.8888889, wrongvaluecontrolAUC0.8888889,
referenceNLLAUC0.5555556. Predeclaredspecificitymargin0.0versusrequired.15,
so038understandinggateFAILED. Exactanswer-specificprobeclosed, nofit or
reduction/prefix revision on these outcomes. Independent12case audit verifies
24priorunforcedreplay/termination controls, native masklineage, offsets, features,
z andAUC arithmetic. Existingrawprior.masks.npz+hashmatch suppliesmasklineage
withoutduplicatingarrays. See results/digit-copy-independent-audit.json.

Descriptiveperdigitmeans show likelihood damage mainly on later digits for both
queried andcontrolvalues. This supports considering a broader copying effect as
a newhypothesis, not a retrospective pass of038. RootrequestsboundedSolMedium
judgment on the ambiguoussignal aftermultiplefailedfamilies; no newexperiment
chosen yet. Newdata andconfirmation remain unopened.
