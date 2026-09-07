# Fixed functional digit-copy understanding slice

Study035failed and is closed. This exploratory, no-fit experiment distinguishes
functional answer-token damage from coarse mask statistics or generic numeric
probability shifts. It is not a signed-loss predictor validation.

Competing explanations are in037. Choose a short teacher-forced functional
measurement before another feature/model search. Correct037sample arithmetic:
discovery has only7zero-loss rows, so use first6positive and first6zero rows in
manifest order,12total, an explicitly outcome-stratified exposed slice.
IDs:000001003005006007 and002004009012032036, allvalue-head-v1-discovery.
Keep nativeKnorm.10, Qwenrevision, original split-prefill B0 and original unforced
128-token quality target. No data/model/action/timing changes or sweep.

Construct probe inputs from prompt text only. Parse final known-schema query key
and facts with existing adapter rules. Use exactly this fixed response prefix:
'The special magic number for KEY mentioned in the provided text is '
Then teacher-force the queried seven-digit value. This is a sandbox branch
with known forced text, not an observed continuation or changed official prompt.
The exact schema already yields the nominal answer, so its utility is narrow.

Control: choose lexicographically first different key in the same context,
require its value differs from queried value, and score its seven digits under
the SAME queried-key response prefix. Do not adapt choice based on model scores.
This distinguishes queried-value-specific damage from generic numeric changes.

For correct and control sequence, use independent full/nativecompressed B0
clones. Process pending lastprompt token plus fixed prefix and previous digits
with original logical positions. Save native masks, per-digit log probabilities,
correct-digit-minus-best-other-vocabulary-logit margins, digit-minus-EOS margins,
and cost separately for clone/compression, forward, reduction and transfers.
Native digit positions come from exact tokenizer offsets of prefix+value;
assert exactly7single-digit tokens and exact reconstruction. CPU fixture may
use the available tiny model; production verifies112KVheads/native mask hash.

One primary scalar: mean over7digits of logp_reference(correctdigit) minus
logp_action(correctdigit). Control scalar identically for wrongvalue. Reference
mean NLL is an information-matched baseline. No per-digit/head selection, no fit,
no inversion, thresholdsweep or probing-prefix revision after observation.

Require exact prior native masks, immutable source/boundary, finite logits,
paired reference/action first-token replay, original tiny CPU realcontinuation
replay and first exposedGPUcase unforced continuation replay before fullslice.
No stored answer/outcome/continuation text enters probe computation. Join old
signed losses only after feature output is persisted; source and data hashes.

Understanding gate: primaryscalar positive-loss AUC>=.80, at least.15above
wrongvaluecontrol AUC, and strictly above reference-NLL AUC. Report all12cases,
per-digit patterns, margins and costs. This case-control gate only justifies
considering a new frozen predictive study. If it fails, close this exactsemantic
probe without trying another prefix/sequence reduction on these outcomes.
