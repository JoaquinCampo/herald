# One sandbox retrieval probe, unchanged B0 target

The025query selected recent suffix positions and failed to rescue content.
Change the observation, preserving the original B0 action and signed final score
loss. Competing explanations: the chat-suffix query is task-poor; attention times
Vnorm is inadequate even under task-oriented queries; distributed evicted mass
matters but sparse one-token restoration was inadequate. This is exposed-data
exploration, not confirmation. No oracle spans/answers enter the observation.

Freeze the same12NIAH prompts at .10. In an independent full B0cache clone,
process the original pending token followed by this exact forced assistant text:
Before answering, identify the information in the prompt that is needed. The needed information is:
Tokenize once without special tokens under the fixed checkpoint tokenizer and
save IDs. No generated answer/reference tokens, prompt-specific extraction,
phrase variants, horizons or head searches. Capture actual Q of the final forced
token with the native RoPE. Discard the probe clone; neither real continuation
contains the cue. Existing reference/.10 outcomes remain the target unchanged.

For each layer/KVhead, calculate025directattention timesVnorm, max over grouped
Qheads, normalized over ORIGINAL B0cached positions only. z is mean across
layer/KVheads of salience mass on native .10 evicted positions. Fixed mechanistic
control: compute exactly the same z from the unforced B0query. This separates
cue value from changing the sparse025aggregation. Save perheadmass and raw
probabilities/Vnorm/salience compactly for audit. Measure synchronized probe,
feature reduction and transfer cost, token count and memory; auditcost separate.

Require full B0source equality/independence/immutability, exact instrumented/plain
forced-probe logits, exact unforced first-token and masks against prior outcomes,
normalized finite distributions and complete12row coverage. Batch forced tokens
with original logical positions. No later GPU generation is needed: labels were
already independently reproduced at this same original boundary/action.

Lock LOO prediction, all transforms fit within11trainingprompts: B0trainingmean;
Ridgealpha1/interceptTrue on one TRAINstandardized unforcedz; same on probez.
No combination, clipping, orientation choice, transformations orpenaltysearch.
MSE primary, MAE secondary. For a candidate feature, require10%lower OOF MSE
than mean baseline, at least8/12 prompt errorwins, and rawz failure AUC>=.80
(directionlargermass=>largerloss). For incremental cue value, probe must also
beat the fixed unforced model in pooled MSE. Report all comparisons, prompt
bootstrap interval over fixed OOF errorpairs (not retraininguncertainty), and
exact one-sided label-permutation rank p-value. All12prompts retained.

If neither feature meets itsgate, close these fixed B0query-attention variants;
no cue/window/head/threshold search. Next strategic revision should consider an
explicitly later action boundary, recognizing a changed estimand. If unforced
wins but cue doesnot addvalue, report thatfinding rather than attributing value
to the cue. Any passingfeature requires a separate frozen broader test before
unseen validation; no success claim from these exposed12prompts.
