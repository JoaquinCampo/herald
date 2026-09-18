# Value-level signed-loss supervision

Pilot043 passed its outcome-variation gate, not a prediction gate. Test whether
separate labels for four values reveal useful head-retention information that
was hidden by single-answer supervision. Do not add nonlinear model flexibility.

Competing explanations: value-specific retention predicts which values survive;
prompt-level severity explains all useful variation; value order/position explains
survival without head information; or apparent head patterns do not transfer.
The distinguishing requirement is prediction within a shared prompt, in addition
to task-level loss prediction on new prompts.

Freeze 128 discovery prompts (seed2026090645) and 64 locked evaluation prompts
(seed2026090646), official niah_multivalue with one key/four seven-digit values,
essay haystack, length4096, original template and128-token generation horizon.
Exclude exact prompt/context overlaps with prior v4 populations, including043.
Use Qwen2.5-7B pinned revision, B0 and native Knorm .05 as in043. Group all four
value rows and any context variants by the complete parent context.

A prompt-only known-schema adapter locates the four native digit-token spans
and orders them by occurrence in the prompt, not output order. Validate the
mapping to each official answer only after feature persistence. The adapter can
already solve this benchmark schema, so a passing predictor would have narrow
benchmark scope, not general semantic understanding.

For each value define four structural features: log prompt-token length,
normalized occurrence order j/3, native span midpoint divided by prompt length,
and actual whole-cache removed fraction. Add two aggregate retention features:
mean missing fraction over112heads for this value, and its mean over four values
in the prompt. Candidate adds all112 per-value head missing fractions. No head
selection, cue, forward probe, clipping, polynomial feature or nonlinear model.
Candidate feature order: four structural, two aggregate, then112head fractions.

The per-value signed target is reference answer hit minus action answer hit,
using pinned official matching after postprocessing. Its four-value mean must
exactly equal the signed official task-score loss. Retain improvements and all
imperfect references. Do not train on action correctness alone or assume a
perfect reference in fresh data. Save features and optional predictions before
unforced paired continuations and scoring.

Fit fixed StandardScaler plus Ridge(alpha=1000, intercept=True). This penalty
is inherited from035's frozen all-head selection, not tuned here. Each prompt
has total training weight one (four rows weighted .25); all transformations and
fitting use discovery only. No parameter search or row-level random split.
Predict task loss by the mean of four raw per-value predictions. Baselines:
discovery mean signed contribution; structural-only; structural plus both
aggregate-retention fields. Candidate receives all118 features. Same scaling,
penalty and prompt weights for each fitted baseline.

Before fitting require at least24 discovery prompts with differing per-value
signed outcomes. Evaluation requires at least12 such prompts,64 complete paired
records, and a frozen model hash before its first outcome. Require candidate
MSE at least10%below EACH baseline and at least43/64 strict squared-error wins
against EACH baseline. Also require within-prompt concordance >=.65 and >=.10
above structural-only: among value pairs with unequal signed targets, score
correct ordering1, reversed0 and tied prediction.5; average pairs per prompt,
then average equally over mixed prompts. This is a graded-target discrimination
gate, prospectively replacing any-loss binary AUC for this new population.

Report MSE, MAE, signed bias, within-prompt concordance, reference quality,
negative contributions and task-loss distribution. Report paired-prompt
bootstrap95%relative MSE gain intervals (10000 draws, seed2026090647), conditional
on the frozen fit. A pass only warrants untouched confirmation, which must also
show a positive lower confidence bound for gain versus each matched baseline.
No confirmation data is generated or opened in this study.

Acceptance mechanics: native token-offset reconstruction, same candidate/action
mask hashes and physical effects, immutable independent source states, exact
reference/noop replay, complete coverage and correct signed aggregation. Run a
real tiny CPU slice and first exposed043GPU replay before the fresh collection.
Measure complete mask-observation/adapter/prediction wall cost from prepared B0;
report prefill and correctness controls separately. Reuse validated helpers.

If any predictive gate fails, close the mask-retention family rather than
searching a nonlinear model, different penalty, head subset or feature on these
outcomes. This is one bounded test of decomposed supervision, not a sequence of
learner retries. If outcome variation fails, do not fit or filter cases.
