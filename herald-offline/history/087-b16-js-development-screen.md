# Frozen B16 distribution-shift screen

Selected after086, before collecting paired B16 logits. This is a development
screen on the32 already-exposed071 discovery prompts, not unseen evaluation.
All prior failed results stay closed. Do not use071 evaluation48 to tune this.

The candidate scalar is raw full-vocabulary Jensen-Shannon divergence between
uncompressed and Knorm.1 next-token distributions at the exact B16 shared state.
Use the existing engine formula directly on logits, internally float64
log-softmax and logaddexp, as in021. No float32 probability roundtrip.
No KL alternative, logarithmic transform, horizon or severity search. Reference
and action probes execute on independent equivalent clones before a live action;
only the current pending token is processed, no future answer tokens supplied.
All16 common generated tokens are observed; tokens1..15 are cached and token16
is pending. The distribution predicts token17. Model/runtime/action stay071's.

Reproduce stored prompt IDs, common prefix, B16 cache fingerprint, exact native
mask digest and old reference/action token17 argmax. Ref/no-op logits must match
exactly; source caches must stay unchanged. Save raw ref/no-op/action logits and
hashes. Separate reference+action observation/reduction time from no-op and state
audit costs and from ordinary prefix advancement. First run one exact exposed
case, then all32 only if its real controls pass. No new full action continuations.

Baseline inputs: mean-only; trailing observed digit-count only; and the six071
structural features plus its z and trailing digit count. Count trailing ASCII
digits in decoded common16 prefix, or0 if none. No gold answer length, remaining
digits, future tokens or label-derived features. Candidate adds raw JS to the
eight-input strong baseline. All regressions use training-fold StandardScaler
and Ridge(alpha=1) with intercept, signed targets and no output clipping.

Fixed leave-one-prompt-out32 predictions. Report MSE/MAE, strict per-prompt wins,
raw positive-JS AUROC, and10000 paired prompt-bootstrap relative-MSE-gain
intervals (seed2026091587), ordinary95% and simultaneous Bonferroni95% over the
three baselines. Screening passes only if every integrity check passes, raw AUC
is>=.80, candidate reduces MSE>=10% versus each baseline, wins>=22/32 against
each, and all simultaneous lower gain bounds are positive. Report failures
without tuning. The next-token disagreement bit is already known constant0.

Hypotheses are those in086: latent distribution fragility, no useful shift,
answer-progress confounding, structural confounding, or runtime artifacts.
If the fixed screen fails, close this scalar/model and return to understanding.
Passing only supports designing fresh grouped prospective evaluation; it never
establishes the user goal or permits reusing exposed prompts as confirmation.
