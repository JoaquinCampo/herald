# Bounded lookahead engineering decision

Owner decision, 2026-09-05: approve only a small engineering assay of a
capped eight-distribution reference-prefix probe. No prediction experiment,
training, reserved-label collection or horizon search is approved by this note.

The one-step pilot failed. A delayed probe asks a different question:
can repeated reads of the evicted context change sensitivity beyond the
first prediction? This is a plausible mechanism, not established evidence
of final-quality prediction. No direct primary-source evidence for that
predictive claim was established in the review. An engineering assay can
establish causal availability, correctness and cost, not predictive value.

## Exact measurement

At the existing boundary KV contains output indices 0 through 30 and index31
is pending. Clone that state independently for the reference and each action.
Apply the candidate once. Compare distributions predicting indices32 through
39 inclusive. Step0 reproduces the original one-step probe. For each further
step use the reference greedy token as the shared next input in every clone;
never use stored completed output or an action-generated token as input.
There are at most eight forwards per arm, not nine.

Stop after the distribution whose reference argmax is an EOS token. Record
the realized number of distributions and exact token positions. Never force
post-EOS tokens or silently drop a short probe. Action EOS does not terminate
the teacher-forced comparison; it is still conditioned on the reference
prefix. This compares distributions along a synthetic no-action trajectory,
not along the action's realized continuation.

These are newly computed pre-commitment virtual tokens. They amend the old
feature availability contract, which excluded reference future tokens.
Their acquisition is charged explicitly; no claim that they are free or
observed at the original decision timestamp is allowed.

## Acceptance before any predictor design

Use tiny real local model tests and one exposed/synthetic full-model Orion
prompt. Verify exact indexing, shared forced tokens, step0 parity with the
accepted engine, per-step no-op equality, source cache/RNG preservation,
real compression and identical continuation before/after acquiring probes.
Exercise early reference EOS explicitly. Report synchronized clone, eviction,
forward/reduction and total wall time, plus peak CUDA allocated bytes. State
that the cost includes preservation and validation where applicable.

Implement in a new module and CLI, reusing accepted engine primitives.
Do not edit locked pilot source or old results. No GPU work locally; only
the owner launches on Orion after review and the normal process preflight.

After engineering evidence, owner must freeze one feature contract including
short-probe handling and matched baseline information. Candidate statistical
design is train on exposed120 and evaluate once on up to76 outcome-unseen
candidates, with no horizon/model sweep. Exact eligibility, information floor,
metrics, comparator handling and prediction-before-score sealing must be
reviewed before that separate approval. Existing pilot folds are provenance;
a single train/test fit would not use heldout folds for fitting.
