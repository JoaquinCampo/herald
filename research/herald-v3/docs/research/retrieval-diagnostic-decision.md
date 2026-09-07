# Retrieval-exposure diagnostic decision

Status: CPU feasibility authorized, GPU diagnostic not yet authorized.
This is a new mechanism hypothesis, not a reopening of the closed JS family.

Test whether losing user-message positions in independently identified retrieval
KV heads explains compression damage. Any diagnostic uses explicitly exposed
r5 development cases. A successful rescue would support further investigation,
not establish predictive performance or natural mediation.

Before designing head discovery or generating more outcomes, verify exact user
span alignment against the saved templated prompt tokens using the pinned
tokenizer. Compute per-head evicted user positions and retained non-user donor
counts. Restoring every evicted user position may exceed the donor pool;
feasibility must be measured, not assumed. A partial swap intervention would
need an explicit rule frozen before any new diagnostic outcome, and a narrower
claim about that intervention.

Any later assay must freeze synthetic retrieval-head discovery separately from
IFEval labels, including seeds, score normalization, thresholds, ties and GQA
mapping with shared KV heads deduplicated. One matched control must use the
same heads and swap counts with prespecified position/recency matching.
Equal entry counts and cache bytes, state independence and ordinary Knorm
parity are prerequisites. A failed design cannot be rescued by searching heads,
layers or token subsets on these exposed labels.

The independent reviewer proposed twelve prompt-action development cells,
six per ratio with two per loss-sign stratum, and three continuations per cell.
This is a candidate size, not a frozen roster or GPU approval. The owner must
settle feasibility, selection and numerical rescue/harm criteria first.
A degradation-specific mechanism need not produce monotonic effects across
improvement, unchanged and degradation strata to inform a signed expectation;
report all signs and prespecify acceptable collateral harm.

Next: CPU structural report in results/retrieval-feasibility. Stop before GPU
if exact span provenance or budget-matched controls cannot be established.
No new dataset or predictor fitting is authorized by this decision.

## Owner refinement, 2026-09-06 10:11 UTC

The proposed two-ratio GPU protocol is not accepted as written. First compute
a label-blind optimistic control-feasibility bound using all heads at ratio
0.25 only. Both non-user restoration sources and donors must be inside the
prompt, excluding generated positions. This avoids silently trading away the
generated prefix when claiming an instruction-access effect.

Keep the proposed 10-percent cache-length position caliper and at most two
matched swaps per head. Use at most eight heads for the optimistic bound,
requiring eight swaps across four heads. All-head feasibility is necessary
but insufficient: actual independently detected retrieval heads may fail it.
This CPU gate precedes synthetic head discovery and any new continuation.
The narrower estimand is a local prompt-position rescue, not full restoration
or a general statement about all instruction tokens.

## Synthetic discovery implementation scope, 2026-09-06 10:35 UTC

The optimistic prompt-only matching gate is feasible for all 69 prompts,
allowing implementation of synthetic head discovery, but not a GPU launch.
The candidate Phase A grid, thresholds, GQA mapping and sentinel parity checks
are adopted for CPU implementation. Replace the unavailable ROUGE filter
prospectively with full exact answer-token inclusion, subject to independent
review of boundary tokenization before freezing or execution. Keep the eight
successes per panel floor; failure must stop, without relaxing the filter.
This is an exact-copy adaptation, not a reproduction of the authors' detector.
No IFEval roster or rescue criterion is frozen by this implementation scope.

## Token boundary amendment before any model run

The released needle contains its answer as text, but its standalone answer
tokenization can differ at the leading-space boundary. Authorize exactly one
newline immediately before the unique answer occurrence, preserving the
remaining released text. Freeze this synthetic-input transformation.
The authoritative target sequence comes from the intended answer character
span's offsets in the fully rendered prompt. Verify uniqueness and exact
text/token round trips before loading a model; never infer correctness from
a standalone encoding or assume the newline makes tokens identical.

No generated-output normalization, delimiter search, or post-result filter
adaptation is allowed. Exact-copy success remains a conservative screen with
the unchanged eight-successes-per-panel floor. A failure scopes conclusions
to exact-copy-detectable retrieval heads. The detector must align the attention
from the forward consuming token t with the prediction of token t+1.

## Explicit third-triple adaptation

The released third needle uses lowercase `because`, while its released answer
starts with uppercase `Because`. Before any model run, the owner accepts
replacing that unique casefold-matched source span with the exact released
answer and inserting the single newline. Preserve all other characters.
Freeze original, released answer and transformed text in provenance. This
applies only to synthetic input construction, never to generated outputs or
the exact-copy success criterion.
