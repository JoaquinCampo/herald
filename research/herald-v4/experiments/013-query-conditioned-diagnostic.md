# Query-conditioned EA diagnostic

## Decision

Run one label-free measurement diagnostic before any new predictor fit. Reuse
the four exposed NIAH cases from 011, with the already observed endpoints:
000/004/011 have standard score 0, oracle rescue score 1, and matched control
score 0; 001 scores 1 on all three branches. The 28 existing state and branch
controls pass. These outcomes are read only after the measurements are frozen.
The diagnostic separates two explanations for the weak global EA feature:
mean prefill queries may miss the later retrieval query, or the all-head mean
may be the wrong summary. It is a mechanistic comparison, not a predictor or
confirmation population.

## Source semantic check

Use the installed `kvpress.ExpectedAttentionPress.score` with the existing
settings: `n_sink=4`, `n_future_positions=128`, `use_covariance=False`, and
`use_vnorm=True`. The all-prompt arm passes the complete prefill hidden-state
sequence to the unchanged score implementation.
A direct call with `hidden_states[:, -32:]` is invalid. The installed scorer
removes the first four rows of its input and uses that input length as `q_len`
when applying RoPE, so a direct slice changes both sink semantics and the
absolute starting position. Use a measurement-only adapter around the same
`score` formula: compute query statistics from the final 32 actual prefill
rows, preserve the original prompt length for `apply_avg_rope`, and do not
remove four additional rows from the selected window. Keys, values, weights,
and compression masks remain untouched.
## Minimal slice

For each fixed case, build one shared last-prompt boundary and capture the
per-layer prefill hidden states and native cache tensors. At every layer,
score the same keys and values twice, once with all prompt queries and once
with the fixed final 32 prefill queries. Use the native Knorm `.10` removed
positions for both score maps and report finite per-KV-head values.

For each query window, freeze the same mean EA feature and two diagnostic
references before reading outcomes:

- `mean_excess`: the existing all-head mean EA removed-mass excess;
- `oracle_full_span_mass`: EA mass assigned to the complete injected sentence
  span, normalized by non-sink EA mass;
- `oracle_removed_span_mass`: the same span mass restricted to the native
  Knorm `.10` removed positions. These oracle quantities are diagnostic
  references only and are never features, branch rules, or selection rules.

Compare the fixed last-32 versus all-prompt values descriptively across the
four cases and preserve signed final loss for the existing standard branch.
Do not fit, rank, tune, or select heads, layers, windows, or thresholds.

## Gates and interpretation

Accept the measurement only if prompt token IDs, shared-boundary tensors,
source-cache tensors, cache storage independence, native Knorm masks, and
reference/no-op/standard continuations match the existing controls. The
instrumented prefill must leave the boundary and source cache fingerprints
unchanged, and all score values and summaries must be finite. Any mismatch
blocks interpretation and is evidence of instrumentation or replay failure.

Report the descriptive final-32 minus all-prompt delta for `mean_excess` and
both oracle span masses on each case. A larger final-32 span-mass delta on the
three rescue-positive cases is consistent with query mismatch. A similar weak
delta on both windows leaves all-head aggregation or another mechanism
unresolved. With four cases, these are mechanism checks, not statistical
claims, and no causal head or predictor claim is allowed.

If neither fixed contrast separates the rescue-positive cases from 001, the
two explanations remain unresolved or the EA score is not measuring the
causal mechanism. Do not reopen EA fitting. Stop after this four-case slice
and design a new prespecified hypothesis only if the measurement is valid.
