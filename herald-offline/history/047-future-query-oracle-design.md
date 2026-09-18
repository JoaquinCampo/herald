# Fixed future-query oracle assay

## Purpose and status

Study045 found transferable within-prompt value ranking from native mask features,
but failed every locked task-level prediction comparison. The mask-retention
family is operationally closed. Before collecting more prompts or fitting another
predictor, test whether weighting the fixed B0 deletion by the queries that
actually produce the scored answer digits explains the missing variation.

This is one privileged, exposed-data mechanism diagnostic. Future reference
queries and correct answer locations are unavailable at the B0 decision point.
The measurement is neither a predictor nor an upper bound on predictive
information. Passing does not reopen study045 or justify deployment. Failing
triggers the stated project stopping rule, not a mathematical claim that no
attention-derived observable could ever work.

Use all 64 now-exposed study045 evaluation prompts, including all 42 mixed
prompts. Keep their stored official data, Qwen2.5-7B revision, B0 boundary,
native Knorm .05 action, 128-token horizon, reference/action outputs, native
masks, and signed labels unchanged. No new outcomes or confirmation data.

## Competing hypotheses

1. **Future-query relevance is missing.** Native masks rank values somewhat,
   but useful damage information requires the attention error induced when the
   actual answer-producing queries read the retained cache.
2. **Autoregressive divergence dominates.** Loss develops through the action
   trajectory, so holding the full-reference queries and generated state fixed
   will not explain prompt-level task loss.
3. **Scoring and response dynamics dominate.** Partial numbers, formatting, and
   the fixed horizon turn small decoder changes into discrete final-score loss;
   a query-conditioned attention error may still fail to track that loss.
4. **Prompt-level vulnerability is elsewhere.** The prior head signal captures
   relative value ordering, while contextual geometry or another prompt-level
   state controls how many values fail.

## Frozen counterfactual

Rebuild each exact B0 boundary using the existing last-prompt convention. If the
chat prompt has L tokens, the boundary cache contains prompt tokens 0 through
L-2. Token L-1 is `pending_token_id` at logical position L-1 and is absent from
the boundary cache. Native Knorm .05 chooses its per-layer, per-KV-head kept
indices only from cached prompt positions 0 through L-2.

Teacher-force the stored perfect reference continuation from an independent
full boundary clone. Process the pending prompt token first, followed by the
reference continuation. The pending token, response-prefix tokens, and prior
answer tokens are post-boundary state: retain their full-reference K/V entries
unchanged in both sides of this diagnostic and never apply the B0 mask to them.
At deeper layers these post-boundary K/V entries remain those produced by the
full teacher-forced run. This intentionally isolates the original prompt-cache
deletion under the realized reference trajectory; it does not reproduce the
action trajectory.

Map each of the four official seven-digit values to exactly one contiguous
seven-single-digit-token occurrence in the stored reference continuation. A
missing, duplicate, noncontiguous, or differently tokenized value blocks the
whole assay; do not replace a case or choose another occurrence after inspection.

For a target digit token d_t, use the layer query produced while processing its
immediately preceding teacher-forced token, because that position's final logits
predict d_t. Thus the first digit uses the query from the last response-prefix
token, and each later digit uses the query from the preceding digit. Apply the
native Qwen RoPE position, attention scale, causal limit, and GQA grouping.

For each such query and layer, hold q and every K/V tensor fixed from the full
teacher-forced run. Compute `o_full` by attending to all prompt-prefix K/V plus
all causal post-boundary K/V, including the current query-token K/V. Compute `o_mask` by attending to only the
native Knorm-kept prompt-prefix K/V plus the identical post-boundary K/V. In both
cases perform the ordinary softmax renormalization, concatenate query-head
outputs, and apply the frozen layer `o_proj`. Perform attention arithmetic and
the final reduction in float32 or higher.

Freeze epsilon as 1e-12. The layer-digit error is
`||o_full - o_mask||_2 / (||o_full||_2 + 1e-12)`. The sole oracle scalar for one
value is the arithmetic mean of that quantity over its seven predicted digits
and all 28 layers. The prompt scalar is the arithmetic mean of its four value
scalars. No head, layer, digit, token, or case weighting or selection.

As a descriptive decision-point control, compute the same 28-layer mean error
for the actual pending-prompt-token query against the full versus masked B0
prompt cache, including the pending token's identical self K/V on both sides. It is not fitted and does not replace either primary gate.

## Integrity and gates

Before interpretation, require all 64 records, exact source and model identity,
exact stored prompt/reference/action IDs and termination, exact native kept-index
hashes and physical lengths, independent immutable boundary clones, exact value
mapping, finite queries and outputs, causal token alignment, and persisted oracle
measurements before joining stored labels. Any failure blocks interpretation and
must preserve the failing case.

Higher oracle error predicts higher per-value signed loss. On the 42 fixed mixed
prompts, compute the study045 equal-prompt within-prompt concordance with ties
worth 0.5. Require concordance at least 0.81, which is also more than 0.10 above
the frozen head-Ridge result of 0.710317. Across all 64 prompts, compute SciPy
Spearman correlation with midranks between prompt oracle scalar and signed final
task loss. Require correlation at least 0.70. Both gates must pass. Report the
pending-query control correlation, all per-value scalars, prompt scalars, labels,
tie counts, and runtime descriptively. No threshold search or uncertainty-driven
revision.

If both gates pass, the result supports investigating a prospective intervention
boundary where relevant queries have begun to arrive. It does not validate a B0
predictor. If either gate fails, stop B0 mask and attention-risk work for this
NIAH task/action family and return to the task, action, or timing assumptions.
Do not try another aggregation, query proxy, fitted model, head subset, severity,
prompt subset, functional-likelihood probe, or post-hoc OLS promotion here.
