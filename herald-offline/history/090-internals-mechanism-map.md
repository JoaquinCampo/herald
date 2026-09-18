# How eviction can cause a later digit omission

Understanding record,2026-09-15. No new predictor experiment or GPU collection.
Source evidence is the live Orion engine and installed Transformers4.57.6,
archived as results/b16-internals-source-audit.json with file hashes. Concrete
old trajectories are in results/b16-first-difference-examples.json.

## What the intervention actually changes

The pinned Qwen model has28 decoder blocks,28 query heads and4 KV heads per
block, hidden width3584. Seven query heads share each KV head's entries.
The engine computes scores=-norm(key), keeps top floor(.9*length) positions,
and gathers BOTH keys and values with the same per-layer/per-KV-head indices.
Thus this action removes the largest key norms, not the smallest. It does not
consult the current query or the answer. Do not silently reinterpret its name.
All recorded native-mask replays refer to this exact operation.

Keys are already RoPE-rotated when stored. The gathered keys keep those values;
compression does not re-rotate them or renumber their logical positions. The
single pending token receives its original logical position, while the attention
mask covers the retained physical slots plus the new token. DynamicLayer.update
appends to the physical cache; it does not insert holes at cache_position.
Knorm's topk order is not chronological. For a single query allowed to attend
all existing slots, a joint permutation of K and V does not change attention in
exact arithmetic. Finite-precision kernel behavior is a separate issue, not a
reason to assume order invariance bit-for-bit. Existing exact native replays
validate consistency, not arbitrary-permutation equivalence.

## The local error has a direction, not just a missing mass

For one fixed query and its per-head weighted value mixture before o_proj,
with the same scale/mask and exact softmax arithmetic, let a_i be full-cache softmax attention weights. Let R be
removed entries, r=sum_R a_i, o_R their normalized weighted mean value vector,
and o_K the corresponding retained mean, including the current token if present.
For 0<r<1, when the retained keys/values and query are otherwise unchanged:

    o_full = (1-r)*o_K + r*o_R
    o_compressed = o_K
    o_compressed - o_full = r*(o_K - o_R)

This follows by renormalizing the retained exponentials by1-r. Missing attention
mass alone omits the difference between removed and retained value vectors.
Equal missing mass can mean zero local output error if those means agree, or a
large directed error if they differ. Value-vector norms also discard direction.
Key norm alone does not determine q dot k because alignment matters.

This identity is exact for that fixed-query comparison, not a decomposition of
an entire multi-layer autoregressive trajectory. In the first layer at the first
post-action forward, the input token/embedding and hence projected query and
new K/V match between branches. Later layers receive changed residuals, so their
queries and newly generated K/V can change as well. Applying the identity there
while pretending queries remain equal would confound direct eviction with
propagated changes. Instrumented same-query counterfactuals would be separate
measurements, not actual branch attention.

## Why unchanged tokens can conceal divergence

Each block computes residual + attention_output, then adds an MLP update after
normalization. Head outputs pass through o_proj before joining the residual.
A changed attention mixture can therefore alter later-layer queries, value
writes, MLP outputs and final token logits. The final greedy token only changes
when a competitor overtakes the previous winner; it ignores all other logit
changes. The two branches can emit the same token yet append different K/V
entries in deeper layers. Subsequent tokens then read different internal history
as well as different original retained source entries.

This explains a possible route from an initially stable argmax to later damage.
It does not prove which layer/head/value direction caused any observed case.
Small B16 JS describes the distribution of token17, not the sensitivity of the
subsequent recurrent computation. A scalar JS also omits which token alternatives
changed. Neither statement makes a new feature predictive by itself.

## What the saved cases establish

- Discovery000: after common text ending198984, reference emits0 and action emits
  a period, at generated index21 (zero-based). The number stops early, but this
  is not an EOS transition: both subsequently terminate normally.
- Discovery002: after58, reference emits7 and action emits4 at index17. The action
  continues4921, consistent with skipping a middle digit. This is not explained
  by merely stopping the whole answer too early.
- Discovery023: reference starts bold formatting and action uses whitespace;
  both preserve3705852. Token divergence here is harmless under the fixed score.

The034 rescue was a different, earlier B0 population: restoring source value-token
K/V entries recovered8/8 selected failures, while surrounding-context rescue and
matched controls recovered none. This supports sufficient source-value repair
in those cases. It does not establish that the current B16 cases share the same
causal mechanism, identify a necessary head, or distinguish whether repair works
by restoring immediate copying or preventing downstream state drift. Healthy
cases losing more entries refute raw missing-count sufficiency on that slice.

## Questions to resolve before another predictor design

Four explanations remain: direct loss of the currently needed source-value
contribution; altered position/copy progression carried in later-layer state;
a shift toward punctuation/answer completion; or numerical amplification of an
initial perturbation. They may coexist, and the middle-digit case already rules
out a completion-only explanation for every failure.

First inspect the existing source-token locations, native masks and available
query/value records for the exact failed examples, checking which facts are
actually observable. Then identify the earliest unresolved link in the chain.
Any further measurement must distinguish direct source access from accumulated
state drift while preserving the real branch trajectories and exact replay.
Do not start a layer sweep, hidden-state fit or new dataset merely because this
map shows that richer information could exist. No validated predictor yet.

## Existing source masks and what the archive cannot tell us

Oracle localization of source digits in000,002,023 reproduced exact original
chat token IDs and seven single-digit source tokens. The omitted final0 in000
is absent in10/112 KV-head/layer pairs. The skipped7 in002 is absent in6/112,
while later digits that are successfully copied are absent in up to15/112.
The correct final2 in023 is absent in11/112. These selected examples do not
justify a fitted retention threshold or identify which head matters. They make
clear that 'digit omitted' does not mean its token vanished from every layer.
Source evidence: results/b16-source-digit-mask-examples.json. Oracle answer
locations are explanatory annotations only, never prospective features.

The old071 raw tensors store probabilities, value_norm, salience, native_masks
and mass_per_head. They do not store value-vector directions, layer output
projections, per-layer residual trajectories or later divergence-step queries.
Consequently the exact vector identity above cannot be evaluated from those
archives alone. Norms cannot reconstruct the missing directions. This is the
specific evidence gap, rather than an absence of another scalar feature.

Independent Luna review confirmed the fixed-query mathematics and causal scope.
It requested clearer pre-o_proj terminology and engine provenance; both are now
explicit, with engine compress_knorm and _pending_forward source/hash added to
the source-audit artifact. No additional inference run was needed.
