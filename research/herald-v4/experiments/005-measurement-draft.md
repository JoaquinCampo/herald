# Measurement draft: action-time evidence for Knorm loss

## Decision
Run one small, action-aligned feature family first: expected-attention mass on the
tokens the exact Knorm action would remove. Use mean-only `ExpectedAttentionPress.score`
(`use_covariance=False`, `use_vnorm=True`, `epsilon=0`), fixed `n_sink=4` and
`n_future_positions=128`, which tests H1 cheaply on Qwen2.5-7B without head discovery.
The outcome remains the signed final score difference, reference minus action,
from the new shared split-boundary reference. Positive values mean degradation;
negative values remain valid improvements. Do not fit or open confirmation evidence
in this measurement.
## Feature and matched baseline
For every layer and KV head, reproduce the exact Knorm top-k mask for each fixed
removal fraction without changing the cache. Exclude the first four sink
positions from mass normalization because the scorer pads them with an artificial
maximum. Let `s_lht` be the scorer output, `M_lh` the removed indices, and
`N_lh={4,...,k_lh-1}`. Record the non-sink value-weighted mean-query EA score excess:
```text
ea_ns_removed_excess = mean_lh(
    sum(s_lh[M_lh intersect N_lh]) / sum(s_lh[N_lh])
    - |M_lh intersect N_lh| / |N_lh|
)
```
The second term is the matched structural null at the same layer, head, length, and
severity. Record sink-removal fraction, removal fraction, prompt/cache length,
removed-position and recency summaries, Knorm removed key-norm mass, cheap removed
value-norm mass, and raw per-head values. This scorer-derived mass is not literal
attention probability; no task name, answer, continuation token, final score, or
future observed attention enters the feature.
H1 predicts larger positive excess for larger positive signed loss. The structural
baseline is exact under uniform importance and exposes whether any association
comes from the action's removal fraction or cache geometry alone.
## Implementable collection path
Pin and record the installed kvpress version and source hashes. NVIDIA's
[`ScorerPress.score`](https://raw.githubusercontent.com/NVIDIA/kvpress/v0.5.2/kvpress/presses/scorer_press.py)
is a pure scorer with signature `(module, hidden_states, keys, values, attentions,
kwargs)` and returns `(batch, kv_heads, sequence)`. [`ExpectedAttentionPress`](https://raw.githubusercontent.com/NVIDIA/kvpress/v0.5.2/kvpress/presses/expected_attention_press.py)
does not require attention weights, so native SDPA can remain unchanged. The read-only
v2 environment is Transformers 4.57.6, torch 2.10.0+cu128; source SHA-256 values
are base `2b33576eed0a57936502636af8f82566ee83d1072435d27ca55e61ad7745f1b0`,
scorer `a5eb57a8d9defdaf1f46414fad1584e0c070643a85116d9a85f2e9321cb29728`,
expected-attention `f0e8525d9e19b68a0f123cea3bb39b76af4adbf888ef9c4f55680f582feedc42`,
and Knorm `26ea0d4f41e6eb120c474556d0b5461ad5ab9d029f4d74767b11f9683b8d824f`.
Use a temporary `register_forward_hook(..., with_kwargs=True)` on each
`model.model.layers[i].self_attn` during split-prefix prefill. Read
`kwargs["hidden_states"]`, `kwargs["past_key_values"]`, and the layer cache keys
and values, call `score(...)`, compute the exact Knorm mask and summary, then
return the original output without changing cache tensors. Do not call
`BasePress.forward_hook` or `compress`, since those mutate the cache. Call the
press's `post_init_from_model(model)` for API compatibility, then reproduce the
context manager's `layer.self_attn.rotary_emb = model.model.rotary_emb` assignment
before scoring. The current path uses `DynamicCache`, `cache.layers[module.layer_idx]`,
`use_cache=True`, and `return_dict=True`; verify these fields.

The covariance-free scorer costs a query projection/statistic pass plus key-query
scoring per layer. It avoids the full attention tensor and covariance `O(T d_head^2)`.
Record hook wall time and GPU peak increment against the same split-prefill without
hooks. H3 can be measured as the existing
`engine._probe_logits` no-op/action first-step logit JS, while true hidden disturbance
would need a layer-output hook. A simulated probe makes H3 available before committing
the action, but costs roughly one extra first-step forward; defer it for cost and
likely redundancy, not because decision-time use is invalid.

## Exploratory slice and gates
First run a transparent hook/cost pilot on one existing development case at removal
fractions 0.25, 0.5, and 0.75. Verify finite scores, non-sink masks matching direct
Knorm, unchanged cache fingerprints and outputs, and recorded hook overhead. Do not
pool this case or treat it as new evidence.

If that passes, generate a fresh namespace with new seeds and provenance: 24 NIAH
and 12 CWE prompts, balanced across removal fractions 0.25, 0.5, and 0.75. Do not
reuse cached 12-row outputs or open a confirmation set.

Before collecting outcomes, require repeated full/full and split/split no-op
parity, preserve the original cross-partition divergence at token 118 as a
separate audit, and require the shared split-boundary reference and no-op to have
independent equal cache copies. Store one scored row per prompt/action, preserve
failed and unscorable rows, and compare the feature to the structural null before
any pooled model or deployment claim.
