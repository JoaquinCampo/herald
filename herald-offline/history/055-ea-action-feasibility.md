# ExpectedAttentionPress as a candidate action

**Status: conditionally feasible for one exploratory action slice.** `kvpress.ExpectedAttentionPress` can provide a per-KV-head retention ranking from the full B0 prefix state, but it is not plug-compatible with the current `run_pair_pilot.py` boundary runner. The next step, if approved by the owner, is a small technical integration slice with a fixed configuration and one removal rate. This note does not authorize a MuSiQue pilot or make a predictor claim.

## What the existing press provides

The installed `kvpress` version is 0.5.2. The relevant source was inspected with these SHA-256 hashes:

- `expected_attention_press.py`: `f0e8525d9e19b68a0f123cea3bb39b76af4adbf888ef9c4f55680f582feedc42`
- `scorer_press.py`: `a5eb57a8d9defdaf1f46414fad1584e0c070643a85116d9a85f2e9321cb29728`
- `base_press.py`: `2b33576eed0a57936502636af8f82566ee83d1072435d27ca55e61ad7745f1b0`

The constructor is `ExpectedAttentionPress(compression_ratio=0.0, n_future_positions=512, n_sink=4, use_covariance=True, use_vnorm=True, epsilon=0.0)`. Its scoring path drops the sink positions, computes the mean and, optionally, covariance of the pre-RoPE query states, evaluates average future RoPE positions, repeats grouped-query attention scores over KV heads, averages the corresponding query groups, and optionally multiplies by value norms. Sink positions are restored by padding them with the maximum score. `ScorerPress` then keeps `int(k_len * (1 - compression_ratio))` entries per KV head with `topk` and gathers those entries.

This gives a usable action mask, with two conditions. The score must be computed from the same B0 prefix whose cache will be acted on, and the exact configuration must be frozen before comparison. In particular, `n_future_positions`, `n_sink`, covariance use, value-norm weighting, epsilon, and the removal rate are estimand-defining choices. The score is per KV head after GQA aggregation, not per query head.

## Why it does not fit the current runner directly

The current v4 runner computes a plain boundary at the end of the prompt prefix, retains the pending final prompt token, and continues with absolute logical positions. The read-only v3 engine used by the runner only has a KNorm compression path. Its continuation helper directly replaces the cache with the compressed result and has no generic action-mask or arbitrary-index gather interface.

Applying `ExpectedAttentionPress` in the source prefill hook would mutate the cache while the reference prefill is still running. That would alter deeper layers and invalidate the paired reference. The safe ordering is therefore:

1. Run the ordinary plain prefill and save the B0 boundary, pending token, and logical position.
2. Run an independent instrumented prefill over the same B0 prefix, with `compression_ratio=0`, only to collect keys, values, hidden states, and scores. The resulting cache must be exactly equal to the plain boundary.
3. Extract the per-KV-head `topk` indices from the scores, map them to original B0 positions, and gather those positions into an independent clone of the plain boundary.
4. Apply the action before processing the pending token. Continue with the existing absolute logical positions and attention-mask sizing for the physically shortened cache.

The action must preserve original position identity. `topk` order can differ from physical position order, while RoPE positions remain tied to the original absolute locations. The Qwen cache also represents physical length and receives `cache_position`, so the action adapter must keep the original logical position and account for the shortened physical prefix.

## Cost and technical risks

Keeping an exact plain reference while collecting scores requires two full prefills, one plain and one instrumented. The instrumented pass adds per-layer query statistics, and covariance can be expensive because it introduces head-dimension covariance work. The action also needs a cache clone and per-head gathers, so peak memory includes the uncompressed boundary and the acted-on clone. These costs must be measured before any deployment interpretation.

The main correctness risks are score collection from the wrong prefix, applying compression after the pending token, changing the frozen RoPE or sink policy, losing per-head index identity, and silently using a different cache or source continuation. `use_covariance=False` can reduce cost, but it is a different predeclared action and must not be presented as equivalent to the default.

## Minimum technical acceptance slice

Before a MuSiQue experiment, run one CPU or small-model technical case using the exact current boundary path, one frozen EA configuration, and one fixed removal rate. Require all of the following:

- plain and instrumented B0 boundaries are exactly equal, and the source cache is unchanged by score collection;
- repeated runs produce finite scores, equal kept counts, identical original-position masks, and a reproducible mask hash;
- the physical cache length and bytes decrease by the expected amount, while sink and kept-position metadata remain valid;
- the no-op instrumented path matches the plain path in generated token IDs and termination;
- the acted-on clone continues from the same absolute logical position and completes without cache or attention-mask errors;
- the action mask agrees with direct `score.topk` output for every layer and KV head;
- the source item and the reference/action cache identities are recorded, with no accidental reuse of the source continuation.

Only after this slice passes should the owner decide whether to run a MuSiQue pilot. That pilot would need a competent reference, fixed horizons, a signed final token-F1 loss, and a matched predeclared action baseline such as KNorm. It must report the extra prefill, scoring, clone, and gather overhead. Any observed quality change would establish the behavior of this action under the tested protocol, not that ExpectedAttentionPress is a validated predictor.

## Recommendation

Treat ExpectedAttentionPress as a technically plausible, expensive exploratory candidate action. Do not integrate it into the shared engine yet. Implement the cache-action adapter locally in a selected experiment runner, preserve the plain B0 boundary as the reference, and make the technical acceptance slice the gate for any later MuSiQue work.
