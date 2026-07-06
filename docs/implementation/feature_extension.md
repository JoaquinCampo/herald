# Feature extension: attention reliance, press scores, probe divergence

Status: design, approved direction (2026-07-05). Motivated by the
controller exhaustion report (`exhaustion_report.md`): reference-
stream logit statistics do not carry prompt-level fragility; the
0.10 savings rung needs features that measure, per prompt and per
moment, how much the generation depends on what compression would
destroy. Rationale: `docs/_why/5_attention_features.md`.

## Revision (2026-07-05, evening): hierarchy after Phase 0

Decisions from the Phase 0 diagnostics and review with the user:

1. HEADLINE HYPOTHESIS: compressor-agnostic fragility from
   attention internals. The exhaustion analysis showed the
   consensus (compressor-blind) label clears the 0.10 rung on all
   three held-out compressors; what failed was distilling it from
   LOGIT internals. Attention-reliance features are the candidate
   internals that measure dependence rather than confidence.
2. PROBE DEMOTED to dataset-only diagnostic. Its ~6 percent
   deployment compute means the controller must beat the no-probe
   baseline by well more than that in KV savings; it is excluded
   from the deployed controller feature set. (Phase 0: token-match
   direction confirmed; raw step-0 h0/d0 scalars falsified as-is,
   they mislead across the compressor shift; any native probe
   would need distribution-level divergence, not raw deltas.)
3. PRESS FEATURES are a secondary ablation, not the headline. They
   are captured for free during hybrid regeneration and quantify
   what compressor-awareness is worth on top of pure internals.
   They measure the eviction about to happen (defined for any
   press), never compressor identity.
4. FAST PATH: regenerate REFERENCES ONLY (all 3 tasks, tap on,
   hours of GPU). Existing hybrid rows (all dq labels) join the
   widened reference features by (prompt_id, s) provided the
   regenerated greedy gen_ids are identical to the stored ones;
   validate token-identity per prompt after the run and exclude
   mismatching prompts. The headline experiment then runs on the
   full 224k-row dataset without re-running any hybrid. The gsm8k
   hybrid pilot (press ablation) follows only if still wanted.

All designs below keep `attn_implementation="sdpa"`: nothing
materializes an N x N attention matrix. Row/column statistics are
reconstructed from one 1 x N attention row per decode step, computed
by a side matvec against the KV cache we already hold.

## Phase 0 (local, no new data): probe-lite from existing artifacts

The sweep already persists the hybrid stream's per-step logit
scalars (`hybrid_features/{comp}__{ratio}/{id}__s{s}.npy`,
`storage.py`), which the switch-dataset builder ignores. Step 0 of
that matrix is the model's reaction to the freshly compressed cache
at the switch point, BEFORE any compressed token is committed. In
deployment this equals a 1-token probe (one extra forward pass with
the compressed cache).

New columns on the existing dataset, causal for a 1-token probe:

- `probe__<base>`: hybrid step-0 value of each per-step scalar
  (entropy, max_prob, margin_prob, chosen_logprob, topk_mass_*).
- `probe__delta_<base>`: step-0 hybrid value minus the reference
  value at position s (same scalar, same position).
- `probe__token_match`: 1.0 if the hybrid's first generated token id
  equals the reference token at position s (from gen_ids), else 0.
  (Both streams are greedy, so this is a real divergence bit.)

These are compressor-SPECIFIC measurements (the probe runs the
actual compressor), which is the point: they sidestep the
compressor-transfer problem instead of fighting it. Validation:
rerun the oracle-tau reachability diagnostics with probe columns
added. If the model-space ceiling moves materially above 0.10,
the probe hypothesis is confirmed cheaply; the Orion regeneration
then mainly adds the richer signals below.

Caveat to keep honest: step-0 features come from the same forward
pass whose token becomes hybrid gen_ids[0]; only step 0 may be used
as a decision input (a w-token probe uses steps < w and must be
declared as w extra forwards of deployment cost).

## Phase 1 (local): attention-reliance hooks (SPOT-adapted)

New module `src/herald/attention_features.py`. A forward hook on
`layer.self_attn` for a small set of layers (quarter-depth
convention: llama 32L -> {8, 16, 24}; qwen3 36L -> {9, 18, 27}),
mirroring how kvpress hooks the same modules. Per decode step and
hooked layer, the hook:

1. Recomputes the current token's attention row per KV head:
   q_t from `get_prerope_query_states(module, hidden_states)` +
   rope from `kwargs["position_embeddings"]`; K from
   `extract_keys_and_values(kwargs["past_key_values"], layer_idx)`;
   row = softmax(q_t K^T / sqrt(d)). Cost: one matvec per layer,
   1 x N memory; the sdpa kernel is untouched. (Same recompute
   pattern SnapKV uses internally.)
2. Reduces the row (mean over heads, plus max over heads for the
   concentration stats) to per-step scalars:
   - `attn_prompt_mass`: attention mass on prompt tokens
   - `attn_sink_mass`: mass on the first 4 positions
   - `attn_local_mass`: mass on the last 64 positions
   - `attn_entropy`: row entropy
   - `attn_top8_mass`: top-8 concentration (max-head variant too)
3. Maintains column accumulators: running per-entry sums of
   incoming attention (the reliance profile of the cache). Reduced
   per step to `attn_reliance_prompt_share` (share of accumulated
   incoming mass on prompt tokens) and
   `attn_reliance_drift` (L1 change of the normalized profile over
   the last 16 steps).

Cross-layer aggregation (SPOT's M and Sigma): mean and variance of
each scalar across the hooked layers, persisted as the feature
(per-layer values are not persisted). Temporal dynamics come free
downstream via `derive_features` (deltas, EWMA, rolling stats) by
adding the new scalars to `DYNAMIC_BASES`.

Persistence: the per-step reference npy gains the new columns.
Because width changes, reference json gains a `feature_names` list;
`load_references` uses it when present and falls back to the legacy
`FEATURE_NAMES` order otherwise. Existing artifacts stay readable.

Overhead budget: 3 of 32 layers, one extra K read + matvec per
hooked layer per step. Expected low single-digit percent wall-clock;
the smoke test measures it (accept <= 10 percent, target <= 5).

## Phase 2 (Orion, needs approval): regeneration with new signals

1. Attention-reliance features on the reference stream (Phase 1
   hooks) for all prompts/tasks.
2. Press-score features at each (compressor, ratio, s): wrap
   `ScorerPress.compress` to capture `scores` (batch, kv_heads, N)
   at the prefill eviction and reduce to:
   - `press_score_entropy`: entropy of the normalized score dist
   - `press_evicted_reliance`: share of the reference stream's
     accumulated incoming attention (Phase 1 column accumulator at
     the matching layer, head-mean) that sits on entries the press
     evicts. "How much of what the generation leans on dies."
   - `press_evicted_share_prompt`: fraction of evicted entries in
     the prompt region.
   These are computable at decision time in deployment (the press
   scoring pass is cheap and runs before committing to eviction).
3. Full probe features (Phase 0 columns computed natively, plus a
   declared w=4 probe variant).

Same sweep machinery (`run_sweep`), resumable storage, batch-1
hybrids, greedy self-check unchanged. Wall-clock is dominated by
the hybrid grid as before; new overhead is the hook cost on
references plus nothing meaningful on hybrids.

## Smoke test plan (before any Orion job)

Local Mac, tiny random-weight llama (hf-internal-testing), CPU:

1. Unit tests: hook row equals eager attention row on a toy config
   (correctness against `output_attentions=True` ground truth);
   column accumulator equals sum of rows; mass features sum to ~1;
   causality (features at step t unchanged by future tokens).
2. Integration: `generate_reference` with hooks on 2 layers runs
   end-to-end, npy width matches `feature_names`, self-check passes.
3. Overhead: wall-clock with/without hooks at a 2k-token context,
   reported in the PR/log.
4. Press-score capture: wrapped press reproduces identical
   evictions (token-for-token hybrid output vs unwrapped press) and
   yields finite score features.

On Orion, the standard 10-second nohup smoke check plus one
single-prompt end-to-end shard before the sweep.

## Outcome (2026-07-05, night)

The headline test ran: tapped references regenerated for all 3 tasks
(token-identical or prefix-salvaged against the legacy grid, zero
drops), dataset rebuilt with 16 `feat__attn_*` columns (215,759 rows),
canonical leave-one-compressor oracle-tau readout. Result: the
attention-reliance aggregates do not lift the compressor-agnostic
worst-case ceiling. Across XGB seeds the classifier worst-case
averages 0.049 with the attention columns vs 0.061 without; the
regressor collapses on knorm held-out (0.044 to 0.001). The per-cell
pattern (streaming_llm consistently helped, knorm consistently hurt,
worst-case knorm-bound) indicates the layer-mean/var aggregates carry
an eviction-adjacent component that misleads under knorm shift rather
than a universal fragility signature. Details in
`results/predictor/experiments/experiment_log.md`
(attn_ceiling_canonical entry).

The follow-up arc (online forecasting from the post-switch hybrid
stream, the within-compressor reframe, and the settled deployable
calibration pipeline) is documented in `online_forecasting.md`.
