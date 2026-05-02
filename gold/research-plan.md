# HERALD Research Plan

Converged design for the NeurIPS submission. Result of a multi-agent
discussion. The plan is staged so each phase produces a publishable
artifact even if later phases fail empirically.

## North Star

Measure compression-attributable damage rigorously, prove whether
cheap online signals predict it, then show predictor-guided control
improves the compute-quality frontier.

## Three Contributions, Each Independently Defensible

1. **Measurement methodology**: matched-prefix replay framework plus
   multi-resolution alignment study.
2. **Predictability**: cheap online signals forecast
   compression-attributable damage at multi-token horizons across
   models and tasks.
3. **Control**: predictor-guided intervention improves the cost-quality
   Pareto frontier.

Each phase has a pre-registered "go" criterion and a fallback that
keeps the paper viable.

## Mechanistic Hypothesis (Frame for Introduction)

KV-cache compression evicts or down-weights past tokens; the attention
layer at time t now operates on a degraded context. This degradation
manifests at output as a perturbation of the logit distribution. We
hypothesize the perturbation has temporal structure: small
perturbations are easily absorbed, but persistent or compounding
perturbations propagate into divergent generation trajectories. Local
logit signals (entropy, top-k concentration, instability between
consecutive timesteps) are the observable footprint of this
perturbation and so are candidate predictors of impending divergence.

## Damage Measurement: Multi-Resolution Hierarchy

### Token level (matched-prefix replay)

For every t in the compressed run, feed the *compressed* generated
prefix to the *uncompressed* model and capture its logits at position
t. Compare against the compressed model's own logits (already saved at
generation time). Conditioning on identical history is what makes the
divergence well-defined.

- KL(p_uncompressed || p_compressed) with explicit tail-mass logging
  for top-k truncation bias control.
- JS divergence (symmetric, bounded, easier to threshold).
- Top-1 rank shift of uncompressed argmax under compressed model.

### Trajectory level

- NLL of baseline output under compressed model (sequence likelihood
  ratio).
- First-divergence-point under greedy decoding.
- Sum of matched-prefix KL along the compressed trajectory.

### Sequence level

- BERTScore against baseline output.
- Embedding cosine similarity.
- Edit distance, ROUGE-L.

### Outcome level (paired counterfactual)

Track all four cells:

- baseline_correct, compressed_correct
- baseline_correct, compressed_wrong (gross harm)
- baseline_wrong, compressed_correct (gross help)
- baseline_wrong, compressed_wrong

Report:

- `gross_harm = P(baseline_correct AND compressed_wrong)`
- `gross_help = P(baseline_wrong AND compressed_correct)`
- `net_delta = gross_harm - gross_help`

The "compression occasionally helps" cell prevents the paper from
sounding one-sided and is statistically cleaner.

### Diagnostic tags (categorical, not primary labels)

Looping, non-termination, format break, drift. Used to characterize
*what kind* of failure occurred, not as training signal.

### Alignment study (methodological centerpiece)

Pairwise correlation and AUROC matrix across all damage metrics,
stratified by task / press / ratio, with bootstrap CIs over prompts.
This is what legitimizes intrinsic JS as a proxy for extrinsic damage.

## Phase 0: Engineering Dry-Run

**Scope**: 20 prompts, 1 task, 2 presses, 3 ratios.

**Goal**: Validate end-to-end pipeline. Storage schema, replay code,
metric computation, alignment code. Compare 4-token vs 8-token replay
sampling for information loss.

**Success criterion**: All metrics computable end-to-end on the slice.
No science claims here. Prevents wasting GPU weeks on a broken schema.

## Phase 1: Measurement Paper Core (Standalone-Publishable)

**Scope**:

- 1 model
- 4 tasks: GSM8K (math), HumanEval (code), IFEval (instruction
  following), LongBench-Single (long-context)
- Full press matrix (all kvpress methods plus random-eviction
  baseline)
- 6 ratios: 0, 0.5, 0.75, 0.875, 0.9375, 0.96875 (geometric near the
  cliff)
- 200 prompts per task
- Greedy decoding (deployment-aligned; no wasted seeds)
- Uniform matched-prefix replay every 8 tokens (committed; Phase 0
  measured Spearman rank correlation 0.953 between every-8 and the
  full per-token sequence on trajectory aggregates, vs 0.967 for
  every-4. The marginal information cost of every-8 vs every-4 is
  smaller than expected metric-to-outcome noise, and it halves
  Phase 1 replay cost. See `results/phase0/sampling_rate_report.json`
  and `gold/phase-0-results.md`.)
- Anomaly-triggered dense replay reserved for analysis only, never
  for training

**Outputs**:

- Multi-resolution damage metrics for every (prompt, press, ratio)
  cell
- Alignment matrix figure with bootstrap CIs

**Success criterion**: Intrinsic metrics significantly predict
extrinsic damage under held-out prompts and remain directionally
stable across at least 3 of 4 tasks. Reported via Spearman correlation
AND AUROC/AUPRC for intrinsic-to-extrinsic classification (e.g., does
future_max_JS_H classify baseline_correct AND compressed_wrong; does
trajectory NLL ratio classify diagnostic failures).

**Fallback**: "Negative methodology" paper. Five plausible measures of
compression damage; how they fail to align; why compression damage is
harder to measure than the field assumes. Still publishable.

## Phase 2: Predictor

**Inputs (Tier 0 + Tier 0.5 + Tier 1, no internals)**:

- Tier 0: top-1 prob, entropy, top-k concentration, log-rank slope of
  tail, top-1/top-2 ratio.
- Tier 0.5: rolling EWMA, rolling variance, rolling max of Tier 0.
- Tier 1: KL between consecutive timestep distributions
  (instability), distance from prompt embedding, output length so far.

Conditioned on (press_id, ratio) as auxiliary inputs so the model
learns press-specific dynamics.

**Targets**:

Phase 0 measured that per-token JS divergence on the matched-prefix
replay saturates at the information-theoretic ceiling (`ln 2 ≈ 0.693`)
for every compressed cell on Qwen2.5-7B-Instruct (134 runs,
2026-05-02; see `gold/phase-0-results.md`). Per-token JS therefore
cannot serve as the regression target — there is nothing to rank
within a saturated cell. Trajectory-level aggregates do retain rank
structure: the ratio-discrimination analysis on the same data
(`results/phase0/analysis/ratio_discrimination.json`) shows
`sum_kl`, `sum_js`, and `nll_ratio` cleanly separating ratio=0.5
from heavier ratios (Mann-Whitney p < 1e-5 for streaming_llm).

The predictor therefore targets trajectory- and sequence-level
quantities, not raw per-token JS:

- Multi-horizon (H in {5, 10, 25, 50}) joint prediction of:
  - Binary: trajectory-aggregate damage above threshold T over the
    window [t+1, t+H], where the aggregate is one of
    `partial_sum_KL_H`, `partial_sum_JS_H`, or `partial_NLL_ratio_H`
    (cumulative over the window).
  - Regression: severity quantiles (q in {0.1, 0.5, 0.9}) of those
    same partial-sum aggregates.
- Sequence-level risk via max or pooled token risk over the
  full trajectory's `sum_kl` / `sum_js` / `rouge_l` drop.

**Models, in increasing complexity**:

1. Logistic regression on Tier 0 (sanity).
2. XGBoost / LightGBM on full feature set (primary).
3. Small GRU on the streaming feature sequence (only if XGBoost leaves
   obvious headroom on lead time or calibration).

Stop at the first model that decisively beats baselines.

**Calibration**: Reliability diagrams, ECE. Split conformal intervals
as optional deployment polish, not headline (coverage guarantees do
not survive distribution shift to held-out model or held-out task).

**Baselines that have to be beaten**:

- Threshold on entropy.
- EWMA on entropy with fitted threshold.
- Online change-point detection on entropy or perplexity.
- Logistic regression on raw tokens.

**Threshold selection**: Validation-set alignment with extrinsic
outcomes plus sensitivity analysis on T, W, K. Heuristic-dependence is
moved, not eliminated; defended by the sensitivity study.

**Success criterion**: Best predictor's AUROC on held-out ratio
exceeds best entropy/EWMA baseline by >= 0.05, with paired bootstrap
CI not crossing zero.

**Fallback**: "Compression damage is not predictable from local logit
features beyond chance baselines." Still a meaningful contribution.

## Phase 3: Generalization

**Scope expansion**:

- 3 model families (e.g., Llama-3.1-8B, Qwen2.5-7B, Mistral-7B).
- Same 4 tasks.
- Same press and ratio matrix.
- 200 prompts per (model, task) cell.

**Evaluation splits, increasing difficulty**:

- Held-out prompts (IID sanity).
- Held-out compression ratio (interpolation along the cliff).
- Held-out press (cross-mechanism).
- Held-out task (cross-distribution).
- Held-out model family (the headline).

**Success criterion**: Held-out model family AUROC stays within 0.10
of held-out prompt AUROC.

**Fallback**: "The predictor must be trained per model family."
Honest scope statement.

## Phase 4: Closed-Loop Control

**Two presses (one mask-based, one eviction-based)**:

- StreamingLLM: mask-based, easy intervention via toggle or relaxed
  ratio.
- SnapKV or ExpectedAttention: eviction-based, recompute fallback or
  temporary full-KV recovery.

**Controller design (per-segment gating, primary)**:

- Every K tokens, compute risk from the previous window.
- If risk > threshold, relax compression for next K tokens.
- Evaluate K in {8, 16, 32}.
- Policies:
  - StreamingLLM: disable compression or lower ratio for next segment.
  - SnapKV / ExpectedAttention: recompute next segment with full KV,
    or fall back to lower compression ratio.

The segment design also makes compute accounting cleaner.

**Per-token gating**: included as oracle/upper-bound ablation only,
not as deployable controller.

**Comparisons (Pareto curve of compute vs accuracy)**:

- Fixed compression at each ratio.
- No compression.
- Random gating at matched compute (load-bearing control: proves the
  predictor's gating policy helps, not that any intervention helps).
- Predictor gating.

**Success criterion**: Predictor gating recovers >= 50% of
compression-induced quality loss at <= 50% of the compute cost of
"always uncompressed," and beats random gating at matched compute by
a paired-test significant margin.

**Headline metric**: "Predictor gating recovers X% of compression-
induced quality loss while preserving Y% of compute savings."

## Cross-Cutting Rigor

- Bootstrap CIs everywhere; paired tests for predictor-vs-baseline
  comparisons.
- Multiple training seeds for the predictor itself (data is
  deterministic under greedy; predictor training is not).
- Pre-registered decision rules and fallbacks for each phase.
- Sensitivity studies on every threshold (T, W, K in onset
  definition; predictor threshold; conformal alpha if used).
- Reproducibility: deterministic kernels, frozen seeds, released code,
  released per-token feature dataset (not raw model outputs).

## Out of Scope for v1

Deliberately deferred to v2 / journal extension:

- Human or LLM-judge severity annotation.
- Open-ended judge-scored tasks (MT-Bench).
- Tier 2 features (hidden states, attention internals) as predictor
  inputs.
- 13B+ model variants.
- Sampled (non-greedy) decoding as primary regime.

The current scope is sufficient for a NeurIPS submission; expansion
goes in v2.

## Implementation Priority

The next concrete blocker is Phase 0: matched-prefix replay pipeline
plus the multi-resolution metric table, validated on a 20-prompt
slice. Until that is clean, scaling to Phase 1 is premature.

The architecture debate is closed; execution begins with the schema
and replay code.
