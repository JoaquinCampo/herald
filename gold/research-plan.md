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

## Contribution Validation Bar

Related work narrows the novelty claim. HERALD should not claim that
"logits contain quality information"; entropy-based and lightweight
per-token predictors already establish that. The claim is that cheap
online signals can forecast *future, compression-attributable* damage
and drive useful intervention across compressors, ratios, tasks, and
models.

Before making the strong paper claim, validate against
`gold/contribution-validation.md`. In particular:

- Beat random, position-only, ratio-only, entropy/EWMA, change-point,
  single-feature, and logistic-regression baselines.
- Treat held-out press / cross-compressor transfer as a headline
  experiment, not an appendix.
- Tie diagnostic tags to prior failure-mode definitions where possible
  and label heuristic thresholds honestly.
- Quantify whether cheap online features contain information beyond
  position and ratio proxies.
- Include at least one real closed-loop intervention demo before
  claiming the control contribution.

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

Sequence metrics are paired against the uncompressed baseline for the
same prompt. They measure output drift and semantic drift; they do not
by themselves prove task failure. ROUGE-L / edit / length difference
are deterministic v1 metrics. Embedding cosine should be computed when
the dependency environment is available and reported separately from
lexical overlap because paraphrase can lower ROUGE-L without lowering
semantic quality.

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

Outcome correctness is task-specific and must use real evaluators:
GSM8K exact-answer extraction, HumanEval execution/pass-fail, Qasper
F1/EM, and IFEval constraint scoring. Presence checks are not valid
correctness labels. If a task evaluator is saturated or placeholder-
only, correctness is reported as unavailable for that task rather than
silently treated as "all correct."

### Diagnostic tags (categorical, not primary labels)

Looping, non-termination, format break, drift. Used to characterize
*what kind* of failure occurred, not as training signal.

### Canonical run-level damage table

Every phase after the fixed-ratio sweep should build one canonical
`run_damage.parquet` table with one row per compressed run, joined to
the paired uncompressed baseline. This is the table used for
measurement validation and for evaluating whether a predictor matters
to users.

Required columns:

- Metadata: `run_id`, `baseline_run_id`, `task`, `prompt_id`, `press`,
  `compression_ratio`.
- Intrinsic trajectory damage: `sum_kl`, `sum_js`, `nll_ratio`,
  `first_divergence_point`.
- Sequence drift: `rouge_l_drop`, edit-distance ratio, length-diff
  ratio, and embedding-cosine drop when available.
- Diagnostic tags: looping, non-termination, format break, drift.
- Task quality where available: baseline score, compressed score,
  `gross_harm`, `gross_help`, and task-specific score deltas.

This table makes the distinction explicit:

- Intrinsic replay labels are suitable training targets because they
  are defined for every task.
- Sequence and semantic drift are non-circular validators that work
  without task graders.
- Task correctness is the strongest validator, but only where a real
  evaluator exists.

### Predictor target discipline

The headline HERALD predictor must not be trained or evaluated as a
looping / non-termination detector. Diagnostic tags are allowed for
onset-aligned analysis, qualitative taxonomy, and auxiliary tasks, but
they are not the main compression-damage label.

The main predictor target is paired, counterfactual damage relative to
the uncompressed baseline for the same prompt:

- **Outcome harm**: `baseline_correct AND compressed_wrong`.
- **Sequence damage**: ROUGE-L / embedding similarity / edit-distance
  degradation versus the paired uncompressed output.
- **Trajectory damage**: future partial sums of matched-prefix
  KL / JS / NLL-ratio over `[t+1, t+H]`.

For every compressed token `t`, the predictor observes only cheap
online features available at time `t`; the label asks whether paired
compression-attributable damage occurs within horizon `H` or reaches a
severity threshold by completion. If task correctness is saturated,
Phase 2 should use continuous sequence/trajectory severity as the
primary target rather than falling back to diagnostic tags.

Reviewer-facing rule: HERALD predicts future counterfactual
compression harm. It does not merely classify abnormal text.

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
- 8 ratios: 0, 0.25, 0.375, 0.5, 0.75, 0.875, 0.9375,
  0.96875. Phase 0 showed the cliff is at or below 0.5 on GSM8K, so
  0.25 and 0.375 are required to map the gentle-damage regime.
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
- Segment-level aggregates for K in {8, 16, 32}: cheap online
  features, trajectory partial sums, repetition/progress markers, and
  answer-state markers. These are computed from existing per-token
  data and are required for Phase 2 and Phase 4.
- Segment-level cost instrumentation: retained KV size where
  available, wall-clock/token, tokens/sec, and memory telemetry where
  the platform exposes it. These are required for cost-quality Pareto
  curves and reported as part of the measurement substrate.
- Generation loop refactored behind a policy abstraction:
  `FixedRatioPolicy` is the default for the headline sweep, and
  `SwitchAtOffsetPolicy` exists only for the pre-registered
  intervention probe below. This is infrastructure, not a controller
  result.

**Outputs**:

- Multi-resolution damage metrics for every (prompt, press, ratio)
  cell
- Canonical `run_damage.parquet` joining intrinsic trajectory damage,
  sequence/semantic drift, diagnostic tags, and task-quality deltas
  where evaluators exist
- Segment metrics and cost metrics for every fixed-ratio cell
- Alignment matrix figure with bootstrap CIs

**Success criterion**: Intrinsic metrics significantly predict
extrinsic damage under held-out prompts and remain directionally
stable across at least 3 of 4 tasks. Reported via Spearman correlation
AND AUROC/AUPRC for intrinsic-to-extrinsic classification (e.g., does
future_max_JS_H classify baseline_correct AND compressed_wrong; does
trajectory NLL ratio classify diagnostic failures).

Post-Phase-1 clarification: if a task's correctness field is
instrumentation-saturated because the evaluator is a placeholder
(e.g., presence check), the strict outcome-harm criterion is reported
as undefined for that task, not as a scientific failure of the
intrinsic signal. The methodological follow-up is to evaluate the same
bar against continuous severity columns in `run_damage.parquet` and to
wire the proper deterministic evaluator. The pre-registered strict
reading still remains reportable; this clarification prevents a
placeholder grader from being mistaken for user-facing correctness.

**Fallback**: "Negative methodology" paper. Five plausible measures of
compression damage; how they fail to align; why compression damage is
harder to measure than the field assumes. Still publishable.

### Phase 1 Intervention Probe (Pre-Registered Side Study)

This probe is included to keep the staged plan aligned with the
ultimate controller goal without turning Phase 1 into a full
controller dataset. It is a measurement of *recoverability*, not a
policy sweep and not a Phase 4 result.

**Purpose**: determine whether reactive control is feasible, or
whether Phase 2 must focus on anticipatory prediction. Phase 0's
per-token JS saturation is indirect evidence only; it does not prove
that a damaged trajectory cannot be salvaged by an intervention.

**Budget**: approximately 900 additional runs, roughly 3% overhead
relative to the Phase 1 fixed-ratio sweep.

**Pre-registration timing**: stratum rules, sampling algorithm,
offsets, intervention definitions, endpoints, and seed/decoding policy
are written down before the Phase 1 sweep launches. Actual prompts are
sampled after the sweep reveals the baseline-conditioned strata.

**Strata**:

- Clear success: baseline and compressed fixed-ratio runs are both
  correct / high-quality.
- Boundary-unstable: correctness or quality flips across adjacent
  ratios.
- Clear compression failure: baseline is correct / high-quality and
  compressed fixed-ratio run fails.

**Tasks**: split the probe across all four Phase 1 tasks. Recovery
dynamics may differ across math, code, instruction following, and
long-context generation; a GSM8K-only probe is not enough to inform
the controller design.

**Intervention vocabulary is press-specific**:

- StreamingLLM / mask-based: lift pressure by disabling or relaxing
  the mask for the next segment.
- Continuous eviction methods such as Knorm / TOVA: lift pressure by
  stopping further eviction for the next segment where the press
  implementation supports it.
- Prompt-time eviction methods such as SnapKV / ExpectedAttention:
  there may be no meaningful "stop further eviction" action after
  prefill. The deployable arm is ratio reduction only if it can be
  implemented without changing the realized history; otherwise the
  meaningful ablation is reprefill.
- Reprefill / recompute fallback: oracle intervention, run on a
  20% subset. It bounds what recovery could achieve if the system is
  willing to pay a high compute cost.

**Offset arms**:

- Deployable offsets: fixed token positions, budget fractions, or
  fixed online thresholds defined before seeing outcomes.
- Offline-diagnostic offsets: generated-length fractions, used only to
  characterize recoverability and never presented as deployable.

**Primary estimand**: paired comparison of
`switch-at-offset-T` versus `continue-fixed` for the same
`(prompt, task, press, ratio, history)` wherever the intervention is
well-defined.

**Probe decision rules**:

- Reactive control feasible if the deployable intervention recovers
  >= 50% of the quality gap between fixed compressed and uncompressed
  baselines on the boundary-unstable stratum, with paired bootstrap CI
  excluding zero, on at least 2 of 4 tasks.
- Reactive control likely infeasible if neither deployable
  interventions nor reprefill recover > 20% of the quality gap on any
  task. Phase 2 should then emphasize anticipatory horizons and early
  warning.
- Mixed outcome: document which tasks and presses support reactive
  versus anticipatory control, then set Phase 2 targets and Phase 4
  policies accordingly.

**Narrative constraint**: the probe must stay a self-contained
intervention study. The Phase 1 headline remains the fixed-ratio
measurement methodology.

## Phase 2: Predictor

**Status after Phase 2 baseline sweep (2026-05-05)**:
`gold/phase-2-results.md` records a passing result. Logistic
regression over all cheap online features (`lr_all_cheap`: Tier 0 +
rolling/EWMA + position + ratio) beats the best entropy/EWMA-style
single-feature baseline (`entropy_mean_8`) by >= 0.05 AUROC on every
split x horizon cell:

- Held-out prompts: min delta +0.087.
- Held-out ratios: min delta +0.054.
- Held-out presses: min delta +0.062.
- Held-out tasks: min delta +0.073.

Per-run aggregation is also meaningful: max OOF score correlates with
`rouge_l_drop` (Spearman 0.73), `sum_js` (0.74), looping (AUROC 0.81),
and non-termination (AUROC 0.84). This validates the predictor as a
run/segment risk estimator.

Important limitation: lead-time analysis against looping /
non-termination onsets does not pass. The JS-trained token predictor
does not fire before those tag onsets; the score is inverted near the
onset because collapsed loops can have low future JS. Therefore the
v1 controller should be framed around segment/run risk gating, not
precise "loop will start in N tokens" onset alarms.

**Inputs (Tier 0 + Tier 0.5 + Tier 1, no internals)**:

- Tier 0: top-1 prob, entropy, top-k concentration, log-rank slope of
  tail, top-1/top-2 ratio.
- Tier 0.5: rolling EWMA, rolling variance, rolling max of Tier 0.
- Tier 1: KL between consecutive timestep distributions
  (instability), distance from prompt embedding, output length so far.

Conditioned on (press_id, ratio) as auxiliary inputs so the model
learns press-specific dynamics.

If the Phase 1 intervention probe shows reactive control is feasible
for some press/task regimes, Phase 2 includes short-horizon reactive
risk targets and segment-level decision targets for those regimes. If
the probe shows recovery is weak, Phase 2 emphasizes anticipatory
targets with enough lead time to intervene before derailment.

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
- Outcome-harm risk (`baseline_correct AND compressed_wrong`) where
  Phase 1 produces a non-saturated correctness distribution. If
  correctness remains saturated, outcome harm is reported as an
  extrinsic validator but not used as the sole training target.

Diagnostic tags (`looping`, `non_termination`, later `format_break` /
`drift`) may be auxiliary outputs or stratification variables, but
they are not the headline training target.

**Models, in increasing complexity**:

1. Logistic regression on Tier 0 (sanity; failed the 0.05 bar against
   entropy in Phase 2).
2. Logistic regression on all cheap features (passed the Phase 2
   predictor criterion; current v1 baseline to beat).
3. XGBoost / LightGBM on full feature set (next-tier model; run only
   to measure headroom over `lr_all_cheap`, not to rescue the core
   claim).
4. Small GRU on the streaming feature sequence (only if XGBoost leaves
   obvious headroom on lead time or calibration).

Stop at the first model that decisively beats baselines unless the
next model is needed to address a known limitation (currently:
lead-time/onset behavior, calibration, or controller utility).

**Calibration**: Reliability diagrams, ECE. Split conformal intervals
as optional deployment polish, not headline (coverage guarantees do
not survive distribution shift to held-out model or held-out task).

**Baselines that have to be beaten**:

- Random predictor.
- Position-only predictor.
- Compression-ratio-only predictor, reported as a metadata/proxy
  baseline rather than as an online damage signal.
- Threshold on entropy.
- EWMA on entropy with fitted threshold.
- Online change-point detection on entropy or perplexity.
- Single-feature thresholds for each cheap feature family.
- Logistic regression on raw tokens.
- Logistic regression on Tier 0 features.

If entropy, position-only, or ratio-only baselines are within a few
AUROC points of the primary predictor, the multivariate predictor claim
is weak and must be scoped accordingly.

**Threshold selection**: Validation-set alignment with extrinsic
outcomes plus sensitivity analysis on T, W, K. Heuristic-dependence is
moved, not eliminated; defended by the sensitivity study.

**User-facing validation**: after per-token predictions are aggregated
to per-run scores (e.g., max risk, mean top-k risk, count above
threshold), evaluate them against `run_damage.parquet`:

- Spearman correlation with continuous drift/severity metrics.
- AUROC/AUPRC against diagnostic catastrophic tags.
- AUROC/AUPRC against task outcome harm where real evaluators exist.
- Calibration/reliability against the chosen intervention threshold.

This validation is not the training objective; it is the evidence that
the intrinsic target corresponds to damage a user would notice.

**Success criterion**: Best predictor's AUROC on held-out ratio
exceeds best entropy/EWMA baseline by >= 0.05, with paired bootstrap
CI not crossing zero. Also report the margin against position-only and
ratio-only baselines; if either dominates, the result shifts toward a
measurement paper or a per-regime calibration paper.

Post-Phase-2 clarification: the point estimates pass on every
split/horizon. Before paper submission, add paired-bootstrap CIs on the
cross-fold mean deltas to make the >=0.05 decision CI-backed rather
than point-estimate-only.

**Fallback**: "Compression damage is not predictable from local logit
features beyond chance baselines." Still a meaningful contribution.

### Phase 2b: Model / Label Follow-Up

Phase 2 passed, but four follow-ups are required before locking the
controller design:

- **Headroom check**: train XGBoost / LightGBM on the same dataset and
  compare against `lr_all_cheap`, especially on held-out ratio and
  held-out press.
- **Label-family check**: repeat the baseline sweep for
  `future_sum_kl_H` and `future_max_js_H` (already present in
  `phase2_tokens.parquet`) and compare lead-time/per-run validation
  against the current `future_sum_js_H` label.
- **Segment-risk check**: aggregate predictions over K in {8,16,32}
  and evaluate segment/run risk, because exact tag-onset lead time is
  weak under the JS label.
- **CI upgrade**: add paired-bootstrap confidence intervals for
  predictor-vs-baseline deltas.

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

**Cross-compressor requirement**: held-out press is load-bearing for
the black-box claim. If cross-press transfer collapses, the paper must
state that HERALD is a per-compressor calibration framework rather
than a universal compressor-agnostic predictor.

**Fallback**: "The predictor must be trained per model family."
Honest scope statement.

## Phase 4: Closed-Loop Control

**Two presses (one mask-based, one eviction-based)**:

- StreamingLLM: mask-based, easy intervention via toggle or relaxed
  ratio.
- SnapKV or ExpectedAttention: eviction-based, recompute fallback or
  temporary full-KV recovery.

The exact action set is inherited from the Phase 1 intervention probe:
each press has a written intervention vocabulary before controller
training begins. The controller is not allowed to use an action that
was not validated or bounded by the probe.

**Controller design (segment/run risk gating, primary)**:

- Every K tokens, compute risk from the previous window and/or the
  maximum predicted risk seen so far in the run.
- If segment/run risk > threshold, relax compression for next K tokens
  or switch to a safer mode for the remainder of the generation.
- Evaluate K in {8, 16, 32}.
- Policies:
  - StreamingLLM: disable compression or lower ratio for next segment.
  - SnapKV / ExpectedAttention: recompute next segment with full KV,
    or fall back to lower compression ratio.

The segment design also makes compute accounting cleaner.

Phase 2's lead-time result changes the emphasis: v1 should not claim
precise onset prediction for looping/non-termination. It should test
whether accumulated segment/run risk can drive a useful compression
policy. Exact onset prediction remains a label/model follow-up, not a
precondition for the controller demo.

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

Phase 0, Phase 1, and the Phase 2 baseline predictor are complete.
The next concrete blockers are:

1. Add paired-bootstrap CIs for the cross-fold predictor-vs-baseline
   deltas in Phase 2.
2. Run Phase 2b: XGBoost / LightGBM headroom check plus alternative
   label families (`future_sum_kl`, `future_max_js`) and segment-risk
   aggregation.
3. Update paper figures/tables from `gold/phase-2-results.md`:
   baseline table, split-transfer table, per-run validation table, and
   lead-time limitation.
4. Decide whether to run the Phase 1 intervention probe now or roll it
   into Phase 4 as the first controller feasibility experiment.
5. For control, implement the smallest segment-risk gating demo on one
   mask-based press first, with random matched-compute gating as the
   load-bearing control.
