# Phase 4 Controller Design

Date: 2026-05-06.
Status: revised after kvpress feasibility research.

## Recommendation

Proceed with Phase 4, but change the controllable substrate.

The original design targeted mid-generation toggling of
`StreamingLLMPress`. That is not implementable in the installed
kvpress: StreamingLLM is a prefill-time physical KV prune, not a live
mask. Once the prompt KV is pruned, disabling the hook during decode
cannot restore dropped entries.

Internet + installed-package research found a better path: kvpress now
ships experimental decode-time compression wrappers:

- `DecodingPress`: periodically compresses the KV cache during token
  generation with a target cache size.
- `PrefillDecodingPress`: combines prefill and decoding presses.
- `DMSPress(..., decoding=True)`: threshold-based dynamic memory
  sparsification that can run during decoding.

Therefore v1 Phase 4 should test HERALD as a controller over a
runtime-controllable decode-time compressor, not over static
StreamingLLM. StreamingLLM remains a fixed-compression baseline because
it was part of Phase 1/2, but it is not the primary controller press.

Recommended v1 controller substrate:

- Primary candidate: `DecodingPress(base_press=KnormPress(),
  target_size=..., compression_interval=16, hidden_states_buffer_size=0
  or 128 after feasibility testing)`, using kvpress >= 0.5.3.
- Secondary candidate: `DMSPress(KnormPress(), threshold=...,
  sliding_window_size=128, decoding=True)` if `DecodingPress` is too
  brittle or too slow.
- Initial target sizes: choose from fixed-cache budgets corresponding
  approximately to Phase 1 ratios, e.g. {256, 512, 1024, 2048}, then
  refine after feasibility profiling.
- Segment size: `K=16`.
- Predictor: `lr_all_cheap` trained on `future_sum_js_25`.
- Calibration: isotonic calibration on held-out calibration prompts.
- Segment score: mean calibrated token risk over the completed segment.
- Run score: max segment score seen so far.
- Primary action: adjust decode-time cache budget for future segments.
  This is future-only control. Decode-time pruning is still physical
  and irreversible, so the controller cannot restore KV that was
  already evicted.

The controller is TCP-like with an important constraint: when risk is
low, it may tighten the budget at the next compression event; when
risk is high, it may delay, skip, or soften future compression events.
It cannot recover tokens already dropped without a separate reprefill
oracle path.

## Core Claim

Phase 4 should prove or falsify this claim:

At matched or lower compute, calibrated HERALD risk-guided cache-budget
control improves the compute-quality Pareto frontier compared with
fixed decode-time compression and random matched-budget control.

The load-bearing comparison is random budget changes with the same
compute/intervention budget. No compression / large-cache decoding is
the upper-quality and upper-cost anchor.

## Controller State

At each decision point, after every completed `K=16` generated tokens,
the controller observes only cheap online state available during
decoding:

- `segment_risk`: mean calibrated token risk over the last completed
  segment.
- `segment_risk_max`: max calibrated token risk in the last completed
  segment, logged for ablation.
- `run_risk`: max `segment_risk` over all completed segments so far.
- `token_pos`: number of generated tokens so far.
- `segment_idx`: zero-based completed segment count.
- `press`: active controllable press name, v1 one of
  `decoding_knorm` or `dms_knorm`.
- `active_cache_budget`: current target cache size or threshold.
- `initial_cache_budget`: starting target cache size or threshold.
- `previous_actions`: ordered list of prior action records.
- `protected_window_size`: number of recent tokens protected from
  eviction if the press exposes this parameter.
- `compression_events`: number of decode-time compressions so far.
- `estimated_cache_tokens`: current retained KV sequence length, if
  exposed by the cache or measured from `past_key_values`.
- `extra_compute_budget_used`: normalized compute exposure relative to
  fixed aggressive compression and no/large-cache compression.
- `cheap feature state`: rolling and EWMA feature state needed by
  `lr_all_cheap`, including entropy, top probability, top-k
  concentration, instability, token position, press id, and budget.

The controller must not observe replay metrics, future JS labels, task
correctness, sequence drift, diagnostic tags, final length, or the
paired uncompressed output.

## Action Space

The conceptual v1 action space is:

- Keep current decode-time cache budget for the next segment.
- Increase compression for future segments by lowering target cache
  size or raising the DMS threshold.
- Relax future compression by raising target cache size, lowering the
  DMS threshold, or skipping the next scheduled compression event.
- Enter safe mode for the rest of generation: use a large target size,
  no further compression events, or `target_size >= current_cache_len`.

This action space is monotone with respect to already-pruned KV.
Raising a budget after a compression event does not restore entries
that were physically gathered away. It only protects future tokens and
future cache growth. Any action that restores evicted KV is a reprefill
or offload-based oracle, not the deployable v1 controller.

Do not include StreamingLLM mid-generation disable in v1. It is not a
valid action under installed kvpress because StreamingLLM physically
prunes prompt KV during prefill and is dormant during decode.

Before any controller smoke, run a feasibility gate:

1. Verify `DecodingPress(KnormPress)` works on Qwen2.5-7B-Instruct.
2. Verify kvpress >= 0.5.3 on Orion before launch, because 0.5.1 has
   a per-sample state reset bug.
3. Set `compression_interval=16` so the control surface matches the
   `K=16` segment policy.
4. Verify compression happens during decode, not only prefill.
5. Log hook fire counts per step and retained cache lengths before and
   after each compression event.
6. Verify cache length / target size changes as expected.
7. Verify fixed-budget manual/generate path produces compatible token
   logs and quality metrics.
8. Measure overhead relative to existing fixed compression.

If `DecodingPress` fails, test `DMSPress(..., decoding=True)`. If both
fail, Phase 4 falls back to reprefill or prompt-level gating and the
paper should not claim runtime compression control.

Optional engineering hedge: a small `BudgetedDynamicCache` wrapper
around HF `DynamicCache` can expose the same budget interface if
kvpress's experimental decode-time wrappers are too brittle. This is
insurance only, not the first implementation path.

## Policy Class

Phase 4 should include only simple policies.

### Policy A: RiskAIMD

Primary v1 policy, analogous to TCP congestion control.

- Start with an aggressive cache budget `B0`.
- After each completed segment, compute calibrated `segment_risk`.
- If `segment_risk < low_threshold` for `M` consecutive segments,
  increase compression one step: lower target size, bounded by
  `B_min`.
- If `segment_risk >= high_threshold`, decrease compression sharply:
  raise target size by a multiplicative factor or jump one/two budget
  levels safer.
- If `run_risk >= critical_threshold`, enter safe mode for the rest of
  generation.
- Interpret every safer move as future-only. The policy prevents or
  delays later pruning; it does not restore earlier KV entries.

This policy directly tests the user's intended HERALD-as-congestion-
control idea.

### Policy B: RiskBudgetStep

Simpler threshold policy for ablation.

- Define ordered budgets from aggressive to safe, e.g.
  `[256, 512, 1024, 2048, full]`.
- After each segment:
  - risk below threshold: move one step more aggressive, unless already
    at minimum.
  - risk above threshold: move one step safer for future compression
    events.
- Add one-segment cooldown after any change.

This is easier to analyze than AIMD and may be better for the first
smoke.

### Policy C: RiskSafeRest

Monotone safety policy.

- Start at aggressive budget.
- If max run risk crosses threshold, switch to safe budget for the rest
  of generation, meaning skip or soften later compression events.
- Otherwise keep aggressive budget.

This is the closest analogue to the earlier `RiskFuseRest`, but the
action is a decode-time cache-budget change rather than disabling
StreamingLLM.

### Threshold / Budget Sweep

For smoke:

- Three budgets: aggressive, medium, safe.
- Five risk thresholds from calibration run-risk quantiles:
  0.50, 0.65, 0.80, 0.90, 0.95.
- Policies: `RiskBudgetStep` and `RiskSafeRest` first. AIMD after the
  fixed-budget feasibility is green.

For publishable:

- Five budgets and eleven thresholds from calibration quantiles.
- Include fixed decode-time compression budgets and no/large-cache
  anchors.

Thresholds and budget grids are defined on calibration prompts only.
Evaluation prompts must not determine them.

## Calibration

Use the Phase 1 and Phase 2 data, but split by prompt id to avoid
leakage.

For every controller experiment:

- Define three disjoint prompt sets per task: train, calibration,
  evaluation.
- Remove every evaluation prompt id from predictor training and
  isotonic calibration.
- Train `lr_all_cheap` on train prompts only, with `future_sum_js_25`
  labels thresholded at the training-fold p90.
- Fit isotonic calibration on calibration prompts only.
- Tune threshold grids and operating points on calibration prompts
  only.
- Evaluate the full threshold sweep on evaluation prompts.

Calibration target:

- Primary: binary `future_sum_js_25` above training p90.
- Threshold tuning utility: quality recovery versus compute on
  calibration prompts, using `quality_delta` where available and
  `rouge_l_drop` as the continuous fallback.

No leakage rules:

- Evaluation prompts are absent from predictor training, isotonic
  fitting, threshold grid construction, budget-grid selection, and
  operating-point selection.
- Baseline uncompressed outputs for evaluation prompts may be used only
  after generation, for paired quality scoring.
- Replay metrics may be used only for analysis, never for online
  action.

## Cost Model

Report three cost views. The primary Pareto curve should use measured
generation wall-clock, with cache-budget metrics shown beside it.

### Primary Cost

- `wall_clock_per_output_token`, measured for generation only.
- `total_generation_wall_clock_seconds`.
- Predictor overhead included in measured controller runs if predictor
  executes online.
- Replay cost excluded from inference cost.

### Cache Budget Cost

Record per segment:

- active press,
- target cache size or DMS threshold,
- observed retained cache length where available,
- number of compression events,
- segment id,
- action id.

Then report:

- average retained cache tokens,
- max retained cache tokens,
- fraction of segments in each budget level,
- `extra_compute_budget_used = (C_policy - C_aggressive_fixed) /
  (C_safe_fixed - C_aggressive_fixed)`.

Use measured wall-clock for `C` when stable; otherwise report both
wall-clock and budget-proxy curves.

### Memory / KV Cost

Report retained KV length from `past_key_values` if accessible. Report
bytes only if reliably measurable across layers. Do not make retained
KV bytes a v1 success criterion unless the measurement is stable.

## Quality Metrics

Quality is paired against the uncompressed baseline for the same
prompt.

Primary metrics:

- `quality_delta = baseline_quality_score - policy_quality_score`.
- `gross_harm_final = baseline_correct_final AND NOT policy_correct_final`.
- `gross_help_final = NOT baseline_correct_final AND policy_correct_final`.
- Task-specific quality score from `run_damage.parquet` conventions.

Continuous drift metrics:

- `rouge_l_drop`.
- `char_edit_ratio`.
- `length_diff_ratio`.
- `embedding_cosine_drop` when available.
- `sum_js` and `sum_kl` from replay, analysis only.

Diagnostic metrics:

- `has_looping`.
- `has_non_termination`.
- `has_format_break`.
- `has_drift`.
- output length and max-token hit rate.

Reporting:

- Report per task.
- Report pooled with task-balanced weighting.
- Report paired bootstrap confidence intervals over prompt ids.

## Baselines

Required baselines:

- No compression / large-cache decode-time budget.
- Fixed decode-time compression at each tested target size.
- Static Phase 1 presses, including StreamingLLM, as reference
  baselines but not dynamic-controller baselines.
- Random budget changes at matched compute budget.
- Entropy-threshold budget control if cheap to implement.
- Constant-rate AIMD without HERALD risk. This tests whether any
  generic budget adaptation works, independent of learned risk.
- LoopGuard-style suffix-pruning / loop-recovery heuristic where
  implementable. This is mandatory because LoopGuard
  (arXiv:2604.10044) is the closest known threat for loop failures.

Optional but recommended:

- Oracle budget control using ground-truth future JS from the fixed
  compressed trace, analysis upper bound only.
- Prompt-level oracle using ground-truth fixed-run quality loss.

Random matched-compute budget:

- For each HERALD threshold/policy, compute intervention budget on
  calibration prompts.
- Match random control to the same distribution of budget levels or
  same average retained-cache budget.
- In evaluation, run at least three random seeds for publishable
  results.

Entropy-threshold control:

- Use entropy segment mean with the same K, threshold selection
  procedure, action space, and calibration split as HERALD.

LoopGuard comparison:

- Treat as a loop-specific baseline, not a universal compressor
  controller.
- Report whether HERALD crosses a segment-risk threshold before the
  LoopGuard trigger would fire.
- Report non-loop failure modes separately, especially instruction
  skipping and format failures, where LoopGuard should not help.

Constant-rate AIMD:

- Use the same budget grid and segment interval as HERALD.
- Remove predictor inputs entirely.
- Tune only generic schedule parameters on calibration prompts.
- If this matches HERALD, the control result is about adaptive budgets,
  not learned compression-damage risk.

## Exact Smoke Experiment

Purpose:

Validate decode-time compression feasibility, logging, calibration,
matched-compute random baselines, and whether HERALD's risk-guided
budget changes have a promising direction. The smoke is not the paper
claim.

Scope:

- Model: `Qwen/Qwen2.5-7B-Instruct`, fp16, greedy decoding.
- kvpress: require >= 0.5.3 on Orion before launch.
- Press candidate: `DecodingPress(KnormPress)` first.
- Backup press: `DMSPress(KnormPress(), decoding=True)` if
  `DecodingPress` fails feasibility.
- `DecodingPress` settings: `compression_interval=16`, target sizes
  chosen from feasibility profiling.
- Initial budget: one medium target size selected from feasibility
  profiling.
- Segment size: `K=16`.
- Max new tokens: `512`.
- Tasks: GSM8K and HumanEval.
- Prompts: 30 evaluation prompts per task, 20 calibration prompts per
  task.
- Predictor: `lr_all_cheap`, `future_sum_js_25`.
- Calibration: isotonic on the 20 calibration prompts per task.
- Policies: `RiskBudgetStep` and `RiskSafeRest`.
- Thresholds: five calibration quantile thresholds.
- Baselines: fixed aggressive budget, fixed medium budget, fixed safe
  budget, no/large-cache budget, random matched-budget, entropy-budget
  control, constant-rate AIMD, and LoopGuard-style heuristic where
  implementable.

Feasibility run before smoke:

- 1 task, 3 prompts, fixed decode-time budgets only.
- Confirm generation completes.
- Confirm decode-time compression events occur.
- Confirm per-step hook fire counts are nonzero at decode compression
  points.
- Confirm retained cache length changes.
- Confirm token logs and quality scoring work.
- Confirm overhead is not pathological (<3x existing generation path
  for smoke; otherwise stop and redesign).

Smoke pass condition:

- Decode-time press feasibility passes.
- Controller artifacts are complete and join to baseline prompt ids.
- At least one HERALD threshold beats random matched-budget gating on
  pooled quality with a positive paired point estimate.
- No task has worse mean quality than fixed aggressive compression by
  more than 2 percentage points or equivalent normalized score.
- Predictor overhead estimate is below 5 percent of generation
  wall-clock.

If the smoke fails because decode-time compression cannot run reliably,
fall back to reprefill or prompt-level gating and do not claim runtime
control. If it runs but HERALD equals random, report a negative control
result.

## Exact Publishable Experiment

Purpose:

Provide the minimal rigorous controller result for the paper.

Scope:

- Model: `Qwen/Qwen2.5-7B-Instruct`, fp16, greedy decoding.
- Press: the decode-time press that passed smoke.
- Budgets: five target sizes / thresholds from feasibility profiling.
- Segment size: `K=16`.
- Max new tokens: `512`.
- Tasks: GSM8K, HumanEval, IFEval, LongBench-Single/Qasper.
- Prompts: 100 evaluation prompts per task, or all 164 HumanEval
  prompts if using the full HumanEval set.
- Calibration: 50 prompts per task, disjoint from evaluation.
- Training: all remaining Phase 1 prompt ids excluding calibration and
  evaluation.
- Predictor: `lr_all_cheap`, `future_sum_js_25`.
- Optional parallel predictor: calibrated XGBoost for a secondary curve
  only.
- Policies: primary `RiskBudgetStep` or `RiskAIMD`, secondary
  `RiskSafeRest`.
- Thresholds: eleven calibration quantile thresholds.
- Random matched-budget seeds: three.
- Entropy baseline: same threshold count and policy class.
- Constant-rate AIMD baseline: same budget grid, no predictor.
- LoopGuard-style baseline: required for looping/non-termination
  comparisons, plus a lead-time comparison against HERALD segment
  risk.
- Oracle baselines: segment JS oracle and prompt quality-loss oracle.

Expected GPU time:

- Unknown until decode-time press feasibility profiling. Do not launch
  publishable-scale runs until fixed-budget decode-time compression
  overhead is measured.

Power and reporting:

- Bootstrap over prompt ids, stratified by task.
- Report task-balanced pooled curves and per-task curves.
- Use paired comparisons among HERALD, random, entropy, fixed budgets,
  and no/large-cache for the same prompt ids.

## Success Criteria

The Phase 4 controller claim passes only if all of the following hold
on evaluation prompts:

- Pareto improvement: at least one HERALD operating point is
  non-dominated by fixed budgets, no/large-cache, entropy control, and
  random matched-budget control.
- Quality recovery: HERALD recovers at least 50 percent of aggressive
  fixed-budget quality loss, where recovery is
  `(Q_policy - Q_aggressive_fixed) / (Q_safe_fixed - Q_aggressive_fixed)`.
- Compute budget: the successful operating point uses at most 50
  percent of the incremental compute gap between aggressive and safe
  fixed budgets.
- Random baseline: HERALD beats random matched-budget control with
  paired bootstrap 95 percent CI lower bound above zero on
  task-balanced quality.
- Non-learned adaptive baseline: HERALD beats constant-rate AIMD, or
  the result is scoped as "adaptive budget control works" rather than
  "learned HERALD risk is necessary."
- Loop baseline: for loop failures, HERALD must either beat the
  LoopGuard-style heuristic or provide earlier warning. For non-loop
  failures, report separate gains where LoopGuard is not applicable.
- No task collapse: no task has a negative HERALD versus aggressive
  fixed-budget quality delta whose paired bootstrap CI excludes zero.
- Catastrophe safety: pooled looping and non-termination rates are not
  higher than aggressive fixed budget by more than 2 percentage points.
- Predictor overhead: online predictor and calibration overhead is
  below 5 percent of generation wall-clock in smoke and below 3 percent
  in the publishable run, or is reported as an explicit deployment
  limitation.

If quality recovery is between 25 and 50 percent and HERALD still
beats random with CI above zero, report the result as a limited
controller improvement, not as a strong Pareto-frontier shift.

## Failure Modes and Interpretation

Decode-time press fails on Qwen2.5:

- Interpretation: current library support is insufficient for runtime
  HERALD control on this model.
- Next step: reprefill/prompt-level fallback or update kvpress.

Predictor beats random:

- Interpretation: cheap online compression-damage risk contains
  actionable control information for runtime cache-budget allocation.

Predictor equals random:

- Interpretation: risk predicts damaged runs, but does not identify
  useful budget changes better than spending compute randomly.

Predictor improves quality but costs too much:

- Interpretation: risk identifies harmful compression, but the action
  grid is too blunt or decode-time compression overhead is too high.

Entropy matches HERALD:

- Interpretation: the controller result does not require the
  multivariate predictor.

Oracle beats HERALD by a large margin:

- Interpretation: intervention is useful, but the current risk score is
  not extracting enough timing information.

## Artifacts

Write these outputs:

- `gold/phase-4-controller-design.md`, this document.
- `results/phase4/feasibility/decoding_press_report.json`.
- `results/phase4/smoke/controller_runs.parquet`.
- `results/phase4/smoke/controller_segments.parquet`.
- `results/phase4/smoke/controller_actions.parquet`.
- `results/phase4/smoke/controller_summary.json`.
- `results/phase4/publishable/controller_runs.parquet`.
- `results/phase4/publishable/controller_segments.parquet`.
- `results/phase4/publishable/controller_actions.parquet`.
- `results/phase4/publishable/pareto_points.parquet`.
- `results/phase4/publishable/bootstrap_cis.parquet`.
- `results/phase4/publishable/phase4_summary.json`.
- `figures/phase4_pareto_quality_compute.png`.
- `figures/phase4_task_curves.png`.
- `figures/phase4_random_matched_compute_delta.png`.
- `gold/phase-4-results.md`, written only after the smoke or
  publishable experiment completes.
