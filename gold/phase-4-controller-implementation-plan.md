# Phase 4 Controller Implementation Plan

Date: 2026-05-06.
Status: revised after kvpress feasibility research. No controller code
has been written yet.

## Current Decision

Do not implement a custom compressor first.

Do not target mid-generation `StreamingLLMPress` toggling. In the
installed kvpress, `StreamingLLMPress` physically prunes KV at
end-of-prefill and is dormant during decode. The dropped prompt KV
cannot be restored by disabling a hook later.

Proceed instead with a decode-time controllable kvpress substrate:

- Primary candidate: `DecodingPress(base_press=KnormPress(), ...)`.
- Backup candidate: `DMSPress(KnormPress(), decoding=True, ...)`.
- Fallback if both fail: reprefill oracle or prompt-level gating, both
  clearly scoped as weaker than runtime cache-budget control.

Important constraint: `DecodingPress` still physically prunes KV.
Budget changes are future-only. A safer budget can delay or skip later
compression events, but it cannot restore entries that were already
evicted. KV restoration requires reprefill or an explicit offload-based
oracle path.

The next step is not a controller. It is a feasibility gate proving
that an existing kvpress decode-time press can be run, logged, and
budgeted on Qwen2.5-7B-Instruct.

## Why The Plan Changed

The original Phase 4 design assumed StreamingLLM was mask-based and
could be relaxed during generation. Source inspection showed the
opposite:

- The standard kvpress hook compresses during prefill, then skips
  decode.
- `StreamingLLMPress` chooses prompt positions once.
- The selected KV tensors are gathered into smaller tensors, so pruned
  entries are physically gone.

Internet and installed-package research found a better route:
experimental kvpress decode-time wrappers already exist. That means
the right near-term path is to test those wrappers before building
anything custom.

## Phase 4 v1 Engineering Sequence

### Step 1: Decode-Time Press Feasibility

Build `scripts/run_phase4_decoding_press_feasibility.py`.

Scope:

- Model: `Qwen/Qwen2.5-7B-Instruct`.
- kvpress: require >= 0.5.3 on Orion before any feasibility launch.
  Installed 0.5.1 has a per-sample state reset bug.
- Tasks: GSM8K first, then HumanEval if GSM8K passes.
- Prompts: 3 prompts for the first gate, then 10 prompts for the
  extended gate.
- Press candidate 1: `DecodingPress(KnormPress)` with
  `compression_interval=16`.
- Press candidate 2: `DMSPress(KnormPress(), decoding=True)`.
- Fixed budgets only. No HERALD risk policy yet.
- Greedy decoding.
- Max new tokens: 256 for the first gate, 512 after it passes.

The feasibility script must record:

- press name and parameters,
- generated tokens,
- stop reason,
- wall-clock per output token,
- peak memory,
- decode-time compression event count,
- per-step hook fire counts,
- observed retained cache length where accessible,
- retained cache length before and after each compression event,
- target cache size or DMS threshold by segment,
- per-token cheap features,
- per-segment aggregates with `K=16`,
- task quality score using the existing `run_damage` conventions.

Acceptance criteria:

- Generation completes without crashes.
- Compression happens during decode, not only prefill.
- Hook fire counts prove decode-time compression is actually happening.
- Observed retained cache length or equivalent event log changes when
  the budget changes.
- Overhead is below 3x the existing fixed-generation path for the
  first smoke.
- Outputs can be scored with existing task scorers.

If `DecodingPress(KnormPress)` fails, run the same gate for
`DMSPress(KnormPress(), decoding=True)`. If both fail, stop and report.

Do not implement dynamic HERALD control until this fixed-budget
feasibility gate is green.

### Step 2: Predictor Export

Export the Phase 2 v1 predictor:

- model: `lr_all_cheap`,
- label: `future_sum_js_25`,
- features: exact Phase 2 cheap feature order,
- scaler mean and scale,
- coefficients and intercept,
- isotonic calibration fitted on calibration prompts only.

Expected artifacts:

- `models/phase4_lr_all_cheap.json`,
- `models/phase4_isotonic.json`,
- `gold/phase-4-splits.json`.

Tests:

- split disjointness by prompt id,
- exported model scores match the Phase 2 offline scorer,
- isotonic interpolation matches sklearn output on fixture rows.

### Step 3: Online Feature Parity

Implement an online state object that reproduces the Phase 2 cheap
features token by token:

- rolling means and standard deviations for windows 8 and 32,
- EWMAs for half-lives 8 and 32,
- token position and relative progress,
- press and budget metadata,
- entropy, top probabilities, alternative entropy, token instability,
  top-k concentration.

Test:

- Feed recorded Phase 2 token rows through the online object.
- Compare against the offline Phase 2 dataset.
- Maximum absolute difference must be below `1e-6` for deterministic
  fields and below `1e-4` where floating reductions differ.

### Step 4: Fixed-Budget Baselines

Before HERALD controls anything, run fixed decode-time budgets:

- aggressive budget,
- medium budget,
- safe budget,
- no or large-cache budget.

These define the quality and compute anchors. The controller cannot
claim Pareto improvement without them.

### Step 5: Non-Learned Controller Baselines

Implement two mandatory baselines before interpreting HERALD:

- Constant-rate AIMD without predictor input. Same budget grid and
  `K=16` segment cadence as HERALD, but schedule driven only by fixed
  rules and calibration-tuned parameters.
- LoopGuard-style loop heuristic where implementable. This is required
  because LoopGuard (arXiv:2604.10044) is the closest known threat for
  loop and non-termination failures.

The report must include:

- HERALD versus constant AIMD on quality and compute.
- HERALD segment-risk lead time versus the LoopGuard trigger.
- Non-loop failure modes separately, especially instruction skipping
  and format failures.

### Step 6: Minimal HERALD Controller Smoke

Only after Steps 1 to 5 pass, implement the first controller:

- policy: `RiskBudgetStep`,
- segment size: `K=16`,
- score: mean calibrated token risk over the previous segment,
- action: move one budget level safer or more aggressive for future
  segments,
- thresholds: five calibration quantiles,
- baselines: fixed budgets, entropy policy, random matched-budget,
  constant-rate AIMD, and LoopGuard-style heuristic.

Do not implement AIMD first. AIMD is useful, but `RiskBudgetStep` is
easier to debug and interpret.

### Step 7: Publishable Controller Run

Run only if the smoke shows a positive direction.

Scope:

- one decode-time press that passed feasibility,
- all four Phase 1 tasks,
- 100 evaluation prompts per task,
- 50 calibration prompts per task,
- fixed budgets, HERALD, entropy, random matched-budget, no or
  large-cache anchor,
- three random seeds for random matched-budget.

## Optional Engineering Hedge

If `DecodingPress` is usable but too brittle to trust as the only
runtime substrate, consider a small `BudgetedDynamicCache` wrapper
around HF `DynamicCache` as an insurance path. The target interface is
the same as `DecodingPress`: `set_budget(layer_idx, k)` or a global
budget setter. Keep this as a hedge, not the first implementation. The
first path remains existing kvpress decode-time presses.

## Tests To Add

CPU-only:

- `tests/test_phase4_splits.py`,
- `tests/test_phase4_predictor_export.py`,
- `tests/test_phase4_online_features.py`,
- `tests/test_phase4_policies.py`,
- `tests/test_phase4_feasibility_dry_run.py`.

GPU or Orion gated:

- decode-time press import and generation smoke,
- fixed-budget smoke on 3 prompts,
- event logging confirms decode-time compression,
- overhead report.

## Artifact Schemas

Feasibility:

- `results/phase4/feasibility/decoding_press_report.json`,
- `results/phase4/feasibility/runs.parquet`,
- `results/phase4/feasibility/segments.parquet`,
- `results/phase4/feasibility/events.parquet`.

Controller smoke:

- `results/phase4/smoke/controller_runs.parquet`,
- `results/phase4/smoke/controller_segments.parquet`,
- `results/phase4/smoke/controller_actions.parquet`,
- `results/phase4/smoke/controller_summary.json`.

Publishable run:

- `results/phase4/publishable/controller_runs.parquet`,
- `results/phase4/publishable/controller_segments.parquet`,
- `results/phase4/publishable/controller_actions.parquet`,
- `results/phase4/publishable/pareto_points.parquet`,
- `results/phase4/publishable/bootstrap_cis.parquet`,
- `results/phase4/publishable/phase4_summary.json`.

Forbidden online fields:

- `gross_harm_final`,
- `quality_delta`,
- `sum_js`,
- `sum_kl`,
- `js_full`,
- `kl_unc_comp_full`,
- `first_divergence_point`,
- any `future_*` label,
- paired uncompressed output.

## Current Blockers

Before running the actual controller smoke:

1. Feasibility gate for `DecodingPress(KnormPress)` or
   `DMSPress(KnormPress(), decoding=True)`.
2. kvpress >= 0.5.3 installed on Orion.
3. Predictor and isotonic export.
4. Phase 4 train/calibration/evaluation prompt splits.
5. Online feature parity with Phase 2.
6. Fixed-budget decode-time baselines.
7. Constant-rate AIMD and LoopGuard-style baselines specified.
8. Orion approval for GPU work.

## Status Report

- StreamingLLM mid-generation disable implementable: no.
- Existing decode-time kvpress path available: likely yes, pending
  feasibility smoke.
- Custom compressor required now: no.
- Recommended next agent task: implement and run the decode-time press
  feasibility gate only.
