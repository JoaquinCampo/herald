# HERALD Kanban

Lightweight coordination board for multi-agent work. Keep this file
manual and current. Do not let it replace the phase plan in
`gold/research-plan.md`; it is only an execution tracker.

## Card Format

Use stable IDs so agents can reference work unambiguously.

- `P1-E...`: Phase 1 engineering / infrastructure
- `P1-S...`: Phase 1 sweep / profiling
- `P1-A...`: Phase 1 analysis
- `P1-P...`: Phase 1 intervention probe
- `P2-M...`: Phase 2 modeling
- `P3-G...`: Phase 3 generalization
- `P4-C...`: Phase 4 control

Each card should include owner, phase, GPU use, dependencies, and
acceptance criteria. GPU-launching cards require explicit user
acknowledgment before execution.

## Ready

### P1-A01: Failure-Onset Event Study

Owner: unassigned
Phase: Phase 1 analysis
GPU: No
Depends on: Phase 0 artifacts; later rerun on Phase 1 artifacts
Blocks: P1-A02
Risk: Low

Acceptance:

- CPU-only script/module exists for onset-aligned event studies.
- Phase 0 smoke output is produced if artifacts are available.
- Summary JSON reports run counts, catastrophic counts, exclusions,
  onset-source counts, feature columns, and caveats.
- No generation or sweep behavior changes.

Notes:

- This is the first analysis task to parallelize while the Phase 1
  sweep runs.
- Phase 0 output is smoke-only; Phase 1 output is the paper figure.

### P1-A02: Feature Lead-Time Curves

Owner: unassigned
Phase: Phase 1 analysis
GPU: No
Depends on: P1-A01
Blocks: P2 target/horizon justification
Risk: Low

Acceptance:

- Per-feature lead-time curves are computed relative to failure onset.
- Curves report the earliest reliable separation between healthy and
  catastrophic traces.
- Horizons `{5, 10, 25, 50}` are validated or revised based on the
  observed lead-time distribution.

### P1-A03: Feature Information-Ceiling Report

Owner: unassigned
Phase: Phase 1 analysis / Phase 2 prep
GPU: No
Depends on: finalized token/segment features and labels
Blocks: strong predictor novelty claim
Risk: Medium

Acceptance:

- Mutual-information or equivalent feature-information analysis exists
  for Tier 0 / Tier 0.5 / Tier 1 features.
- Position-only and ratio-only proxy baselines are included.
- Report states whether cheap online features contain signal beyond
  trivial proxies.

### P1-A04: Cross-Press Transfer Matrix Scaffold

Owner: unassigned
Phase: Phase 1 analysis / Phase 2 prep
GPU: No
Depends on: Phase 1 fixed-ratio sweep artifacts
Blocks: cross-compressor novelty claim
Risk: Medium

Acceptance:

- Train-on-press / test-on-press evaluation scaffold exists.
- Random, position-only, ratio-only, entropy/EWMA, single-feature, and
  logistic baselines can be evaluated in the same matrix.
- Output table clearly distinguishes within-press from held-out-press
  performance.

### P1-A05: Failure-Mode Definition Hardening

Owner: unassigned
Phase: Phase 1 analysis / paper prep
GPU: No
Depends on: related-work appendix and current detector definitions
Blocks: diagnostic-tag credibility
Risk: Low

Acceptance:

- Looping, non-termination, instruction amnesia, format break, and
  drift definitions are mapped to prior literature where possible.
- Heuristic thresholds are labeled honestly.
- Sensitivity checks are specified for thresholds that remain
  heuristic.

### P1-P01: Probe Pre-Registration Audit

Owner: unassigned
Phase: Phase 1 intervention probe
GPU: No
Depends on: `gold/phase-1-intervention-probe.md`
Blocks: intervention probe launch
Risk: Medium

Acceptance:

- Stratum rules, sampling algorithm, offsets, action vocabulary,
  endpoints, and decision rules are internally consistent.
- Press-specific action table is explicit.
- Probe remains separate from the fixed-ratio headline.

## In Progress

### P1-S01: Orion Sanity Profile Slice

Owner: phase1 agent
Phase: Phase 1 profiling
GPU: Yes
Depends on: Block 1 engineering prep, equivalence smoke green
Blocks: P1-S02, P1-S03
Risk: Medium

Acceptance:

- GSM8K sanity slice completes or fails with actionable diagnostics.
- Report includes completion/failure count, wall-clock/token by cell,
  replay overhead, peak memory, storage growth, and budget JSON schema
  status.
- No expansion launches automatically after completion.

Current status:

- Running on Orion as reported by the phase agent.

## Blocked

### P1-S02: Minimum Per-Task Budget Gate

Owner: phase1 agent
Phase: Phase 1 profiling
GPU: Yes
Depends on: P1-S01 clean result
Blocks: P1-S03
Risk: Medium

Acceptance:

- Runs 4 tasks x 1 press x 3 ratios, after explicit user ack.
- Cost telemetry separates prefill, decode, and replay time where
  possible, especially for LongBench.
- Per-task wall-clock matches the written estimate within 10%, or
  anomalies are documented.

Blocked reason:

- Waiting for P1-S01 results and user ack.

### P1-S03: Full Phase 1 Fixed-Ratio Sweep

Owner: unassigned
Phase: Phase 1 measurement core
GPU: Yes
Depends on: P1-S02 clean result, watchdog prerequisite, dataset cache
Blocks: P1 alignment study, P1-P02, P2 modeling
Risk: High

Acceptance:

- 1 model x 4 tasks x full press matrix x 8 ratios x 200 prompts.
- Baselines run first and compressed cells link to baseline run IDs.
- Skip-existing honors `replay_status='ok'`.
- Live cost monitoring stops if any cell exceeds the profile by >25%.
- Metrics finalize and build complete after sweep.

Blocked reason:

- Waiting for budget gates and explicit user launch approval.

### P1-P02: Intervention Probe Execution

Owner: unassigned
Phase: Phase 1 intervention probe
GPU: Yes
Depends on: P1-S03 complete, P1-P01 green
Blocks: Phase 1 results write-up, Phase 2 target design
Risk: High

Acceptance:

- Probe follows pre-registered strata, offsets, action vocabulary, and
  budget.
- If boundary-unstable candidates are insufficient for a task/press,
  the pair is flagged anticipatory-only; strata are not padded.
- Probe outputs remain separate from fixed-ratio headline plots.

Blocked reason:

- Requires completed fixed-ratio sweep and pre-registration audit.

### P2-M01: Predictor Baseline Suite

Owner: unassigned
Phase: Phase 2 modeling
GPU: No initially
Depends on: Phase 1 fixed-ratio artifacts, P1-A03
Blocks: predictor novelty claim
Risk: Medium

Acceptance:

- Random, position-only, ratio-only, entropy threshold, EWMA, change
  point, single-feature thresholds, logistic regression, and primary
  model are evaluated consistently.
- Best model beats strongest cheap baseline by the pre-registered
  margin or the claim is downgraded.

Blocked reason:

- Needs Phase 1 labels/features.

## Review

### P1-E01: Phase 1 Engineering Prep

Owner: phase1 agent
Phase: Phase 1 engineering
GPU: No
Depends on: Phase 0 complete
Blocks: P1-S01
Risk: Medium

Acceptance:

- Policy abstraction preserves Phase 0 fixed-ratio behavior.
- Segment metrics exist for K in {8, 16, 32}.
- Run-level cost fields are captured.
- Multi-task dataset support exists.
- Equivalence smoke is green.

Review notes:

- Reported green by phase agent after replay-timing additions.
- Keep in Review until the sanity profile confirms no cost-telemetry
  regression.

## Done

### P0-D01: Phase 0 Dry Run

Owner: completed
Phase: Phase 0
GPU: Yes
Depends on: none
Blocks: Phase 1
Risk: Closed

Acceptance:

- Matched-prefix replay substrate built.
- Phase 0 sweep completed and documented.
- Sampling-rate decision made.
- Phase 1 low-ratio grid updated.
- Carry-forward risks documented in `gold/phase-0-results.md`.

### P1-D01: Contribution Validation Bar

Owner: codex
Phase: Cross-cutting
GPU: No
Depends on: related-work appendix
Blocks: paper claim discipline
Risk: Closed

Acceptance:

- `gold/contribution-validation.md` exists.
- Research plan references required baselines, transfer experiments,
  feature-information checks, and closed-loop evidence.
- Ultimate goal states what makes HERALD a true contribution versus a
  measurement-only paper.
