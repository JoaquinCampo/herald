# Phase 1 Offline Analysis Scaffold

Status: scaffold complete on Phase 0 smoke data. These analyses are
CPU-only consumers of finalized parquet artifacts. They do not launch
GPU work and do not modify generation or sweep behavior.

The purpose of this document is to prevent two mistakes:

1. Treating Phase 0 smoke outputs as paper claims.
2. Forgetting to rerun the analysis stack immediately when Phase 1
   data lands.

## Bottom Line

Four high-value analysis products are now built and smoke-tested:

- Failure-onset event study.
- Per-feature lead-time curves.
- Feature-information / information-ceiling analysis.
- Cross-press transfer matrix.

None of the Phase 0 outputs supports a final paper claim. They are
infrastructure checks and hypothesis generators. The real validation
requires Phase 1's four tasks, full press matrix, low-ratio grid, and
matched controls.

## Artifact 1: Failure-Onset Event Study

Reference: `gold/event-study-analysis.md`.

Purpose: align catastrophic compressed generations at estimated onset
and plot cheap online feature trajectories in a +/- token window. This
is the candidate canonical figure for "what compression collapse looks
like at the logit level."

Phase 0 smoke read:

- 134 total runs.
- 77 catastrophic runs selected by tags (`looping` or
  `non_termination`).
- 49 looping onsets and 28 non-termination-proxy onsets.
- 140,632 event-aligned rows in the +/-200 token window.
- No defensible healthy control group in Phase 0; the compressed
  non-catastrophic stratum is too thin.
- Non-termination proxy must stay separated from looping because it is
  structural, not mechanistic.

Phase 1 must validate:

- Whether feature trajectories differ from matched healthy compressed
  controls.
- Whether the onset signature is stable across task, press, and ratio.
- Whether looping, non-termination, and wrong-answer/degradation
  failures need separate panels rather than a pooled event study.

Phase 1 rerun command:

```sh
.venv/bin/python scripts/build_event_study.py \
    --input results/phase1 \
    --output results/analysis/phase1/event_study \
    --window-before 200 \
    --window-after 200
```

## Artifact 2: Per-Feature Lead-Time Curves

Reference: `gold/lead-time-analysis.md`.

Purpose: quantify how many tokens before onset each cheap online
feature begins separating catastrophic compressed runs from healthy
compressed controls.

Phase 0 smoke read:

- 134 total compressed runs.
- 77 catastrophic runs with derivable onset.
- 37 healthy compressed controls.
- Only 2 of 5 `(task, press, ratio)` strata satisfy the default
  well-powered condition.
- `control_limited=True`.
- At default thresholds (`AUROC >= 0.65`, CI lower `> 0.5`,
  persistence `= 5`), only entropy shows a qualifying smoke lead time:
  7 tokens before onset.
- All other tracked Tier 0 features have no qualifying Phase 0 lead
  time.

Phase 0 claim boundary:

- "Entropy leads by 7 tokens" is not a paper result. It is a smoke
  output from a control-limited slice.

Phase 1 must validate:

- Whether any feature has stable lead time once controls are matched
  across the full task/press/ratio grid.
- Whether lead time differs by failure family and compressor.
- Whether the selected prediction horizons (`H in {5,10,25,50}`) are
  empirically justified or should be adjusted.

Phase 1 rerun command:

```sh
.venv/bin/python scripts/build_lead_time_curves.py \
    --input results/phase1 \
    --output results/analysis/phase1/lead_time \
    --window-before 200 \
    --window-after 50 \
    --n-bootstrap 1000
```

## Artifact 3: Feature-Information / Information-Ceiling Analysis

Reference: `gold/information-ceiling-analysis.md`.

Purpose: measure whether cheap online features contain information
about future compression damage beyond trivial proxies such as
position, press, task, and ratio. This is the direct rebuttal to the
LimitsLearned-style concern that the signal might not be in the online
features.

Phase 0 smoke read:

- 114 compressed runs.
- 43,864 token rows loaded.
- 19,625 token rows after dropping tokens at or after onset.
- Positive future-damage labels:
  - `H=5`: 221
  - `H=10`: 399
  - `H=25`: 905
  - `H=50`: 1709
- `phase0_smoke=True` because only one task is present.

Directional Phase 0 incremental AUROC gain of all features over
position+metadata:

- `H=5`: +0.056
- `H=10`: +0.048
- `H=25`: +0.029
- `H=50`: +0.018

Interpretation:

- Online features add measurable short-horizon signal in Phase 0, but
  metadata/position already explain much of the label.
- The gain decays with horizon, consistent with "deep in a
  heavy-compression run" becoming a strong proxy.
- This is a warning, not a win: Phase 1 must show the incremental
  online signal survives broader tasks, presses, and stricter splits.

Phase 1 must validate:

- Whether online features add paired, statistically defensible
  information beyond position+metadata.
- Whether incremental information survives held-out press and
  held-out ratio.
- Whether `online-only` remains weak or improves with a less saturated
  low-ratio regime.

Phase 1 rerun command:

```sh
.venv/bin/python scripts/build_information_ceiling.py \
    --input results/phase1 \
    --output results/analysis/phase1/information_ceiling \
    --horizons 5,10,25,50 \
    --n-bootstrap 1000
```

## Artifact 4: Cross-Press Transfer Matrix

Reference: `gold/cross-press-transfer-analysis.md`.

Purpose: train on one press and test on every other press. This is the
load-bearing test for HERALD's black-box compressor-agnostic claim.

Phase 0 smoke read:

- 114 compressed runs after filtering.
- 19,625 token rows after the at/past-onset filter.
- Two presses only:
  - `snapkv`: 19 runs
  - `streaming_llm`: 54 runs
- 64 matrix cells total.
- 0 insufficient cells under the smoke thresholds.

Directional Phase 0 pattern:

- Online features transfer poorly at short horizons:
  - `H=5` diagonal AUROC 0.61, off-diagonal AUROC 0.36,
    transfer gap +0.25.
- Position+metadata transfers cleanly from `H=10` onward:
  transfer gap below 0.02.
- All-features diagonal is high (roughly 0.83-0.84), but
  off-diagonal gap is large at short horizons and shrinks as position
  dominates at longer horizons.

Interpretation:

- Phase 0 is directionally consistent with "online short-horizon
  signals are press-calibrated."
- The matrix is too small and imbalanced to support a transfer claim.
- Phase 1's full press matrix is the actual test.

Phase 1 must validate:

- Whether cross-press transfer holds when more compressors are present.
- Whether failure signatures are method-agnostic or compressor-specific.
- Whether HERALD should be framed as a universal black-box predictor or
  a per-compressor calibration framework.

Phase 1 rerun command:

```sh
.venv/bin/python scripts/build_cross_press_transfer.py \
    --input results/phase1 \
    --output results/analysis/phase1/cross_press_transfer \
    --horizons 5,10,25,50 \
    --n-bootstrap 1000
```

## What Phase 1 Must Decide

When Phase 1 data lands, the offline analysis stack should answer five
questions before any strong paper claim is written:

1. Do event-study signatures persist when matched healthy controls are
   available?
2. Do any cheap online features provide stable pre-onset lead time?
3. Do online features add signal beyond position, task, press, and
   ratio proxies across tasks?
4. Does cross-press transfer work, or is HERALD press-specific?
5. Does the low-ratio grid reveal a non-saturated regime where
   prediction is meaningful rather than merely identifying already
   broken runs?

If the answers are positive, HERALD can support the strong predictor
and controller trajectory. If not, the work should pivot toward the
measurement-methodology or negative-methodology framing described in
`gold/contribution-validation.md`.

## Operational Rule

These analyses should run while GPU sweeps continue, but they should
not block generation unless they reveal a schema break. Their job is
to consume finalized artifacts, validate paper claims, and feed back
into Phase 2 target design.
