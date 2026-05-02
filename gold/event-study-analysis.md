# Event Study Analysis

Aligns catastrophic compressed runs at estimated failure onset and
plots cheap online feature trajectories in a window around onset.
The first version is CPU-only Phase 0 smoke infrastructure; the
canonical figure waits for Phase 1 scale.

## Why It Matters For HERALD

The contribution-validation bar requires that HERALD show "online
features contain information beyond trivial position/ratio proxies"
and that the predictor's signal is mechanistically interpretable
(`gold/contribution-validation.md`, `gold/ultimate-goal.md`).

The event-study figure is the qualitative companion to the
quantitative predictor result. It shows, at the logit level, what
compression collapse looks like before and after the moment a run
goes off-rail. If cheap features carry no event structure around
onset, the predictor's AUROC is suspect (it is probably learning
position or ratio). If they do carry structure, the figure is the
mechanistic story the paper needs to defend the "predictability is
real" claim.

It also is the visual the closed-loop chapter needs: the lead-time
panels in Phase 4 are derived from the same alignment, just with
predicted-onset replacing detected-onset. Same axis, same window,
different curve. Building the alignment now means Phase 2/3/4
plots reuse the same code path.

## What This Analysis Does Claim

- For runs with a derivable failure onset, cheap online features
  exhibit a structured trajectory around that onset.
- The trajectory is qualitatively different across onset sources;
  pooling looping with non-termination-proxy onsets dilutes the
  signal, so the headline figure stratifies by onset source.

## What This Analysis Does NOT Claim

- That every compressed run has a discrete onset. Wrong-answer runs
  are excluded from the study because the error is only detectable
  after the full answer is produced.
- That the non-termination proxy onset (`0.75 * max_new_tokens`)
  marks a real transition. It is a labeling-time fallback; in this
  view it is included as a separate cohort, not pooled with looping.
- That the figure proves predictability. AUROC, lead time, and
  feature-importance work belong to Phase 2 evaluation, not here.
- That Phase 0 has a control group. The compressed-non-catastrophic
  stratum on Phase 0 (134 runs, 1 task, 2 presses, ratios above the
  cliff) is too small to be a defensible control. The summary JSON
  documents this absence explicitly.

## Phase 0 Smoke vs Phase 1 Final

Phase 0 (current):

- 134 runs, 1 task (GSM8K), 2 presses, 3 ratios.
- Onset sources observed: 49 looping, 28 non-termination-proxy
  (out of 77 catastrophic runs after onset filter).
- No matched control group. The figure shows looping vs
  non-termination-proxy traces only.
- Feature set: entropy, top1_prob, top1_top2_margin, kl_div, h_alts,
  delta_h, top10_jaccard, tail_mass.
- Window: ±200 tokens.
- Bootstrap CIs at 200 resamples per cell.

Phase 1 final analysis (deferred):

- 4 tasks, full press matrix, 8-ratio grid including the gentle-
  damage regime (0.25, 0.375).
- Compressed-non-catastrophic control group, matched per
  (task, press, ratio), aligned at the same absolute token
  positions as their catastrophic counterparts. Treated as a
  separate cohort, never silently mixed.
- Stratification by task and ratio added on top of the press +
  onset-source stratifications.
- Onset definition refined where the diagnostic-tag work lands a
  better source (e.g., detector-based looping onset hardened
  against false positives, or a paired-counterfactual
  `baseline_correct AND compressed_wrong` definition once paired
  outcomes are reliable).
- Sensitivity analysis on window size, bootstrap count, and
  rouge-threshold gating.

## Connection To Lead-Time Curves

The Phase 2/3/4 lead-time analysis asks: at relative position `t`
before onset, what is the predicted hazard? The event-study
alignment is the same axis. The lead-time figure replaces the cheap-
feature mean trajectory with the predicted hazard probability at
each relative position; the bootstrap CI band is computed the same
way. Sharing this alignment code means the Phase 4 plot is a
near-trivial extension of the Phase 0/1 figure, with `predicted_p`
substituted for `feature_value`.

The intervention-probe analysis in `gold/phase-1-intervention-
probe.md` reuses the same alignment for "switch-at-offset-T vs
continue-fixed" trace comparisons. Building the alignment as a
reusable analysis primitive here pays off three more times in the
paper.

## Files

- `src/herald/analysis/event_study.py` — onset derivation, alignment,
  bootstrap aggregation, plotting.
- `scripts/build_event_study.py` — CPU CLI (`--input`, `--output`,
  `--window-before`, `--window-after`, `--features`,
  `--rouge-threshold`).
- `tests/analysis/test_event_study.py` — alignment, exclusion,
  aggregation shape, bootstrap on tiny samples, CLI smoke.

## Reproducing The Phase 0 Smoke Output

```sh
.venv/bin/python scripts/build_event_study.py \
    --input results/phase0 \
    --output results/analysis/event_study \
    --window-before 200 --window-after 200
```

Outputs land under `results/analysis/event_study/`:

- `event_study.parquet` (long event-aligned rows)
- `event_study_agg_{pooled,by_press,by_onset_source}.parquet`
- `event_study_features.png` (headline; stratified by onset source)
- `event_study_features_by_press.png`
- `event_study_summary.json` (counts, exclusions, schema notes,
  Phase 0 caveat)

## Schema Assumptions

- `runs.parquet` follows `metrics/io.py:RUNS_SCHEMA`. Notably,
  `catastrophe_onsets` is *not* a column, so per-run looping onset
  is re-derived from `generated_token_ids` at analysis time. This
  is deterministic but ties the analysis to the `detect_looping`
  parameters (`window_size=20, min_repeats=3`).
- Tokens parquet is partitioned `press=*/ratio=*/*.parquet` under
  `<input>/final/tokens/`. The script reads every partition once
  and filters to selected `run_id`s; this is the existing finalized
  layout and is reused for Phase 1.
