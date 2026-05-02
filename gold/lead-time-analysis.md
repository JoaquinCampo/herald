# Lead-Time Analysis

Per-feature quantification of how many tokens before catastrophic
onset each cheap online feature begins to separate catastrophic
compressed runs from healthy compressed controls. CPU-only,
reuses the event-study alignment primitives.

## Why It Matters For HERALD

The contribution-validation bar (`gold/contribution-validation.md`)
requires that HERALD show "online features contain information
beyond trivial position/ratio proxies" and that the predictor's
signal is "future, transferable, control-useful". The lead-time
curve is the quantitative companion to the qualitative event-study
trace: where the event study shows feature *means* drift around
onset, the lead-time analysis asks at which negative offset the
feature *separates* catastrophic from healthy populations under a
disciplined threshold.

A cheap feature that crosses AUROC >= 0.65 (with bootstrap CI
lower bound > 0.5) for at least 5 consecutive evaluated positions
before onset is contributing actionable signal at that horizon.
Features that never cross are either uninformative for this
failure family or only diagnostic post-onset, in which case they
cannot drive a controller.

## What This Analysis Does Claim

- For features that pass the persistence test, there is a finite
  number of tokens before the failure onset at which the feature
  alone (no learned predictor) already separates catastrophic from
  healthy compressed runs.
- That number is a lower bound on the effective lead time of any
  HERALD predictor that uses only that feature; a learned model
  that combines features should match or beat it.
- The lead-time numbers are computed with the same event-study
  alignment used for the trajectory plots, so the two figures sit
  on a shared axis and can be read together.

## What This Analysis Does NOT Claim

- It is not a predictor. AUROC at relative position `r` is
  evaluated using the *raw* feature value at that token, not a
  learned score. The HERALD predictor will use multi-feature
  context, so its lead time can exceed the per-feature numbers
  here.
- It is not a Phase 1 result. The Phase 0 stratum coverage is
  structurally thin (single task, two presses, three ratios). The
  summary JSON flags `control_limited=True` whenever fewer than
  `min_well_covered_strata` (default 3) (task, press, ratio)
  cells satisfy the per-cell minimum.
- It does not pretend the non-termination proxy is a real onset.
  Catastrophic onset is derived using the same priority as the
  event study: `detect_looping_onset` first, then the
  `nt_onset_frac * max_new_tokens` proxy. Mixing the two cohorts
  is acceptable here because the AUROC question is about
  feature-level separability, not trajectory shape.

## Method

1. **Catastrophic selection** — same as
   `event_study.select_catastrophic_runs` (compressed, with at
   least one of `looping` or `non_termination`, optional
   `rouge_threshold`).
2. **Healthy controls** — `lead_time.select_healthy_controls`:
   compressed runs with neither `looping` nor `non_termination`
   tags. `wrong_answer` is tolerated; it is an answer-quality
   signal, not a trajectory-collapse signal, and excluding it
   would shrink the control pool below usability.
3. **Stratum matching** — by `(task, press, compression_ratio)`.
   Each control receives a virtual onset equal to the median
   catastrophic onset in its stratum (clipped to the control's
   own length). Strata with no catastrophic run, or no control,
   contribute zero pairs.
4. **Alignment** — `relative_pos = token_pos - onset_token`,
   filtered to `[-window_before, +window_after]`. Optional
   `position_stride` subsamples the axis.
5. **Per-feature orientation** — `infer_feature_direction` picks a
   per-feature sign so the AUROC metric is *directional*: scores
   are pre-oriented (multiplied by +1 or -1) so that higher means
   "more catastrophic". The direction is chosen empirically from
   the pooled cat-vs-ctrl mean difference. Surfaced in the summary
   under `feature_directions` so reviewers can see which features
   were inverted (e.g., `top1_prob = -1` because catastrophic runs
   tend to have lower top-1 probability).
6. **AUROC + CI** — `compute_auroc_with_ci` returns the per
   `(feature, relative_pos)` directional AUROC and the percentile
   bootstrap CI from resampling run_ids (preserves within-run
   dependence). Because the metric is directional, the threshold
   "AUROC >= 0.65 AND CI lower > 0.5" keeps its frequentist
   meaning: under no signal, the bootstrap distribution is
   centered on 0.5, so the `> 0.5` lower-bound test is not biased
   upward.
7. **Lead time** — earliest negative `relative_pos` where AUROC
   >= `auroc_threshold` AND `ci_lo > ci_lower_threshold`, and the
   condition holds for at least `persistence` consecutive
   evaluated positions whose largest is still <= 0.

## Files

- `src/herald/analysis/lead_time.py` — selection, virtual onset
  assignment, alignment, AUROC + bootstrap, lead-time logic,
  plotting.
- `scripts/build_lead_time_curves.py` — CPU CLI mirroring
  `build_event_study.py` with extra threshold and persistence
  flags.
- `tests/analysis/test_lead_time.py` — synthetic-pair fixtures,
  AUROC convergence test, lead-time persistence semantics, CLI
  smoke, blockers + insufficient-pairs paths.

## Reproducing The Phase 0 Smoke Output

```sh
.venv/bin/python scripts/build_lead_time_curves.py \
    --input results/phase0 \
    --output results/analysis/lead_time \
    --window-before 200 --window-after 50 --n-bootstrap 200
```

Phase 0 smoke result on `results/phase0`:

- 134 total compressed runs.
- 77 catastrophic with derivable onset (49 looping, 28
  non_termination_proxy).
- 37 healthy compressed controls, all matched into 5
  `(task, press, ratio)` strata.
- 2 of 5 strata satisfy `min_per_stratum=3` for both cat & ctrl,
  below the default `min_well_covered_strata=3`, so the run is
  flagged `control_limited=True`.
- Per-feature lead time at default thresholds (AUROC >= 0.65,
  CI lower > 0.5, persistence = 5):
  - `entropy`: 7 tokens before onset.
  - All other Tier 0 features (`top1_prob`, `top1_top2_margin`,
    `kl_div`, `h_alts`, `delta_h`, `top10_jaccard`, `tail_mass`):
    no qualifying lead time. Likely a power problem given the
    77/37 split inside two presses, not a scientific claim that
    these features lack signal.

## Outputs

Under `<output>/`:

- `lead_time_by_feature.parquet` — long rows of
  `(feature, relative_pos, n_pos, n_neg, auroc, ci_lo, ci_hi)`.
- `lead_time_by_feature_press.parquet` — same, stratified by
  press.
- `lead_time_summary.json` — counts (catastrophic / control /
  matched / per stratum), matching strategy, excluded counts,
  feature list, AUROC threshold, persistence window, lead time
  per feature, control-limited flag and caveats.
- `lead_time_curves.png` — one subplot per feature; AUROC vs
  relative_pos with CI band, threshold + onset markers, lead-
  time annotation in the panel title.
- `lead_time_curves_by_press.png` — same, one trace per press.

## Connection To The Predictor

The headline HERALD predictor figure (Phase 2) uses the same
relative-position axis and the same alignment primitives. Where
the lead-time figure plots per-feature AUROC, the predictor
figure plots `predicted_p` (or its calibrated AUROC) on the same
axis, so the two figures stack.

If the predictor's lead time does not exceed the per-feature
ceiling in this analysis, the Phase 2 claim collapses to "logistic
regression on Tier 0 features is sufficient" and the framing must
adjust. The lead-time analysis is therefore both a sanity check
(do features carry mechanism-level signal at horizon?) and a
required baseline against which the learned model is compared.
