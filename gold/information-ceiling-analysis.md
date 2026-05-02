# Information-Ceiling Analysis

Quantifies whether cheap online features carry information about
future compression damage *beyond* trivial position and metadata
proxies (press, ratio, task). This is the HERALD rebuttal to the
LimitsLearned-style concern that the predictor's signal might be
indistinguishable from `position + ratio` alone.

## Why It Matters For HERALD

The contribution-validation bar (`gold/contribution-validation.md`,
"Required Feature-Information Check") demands a quantification of
whether Tier 0 / Tier 0.5 features add information beyond
position-only and ratio-only baselines, before any strong predictor
claim can stand. If `auroc(position+metadata+online) ==
auroc(position+metadata)`, then HERALD has no story; the predictor's
apparent signal is just a proxy for "deeper into a heavily
compressed run."

## What This Analysis Does Claim

- A point estimate of the *headline incremental gain*
  `auroc_all - auroc_position_plus_metadata` per horizon, with run-
  level bootstrap CIs.
- A per-feature MI estimate against the future-damage label per
  horizon, comparable across position / metadata / online groups.
- A permutation importance ranking from a single regularized LR
  fit on the smallest horizon, as a robustness check on MI.
- Stratified per-press scores when each press has at least
  `min_runs_per_press` runs.

## What This Analysis Does NOT Claim

- It is not the predictor. The CV AUROC numbers come from a tiny
  balanced LR, not the eventual XGBoost/HERALD model. They are a
  ceiling-style read on what *cheap, linearly-combinable* online
  features can do.
- It is not Phase 1. Phase 0 has a single task and only two
  presses. The summary marks `phase0_smoke=True` whenever any press
  has fewer than `min_runs_per_press` runs or only one task is
  present; treat the headline gain as smoke-only until the Phase 1
  grid lands.
- It does not measure post-onset discriminability. Tokens at or
  past derived onset are excluded from the analysis frame; the
  ceiling is a *future-damage* ceiling, not a post-hoc one.
- It does not put a CI on the headline incremental gain. The
  per-group bootstrap CIs in `information_by_group.parquet` are
  independent, not paired across groups; deciding whether the
  gain itself excludes zero needs a paired bootstrap (resample
  runs once, score `pos+meta` and `pos+meta+online` on the same
  resamples, take the difference). Phase 1 followup; the Phase 0
  smoke read is directional only.

## Method

1. **Onset derivation** — same as event-study /
   `derive_onset(catastrophes, token_ids, max_new_tokens)`. Looping
   first, non-termination proxy second; censored runs map to no
   onset.
2. **Per-token table** — restrict to compressed runs (`press !=
   'none'`). Compute position features (`token_pos`,
   `relative_progress`, `output_length_so_far`), encode metadata
   (`compression_ratio`, `press_code`, `task_code`), pull online
   features as-is. Drop tokens at/past onset for catastrophic runs.
3. **Future-damage labels** — `future_damage_h{H} = 1` iff `onset
   in (token_pos, token_pos+H]`; censored runs and tokens beyond
   the H window get 0.
4. **Mutual information** — `sklearn.feature_selection.
   mutual_info_classif(random_state=cfg.seed)`, with categorical
   features (`press_code`, `task_code`) declared as discrete. The
   MI table is per `(feature, horizon)`.
5. **Group AUROC** — `LogisticRegression(class_weight='balanced',
   C=1.0)` wrapped in a `StandardScaler`, scored with
   `GroupKFold(n_splits, groups=run_id)` so train/val never share a
   run. Bootstrap CIs resample run_ids (not tokens), refitting CV
   on every iteration to get a true run-level interval.
6. **Permutation importance** — single train/val fold from
   `GroupKFold`, AUROC drop after permuting each feature in the
   validation set.
7. **Per-press stratification** — repeats the group-AUROC scoring
   inside each press with `press_code` removed (it is constant in
   the cell). Skipped when a press has zero rows.

## Critical Decisions Locked In

- `compression_ratio` lives in **metadata**, never in **online**.
  The whole point is to show online features add something *beyond*
  ratio + position; conflating the two would rig the comparison.
- Group CV by `run_id` is non-negotiable. Tokens within a run
  share a derived onset, so token-level CV would leak the label
  across folds and inflate AUROC across the board.
- `class_weight='balanced'`. At H=5 the positive class is roughly
  `n_cat * 5` tokens against tens of thousands of negatives; without
  balancing the LR collapses to predicting the majority class.
- Bootstrap by run, not token. Token-level bootstrap collapses CIs
  to nothing because tokens within a run are not independent.
- Tokens at/past onset are dropped. They are trivially separable
  post-hoc (entropy spikes during looping); including them would
  inflate the ceiling and miss the predictive question entirely.

## Files

- `src/herald/analysis/information_ceiling.py` — dataset
  construction, MI, group AUROC + bootstrap CI, permutation
  importance, plotting, top-level entry point.
- `scripts/build_information_ceiling.py` — CPU CLI mirroring
  `build_lead_time_curves.py`.
- `tests/analysis/test_information_ceiling.py` — synthetic
  fixtures, MI separation test, group AUROC ordering test, smoke
  CLI run.

## Reproducing The Phase 0 Smoke Output

```sh
.venv/bin/python scripts/build_information_ceiling.py \
    --input results/phase0 \
    --output results/analysis/information_ceiling \
    --horizons 5,10,25,50
```

Outputs land under `results/analysis/information_ceiling/`:

- `information_by_feature.parquet` — per (feature, horizon) MI + n
  + n_pos.
- `information_by_group.parquet` — per (group, horizon) CV AUROC
  + bootstrap CI + n_features + n_samples + n_runs.
- `information_by_group_press.parquet` — same, stratified by
  press; written only when at least one press has rows.
- `permutation_importance.parquet` — per feature AUROC drop on a
  single GroupKFold fold at the smallest horizon.
- `information_summary.json` — input paths, config, headline per
  horizon (`auroc_position_only`, `auroc_metadata_only`,
  `auroc_position_plus_metadata`, `auroc_all`,
  `incremental_auroc_gain`, `mi_sum_*`, `incremental_mi_gain`),
  feature columns used/missing, `phase0_smoke` flag, caveats.
- `information_by_horizon.png` — grouped bar chart of group-CV
  AUROC by horizon.
- `information_by_feature.png` — per-feature MI bar plot, one
  panel per horizon.

## Headline Read

The single number the rebuttal hinges on is per horizon:

```
incremental_auroc_gain[H] =
    auroc_position_metadata_online[H] - auroc_position_metadata[H]
```

If this is `> 0` with bootstrap CI excluding zero across multiple
horizons, online features carry actionable information beyond the
trivial proxies; if it is `~0`, HERALD is largely re-encoding
position + ratio and the framing must adjust toward measurement /
methodology rather than a predictor claim. Phase 0 smoke output is
intended to ship the plumbing and a directional read; the
defensible number lands with Phase 1.
