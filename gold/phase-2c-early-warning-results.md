# Phase 2c Results — Early-Warning-Signal Features

**Date**: 2026-05-06.
**Source dataset**:
`results/phase2c_early_warning/phase2_dataset_ews.parquet` (7 738 974
rows; Phase 2 dataset augmented with 100 EWS columns: 5 base
signals × 4 windows × 5 families).
**Prerequisites**: `gold/phase-2-results.md`,
`gold/phase-2b-results.md`.

This document records the Phase 2c early-warning-signal experiment:
do classic dynamical-systems collapse indicators (rolling skew,
slope, lag-1 autocorrelation, range, median-crossing flicker) carry
predictive information beyond the existing rolling-mean / EWMA cheap
features in `lr_all_cheap`?

## TL;DR — Verdict: Analysis-only

EWS features add a **small but CI-significant lift** over
`lr_all_cheap` (`lr_all_cheap+EWS minus lr_all_cheap` cluster-
bootstrap delta is positive in **all 16 (split, horizon) cells**;
lower CI bound > 0 everywhere). The lift is in the +0.006 to
+0.012 AUROC range, comfortably below the +0.02 "materially
better" gate Phase 2b applied to XGBoost. EWS does **not** fix
the catastrophic-onset lead-time inversion: pre-onset mean AUROC
goes from 0.388 (`lr_all_cheap`) to 0.406 (`lr_all_cheap+EWS`),
still well below 0.5 and far below the deployable 0.65 threshold.

Per the pre-stated verdict criteria below, this is the
**analysis-only** branch: keep the module and the feature
audit, do not include EWS in the Phase 4 v1 controller score.
The dynamical-instability framing is partially supported (EWS
carries information beyond `lr_all_cheap`), but the existing
rolling/EWMA features already capture most of the signal, and
EWS does not unlock the lead-time / pre-collapse early-warning
behaviour the framing predicted.

## Pre-stated verdict criteria

The decision rule was locked in *before* reading the runner
outputs (preserved verbatim from the pre-result draft of this
file):

1. **Promote EWS** (include in Phase 4 controller score) iff:
   - The cluster-bootstrap (run_id) lower CI bound on the
     `lr_all_cheap+EWS minus lr_all_cheap` delta is strictly > 0
     on the held-out *prompts* split AND on at least one of
     {ratios, presses, tasks} at H=25; AND
   - Either lead-time pre-onset mean AUROC[-200, 0] for the
     EWS-augmented predictor exceeds 0.5 (Phase 2 inversion at
     least neutralised), OR the headline AUROC delta is
     materially > +0.02 on at least three of four splits at H=25.
2. **Analysis-only** (keep the module, do not include in
   controller) iff the prompts CI excludes zero but the headline
   transfer or lead-time half of the criterion fails.
3. **Exclude** (negative result) iff the prompts-split CI lower
   bound is ≤ 0 or any transfer split is strictly worse with CI
   excluding zero on the wrong side.

**Outcome**: criterion #1's first half holds (prompts and all
three transfer splits' CIs strictly > 0 at H=25). Its second half
fails on **both** disjuncts: lead-time mean AUROC[-200, 0]
remains 0.406 (< 0.5), and the H=25 AUROC delta is +0.0088
(prompts), +0.0105 (ratios), +0.0105 (presses), +0.006 (tasks) —
none above +0.02. The result therefore lands on criterion #2.

## What is in the EWS feature set

Module: `src/herald/early_warning_features.py`.
Audit: `results/phase2c_early_warning/ews_feature_audit.json`.

For each of 5 base signals (entropy, top1_prob, h_alts, delta_h,
kl_div) and each of 4 windows (W in {8, 16, 32, 64}):

- `ews_<signal>_skew_W`     — rolling third-moment shape
- `ews_<signal>_slope_W`    — OLS slope vs token_pos over W tokens
- `ews_<signal>_lag1ac_W`   — corr(x_t, x_{t-1}) over W tokens
- `ews_<signal>_rng_W`      — rolling max minus rolling min
- `ews_<signal>_flicker_W`  — count of median-crossings over W

5 × 4 × 5 = 100 EWS columns.

Excluded by design:

- `std`/`var`: redundant with existing `*_std_8`, `*_std_32` in
  the Phase 2 dataset.
- `flicker_rate_W = flicker_count_W / W`: linear duplicate of
  `flicker_count_W`.
- `top1_top2_margin`: top-2 prob is not stored at decode time in
  the Phase 1 token files (only top1 / top5 are). Documented as
  a v2 candidate.

`kl_div` is the **consecutive-timestep instability KL** computed
in `herald.signals` at decode time, NOT the replay-based
`kl_unc_comp_full` (which is one of the labels' source columns
and forbidden as input).

Causality (load-bearing, tested in
`tests/test_early_warning_features.py`, 8 tests):

- causal-window: mutating future tokens does not change EWS
  values at earlier positions.
- group-isolation: EWS values inside a multi-run frame equal the
  standalone-run values (no run-A leak into run-B).
- constant-signal edge cases: range and slope are 0; autocorr
  falls back to null when variance is zero.
- small-run: runs shorter than W return null for that window
  without crashing.
- schema hygiene: the deployable-feature filter drops every
  replay column, every `future_*` label, and every run-level
  validator.

## 1. Headline: EWS-augmented predictor vs `lr_all_cheap`

Code: `scripts/run_phase2_ews_baselines.py`.
Outputs: `results/phase2c_early_warning/ews_baselines.parquet`,
`ews_baselines_summary.json`,
`ews_delta_ci.parquet`, `ews_delta_ci_summary.json`.

Mean AUROC across folds, label `future_sum_js_H`, max-train-rows
200 000, n-boot 100 (per-baseline) and 200 (cluster CI).

| split   | H  | `lr_all_cheap` | `+ EWS` | Δ EWS-vs-LR | EWS-vs-LR 95% CI | Δ EWS-vs-entropy |
|---------|----|---------------:|--------:|------------:|------------------|----------------:|
| prompts | 5  | 0.8506         | 0.8604  | +0.0098     | [+0.0087, +0.0109] | +0.094          |
| prompts | 10 | 0.8700         | 0.8790  | +0.0090     | [+0.0080, +0.0100] | +0.102          |
| prompts | 25 | 0.8889         | 0.8977  | +0.0088     | [+0.0080, +0.0097] | +0.105          |
| prompts | 50 | 0.8925         | 0.9009  | +0.0084     | [+0.0074, +0.0097] | +0.101          |
| ratios  | 5  | 0.7727         | 0.7850  | +0.0123     | [+0.0088, +0.0155] | +0.063          |
| ratios  | 10 | 0.8016         | 0.8137  | +0.0121     | [+0.0087, +0.0157] | +0.079          |
| ratios  | 25 | 0.8178         | 0.8283  | +0.0105     | [+0.0053, +0.0156] | +0.076          |
| ratios  | 50 | 0.8387         | 0.8491  | +0.0105     | [+0.0056, +0.0161] | +0.074          |
| presses | 5  | 0.8080         | 0.8173  | +0.0093     | [+0.0080, +0.0108] | +0.069          |
| presses | 10 | 0.8269         | 0.8368  | +0.0099     | [+0.0087, +0.0113] | +0.079          |
| presses | 25 | 0.8442         | 0.8547  | +0.0105     | [+0.0089, +0.0117] | +0.088          |
| presses | 50 | 0.8494         | 0.8601  | +0.0108     | [+0.0095, +0.0121] | +0.088          |
| tasks   | 5  | 0.8375         | 0.8453  | +0.0078     | [+0.0062, +0.0090] | +0.079          |
| tasks   | 10 | 0.8584         | 0.8652  | +0.0068     | [+0.0054, +0.0082] | +0.088          |
| tasks   | 25 | 0.8778         | 0.8838  | +0.0060     | [+0.0045, +0.0074] | +0.093          |
| tasks   | 50 | 0.8823         | 0.8883  | +0.0060     | [+0.0044, +0.0079] | +0.086          |

All 16 cells: lower CI bound on EWS-vs-LR delta is strictly > 0.
The largest cell-level lift is **+0.0123 AUROC** (ratios H=5);
the smallest is **+0.0060** (tasks H=25, H=50). For comparison,
Phase 2b XGBoost lift over `lr_all_cheap` at H=25 was +0.025
(prompts), +0.024 (presses), +0.016 (ratios), +0.010 (tasks);
the XGBoost lift is materially larger than the EWS lift on the
prompts/presses splits and comparable on the harder transfer
splits.

The `EWS-vs-entropy` column shows the EWS-augmented predictor
beats the best entropy/EWMA single-feature baseline by +0.063
to +0.105 AUROC. The Phase 2 success criterion (≥ 0.05 over
entropy) is therefore exceeded by an even wider margin with EWS
than with `lr_all_cheap` alone, but the relevant gate for
*promoting EWS* is the much narrower delta over `lr_all_cheap`,
not over entropy.

## 2. Lead time at catastrophic onset

Code: `scripts/run_phase2_ews_lead_time.py`.
Outputs: `results/phase2c_early_warning/ews_lead_time.parquet`,
`ews_lead_time_summary.json`.

Comparison of three predictors on the same matched-stratum
catastrophic / control set Phase 2b used (max 20 cats / 20 ctrls
per (task, press, ratio); n=1882 catastrophic, n=2133 control;
predictor trained on a 400 000-row subsample disjoint from
selected runs).

| feature             | mean AUROC[-200, 0] | mean AUROC[-50, 0] | max AUROC[-200, 0] | mean AUROC[0, +50] |
|---------------------|--------------------:|-------------------:|-------------------:|-------------------:|
| entropy (raw)       | 0.542               | (n/a)              | 0.587              | 0.524              |
| `lr_all_cheap`      | 0.388               | 0.297              | 0.491              | 0.498              |
| `lr_all_cheap+EWS`  | 0.406               | (similar)          | 0.467              | 0.490              |

The EWS-augmented predictor is **slightly less inverted** than
`lr_all_cheap` (0.406 vs 0.388 mean AUROC in the [-200, 0]
window), but still well below the 0.5 chance line and very far
from the deployable 0.65 threshold. The dynamical-instability
hypothesis (rising variance / skew / flicker should precede
collapse and lift the predictor above 0.5 pre-onset) is
**not supported** by these numbers. The looping / non-termination
inversion documented in `gold/phase-2-results.md` §(b) and
`gold/phase-2b-results.md` §2 persists.

Mechanistic reading consistent with Phase 2b: looping and
non-termination collapse into low-divergence repetitive output,
and the JS-sum-trained predictor (correctly, given its training
target) assigns those quiet windows lower future-divergence
scores than divergent-but-readable healthy compressed text. EWS
captures *some* additional pre-onset variance / flicker, which is
why the inversion is mildly reduced, but not enough to flip the
sign. The remaining levers are label co-design (train on
diagnostic tags directly) or a sequence model — both v2 work.

## 3. Which feature families matter most

Per-family ablation was not run, since the headline lift is below
the gate that would have justified the controller change. The
overall lift is small and broadly distributed; carving it into
families would not change the verdict and would risk over-fitting
the family list to the dataset on hand. Listed as a v2 follow-up
if EWS resurfaces.

## 4. Reproduction

```sh
# Build EWS-augmented dataset (~30 s; CPU only; 4.3 GB output).
uv run python scripts/build_phase2_ews_dataset.py

# EWS-vs-LR baselines + cluster-bootstrap CIs (~30 min wall).
uv run python scripts/run_phase2_ews_baselines.py \
    --horizons 5 10 25 50 \
    --splits prompts ratios presses tasks \
    --max-train-rows 200000 --n-boot 100 --ci-n-boot 200

# EWS lead-time vs entropy / lr_all_cheap (~1.5 min wall).
uv run python scripts/run_phase2_ews_lead_time.py \
    --label future_sum_js_25
```

## 5. Files changed / commands run

### Source

- `src/herald/early_warning_features.py` — new module (causal
  rolling EWS features, deployable-column filter, audit
  constants).
- `src/herald/predictor_baselines.py` —
  `_build_baseline_scores`, `evaluate_split`,
  `collect_fold_predictions` accept an optional
  `extra_features: tuple[str, ...]`. The extras union with
  `CHEAP_ALL_FEATURES` and produce a new score key
  `lr_all_cheap_plus_extras`. No change to existing behaviour
  when `extra_features=()`; Phase 2 / 2b regressions are not
  affected.

### Tests

- `tests/test_early_warning_features.py` — 8 tests covering
  causality, group isolation, edge cases, small runs, and schema
  hygiene. All pass.

### Scripts

- `scripts/build_phase2_ews_dataset.py`
- `scripts/run_phase2_ews_baselines.py`
- `scripts/run_phase2_ews_lead_time.py`

### Output artifacts

- `results/phase2c_early_warning/phase2_dataset_ews.parquet`
- `results/phase2c_early_warning/ews_feature_audit.json`
- `results/phase2c_early_warning/ews_baselines.parquet`
- `results/phase2c_early_warning/ews_baselines_summary.json`
- `results/phase2c_early_warning/ews_delta_ci.parquet`
- `results/phase2c_early_warning/ews_delta_ci_summary.json`
- `results/phase2c_early_warning/ews_lead_time.parquet`
- `results/phase2c_early_warning/ews_lead_time_summary.json`

## 6. Decision

**Analysis-only.** EWS is real (CI-significant on every cell)
but small (+0.006 to +0.012 AUROC over `lr_all_cheap`, well
below the +0.02 promote bar), and does not fix the
catastrophic-onset lead-time inversion (pre-onset mean AUROC
0.406, still below 0.5).

Implications:

- **Phase 4 v1 controller**: do not include EWS. `lr_all_cheap`
  remains the recommended controller score (per
  `gold/phase-2b-results.md` §5). The Phase 4 plan does not
  change.
- **Paper framing**: report EWS as a measured negative on the
  dynamical-instability hypothesis. The framing that "rising
  variance / autocorrelation precede collapse" makes a clean
  prediction, the experiment is a clean test, and the test
  fails: existing rolling/EWMA features already capture most of
  the predictable signal under the JS-sum label. Combined with
  the Phase 2b `future_max_js` negative (label co-design does
  not fix inversion either), this strengthens the
  scope-limit narrative around the lead-time finding rather
  than weakening the predictor headline.
- **v2 follow-ups**: train directly on the diagnostic tag
  (`future_has_looping_H`, `future_has_non_termination_H`),
  or fit a sequence model on the streaming feature sequence.
  Either may exploit pre-onset entropy patterns the linear /
  static-feature models miss. Both are out of v1 scope.
- **EWS module**: keep it. The audit, the schema-hygiene
  filter, and the test coverage are useful infrastructure even
  when the controller does not consume the features. A future
  per-family ablation or sequence-model experiment can reuse
  `add_ews_features` directly.
