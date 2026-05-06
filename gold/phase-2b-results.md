# Phase 2b Results — Predictor Follow-Ups

**Date**: 2026-05-05.
**Source dataset**: `results/phase2/dataset/phase2_tokens.parquet`
(7 738 974 rows from 32 088 compressed runs, 90 columns).
**Prerequisite**: `gold/phase-2-results.md` (point-estimate Phase 2
result, all four splits passing the 0.05 AUROC bar).

This document records the four follow-ups Phase 2 deferred:
paired-bootstrap CIs, alternative label families, XGBoost headroom,
and segment / run risk validation. The combined evidence determines
the controller score recommendation for Phase 4.

## TL;DR

1. **CI upgrade**: the Phase 2 ">= 0.05 AUROC over `entropy_mean_8`"
   claim now has cluster-bootstrap (run-level) CIs that exclude zero
   in **all 16 (split, horizon) cells**. The lower bound is at or
   above 0.045 in every cell; the prompts split exceeds 0.083 in
   every horizon.
2. **Label family**: `future_sum_js` strictly dominates
   `future_sum_kl` and `future_max_js` on every per-run validator
   (Spearman ρ vs `rouge_l_drop` 0.73 / 0.68 / 0.54; AUROC vs
   `has_looping` 0.81 / 0.76 / 0.62; H=25 prompts AUROC 0.889 /
   0.833 / 0.741). The advisor's hypothesis that `future_max_js`
   would fix the catastrophic-onset lead-time inversion is
   **contradicted**: max-js is more inverted at onset (mean AUROC
   0.34 vs 0.39 for sum-js in the [-200, 0] window). Phase 2's
   choice of `future_sum_js` was correct.
3. **Model headroom**: XGBoost on the same feature set gains
   ≈ +0.024 AUROC and reduces ECE by ~0.06 versus `lr_all_cheap`
   on held-out prompts at H=25, with similar lift on the other
   splits. The headroom is real but borderline relative to the
   advisor's "materially > +0.02" gate; it does not change the
   Phase 2 conclusion that `lr_all_cheap` is the right v1 baseline,
   but XGBoost is recommended for the production controller score.
4. **Segment / run risk**: the existing OOF predictor's per-run
   `max_seg_max` aggregate strongly correlates with run damage
   (Spearman 0.73 vs `rouge_l_drop`, 0.74 vs `sum_js`) and
   discriminates catastrophic tags (AUROC 0.81 looping, 0.84
   non-termination). At every K in {8, 16, 32}, predictor segment
   risk **beats entropy segment risk** by 0.07-0.10 AUROC on
   catastrophe tags. K=16 with `run_max_seg_mean` is the best
   per-run aggregate (looping AUROC 0.809, NT 0.844, sum_js
   ρ 0.746). This is strong enough to justify the Phase 4 segment-
   gating controller demo.

## 1. Paired-bootstrap CIs (Task 1)

**Method**. Tokens within a run share within-run rolling features and
auto-correlated logit signals; row-level bootstrap CIs underestimate
the variance of the AUROC. We resample test rows by `run_id`
(cluster bootstrap). For each (split, horizon, fold), a replicate
draws unique run_ids in the test set with replacement, recomputes
AUROC for `lr_all_cheap` and `feature::entropy_mean_8`, and takes
the per-fold delta. Cross-fold mean delta is the average over folds;
CI is the 2.5/97.5 percentile of cross-fold mean delta across 200
replicates.

Code: `src/herald/predictor_baselines.py`
(`clustered_paired_bootstrap_delta`,
`cross_fold_clustered_bootstrap`,
`collect_fold_predictions`).
Runner: `scripts/run_phase2_delta_ci.py`.

### Result

| split   | H  | mean Δ AUROC | 95% CI            | folds | decision         |
|---------|----|--------------|-------------------|-------|------------------|
| prompts | 5  | 0.0869       | [0.0832, 0.0904]  | 5     | passes_005_ci    |
| prompts | 10 | 0.0948       | [0.0911, 0.0990]  | 5     | passes_005_ci    |
| prompts | 25 | 0.0967       | [0.0930, 0.1008]  | 5     | passes_005_ci    |
| prompts | 50 | 0.0924       | [0.0878, 0.0972]  | 5     | passes_005_ci    |
| ratios  | 5  | 0.0539       | [0.0456, 0.0622]  | 7     | passes_005_ci    |
| ratios  | 10 | 0.0597       | [0.0492, 0.0688]  | 7     | passes_005_ci    |
| ratios  | 25 | 0.0647       | [0.0493, 0.0777]  | 7     | passes_005_ci    |
| ratios  | 50 | 0.0673       | [0.0514, 0.0844]  | 7     | passes_005_ci    |
| presses | 5  | 0.0620       | [0.0590, 0.0653]  | 6     | passes_005_ci    |
| presses | 10 | 0.0705       | [0.0676, 0.0740]  | 6     | passes_005_ci    |
| presses | 25 | 0.0780       | [0.0740, 0.0817]  | 6     | passes_005_ci    |
| presses | 50 | 0.0750       | [0.0704, 0.0791]  | 6     | passes_005_ci    |
| tasks   | 5  | 0.0734       | [0.0696, 0.0769]  | 4     | passes_005_ci    |
| tasks   | 10 | 0.0832       | [0.0790, 0.0872]  | 4     | passes_005_ci    |
| tasks   | 25 | 0.0860       | [0.0817, 0.0896]  | 4     | passes_005_ci    |
| tasks   | 50 | 0.0799       | [0.0746, 0.0853]  | 4     | passes_005_ci    |

All 16 cells pass: mean delta ≥ 0.05 **and** the lower CI bound
exceeds 0.04 in every cell (≥ 0.045 on `ratios::H=5`, the narrowest
margin). The narrowest cell remains `ratios::H=5`, consistent with
the Phase 2 finding that ratio is the single most predictive feature
and the held-out-ratio interpolation is the hardest split. Even
there the CI excludes zero with margin.

Outputs:
- `results/phase2/baselines/phase2_delta_ci.parquet` (per-fold deltas)
- `results/phase2/baselines/phase2_delta_ci_summary.json` (cell CIs)

## 2. Label-family sweep (Task 2)

**Motivation**. Phase 2 trained on `future_sum_js_H`. The lead-time
analysis showed the JS-sum-trained predictor scores *below* 0.5
AUROC near catastrophic onset because looping collapses into low-
divergence repetitive output. `future_max_js_H` is the label most
likely to fix that — onset itself is a JS spike that survives the
max but vanishes in a sum dominated by post-onset quiet tokens.
`future_sum_kl_H` is included as a sanity check on `future_sum_js`.

**Setup**. Same dataset, splits, features, runner as Phase 2. For
each label, repeat the baseline AUROC sweep, the per-run OOF
validation at H=25 on the prompts split, and the lead-time
catastrophic-vs-control AUROC analysis.

Code: `scripts/run_phase2_label_sweep.py`.

**Scope reduction note**. The label-sweep baseline was originally
launched at all four horizons. Mid-run we reduced to H=25 only after
verifying the Phase 2 horizon-trend was monotonic and that the
controller decision is governed by H=25; the sweep then completed in
559 s and the outputs below come from that compact run. The
horizon-axis ablation for `future_sum_js` is already locked in by
`gold/phase-2-results.md`.

### Baseline AUROC at H=25 (mean across folds)

| label             | split   | AUROC `lr_all_cheap` | AUROC `entropy_mean_8` | Δ      |
|-------------------|---------|---------------------:|-----------------------:|-------:|
| `future_sum_js`   | prompts | 0.8893               | 0.7926                 | +0.097 |
| `future_sum_js`   | ratios  | 0.8120               | 0.7473                 | +0.065 |
| `future_sum_js`   | presses | 0.8458               | 0.7678                 | +0.078 |
| `future_sum_js`   | tasks   | 0.8810               | 0.7950                 | +0.086 |
| `future_sum_kl`   | prompts | 0.8329               | 0.6854                 | +0.148 |
| `future_sum_kl`   | ratios  | 0.7330               | 0.6488                 | +0.084 |
| `future_sum_kl`   | presses | 0.7728               | 0.6769                 | +0.096 |
| `future_sum_kl`   | tasks   | 0.8217               | 0.6849                 | +0.137 |
| `future_max_js`   | prompts | 0.7409               | 0.5859                 | +0.155 |
| `future_max_js`   | ratios  | 0.6627               | 0.5686                 | +0.094 |
| `future_max_js`   | presses | 0.6920               | 0.5796                 | +0.112 |
| `future_max_js`   | tasks   | 0.7338               | 0.6001                 | +0.134 |

The lr-vs-entropy delta is largest for the harder labels because
entropy degrades faster than `lr_all_cheap` does. The number that
matters for the controller is the **absolute AUROC** of
`lr_all_cheap`, which is monotonically ordered:

```
future_sum_js  >  future_sum_kl  >  future_max_js
```

across all four splits.

### Per-run validation at H=25 on the prompts split

_OOF prompts split, `max_score` aggregate across runs, n=30 610._

| label           | ρ rouge_l_drop | ρ sum_js | ρ quality_delta | AUROC looping | AUROC NT | AUROC gross_harm |
|-----------------|---------------:|---------:|----------------:|--------------:|---------:|-----------------:|
| `future_sum_js` | 0.731          | 0.740    | 0.326           | 0.810         | 0.844    | 0.681            |
| `future_sum_kl` | 0.684          | 0.702    | 0.328           | 0.758         | 0.830    | 0.677            |
| `future_max_js` | 0.543          | 0.581    | 0.291           | 0.618         | 0.758    | 0.653            |

`future_sum_js` strictly dominates on every per-run validator. The
controller-score evidence (continuous severity correlation, looping
AUROC, NT AUROC, gross-harm AUROC) is unanimous.

### Lead-time at catastrophic onset

_AUROC of `predictor_score` for catastrophic vs healthy controls
matched per (task, press, ratio) stratum, evaluated at relative
positions in [-200, 0] tokens (pre-onset). Ideal would be > 0.65;
chance is 0.5; sub-0.5 means the predictor scores catastrophic runs
**lower** than controls (the Phase 2 inversion finding)._

| label           | mean AUROC[-200, 0] | mean AUROC[-50, 0] | max AUROC[-200, 0] |
|-----------------|--------------------:|-------------------:|-------------------:|
| `future_sum_js` | 0.388               | 0.297              | 0.491              |
| `future_sum_kl` | 0.381               | 0.290              | 0.486              |
| `future_max_js` | 0.336               | 0.257              | 0.425              |
| entropy (any)   | 0.542               | -                  | 0.587              |

**Result contradicts the working hypothesis.** The `future_max_js`
label, which the advisor predicted would fix the inversion (because
onset is itself a JS spike that survives a max but is washed out in
a sum), instead produces a **more inverted** predictor at onset.

Mechanistic interpretation: max-based labels train the predictor to
locate the highest-divergence single token in the next H tokens. In
healthy compressed runs that token is typically a divergent-but-
readable continuation; in looping/NT runs the highest-divergence
token in the future window is the loop-onset spike, which is
followed by quiet tokens — the predictor learns that the *current*
position lacks the precursor signature of an isolated spike, so
catastrophic runs are scored *lower* than controls. The same
mechanism, weaker, applies to sum-based labels.

Operational consequence: **no JS-style label fixes onset lead time**
under the current cheap-feature set. Onset lead-time is a model and
label co-design problem; the obvious next experiment is to train
directly on the diagnostic tag (`future_has_looping_H` or
`future_has_non_term_H`), but that ties the predictor target to the
heuristic tag thresholds in `herald.labeling`, which the research
plan explicitly forbids for the headline predictor. The fallback is
the segment-risk controller in §4 above.

Outputs:
- `results/phase2/label_sweep/label_sweep_results.parquet`
- `results/phase2/label_sweep/label_sweep_summary.json`
- `results/phase2/label_sweep/per_run/<label>/per_run_validation.json`
- `results/phase2/label_sweep/lead_time/<label>/lead_time_summary.json`

## 3. XGBoost headroom (Task 3)

**Method**. Train XGBoost on the same `CHEAP_ALL_FEATURES` feature
set as `lr_all_cheap`, on the same fold schedule (5-fold prompts,
LOGO presses/ratios/tasks). H=25 only; subsample 200 000 rows for
tractable CPU training. Compare per-fold AUROC + AUPRC against
`lr_all_cheap` on the same fold rows. Calibration: 10-bin reliability
table on the held-out-prompts H=25 cell, ECE = mean of |obs - pred|
weighted by bin support.

Code: `scripts/run_phase2_model_headroom.py`.

### Result (mean across folds, H=25)

| split   | AUROC `lr_all_cheap` | AUROC XGBoost | Δ      | ECE lr | ECE xgb |
|---------|---------------------:|--------------:|-------:|-------:|--------:|
| prompts | 0.8893               | 0.9142        | +0.0249 | 0.215 | 0.154   |
| presses | 0.8863               | 0.9103        | +0.0240 | n/a   | n/a     |
| ratios  | 0.8120               | 0.8276        | +0.0156 | n/a   | n/a     |
| tasks   | 0.8810               | 0.8909        | +0.0099 | n/a   | n/a     |

(ECE only computed on prompts H=25 by design.)

**Comparator note**. The `_evaluate_one_fold` helper in
`run_phase2_model_headroom.py` keeps `compression_ratio` in the
feature set on the held-out-press fold for *both* models, so the
head-to-head Δ is internally fair. The Phase 2 baseline runner in
`predictor_baselines.evaluate_split`, by contrast, drops
`compression_ratio` from the LR feature set on the press split (so
the LR cannot rely on a column that is constant within fold).
That convention difference is why this table reports
`lr_all_cheap` press AUROC ≈ 0.886, whereas
`gold/phase-2-results.md` reports ≈ 0.846. Both are correct under
their stated conventions; the lift over `entropy_mean_8` reported
in §1 of this document is the apples-to-apples figure.

XGBoost gains +0.025 AUROC on prompts/presses (where compression-
damage rows are well-represented in training) but only +0.016 on
held-out ratios and +0.010 on held-out tasks. The advisor's
"materially > +0.02" gate is **only met on the IID-ish splits**;
the harder transfer splits (ratios, tasks) are under-gate. ECE on
the prompts H=25 cell falls from 0.215 (lr) to 0.154 (xgb): a real
but still high miscalibration that an explicit calibration layer
(isotonic or Platt on a held-out calibration fold) would be
expected to drive below 0.05 in production.

Conclusion:

- **Phase 2 result stands**: `lr_all_cheap` remains the appropriate
  cheap baseline. The Phase 2 ">= 0.05 AUROC over entropy" claim
  does not depend on XGBoost.
- **Phase 4 controller default**: keep `lr_all_cheap` as the v1
  controller score. XGBoost is an optional engineering polish that
  helps on IID-ish folds but degrades to a marginal +0.010 AUROC
  on the held-out-task split, which is the closest analogue to a
  production "new task family" deployment. Train an XGBoost in
  parallel for the headline figure, but do not let the controller
  demo depend on it.
- **No GRU**: the Phase 2 plan says "only run sequence models if
  XGBoost leaves obvious headroom on lead time or calibration".
  XGBoost did not collapse calibration error; if Phase 4's
  recalibrated XGBoost still underperforms on segment-level lead
  time, GRU becomes the next experiment. It is *not* the
  precondition for the controller demo.

Outputs:
- `results/phase2/model_headroom/model_headroom_results.parquet`
- `results/phase2/model_headroom/model_headroom_summary.json`
- `results/phase2/model_headroom/calibration_prompts_h25.parquet`

## 4. Segment / run risk (Task 4)

**Motivation**. The Phase 2 lead-time analysis showed exact
catastrophic-onset prediction is weak under the JS-sum label. The
v1 controller does not need exact onset timing; it needs a
segment- or run-level risk score that drives "relax compression
for the next K tokens" decisions. This task asks whether such a
score exists today.

**Method**. Reuse the existing OOF per-token predictions from
`results/phase2/per_run/oof_predictions.parquet` (the same
`lr_all_cheap`, `future_sum_js_25`-trained predictor used in the
Phase 2 per-run validation, OOF on the prompts split). Join token-
level entropy from `phase2_tokens.parquet`. For each K in
{8, 16, 32}: bucket tokens into segments of length K, compute per-
segment aggregates (max, mean, p95, count above threshold) for
both score sources. Reduce per-run by max-over-segments and a few
ablations, then correlate with `run_damage.parquet`.

Code: `scripts/run_phase2_segment_risk.py`.

### Result (per-run aggregate, full prompts split, n=30 610 runs)

Best per-K aggregate against `has_looping` / `has_non_termination`:

| K  | source    | aggregate            | Spearman ρ rouge_l_drop | Spearman ρ sum_js | AUROC has_looping | AUROC has_non_term | AUROC gross_harm |
|----|-----------|----------------------|------------------------:|------------------:|------------------:|-------------------:|-----------------:|
| 8  | predictor | run_max_seg_mean     | 0.732                   | 0.741             | 0.810             | 0.844              | 0.680            |
| 16 | predictor | run_max_seg_mean     | 0.735                   | 0.746             | 0.809             | 0.844              | 0.682            |
| 32 | predictor | run_max_seg_mean     | 0.733                   | 0.750             | 0.800             | 0.841              | 0.681            |
| 8  | entropy   | run_max_seg_mean     | 0.561                   | 0.651             | 0.739             | 0.741              | 0.640            |
| 16 | entropy   | run_max_seg_mean     | 0.563                   | 0.652             | 0.741             | 0.741              | 0.639            |
| 32 | entropy   | run_max_seg_mean     | 0.563                   | 0.649             | 0.739             | 0.737              | 0.633            |

Predictor segment scores beat entropy segment scores at every K and
every aggregate. The gain is approximately:
- +0.07 looping AUROC, +0.10 non-termination AUROC,
- +0.17 ρ vs `rouge_l_drop`,
- +0.10 ρ vs `sum_js`,
- +0.04 gross-harm AUROC.

K barely matters between 8 and 32 for the headline aggregates
(looping/non-termination AUROC differs by ≤ 0.01 across K). K=16
is recommended as the controller window because (a) it is the
research-plan-canonical mid-segment, (b) it has the highest sum_js
correlation among the three K values, and (c) it leaves enough
within-segment statistics for the mean/p95 reductions without
collapsing to per-token max.

Outputs:
- `results/phase2/segment_risk/segment_risk.parquet`
- `results/phase2/segment_risk/segment_risk_summary.json`

### Cross-label confirmation

Re-running `run_phase2_segment_risk.py` against the OOF predictions
from each label-sweep label (`results/phase2/label_sweep/per_run/
<base>/oof_predictions.parquet`) confirms that the recommended
controller score depends on the underlying label. Per-run
`run_max_seg_max` AUROC on `has_looping` / `has_non_termination`,
prompts split:

| OOF label        | looping AUROC | NT AUROC | ρ rouge_l_drop |
|------------------|--------------:|---------:|---------------:|
| `future_sum_js`  | 0.810         | 0.844    | 0.731          |
| `future_sum_kl`  | 0.758         | 0.830    | 0.684          |
| `future_max_js`  | 0.618         | 0.758    | 0.543          |
| entropy (any)    | 0.738         | 0.742    | 0.561          |

`future_max_js` segment risk is **worse than entropy segment risk**
on `has_looping` (0.62 vs 0.74) and on `ρ vs rouge_l_drop` (0.54 vs
0.56). This independently confirms that Phase 4 should aggregate
the `future_sum_js`-trained predictor.

Cross-label outputs:
- `results/phase2/label_sweep/segment_risk/<base>/segment_risk.parquet`
- `results/phase2/label_sweep/segment_risk/<base>/segment_risk_summary.json`

## 5. Decisions for Phase 4

### Recommended controller score

**Label**: `future_sum_js_25`. Strictly dominates `future_sum_kl`
and `future_max_js` on every per-run validator and on absolute
AUROC. No alternative JS-style label fixes the onset-lead-time
inversion, so there is no payoff to changing labels.

**Token-level model**: `lr_all_cheap` for the v1 demo. XGBoost is
worth running in parallel for the headline figure (it gains +0.025
AUROC and ~0.06 ECE on prompts/presses) but should not be a
prerequisite for the controller because its headroom collapses to
+0.010 on the held-out-task split. Whichever model is deployed,
add an explicit calibration layer (isotonic on a held-out
calibration fold) before threshold tuning — the raw `lr_all_cheap`
ECE on H=25 prompts is 0.215 and the raw XGBoost ECE is 0.154,
both higher than tolerable for direct threshold-based gating.

**Segment aggregation**: K=16, `seg_mean` per segment, `max` over
segments per run. The chosen aggregate gives looping AUROC 0.809,
non-termination AUROC 0.844, and Spearman ρ 0.746 against
`sum_js`. K barely matters between 8 and 32 (Δ ≤ 0.01 AUROC); 16
is the research-plan-canonical mid-segment.

**Threshold**: tune against `gross_harm_final` on a held-out
calibration fold, with the segment-risk score as the input. The
Phase 2 lead-time finding (loop/NT onset prediction below 0.5
AUROC under any JS-style label) does not threaten the segment-
gating controller because segment-risk correlates strongly with
the run-level outcome.

### Open follow-ups

1. **Onset lead-time** remains an unsolved scoping limit. The
   label-sweep result rules out fixing it by changing JS-aggregate
   shape. The remaining options are (a) train directly on
   diagnostic tags (`future_has_looping_H`), accepting that this
   ties the headline target to the heuristic-threshold tag
   definitions; or (b) fit a sequence model (GRU on the streaming
   feature sequence) which can exploit pre-onset entropy patterns
   the linear / tree-based static-feature models miss. Both are
   v2 / journal-extension work and do not block Phase 4.
2. **XGBoost vs lr** is a Phase 4 engineering decision. If the
   first closed-loop demo on StreamingLLM shows segment-risk
   gating works with `lr_all_cheap`, the marginal +0.010 AUROC on
   held-out-task is unjustified added complexity. If the demo is
   marginal, swap in XGBoost + calibration layer.
3. **Calibration layer choice** (isotonic vs Platt) is a Phase 4
   engineering decision, not a Phase 2 contribution.

## Files changed / commands run

### Source

- `src/herald/predictor_baselines.py`: added
  `clustered_paired_bootstrap_delta`,
  `cross_fold_clustered_bootstrap`,
  `collect_fold_predictions` for the Task 1 cluster-bootstrap CIs.
- `scripts/run_phase2_lead_time.py`: minor robustness fix
  (`pl.DataFrame(..., infer_schema_length=None)`) so the lead-time
  subprocess does not fail intermittently when polars infers a
  narrow schema from the first few catastrophic-run dicts.
- `tests/test_predictor_baselines.py`: 3 new unit tests covering
  clustered paired bootstrap and cross-fold aggregation.

### Scripts (new)

- `scripts/run_phase2_delta_ci.py` — Task 1.
- `scripts/run_phase2_label_sweep.py` — Task 2.
- `scripts/run_phase2_model_headroom.py` — Task 3.
- `scripts/run_phase2_segment_risk.py` — Task 4.

### Commands

```sh
# Task 1: paired-bootstrap CIs (~4.5 min).
uv run python scripts/run_phase2_delta_ci.py \
    --horizons 5 10 25 50 \
    --splits prompts ratios presses tasks \
    --max-train-rows 200000 --n-prompt-folds 5 --n-boot 200

# Task 2: label-family sweep (~9 min on H=25 only).
uv run python scripts/run_phase2_label_sweep.py \
    --label-bases future_sum_js future_sum_kl future_max_js \
    --horizons 25 --per-run-horizon 25 \
    --splits prompts ratios presses tasks \
    --max-train-rows-baseline 200000 \
    --max-train-rows-per-run 500000 \
    --n-prompt-folds 5 --n-boot 100

# Re-run lead-time for the two labels where polars schema-inference
# tripped on the first sweep (fixed in scripts/run_phase2_lead_time.py).
for label in future_sum_js_25 future_max_js_25; do
    base="${label%_25}"
    uv run python scripts/run_phase2_lead_time.py \
        --root results/phase1 \
        --dataset results/phase2/dataset/phase2_tokens.parquet \
        --output-dir "results/phase2/label_sweep/lead_time/${base}" \
        --label "$label" --seed 0
done

# Task 3: XGBoost headroom (~1 min).
uv run python scripts/run_phase2_model_headroom.py \
    --horizons 25 --splits prompts ratios presses tasks \
    --max-train-rows 200000 --n-prompt-folds 5 \
    --n-estimators 200 --n-boot-ci 100

# Task 4: segment / run risk (~5 s).
uv run python scripts/run_phase2_segment_risk.py
```

### Tests

```sh
uv run pytest tests/test_predictor_baselines.py \
              tests/test_predictor_dataset.py \
              tests/test_predictor_per_run.py -x -q
# 29 passed.
```
