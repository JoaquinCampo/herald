# Phase 2 Results — Predictor Baselines

**Date**: 2026-05-05.
**Source dataset**: `results/phase2/dataset/phase2_tokens.parquet`
(7 738 974 rows from 32 088 compressed runs, 90 columns).
**Source spec**: `gold/phase-2-dataset.md` (post-audit revision).

## Headline

The Phase 2 multivariate cheap-feature predictor (`lr_all_cheap`,
24-feature logistic regression on Tier 0 + rolling/EWMA + position +
ratio) **beats the best cheap single-feature baseline
(`entropy_mean_8`) by ≥ 0.05 AUROC on every (split × horizon) cell**.

The pre-registered Phase 2 success criterion
(`gold/research-plan.md` Phase 2 Success: AUROC improvement ≥ 0.05
over best entropy/EWMA-style baseline) is met on the held-out-prompts
split as required, **and also passes on held-out ratios, presses, and
tasks** — the three transfer splits the contribution-validation
checklist marks as load-bearing for the black-box compressor claim.

| split | H=5 | H=10 | H=25 | H=50 | min Δ AUROC |
|---|---:|---:|---:|---:|---:|
| held-out prompts | +0.087 | +0.095 | +0.097 | +0.092 | **+0.087** |
| held-out ratios  | +0.054 | +0.060 | +0.065 | +0.067 | **+0.054** |
| held-out presses | +0.062 | +0.071 | +0.078 | +0.075 | **+0.062** |
| held-out tasks   | +0.073 | +0.083 | +0.086 | +0.080 | **+0.073** |

All deltas are mean across folds of `lr_all_cheap` minus
`feature::entropy_mean_8`, both evaluated on the same held-out
binary label thresholded at p90 of `future_sum_js_H` over the
training fold only.

## Best baseline per (split × horizon)

`lr_all_cheap` is the best baseline on every cell. Absolute AUROCs
of best vs best-cheap below.

| split | H | lr_all_cheap AUROC | entropy_mean_8 AUROC |
|---|---:|---:|---:|
| prompts | 5  | 0.852 | 0.765 |
| prompts | 10 | 0.871 | 0.777 |
| prompts | 25 | 0.889 | 0.793 |
| prompts | 50 | 0.892 | 0.800 |
| ratios  | 5  | 0.768 | 0.714 |
| ratios  | 10 | 0.790 | 0.730 |
| ratios  | 25 | 0.812 | 0.747 |
| ratios  | 50 | 0.824 | 0.756 |
| presses | 5  | 0.809 | 0.747 |
| presses | 10 | 0.827 | 0.756 |
| presses | 25 | 0.846 | 0.768 |
| presses | 50 | 0.849 | 0.774 |
| tasks   | 5  | 0.840 | 0.767 |
| tasks   | 10 | 0.860 | 0.776 |
| tasks   | 25 | 0.881 | 0.795 |
| tasks   | 50 | 0.885 | 0.805 |

## Tier 0 alone is not enough

`lr_tier0` (Tier 0 raw features only — no rolling, no EWMA, no
position/ratio metadata) **fails the 0.05 bar on every cell**:

| split | H | lr_tier0 Δ vs entropy_mean_8 |
|---|---:|---:|
| prompts | 5  | -0.026 |
| prompts | 10 | -0.031 |
| prompts | 25 | -0.035 |
| prompts | 50 | -0.034 |
| ratios  | 5  | -0.021 |
| ratios  | 50 | -0.027 |
| presses | 5  | -0.023 |
| presses | 50 | -0.031 |
| tasks   | 5  | -0.039 |
| tasks   | 50 | -0.048 |

Implication for the paper: the **rolling and EWMA features** (and
the position + ratio scalars in `lr_all_cheap`) carry the gain,
not the raw Tier 0 logit features by themselves. This is the
predictor-vs-feature-engineering distinction the
`gold/contribution-validation.md` "Required Feature-Information
Check" calls for: cheap *online* features carry information beyond
what raw single-token logit features alone provide.

## What the predictor is doing — caveats

Two findings shape how to scope the claim.

### (a) Per-run aggregation works (Task 4)

`scripts/run_phase2_per_run_validation.py` trains an OOF
`lr_all_cheap` on a 500k-row prompts split, aggregates the per-token
score per run (max, mean, p95, mean-top-10, count above 0.5), and
correlates the per-run aggregate with `run_damage.parquet`:

| score | Spearman ρ rouge_l_drop | Spearman ρ sum_js | AUROC has_looping | AUROC has_non_term | AUROC gross_harm |
|---|---:|---:|---:|---:|---:|
| **max** | **0.731** | **0.739** | **0.810** | **0.844** | 0.681 |
| p95 | 0.676 | 0.707 | 0.735 | 0.795 | 0.650 |
| n_above_0.5 | 0.690 | 0.766 | 0.762 | 0.813 | 0.656 |
| mean_top_10 | 0.653 | 0.676 | 0.706 | 0.768 | 0.609 |
| mean | 0.447 | 0.448 | 0.504 | 0.615 | 0.530 |

The token-level future-window label rolls up to per-run scores that
discriminate user-facing damage strongly: max-score AUROC 0.81 for
looping, 0.84 for non-termination, ρ 0.74 with `sum_js`, ρ 0.73 with
`rouge_l_drop`. Outcome harm (`gross_harm_final`) is harder
(AUROC 0.68) because the binary outcome is sparse and task-dependent.

### (b) Lead-time at the catastrophic onset moment is reversed (Task 5)

`scripts/run_phase2_lead_time.py` runs a relative-position AUROC
analysis on catastrophic (looping or non_termination) runs vs healthy
compressed controls in the same (task, press, ratio) stratum, with
the predictor trained on a disjoint pool. Result:

- Neither `entropy` nor `predictor_score` reaches the 0.65 AUROC
  threshold with persistence 5 in the pre-onset window
  `[-200, 0]` realized tokens.
- `entropy` AUROC hovers at 0.55 across the pre-onset window.
- **`predictor_score` AUROC is *below* 0.5 across the pre-onset
  window** (around 0.27–0.33), meaning catastrophic runs have
  *lower* predicted future-divergence scores than healthy
  controls at the relative positions immediately before onset.

Mechanistic reading: the JS-based `future_sum_js_25` label rewards
"high future divergence per token", which is what compression damage
typically looks like in healthy compressed runs (the model produces
divergent but readable text). Looping and non-termination collapse
into low-entropy, low-divergence repetitive output once the failure
mode begins, so the predictor (which keys on entropy + KL features)
*correctly* assigns them low future-divergence scores. The damage
is real and visible at the *run* level (Task 4 above) but not at the
*token-immediately-before-onset* level for these particular tags.

This means the per-token JS label is a poor early-warning signal
for the looping/non-termination failure modes specifically, even
though it predicts the compression-damage *magnitude* well at the
run level. A different label family (e.g. future top-1 unchanged
rate, or directly predicting the future tag) would likely be needed
to get pre-onset lead time on these specific failure modes. This is
a scope-limit finding worth keeping in the paper, not a bug.

## Cross-cell consistency

The `lr_all_cheap` improvement is consistent across all four splits
and all four horizons. The narrowest margins are on the held-out
ratios split (smallest gain at H=5, +0.054), which is consistent
with the research-plan worry that ratio is the single most
predictive feature and held-out ratio is the hardest interpolation
problem. Even there the 0.05 bar holds with margin.

## What was NOT done in this Phase 2 sweep

- **No XGBoost.** The contribution-validation rule says "do not
  train XGBoost until cheap baselines are summarized." With
  `lr_all_cheap` already passing on all cells, the XGBoost step is
  the next experiment, not a precondition for Phase 2 closure.
- **No bootstrap-CI on the headline deltas.** The runner emits
  per-fold AUROCs and per-fold bootstrap CIs around each baseline's
  AUROC, but the headline summary collapses across folds via mean.
  Paired-bootstrap CI on the cross-fold mean delta is a follow-up
  before the paper claim ships.
- **No multi-label sweep** beyond `future_sum_js`. `future_sum_kl_H`
  and `future_max_js_H` are present in the dataset but the runner
  was launched with `--label-bases future_sum_js` only. The
  per-feature direction (especially `future_max`) is a paper
  follow-up.
- **No GRU / sequence model.** Phase 2 plan says only run sequence
  models if the simpler model leaves obvious headroom on lead time
  or calibration. Lead-time finding (b) above is the place where
  headroom may exist; defer to a controlled follow-up.

## Pipeline reproduction

```sh
# Build dataset (160 s, 7.7M rows, 1.1 GB).
uv run python scripts/build_phase2_dataset.py \
    --root results/phase1 \
    --output-dir results/phase2/dataset

# Audit (CPU; ~1 min). Closes OPEN-1..7 in gold/phase-2-dataset.md.
uv run python scripts/phase2_audit.py \
    --root results/phase1 \
    --output-dir results/phase2/audit

# Baselines (730 s, 1760 result rows).
uv run python scripts/run_phase2_baselines.py \
    --horizons 5 10 25 50 \
    --splits prompts ratios presses tasks \
    --max-train-rows 200000 \
    --n-boot 100 \
    --output-dir results/phase2/baselines

# Per-run validation (8 s; OOF on prompts split).
uv run python scripts/run_phase2_per_run_validation.py \
    --label future_sum_js_25 \
    --max-train-rows 500000

# Lead-time analysis (~5 s).
uv run python scripts/run_phase2_lead_time.py \
    --label future_sum_js_25
```

## Outputs

- `results/phase2/audit/phase2_audit.json` (OPEN-1..7 audit)
- `results/phase2/dataset/phase2_tokens.parquet` (7.7M rows, 90 cols)
- `results/phase2/dataset/phase2_dataset_summary.json`
- `results/phase2/dataset/parts/press={p}__ratio={r}.parquet` (×42)
- `results/phase2/baselines/phase2_baseline_results.parquet`
- `results/phase2/baselines/phase2_baseline_summary.json`
- `results/phase2/per_run/oof_predictions.parquet`
- `results/phase2/per_run/per_run_scores.parquet`
- `results/phase2/per_run/per_run_validation.json`
- `results/phase2/lead_time/lead_time_by_feature.parquet`
- `results/phase2/lead_time/lead_time_summary.json`

## Decision

Phase 2 success criterion **passes** on all four splits and all four
horizons. The contribution validity bar is met for the predictor
contribution. Next pre-registered steps:

1. Add paired-bootstrap CIs on the cross-fold mean deltas to upgrade
   the "passes 0.05" decision to a CI-backed claim.
2. Train XGBoost / LightGBM as the spec-permitted next-tier model
   (compare against `lr_all_cheap`).
3. Phase 3 generalization: extend to additional model families.
4. Phase 4 closed-loop control demo.

## 2026-05-05 follow-up: Phase 2b

Steps 1 and 2 above are now complete. See `gold/phase-2b-results.md`
for the full record. Headlines:

- **Cluster-bootstrap CIs (run-level)** on the cross-fold mean
  `lr_all_cheap` minus `entropy_mean_8` delta exclude zero in
  **all 16 (split, horizon) cells**. The narrowest cell remains
  `ratios::H=5` (Δ 0.054, 95% CI [0.046, 0.062]).
- **XGBoost** lifts AUROC by +0.025 on prompts/presses, but only
  +0.016 on held-out ratios and +0.010 on held-out tasks. The
  improvement is real but borderline relative to the "materially
  > +0.02" gate; `lr_all_cheap` remains the recommended v1
  controller score.
- **Segment / run risk**: per-run aggregates of OOF predictor
  scores beat per-run aggregates of entropy by ~0.07 AUROC on
  catastrophic tags at every K in {8, 16, 32}. K=16 with
  `seg_mean` then run-level max is the recommended controller
  aggregate.
- **Label sweep** (`future_sum_kl`, `future_max_js`): see
  `gold/phase-2b-results.md` §2.
