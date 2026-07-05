# Locked controller metrics (HERALD v1 target, v2 metric suite)

Status: LOCKED as of 2026-07-05. This file defines the metrics that
matter for the deployment use case. It was written after the MAE-gate
results (`result_report.md`) but before any model was evaluated under
these metrics; from this point on, models are optimized against this
target and the definitions here do not move. Numbers for the reference
policies live in `controller_metric_lock.json`.

## The decision being scored

During generation, a controller must pick the switch position: keep
the full KV cache up to token `s`, run compressed from `s` onwards.
Earlier switches save more memory and risk more damage. The dataset
samples `s` every 16 tokens per (model, task, prompt_id, compressor,
ratio) decision group, with measured `dq = q_ref - q_hybrid` per
position.

- Savings of switching at `s`: `1 - s / ref_len` (fraction of the
  generation run compressed). Never switching saves 0.
- Realized cost of a group: `dq` at the chosen position, 0 if the
  policy never switches.

`ref_len` and true `dq` are evaluation-side quantities only; the
forbidden-input policy for models is unchanged.

## Policy

Given a predictor score `predicted_dq` for every sampled position and
a threshold `tau`: switch at the EARLIEST sampled position whose
predicted dq is at most `tau`; if no position qualifies, never switch.

`tau` is selected TRAIN-ONLY per leave-one-compressor-out split: on
the split's train rows (the two other compressors), over a grid of
train-prediction quantiles, pick the tau with the highest mean train
savings subject to mean train realized cost <= epsilon; ties break to
the smaller tau. The held-out compressor never influences tau.

## Locked metrics

Splits, seed, and grouping are the canonical ones (seed 0, test group
fraction 0.25, leave-one-compressor-out, prompt groups disjoint).

1. **Savings at quality budget (primary).** epsilon = 0.01 mean
   realized dq per held-out test set (at most one point of quality on
   the 0-1 task scale, chosen by convention, not tuned). Report per
   held-out compressor: mean savings, mean realized cost, budget
   adherence (cost <= epsilon), and prompt-cluster bootstrap CIs (200
   resamples). The headline number is the WORST-CASE mean savings
   across the three held-out compressors among splits that respect
   the budget; a split that busts the budget scores 0 savings for the
   headline.
2. **Catastrophe recall at 10 percent FPR (safety).** Row-level
   recall of major damage (dq >= 0.5) at false-positive rate <= 0.10,
   ranking by predicted dq, per held-out compressor.

Nothing else is locked. MAE vs the mean baseline remains available as
a legacy diagnostic (`switch_baseline_lock.json`) but is no longer a
target: it can be improved by re-aggregation without any new
information, as demonstrated by the median-mix result.

## Reference policies (the bar)

Locked in `controller_metric_lock.json`:

- `never`: never switch (savings 0, cost 0).
- `always_s0`: always switch at s = 0 (max savings, uncontrolled
  cost).
- `oracle`: earliest position with true dq <= 0, never if none
  (deployment-infeasible upper reference).
- `locked_mean_baseline`: policy driven by the locked
  task_ratio_position_bucket grouped-mean predictor.
- `median_mean_mix`: policy driven by the median + 0.1 (mean - median)
  predictor (`src/herald/switch_predictor.py`).

## Success criteria for future predictors

A candidate beats this target when, evaluated through the unchanged
`herald.controller_metrics` evaluator on the canonical splits:

1. Budget respected on all 3 held-out compressors (mean realized cost
   <= epsilon).
2. Mean savings strictly above the best BUDGET-RESPECTING locked
   reference policy per compressor on all 3 held-out compressors. A
   reference that busts the budget on a compressor contributes 0
   savings to this bar there. (Clarified 2026-07-05, same day as the
   lock and before any candidate was scored: comparing a feasible
   policy against an infeasible policy's savings was unintended.)
3. Prompt-cluster bootstrap CI of the savings delta vs that best
   budget-respecting reference excludes zero on at least 2 of 3
   compressors.
4. Catastrophe recall at 10 percent FPR at least as high as the best
   locked reference on at least 2 of 3 compressors, and never lower
   by more than 0.05 on any.

Do not change epsilon, the tau-selection rule, the savings
definition, the splits, or the reference policies to make a model
pass.
