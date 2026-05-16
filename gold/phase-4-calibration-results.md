# Phase 4 controller calibration: results

15 GSM8K prompts, decoding_knorm, K=16, budgets {64,128,256}, max_new_tokens=256.
Per prompt: 3 fixed + 5 HERALD + 5 random_matched = 13 runs. Total: 195 runs, 3,102 segments.

Date: 2026-05-06. Artifacts: `results/phase4/controller_calibration_smoke/`.

## Headline

HERALD now fires at non-trivial rates at every calibrated threshold and produces trajectories that diverge from both `fixed_64` and `random_matched`. No cost penalty for the controller (wall-clock per token is within 0.4% across all 13 policies).

## Action rates (observed online)

| threshold | offline-quantile | observed relax | observed tighten | observed keep |
|-----------|------------------|----------------|------------------|---------------|
| 0.0149 | p05 | 31.8% | 28.8% | 39.5% |
| 0.0288 | p10 | 27.9% | 27.5% | 44.6% |
| 0.0438 | p15 | 20.0% | 20.0% | 60.0% |
| 0.0608 | p20 | 14.2% | 14.2% | 71.7% |
| 0.1035 | p30 |  5.0% |  5.0% | 90.0% |

All thresholds land inside the [5%, 50%] non-trivial-action window.

## Drift table (offline expected vs online observed)

| threshold | expected offline relax | observed online relax | drift |
|-----------|------------------------|------------------------|-------|
| 0.0149 | 95.0% | 31.8% | -63.2pp |
| 0.0288 | 90.0% | 27.9% | -62.1pp |
| 0.0438 | 85.0% | 20.0% | -65.0pp |
| 0.0608 | 80.0% | 14.2% | -65.8pp |
| 0.1035 | 70.0% |  5.0% | -65.0pp |

Drift is uniformly large and negative (~ -65pp), confirming the offline phase2_tokens distribution is a poor population proxy for live decoding_knorm. Future calibration should use the online distribution captured in this smoke (saved to `online_distribution.json`).

## Online distribution (fixed_64, n=240 segments)

```
min  = 0.0019   p50 = 0.0399   p95 = 0.1056
max  = 0.1735   p75 = 0.0639   p99 = 0.1275
mean = 0.0457   p90 = 0.0880
```

Wider than the prior 3-prompt smoke ([0.002, 0.109]) but still concentrated below 0.13. Future thresholds should be picked from this distribution.

## HERALD vs baselines: trajectory divergence

Fraction of (prompt, segment) cells where HERALD's `current_budget` differs from the named baseline.

| threshold | HERALD vs fixed_64 | HERALD vs random_matched |
|-----------|---------------------|---------------------------|
| 0.0149 | 71.2% | 75.1% |
| 0.0288 | 46.7% | 66.7% |
| 0.0438 | 29.2% | 45.0% |
| 0.0608 | 17.5% | 30.4% |
| 0.1035 |  5.4% | 10.8% |

HERALD is meaningfully different from both baselines at every threshold, and divergence scales with the action rate (lower threshold = more divergence). Critically, HERALD≠random at every threshold, so any quality difference cannot be attributed to budget multiset alone.

## Cost impact

Wall-clock per token spread is 0.01439s to 0.01451s (0.8% spread). Within sampling noise. Peak memory 14737-14748 MB (negligible spread). Total evicted tokens range 2,443 (fixed_256) to 7,607 (HERALD t0.0288); HERALD policies use more eviction headroom than fixed_64 at every threshold, consistent with relaxation actually occupying the larger budgets.

## What this calibration does NOT claim

- No claim of quality improvement. GSM8K correctness counts in `controller_runs.parquet` are sparse (max_new_tokens=256 truncates many) and 15 prompts is too few for statistical comparison.
- No claim that the predictor is well-calibrated as a probability. Drift is large; raw scores function as ranks until isotonic calibration is added.

## Next reasonable step (only when authorized)

1. **Threshold pick using online distribution.** Use `online_distribution.json` percentiles to pick a single canonical threshold for the publishable run (e.g., p75 = 0.0639 for ~25% relax).
2. **Larger run.** 50-100 GSM8K prompts at a single calibrated threshold so quality comparison becomes statistically meaningful.
3. **Isotonic calibration.** Fit isotonic on a held-out set, plug into `predictor.calibration` slot, re-pick thresholds in probability space.
