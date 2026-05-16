# Phase 4 controller calibration

## Goal

Pick threshold candidates for `RiskBudgetStepPolicy` from the offline
held-out segment-score distribution, then verify them on a small Orion
controller smoke (10-20 GSM8K prompts). No quality claims yet.

## Inputs

- Predictor: `models/phase4_lr_all_cheap.json` (lr_all_cheap on
  `future_sum_js_25` thresholded at training-fold p90).
- Tokens: `results/phase2/dataset/phase2_tokens.parquet`.
- Held-out fold: GroupKFold over `prompt_id`, `n_splits=5`, fold 0
  (152 prompts, 297,990 rows after `press == "knorm"` filter).

## Offline distribution (results/phase4/calibration/)

Fold0 holdout, K=16 non-overlapping segments, n=18,255 segments.

| stat | value |
|------|-------|
| min  | 0.0001 |
| p01  | 0.0010 |
| p05  | 0.0149 |
| p10  | 0.0288 |
| p25  | 0.0921 |
| p50  | 0.2413 |
| p75  | 0.5394 |
| p90  | 0.7793 |
| p95  | 0.8570 |
| p99  | 0.9445 |
| max  | 1.0000 |

## Offline / online overlay

The prior 3-prompt smoke produced 14 online segment scores in
`[0.002, 0.109]` under decoding_knorm. That range maps to:

| online | offline-quantile |
|--------|------------------|
| 0.002  | 0.0025 |
| 0.109  | 0.3098 |

Online distribution concentrates in the bottom 31% of the offline
distribution. **Quantiles at p50 and above never fire in practice.**
Calibration thresholds therefore use action-rate-target framing
(advisor-approved) rather than the nominal p50/p75/p90/p95/p99.

## Threshold candidates (action-rate targets)

Five thresholds spanning the prior online range:

| label | offline-quantile | threshold | offline relax-rate |
|-------|------------------|-----------|--------------------|
| t05   | p05 | 0.0149 | 95.0% |
| t10   | p10 | 0.0288 | 90.0% |
| t15   | p15 | 0.0438 | 85.0% |
| t20   | p20 | 0.0608 | 80.0% |
| t30   | p30 | 0.1035 | 70.0% |

These are the *expected* relax rates from the offline distribution;
the smoke records *observed* online relax rates and emits a drift
table.

## Known parity gaps

- `compression_ratio` (feature 24) is a constant per offline run but
  varies online as DecodingPress evicts. Offline distribution is a
  proxy, not a perfect twin.
- Offline scores come from runs at varied static compression ratios
  on different presses; online runs use decoding_knorm with K=16
  segment-level mutation.

## Smoke command

Tunnel up + offline mode:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/run_phase4_controller_smoke.py \
  --predictor models/phase4_lr_all_cheap.json \
  --num-prompts 15 \
  --max-new-tokens 256 \
  --budgets 64 128 256 \
  --k 16 \
  --thresholds 0.014907843902907615 0.028783428101760874 \
               0.043847324934322404 0.06081355852354169 \
               0.10352869553171441 \
  --offline-threshold-table results/phase4/calibration/threshold_table.json \
  --output-dir results/phase4/controller_calibration_smoke
```

Per prompt: 3 fixed + 5 herald + 5 random = 13 runs. Total: 195 runs
across 15 prompts, ~3-5 minutes per run on the previous smoke pace,
so budget ~12-16 GPU hours. If that's too long, drop `--num-prompts`
to 10.

## Acceptance criteria

1. Smoke completes without OOM / errors.
2. `online_distribution.json` reports >= 100 fixed_64 segments.
3. `drift_table.json` shows at least one threshold with observed
   relax-rate in `[0.05, 0.50]` (non-trivial action rate).
4. HERALD trajectories diverge from `fixed_64` at thresholds where
   relax rate is non-zero, AND from `random_t<T>` at the same threshold.

Quality claims are out of scope for this stage. Re-run after isotonic
calibration to make the thresholds probability-meaningful.
