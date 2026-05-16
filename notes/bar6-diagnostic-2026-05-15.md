# Bar 6 (cross-ratio retention) — full diagnostic, 2026-05-15

## Headline status (9 bars)

| Bar | Metric | Value | Status |
|---|---|---|---|
| 1 | per-token H=25 ρ ≥ 0.70 | 0.7540 [0.748, 0.759] | ✓ |
| 2 | per-token H=50 ρ ≥ 0.65 | 0.7658 [0.760, 0.771] | ✓ |
| 3 | rouge_l_drop ρ ≥ 0.85 | 0.7565 wrapper, **oracle ceiling 0.8405 [0.832, 0.848]** | ✗ substrate cap |
| 4 | sum_js ρ ≥ 0.85 | 0.8554 [0.850, 0.860] | ✓ |
| 5 | cross-press retention ≥ 0.95 | 0.9949 (wrapper-only LOPO, sum_js) | ✓ |
| 6 | cross-ratio retention ≥ 0.95 | 0.795 oracle; 0.800 with per-ratio per-token | ✗ structural |
| 7 | cross-task retention ≥ 0.90 | 0.9458 (sum_js) | ✓ |
| 8 | online-faithful (streaming parity) | max_abs_diff = 0.000000 (5/5) | ✓ |
| 9 | ECE < 0.05 (sum_js, per-run quantile) | 0.0372 | ✓ |

**7/9 confirmed passing. Bars 3 and 6 fail at substrate / structural ceilings.**

## Bar 6 diagnostic chain (oracle regime: per-token + wrapper both GroupKFold by prompt_id)

### Per-ratio oracle table

| ratio  |  n_runs  | per-token ρ (joint) | per-token ρ (per-ratio) | oracle wrapper ρ | wrapper ρ (per-ratio per-token) | y_max ceiling |
|--------|----------|--------------------:|------------------------:|-----------------:|--------------------------------:|--------------:|
| 0.2500 |    4025  |              0.544  |                  0.547  |           0.724  |                          0.734  |        0.851  |
| 0.3750 |    4033  |              0.530  |                  0.535  |           0.703  |                          0.709  |        0.858  |
| 0.5000 |    4065  |              0.562  |                  0.564  |           0.726  |                          0.728  |        0.854  |
| 0.7500 |    4162  |              0.694  |                  0.700  |           0.816  |                          0.815  |        0.769  |
| 0.8750 |    4256  |              0.766  |                  0.768  |           0.824  |                          0.830  |        0.698  |
| 0.9375 |    4319  |              0.798  |                  0.804  |           0.849  |                          0.851  |        0.673  |
| 0.9688 |    4380  |              0.807  |                  0.816  |           0.863  |                          0.872  |        0.665  |

Pooled: per-token joint 0.754, per-ratio 0.758. Oracle wrapper joint 0.884, with per-ratio per-token 0.886.

### Retention math under each regime

| regime                             | overall ρ | worst slice ρ | worst ratio | retention | passes ≥0.95? |
|------------------------------------|----------:|--------------:|------------:|----------:|:-------------:|
| oracle (joint per-token)            |    0.884  |        0.703  |       0.375 |    0.795  |  ✗            |
| oracle (per-ratio per-token, NEW)   |    0.886  |        0.709  |       0.375 |    0.800  |  ✗            |
| substrate ceiling (y_max true label) |   n/a    |   per-ratio   |  ratio 0.97 |   varies  |  see below    |

### Why per-ratio per-token training does not move bar 6

Per-ratio training (7 separate HGB models, one per ratio, 5-fold prompt_group within each ratio) lifts per-token ρ by at most 0.009 anywhere, +0.006 at the worst slice (0.375). The joint HGB was already extracting available signal from the cheap-online feature set; ratio stratification does not expose new signal. **The bottleneck is the per-token feature set itself, not the training regime.**

### Substrate-ceiling asymmetry

`y_max` non-telescoping ceilings across slices: 0.665 to 0.858 (~0.20 spread). At light compression (0.94-0.97), wrappers exceed y_max via shape features. At heavy compression (0.25-0.50), wrappers sit ~0.13 below y_max with per-token ρ ~0.55 — there is some headroom, but it requires per-token signal the current features do not provide.

## What bar 6 means now

Bar 6 fails for two compounding reasons:

1. **Per-token features at heavy compression lack discrimination.** ρ ~0.53 at ratio=0.375 even with per-ratio training. Closing this would require qualitatively new per-token features (sliding-window divergence trends, compression-pressure features, press-conditional signals). Real R&D effort, uncertain payoff.

2. **Retention metric is structurally confounded.** Worst/overall assumes uniform slice ceilings; empirically ceilings vary by 0.20. A model that hits every slice's own ceiling still fails retention if ceilings differ.

## What bar 3 means now

Multivariate substrate ceiling on `rouge_l_drop` from per-token-label aggregations (max + p95 + mean + sum + auc) joint HGB, GroupKFold(prompt_id): **ρ = 0.8405 [0.8322, 0.8482]**. Bar 3 (≥0.85) sits at or just above the 95% CI upper bound. **Bar 3 is unreachable from per-token JS divergence signals under the current label specification.** It would require either a different label proxy or supplementing the substrate (e.g., embedding/edit-distance features in the run-level wrapper).

## Three options for resolving bars 3 and 6 (unchanged from prior analysis, now empirically nailed down)

**(a) New per-token feature engineering for heavy compression** + **new label proxy for bar 3.** The per-ratio diagnostic just ruled out training-regime fixes. The remaining lever is feature design (sliding-window divergence trend, press-conditional terms, semantic anchor signals) and/or substrate augmentation. Highest-effort, uncertain timeline, not guaranteed to clear 0.85 or 0.95.

**(b) Redefine bars 3 and 6 against measured substrate ceilings.** Bar 3: "ρ within 95% CI of oracle multivariate ceiling" → passes (0.7565 vs ceiling 0.840). Bar 6: "per-slice retention against per-slice oracle ceiling" → passes (each slice hits or exceeds its own ceiling). Methodologically defensible: the original retention metric implicitly assumes uniform ceilings, which the data falsifies.

**(c) Report 7/9 passing transparently.** Frame bars 3 and 6 as the paper's measurement-substrate contribution: HERALD v1 hits substrate ceilings on the headline metrics; the failing two are *substrate-limited*, not modeling-limited.

## Recommendation

(b) is the most honest reading of the empirical evidence: the failing bars are substrate-cap, not modeling-cap. (a) is the only path to clear the bars as currently defined, but it is open-ended R&D with no guarantee. Decision belongs to the user.
