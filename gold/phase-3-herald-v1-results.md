# HERALD v1 Results

Final eval report for HERALD v1: per-token regressor on
`future_sum_js_H` plus per-run trajectory-shape wrapper. All
numbers are full-data, 5-fold OOF (6.93M tokens, 29,240 runs,
GroupKFold by `prompt_id`) with 500-iter cluster-bootstrap CIs by
`prompt_id`. Counts in trajectory features are normalised by
`n_tokens` to prevent implicit run-length leakage (see V1→V2 note
below).

## Headline contract (per `/goal`)

| # | metric | bar | substrate ceiling | achieved (95% CI) | pass |
|---|---|---|---|---|---|
| 1 | per-token ρ vs `future_sum_js_25` | ≥ 0.70 | 1.00 | 0.7540 [0.7480, 0.7595] | ✓ |
| 2 | per-token ρ vs `future_sum_js_50` (cross-horizon) | ≥ 0.65 | 1.00 | 0.7658 [0.7595, 0.7713] | ✓ |
| 3 | per-run ρ vs `rouge_l_drop` (trajectory wrapper) | ≥ 0.85 | 0.78 | 0.8285 [0.8208, 0.8343] | ✗ on bar, beats substrate ceiling by +0.05 |
| 4 | per-run ρ vs `run_damage.sum_js` (trajectory wrapper) | ≥ 0.85 | 0.90 | 0.8643 [0.8590, 0.8688] | ✓ |
| 5 | cross-press retention (held-out press, `sum_js`) | ≥ 0.95 | n/a | 0.998 (min 0.863 / overall 0.864) | ✓ |
| 6 | cross-ratio retention (held-out ratio, `sum_js`) | ≥ 0.95 | n/a | 0.814 (min 0.666 / overall 0.818) | ✗ |
| 7 | cross-task retention (held-out task, `sum_js`) | ≥ 0.90 | n/a | 0.897 (min 0.726 / overall 0.809) | ✗ marginal |
| 8 | per-run ECE (10-bin quantile, wrapper vs `sum_js`) | < 0.05 | n/a | 0.036 | ✓ |
| 9 | online O(1) per-token (streaming parity at tol=5e-4) | exact | n/a | abs-diff = 0.0 | ✓ |

Final tally: **6 of 9 bars cleared**, 3 missed.

### Per-run wrapper, all targets (V2 normalised counts, LOPO press)

| target | ρ_wrapper (95% CI) | ρ_baseline (pred_max) | ECE | min-press retention |
|---|---|---|---|---|
| `sum_js` | 0.8643 [0.8590, 0.8688] | 0.7316 | 0.036 | 0.998 |
| `rouge_l_drop` | 0.8285 [0.8208, 0.8343] | 0.7578 | 0.045 | 0.944 |
| `sum_kl` | 0.8209 [0.8147, 0.8259] | 0.7467 | 0.042 | 0.974 |
| `char_edit_ratio` | 0.8193 [0.8106, 0.8261] | 0.7529 | 0.048 | 0.922 |

The wrapper lifts every target by 0.06–0.13 over the per-token
`pred_max` baseline. ECE is under 0.05 for three of four targets
and 0.048 for the fourth.

## Bars 3, 6, 7 (the misses)

**Bar 3 — `rouge_l_drop ≥ 0.85`.** Substrate ceiling is 0.78 (max
of true label, see `gold/phase-3-substrate-ceiling-rich.md`). The
wrapper at 0.829 exceeds this by 0.05 because it learns
multi-feature trajectory shape, not a single aggregation. The
0.85 bar is reachable only by retraining the per-token regressor
on `rouge_l_drop` directly, abandoning the substrate-honest
framing. We do not pursue that path: `rouge_l_drop` is reported
as a noisy extrinsic validator with its substrate ceiling stated.

**Bar 6 — cross-ratio retention ≥ 0.95 (`sum_js`).** Held-out
0.375 fold ρ = 0.666 vs overall 0.818, retention 0.814. The
two heaviest-compression ratios (0.25, 0.375) are
qualitatively distinct from the rest: the failure modes shift
from gradual KV-degradation to abrupt collapse, and the
predictor trained on lighter ratios is OOD for these. Lighter
held-out ratios (0.875, 0.9375, 0.96875) all retain >0.99.

Per-token feature lever empirically exhausted via three
independent attacks under the oracle regime (per-token AND
wrapper both `GroupKFold(prompt_id)`, every ratio
in-distribution). (i) Per-ratio per-token retraining (7
separate HGB, one per ratio, 5-fold prompt_group within each)
lifts worst-slice per-token ρ by +0.006 from 0.530 to 0.535,
oracle wrapper retention 0.795 → 0.800. (ii) Extended-feature
joint training (+20 causal extras: `delta_h` + rolling/EWMA
variants, std-rollings for every base signal, `avg_logp`,
`output_length_so_far`) lifts worst-slice per-token ρ by
+0.006 to 0.536, oracle wrapper retention 0.7998. (iii)
Slope-feature attack (+18 features: 8 closed-form window-OLS
slopes on entropy/kl_div/delta_h/top1_prob over windows
{8, 32}, 4 ratio×signal interactions, 6 press one-hot dummies),
joint GroupKFold(prompt_id) training, lifts worst-slice
per-token ρ to 0.632 at ratio=0.25 (0.635 at ratio=0.375),
pooled ρ=0.781. This is the largest single lift (+0.10 over
canonical at the worst slice) but still misses the
pre-committed kill threshold of 0.65, and is far below the
~0.84 needed to clear the 0.95 retention bar. All three
routes converge: per-token signal at heavy compression
intrinsically lacks discrimination; trajectory shape, ratio
interactions, and press identity together are insufficient
to push past ρ ≈ 0.63 at ratio=0.25. Closing the gap
requires qualitatively new per-token substrate (semantic
anchors, compression-pressure terms, press-conditional
signals), not training-regime or feature-shelf engineering.
Diagnostic artefacts: `results/phase3/oracle_reachability_bar6.json`,
`results/phase3/oracle_reachability_bar6_per_ratio.json`,
`results/phase3/oracle_reachability_bar6_extfeat.json`,
`results/phase3/per_token_per_ratio_slopefeat.json`,
`results/phase3/preds/prompt_group_slopefeat__h25.parquet`.

Additional structural confound on the retention metric: per-slice
substrate ceilings (`y_max` univariate) span 0.665 (ratio 0.97) to
0.858 (ratio 0.375), a 0.20 spread. Retention = worst/overall
implicitly assumes uniform ceilings; the data falsifies that
assumption, so even a model that hits every slice's own ceiling
fails this bar as stated.

**Bar 7 — cross-task retention ≥ 0.90 (`sum_js`).** Held-out
ifeval ρ = 0.726 vs overall 0.809, retention 0.897, 0.003 below
bar. ifeval prompts have a constrained-format failure mode
(skipping format constraints) that does not show up in the JS
divergence trajectory the way other tasks’ degradations do.

## V1 → V2: implicit run-length leakage caught

V1 of the trajectory wrapper reported sum_js = 0.889 and
rouge_l_drop = 0.836. Those numbers smuggled in `n_tokens` as a
proxy: three of the 17 features were counts (`pred_above_p90`,
`pred_longest_run_above_p75`, `pred_count_local_maxima`) which
scale with run length. With `n_tokens` explicitly dropped per
the no-leakage protocol but counts left unnormalised, run length
leaked back in implicitly. V2 normalises each count to a rate
(divide by `n_tokens`), so all 17 features are scale-invariant
to run length. The corrected numbers above are sum_js 0.864
and rouge_l_drop 0.829, a drop of 0.025 and 0.007 respectively.
The 0.025 drop on sum_js is the size of the implicit leakage.

The V2 numbers still clear the `sum_js ≥ 0.85` bar comfortably.

## No-leakage protocol

We require (a) the strict pred-aggregates-only wrapper to clear
the bar, AND (b) it to beat the meta-only wrapper (press +
ratio + task + n_tokens, no pred features) by ≥ 0.05.

- Meta-only baseline (sum_js): 0.811
- Strict pred-aggregates-only V2 (sum_js): 0.864
- Margin: +0.053 → clears the 0.05 margin requirement.

The /goal pass on bar 4 is honest: the result is not driven by
metadata, it is driven by pred-trajectory shape.

## Substrate ceilings (per-run, max aggregation of true label)

From `gold/phase-3-substrate-ceiling-rich.md`. Reported here so
the wrapper numbers can be read against them.

| target | bar | best ceiling (max-agg) | best ceiling (any rich-agg) |
|---|---|---|---|
| `sum_js` | 0.85 | 0.899 | 0.998 (AUC, telescoping) |
| `sum_kl` | 0.85 | 0.921 | 0.997 (AUC, telescoping) |
| `rouge_l_drop` | 0.85 | 0.781 | 0.781 |
| `char_edit_ratio` | 0.75 | 0.776 | 0.776 |

`rouge_l_drop` and `char_edit_ratio` have hard ceilings below
the 0.85 bar; `sum_js` and `sum_kl` are reachable in principle
and the wrapper closes the gap to within ~0.04 of the
non-telescoping max-agg ceiling.

## Methodology

- **Substrate**: `results/phase2/dataset/phase2_tokens.parquet`
  (32,088 runs × ~240 tokens, 6.9M tokens after H=25 null-drop).
- **Per-token model**: scikit-learn
  `HistGradientBoostingRegressor`, `loss="squared_error"`,
  `max_iter=400`, `learning_rate=0.05`, `max_depth=8`,
  `min_samples_leaf=200`, `l2_regularization=1.0`,
  `early_stopping=True`, `validation_fraction=0.1`,
  `n_iter_no_change=20`.
- **Target transform**: `y = np.log1p(future_sum_js_H)`.
- **Features (per-token)**: `CHEAP_ALL_FEATURES` (24 columns: 9
  Tier-0 + 12 rolling/EWMA + position + ratio).
- **Per-run wrapper**: HGB, `learning_rate=0.05`, `max_iter=400`,
  `max_depth=4`, `min_samples_leaf=20`, `l2=1.0`. 17 features
  (8 strict pred-aggregates + 9 trajectory-shape, counts
  normalised by `n_tokens`).
- **Splits**:
  1. `prompt_group`: 5-fold `GroupKFold(prompt_id)`, headline.
  2. `loo_press`: leave-one-press-out (6 folds).
  3. `loo_ratio`: leave-one-ratio-out (7 folds).
  4. `loo_task`: leave-one-task-out (4 folds).
- **Bootstrap CIs**: cluster-bootstrap by `prompt_id`,
  `n_boot=500`.
- **Per-run aggregation**: trajectory wrapper (17 pred features,
  no metadata).

## Streaming demo

`src/herald/herald_v1_streaming.py` provides
`StreamingHeraldRegressor`, an O(1) per-token wrapper around
`OnlineFeatureState`. Parity test (vs batched predictions on 3
held-out runs) at tol=5e-4: max abs-diff = 0.0, exact match.

## §6.5 decision

Keep commits c7d0d7b, d707858 as slice diagnostics, do not
revert, do not promote to headline. Full argument:
`gold/phase-3-section-65-decision.md`.

## Paper framing

Six of nine bars cleared. Bar 4 (`sum_js ≥ 0.85`) is the
headline; bars 1, 2, 5, 8, 9 are clean supporting evidence.
Bars 3, 6, 7 are reported transparently with their structural
or distributional explanation. Per
`gold/contribution-validation.md`, this is the "solid measurement
+ predictor" framing with an honest scorecard, not "groundbreaking
nine-for-nine".
