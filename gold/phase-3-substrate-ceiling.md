# HERALD v1 Substrate Ceiling

Upper bound for per-run Spearman achievable by aggregating any of the `future_*_H` labels per run. The HERALD v1 model can never exceed these numbers, because they are computed from the TRUE label, not predictions.

All correlations are Spearman ρ between `aggregate_over_tokens_in_run(future_*_H)` and `run_damage[target]`. `y_sum` is excluded because it telescopes to ~H × `sum_js`, which is tautological.

## Best ceiling per (target, family)

| family | target | best H | best agg | ρ_ceiling |
|---|---|---|---|---|
| `future_max_js` | `char_edit_ratio` | 5 | `max` | 0.6204 |
| `future_sum_js` | `char_edit_ratio` | 5 | `max` | 0.7756 |
| `future_sum_kl` | `char_edit_ratio` | 10 | `max` | 0.7265 |
| `future_max_js` | `rouge_l_drop` | 5 | `max` | 0.6016 |
| `future_sum_js` | `rouge_l_drop` | 5 | `max` | 0.7806 |
| `future_sum_kl` | `rouge_l_drop` | 10 | `max` | 0.7273 |
| `future_max_js` | `sum_js` | 5 | `max` | 0.6574 |
| `future_sum_js` | `sum_js` | 50 | `max` | 0.8986 |
| `future_sum_kl` | `sum_js` | 50 | `max` | 0.8296 |
| `future_max_js` | `sum_kl` | 5 | `max` | 0.7531 |
| `future_sum_js` | `sum_kl` | 50 | `max` | 0.9108 |
| `future_sum_kl` | `sum_kl` | 50 | `max` | 0.9209 |

## Full matrix

Columns: family / target. Rows: H × agg.

| H | agg | future_sum_js<br>rouge_l_drop | future_sum_js<br>sum_js | future_sum_js<br>sum_kl | future_sum_js<br>char_edit_ratio | future_sum_kl<br>rouge_l_drop | future_sum_kl<br>sum_js | future_sum_kl<br>sum_kl | future_sum_kl<br>char_edit_ratio | future_max_js<br>rouge_l_drop | future_max_js<br>sum_js | future_max_js<br>sum_kl | future_max_js<br>char_edit_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | max | 0.781 | 0.820 | 0.868 | 0.776 | 0.720 | 0.779 | 0.875 | 0.723 | 0.602 | 0.657 | 0.753 | 0.620 |
| 5 | p95 | 0.506 | 0.685 | 0.736 | 0.497 | 0.465 | 0.635 | 0.741 | 0.460 | 0.313 | 0.478 | 0.570 | 0.318 |
| 5 | mean | 0.467 | 0.623 | 0.657 | 0.448 | 0.482 | 0.630 | 0.718 | 0.471 | 0.347 | 0.504 | 0.555 | 0.334 |
| 10 | max | 0.776 | 0.841 | 0.882 | 0.768 | 0.727 | 0.790 | 0.885 | 0.726 | 0.602 | 0.657 | 0.753 | 0.620 |
| 10 | p95 | 0.537 | 0.739 | 0.778 | 0.528 | 0.493 | 0.683 | 0.784 | 0.489 | 0.324 | 0.509 | 0.601 | 0.334 |
| 10 | mean | 0.443 | 0.621 | 0.650 | 0.427 | 0.455 | 0.626 | 0.711 | 0.446 | 0.298 | 0.467 | 0.523 | 0.290 |
| 25 | max | 0.747 | 0.872 | 0.900 | 0.737 | 0.712 | 0.808 | 0.902 | 0.708 | 0.602 | 0.657 | 0.753 | 0.620 |
| 25 | p95 | 0.551 | 0.804 | 0.824 | 0.544 | 0.512 | 0.748 | 0.837 | 0.510 | 0.357 | 0.561 | 0.646 | 0.375 |
| 25 | mean | 0.401 | 0.633 | 0.649 | 0.387 | 0.410 | 0.638 | 0.713 | 0.403 | 0.264 | 0.451 | 0.511 | 0.262 |
| 50 | max | 0.699 | 0.899 | 0.911 | 0.690 | 0.680 | 0.830 | 0.921 | 0.678 | 0.602 | 0.657 | 0.753 | 0.620 |
| 50 | p95 | 0.522 | 0.852 | 0.848 | 0.519 | 0.488 | 0.798 | 0.872 | 0.491 | 0.366 | 0.573 | 0.656 | 0.383 |
| 50 | mean | 0.365 | 0.694 | 0.684 | 0.356 | 0.371 | 0.692 | 0.750 | 0.371 | 0.264 | 0.468 | 0.528 | 0.265 |

## Implications for HERALD v1 headline bars

The /goal spec sets per-run Spearman bars at ≥ 0.85 vs `rouge_l_drop` and `sum_js`. These ceilings tell us which are reachable:

| target | bar | best ceiling | reachable? |
|---|---|---|---|
| `sum_js` | 0.85 | 0.8986 | ✓ |
| `sum_kl` | 0.85 | 0.9209 | ✓ |
| `rouge_l_drop` | 0.85 | 0.7806 | ✗ |
| `char_edit_ratio` | 0.75 | 0.7756 | ✓ |

## Strategic note

The honest framing: `future_sum_js_H` and `future_sum_kl_H` are process metrics (trajectory divergence accumulated over the next H tokens). `rouge_l_drop` is an end-of-sequence quality delta. A per-token regressor over the JS/KL labels cannot exceed the substrate-ceiling Spearman against `rouge_l_drop`, regardless of model capacity. Closing that gap requires either a different label, a run-level wrapper stage, or accepting that `rouge_l_drop` is reported as an extrinsic validator below the 0.85 bar.
