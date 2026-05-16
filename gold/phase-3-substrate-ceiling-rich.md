# HERALD v1 Rich Substrate Ceiling

Extends `gold/phase-3-substrate-ceiling.md` with five aggregations beyond max/p95/mean: sum, AUC, dwell above the global q90, dwell above the per-run q90 (constant, informational only and excluded here), and local-peak count. The question this answers: does a richer aggregator of the TRUE label raise the per-run ceiling vs `rouge_l_drop`, `sum_js`, and friends?

## Best ceiling per target across all (family, H, agg)

| target | best ρ | family | horizon | agg |
|---|---|---|---|---|
| `char_edit_ratio` | 0.7756 | `future_sum_js` | 5 | `y_max` |
| `rouge_l_drop` | 0.7806 | `future_sum_js` | 5 | `y_max` |
| `sum_js` | 0.9981 | `future_sum_js` | 5 | `y_auc` |
| `sum_kl` | 0.9969 | `future_sum_kl` | 5 | `y_auc` |

## Best ceiling per (target, agg)

| target | agg | best ρ | family | horizon |
|---|---|---|---|---|
| `char_edit_ratio` | `y_auc` | 0.7446 | `future_sum_js` | 5 |
| `char_edit_ratio` | `y_dwell_abs_q90` | 0.5405 | `future_sum_js` | 25 |
| `char_edit_ratio` | `y_max` | 0.7756 | `future_sum_js` | 5 |
| `char_edit_ratio` | `y_mean` | 0.4709 | `future_sum_kl` | 5 |
| `char_edit_ratio` | `y_p95` | 0.5438 | `future_sum_js` | 25 |
| `char_edit_ratio` | `y_peak_count` | 0.5841 | `future_sum_kl` | 5 |
| `char_edit_ratio` | `y_sum` | 0.7446 | `future_sum_js` | 5 |
| `rouge_l_drop` | `y_auc` | 0.7461 | `future_sum_js` | 5 |
| `rouge_l_drop` | `y_dwell_abs_q90` | 0.5521 | `future_sum_js` | 25 |
| `rouge_l_drop` | `y_max` | 0.7806 | `future_sum_js` | 5 |
| `rouge_l_drop` | `y_mean` | 0.4822 | `future_sum_kl` | 5 |
| `rouge_l_drop` | `y_p95` | 0.5511 | `future_sum_js` | 25 |
| `rouge_l_drop` | `y_peak_count` | 0.5801 | `future_sum_kl` | 5 |
| `rouge_l_drop` | `y_sum` | 0.7461 | `future_sum_js` | 5 |
| `sum_js` | `y_auc` | 0.9981 | `future_sum_js` | 5 |
| `sum_js` | `y_dwell_abs_q90` | 0.7341 | `future_sum_js` | 50 |
| `sum_js` | `y_max` | 0.8986 | `future_sum_js` | 50 |
| `sum_js` | `y_mean` | 0.6939 | `future_sum_js` | 50 |
| `sum_js` | `y_p95` | 0.8518 | `future_sum_js` | 50 |
| `sum_js` | `y_peak_count` | 0.7037 | `future_sum_kl` | 5 |
| `sum_js` | `y_sum` | 0.9981 | `future_sum_js` | 5 |
| `sum_kl` | `y_auc` | 0.9969 | `future_sum_kl` | 5 |
| `sum_kl` | `y_dwell_abs_q90` | 0.7768 | `future_sum_kl` | 50 |
| `sum_kl` | `y_max` | 0.9209 | `future_sum_kl` | 50 |
| `sum_kl` | `y_mean` | 0.7497 | `future_sum_kl` | 50 |
| `sum_kl` | `y_p95` | 0.8718 | `future_sum_kl` | 50 |
| `sum_kl` | `y_peak_count` | 0.6277 | `future_sum_kl` | 5 |
| `sum_kl` | `y_sum` | 0.9969 | `future_sum_kl` | 5 |

## Reachability of the /goal bars

| target | bar | best ceiling | reachable? |
|---|---|---|---|
| `sum_js` | 0.85 | 0.9981 | ✓ |
| `sum_kl` | 0.85 | 0.9969 | ✓ |
| `rouge_l_drop` | 0.85 | 0.7806 | ✗ |
| `char_edit_ratio` | 0.75 | 0.7756 | ✓ |

## Interpretation

If the rich-aggregator ceiling for `rouge_l_drop` is still below 0.85, then any per-token regressor (no matter how good, no matter how rich the run-level aggregator) is structurally bounded below the bar. That justifies the option-(c) framing in `gold/phase-3-herald-v1-results.md`.

If the ceiling lifts above 0.85, the bar is reachable in principle and we should explore the corresponding aggregator family in HERALD's run-level wrapper.
