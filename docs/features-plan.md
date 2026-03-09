# features.py — Feature Engineering Plan

`signals.py` extracts 16 raw per-token features. `features.py` will add rolling statistics and prepare the ML-ready dataset.

## Raw features from `TokenSignals`

| # | Feature | Type | Source |
|---|---------|------|--------|
| 1 | `entropy` | float | H_overall |
| 2 | `top1_prob` | float | top-1 probability |
| 3 | `top5_prob` | float | sum of top-5 probabilities |
| 4-8 | `logprob_0` .. `logprob_4` | float | top-5 log-probabilities (from `top5_logprobs`) |
| 9 | `h_alts` | float | entropy excluding top-1 |
| 10 | `avg_logp` | float | mean log-probability |
| 11 | `delta_h` | float | H(t) - H(t-1) |
| 12 | `delta_h_valid` | bool→float | whether delta_h is meaningful |
| 13 | `kl_div` | float | KL(p_t \|\| p_{t-1}) |
| 14 | `top10_jaccard` | float | Jaccard of top-10 token IDs vs prev step |
| 15 | `eff_vocab_size` | float | exp(entropy) |
| 16 | `tail_mass` | float | 1 - sum(top20_probs) |
| 17 | `logit_range` | float | max_logit - mean_logit |

## Rolling window features (8-token causal window)

For each of `entropy`, `top1_prob`, `h_alts`, `delta_h`, `kl_div`, `top10_jaccard`:
- `{name}_mean_8`: rolling mean over last 8 tokens
- `{name}_std_8`: rolling std over last 8 tokens

12 rolling features total.

## Positional feature

- `token_position`: absolute position in the generated sequence (0-indexed)

## Functions

- `flatten_signals(signals: list[TokenSignals]) -> list[dict]` — Unpack `TokenSignals` into flat feature dicts. Expand `top5_logprobs` list into `logprob_0`..`logprob_4`. Replace `None` temporal features (first token) with 0.0.
- `add_rolling_features(rows: list[dict]) -> list[dict]` — Compute 8-token causal rolling mean/std for the 6 target signals. Append to each row. First 7 tokens use partial windows.
- `build_dataset(results_dir: Path, ...) -> Dataset` — Full pipeline: load sweep JSONs → flatten_signals → add_rolling → label (from labeling.py) → return Dataset(X, y, traces, feature_names, ...).

## Estimated feature count

| Group | Count |
|-------|-------|
| Raw numeric | 17 |
| Rolling (6 × 2) | 12 |
| Positional | 1 |
| **Total** | **30** |

## Design decisions

- **No `compression_ratio`**: static per-trace constant, acts as label proxy (kvguard lesson — 60% feature importance, turns predictor into lookup table).
- **No `rep_count`**: concurrent with looping, not predictive. Spikes at/after onset, inflates AUROC for "forecast H tokens ahead" claim.
- **No `is_thinking_token`**: rank 31/40 in kvguard, negligible importance. Binary flag per-token adds noise, not signal.
- **No `rank_of_chosen`**: always 0 under greedy decoding (dead feature).
- **Top-5 logprobs instead of top-20**: individual logprobs beyond top-5 each contribute ~0.001 importance in kvguard. Top-5 captures the distribution shape; the rest is noise.
