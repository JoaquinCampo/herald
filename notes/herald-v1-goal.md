# HERALD v1 Goal

Build HERALD v1: per-token, online, press-agnostic regressor of compression damage over next H tokens. Ships in paper; NeurIPS "groundbreaking" tier. Read `gold/` and data first; pick model, features, recipe yourself. Do not bias toward prior attempts.

## Target (non-negotiable, `gold/research-plan.md:156-180`)
Predict paired counterfactual compression harm. Labels in `results/phase2/dataset/phase2_tokens.parquet`: `future_sum_js_H`, `future_sum_kl_H`, `future_max_js_H` for H in {5,10,25,50}. Per-token matched-prefix divergence vs uncompressed reference. NOT a looping/catastrophe classifier; binary tags are eval slices only.

## Headline Metrics
- per-token Spearman vs `future_sum_js_H` H=25: >= 0.70
- per-token Spearman vs `future_sum_js_H` H=50: >= 0.65
- per-run Spearman vs `rouge_l_drop`: >= 0.85
- per-run Spearman vs `run_damage.sum_js`: >= 0.85
- cross-press retention: >= 0.95
- cross-ratio retention: >= 0.95
- cross-task retention (gsm8k -> humaneval/ifeval/longbench): >= 0.90
- online-faithful: causal features only, O(1) per token
- calibration: ECE < 0.05 per-run quantile binning

Cross-model transfer is follow-up (orion GPU unavailable).

## Local Data
- `results/phase1/final/{tokens,replay}/` per-token features + paired KL/JS
- `results/phase1/final/runs.parquet` run metadata
- `results/phase1/metrics/*.parquet` run_damage, severity, tags, trajectory, alignment
- `results/phase2/dataset/phase2_tokens.parquet` canonical headline dataset
- `results/phase2_v2/segments_k16_ext.parquet` segment-level K=16
- `results/phase2_v2/onsets.parquet` onset table

## Prior Art (read, do not blindly extend)
- `gold/research-plan.md` headline contract, transfer slices, no-leakage protocol
- `gold/phase-2-dataset.md` token schema, splits, footguns (`nll_ratio` sign-flip)
- `gold/phase-2-results.md` `lr_all_cheap` on WRONG binary target; features only
- `gold/phase-2c-early-warning-results.md` lead-time inversion
- `gold/phase-2d-streaming-online-results.md` §6.5 binary loop predictor
- `gold/run-damage-table.md` per-run validator schema
- `PRINCIPLES.md` read first

## Open Questions
- Revert c7d0d7b + d707858 (§6.5 binary) or keep as slice diagnostic. Argue.
- Per-token (`phase2_tokens`) vs segment-level (`phase2_v2`) substrate.
- Splits: cluster bootstrap by `run_id`, GroupKFold by `prompt_id`, held-out press/ratio/task. No leakage on any transfer slice.
- Better label than `future_sum_js_H`? Justify before deviating.

## Constraints
CPU/MPS only (16GB RAM). orion GPU unavailable. Python 3.12+, no `__future__` annotations, line 78, ruff+mypy strict. `poe check` passes before commit. Never commit `results/`. Regression headline only; no diagnostic-tag training targets.

## Deliverables
1. Trained regressor + eval report on every headline metric (held-out prompts + every transfer slice) with cluster-bootstrap CIs.
2. Feature-class ablation (which fraction of Spearman each carries).
3. Decision + short writeup on §6.5 (ship as slice / revert).
4. Streaming demo: O(1) per-token damage forecast on live stream.
5. Calibration report (ECE per-run quantile binning, held-out folds).
