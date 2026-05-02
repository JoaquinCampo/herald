# Phase 0 Results

Sweep run: 2026-05-02, Orion (RTX 5090), Qwen2.5-7B-Instruct, fp16,
20 GSM8K prompts × {none, streaming_llm, snapkv} × {0.5, 0.875,
0.9375}, max_new_tokens=512, greedy, dense per-token replay.

## Completion

134 / 140 runs `replay_status = ok`. Six runs failed on
`gsm8k_1116` and `gsm8k_65` × snapkv × {0.5, 0.875, 0.9375} due to
kvpress' `SnapKVPress.score` asserting `q_len > window_size` with
the default `window_size=64` and the chat-templated prompt
tokenizing to exactly 64 tokens. Fixed in
`src/herald/experiment.py:get_press` by setting
`SNAPKV_WINDOW_SIZE = 32` (regression test in
`tests/test_experiment.py`). The fix is local to the press
constructor; we did not filter prompts (filtering would skew Phase 1
cell sizes). No other failure modes observed.

| press × ratio | n |
|--|--|
| none × 0.0 | 20 |
| streaming_llm × 0.5 / 0.875 / 0.9375 | 20 / 20 / 20 |
| snapkv × 0.5 / 0.875 / 0.9375 | 18 / 18 / 18 |

## Headline measurements

**Replay noise floor (gates 3 / 4).** On 20 baseline (no-press)
runs, `cell_js_max = 0.092`, `cell_js_median = 0.079`. The previous
`JS_NOISE_FLOOR = 1e-3` was an MPS plausibility floor and is wrong
for CUDA fp16 even on a perfectly deterministic baseline. Bumped to
`1e-1` (`tests/metrics/test_phase0_smoke.py`). One order above the
observed maximum, one order below saturation at `ln 2`.

**Per-token JS saturates.** Every compressed cell sits at
per-token JS ~ ln 2 ≈ 0.693 along the trajectory. Per-token JS
therefore cannot serve as a predictor target — there is no rank
structure to learn within a saturated cell. Trajectory-level
aggregates (`sum_kl`, `sum_js`, `nll_ratio`) do carry rank
structure (see Analysis #7).

**Alignment matrix (Spearman, bootstrap 95% CI on 134 runs).**

| pair | ρ | 95% CI |
|--|--|--|
| `rouge_l` ↔ `embedding_cosine` | 0.92 | [0.87, 0.95] |
| `sum_kl` ↔ `sum_js` | 0.91 | [0.86, 0.94] |
| `rouge_l` ↔ `edit_distance_ratio` | -0.90 | [-0.94, -0.84] |
| `embedding_cosine` ↔ `edit_distance_ratio` | -0.85 | [-0.90, -0.76] |
| `sum_kl` ↔ `nll_ratio` | -0.63 | [-0.77, -0.46] |
| `sum_kl` ↔ `rouge_l` | -0.34 | [-0.52, -0.15] |

Sequence-level metrics agree strongly with each other; trajectory
divergence aggregates correlate as expected with `nll_ratio`; the
trajectory-to-sequence link is real but noisy (ρ ≈ -0.34 for
`sum_kl` vs `rouge_l`), which is exactly the gap the predictor is
supposed to close.

**Outcome cells (compressed correctness vs paired baseline).**
`gross_harm` is 1.0 in five of six compressed cells; the only cell
with surviving correct generations is `streaming_llm @ 0.5`
(`gross_harm = 0.79`). `gross_help = 0` everywhere. The cliff lies
at or below ratio 0.5 on this slice.

**Sub-sampling rank correlations** (`results/phase0/metrics/sampling_rate_report.json`).
`trajectory_rank_corr` of every-N vs full per-token sequence:
0.967 (every-4), 0.953 (every-8). **Decision (committed for
Phase 1):** every-8. The 1.4 pp Spearman drop is below the
metric-to-outcome noise floor and halves Phase 1 replay cost. Per-
position `spearman_future_max_js` does drop more (≈0.70 → ≈0.56);
Phase 2 will revisit if the predictor target depends on per-
position fidelity. Recorded in `gold/research-plan.md` Phase 1.

## Analysis #7 — do trajectory aggregates discriminate ratios?

`results/phase0/analysis/ratio_discrimination.json`.

Mann-Whitney U (two-sided) on `sum_kl`, `sum_js`, `nll_ratio` for
each press across pairs of ratios.

`streaming_llm` (n=20 per ratio):
- `sum_kl` 0.5 vs 0.875: p = 1.2e-6.
- `sum_kl` 0.5 vs 0.9375: p = 0.064.
- `sum_kl` 0.875 vs 0.9375: p = 5.6e-4.
- Distribution is non-monotone: median `sum_kl` is 73.7 / 266.6 / 97.1
  for ratios 0.5 / 0.875 / 0.9375. Heavier compression at 0.9375
  produces *less* trajectory KL than 0.875 for streaming_llm. Likely
  artifact: at 0.9375 the model collapses faster into a low-entropy
  loop, accumulating less per-step KL over fewer effective tokens.
  Worth re-checking in Phase 1.

`snapkv` (n=18 per ratio):
- `sum_kl` 0.5 vs 0.9375: p = 0.048; 0.875 vs 0.9375: p = 0.81.
- `sum_js` 0.5 vs 0.875: p = 0.052; 0.875 vs 0.9375: p = 0.58.
- Monotone-ish (0.5 < 0.875 ≤ 0.9375 in median) but the upper two
  ratios do not separate.

Per-token JS is saturated; trajectory aggregates are not. The
methodological story holds — Phase 2 can target trajectory-level
quantities — but two caveats are real and need carrying forward:
(a) streaming_llm shows a non-monotone hump at 0.875; (b) snapkv
collapses both heavy ratios into one statistical bucket on this
small sample.

## Analysis #8 — cheap-features sanity predictor

`results/phase0/analysis/cheap_predictor_sanity.json`.

Tier 0 token features (`entropy`, `top1_prob`, `top5_prob`,
`h_alts`, `avg_logp`, `delta_h`, `kl_div`, `top10_jaccard`,
`eff_vocab_size`, `tail_mass`, `logit_range`) aggregated per run via
mean / max / p95 / EWMA(α=0.1). 44 features, 114 compressed runs.
Logistic regression with leave-one-prompt-out CV (20 folds, group =
`prompt_id`). Label = `rouge_l < median(rouge_l) = 0.171`. Median
split chosen because (a) `correct` is null on 36 / 114 runs and
True on only 3 / 114, both useless for binary classification; (b)
median split gives a balanced 57 / 57 base rate and lets the model
expose any signal at all.

| metric | OOF | chance |
|--|--|--|
| AUROC | 0.869 | 0.500 |
| AUPRC | 0.836 | 0.500 |

Cheap features carry a strong signal for trajectory damage even on
this tiny dataset. The methodological story holds.

## Phase 1 ratio grid recommendation

The current Phase 1 grid is `{0, 0.5, 0.75, 0.875, 0.9375, 0.96875}`.

Phase 0 evidence:
- `gross_harm = 1.0` on snapkv at every ratio (including 0.5);
- `gross_harm = 0.79` on streaming_llm at 0.5;
- `sum_kl` already saturates the rank ordering at 0.875 vs 0.9375
  for snapkv (p ≈ 0.81);
- the cliff is at or below ratio 0.5 on this slice.

**Recommendation: add 0.25 and 0.375 to the Phase 1 ratio grid.**
Final grid: `{0, 0.25, 0.375, 0.5, 0.75, 0.875, 0.9375, 0.96875}`.

Reasoning:
- The gentle-damage regime (where `gross_harm` is plausibly < 1
  and `rouge_l` is well above floor) lives below 0.5 and is
  unobserved. Without it, neither the sequence-level alignment nor
  any predictor can be trained on a balanced (damaged vs healthy)
  population.
- 0.25 / 0.375 cost two extra ratios on the same prompts; relative
  cost is small compared with the cost of a Phase 1 sweep that
  cannot resolve the cliff.
- The upper-end {0.96875} stays — Phase 0 did not measure it and we
  expect deeper saturation, but it is informative for the cliff-
  shape figure.

This is the single decision Phase 0 forces. If Phase 1 confirms
that the gentle regime exists, Phase 2's predictor target choice
(trajectory aggregates) is supported by both Analysis #7 and the
fact that the predictor will see balanced training data.

## Open carry-forward items

- Streaming_llm sum_kl non-monotone hump at 0.875: investigate in
  Phase 1 with a finer grid; could be measurement (replay vs
  collapsed loop) or real (different damage mechanisms by ratio).
- Snapkv 0.875 ↔ 0.9375 indistinguishability: revisit at N=200
  prompts; if it persists, snapkv's compression dimension may be
  effectively 1-D for ratios above 0.5, which is interesting on its
  own.
- Per-position sub-sampling fidelity (~0.56 at every-8) is well
  below the trajectory rank correlation; Phase 2 may need
  every-4 if per-position prediction targets are adopted, or
  anomaly-triggered dense replay.
- Controller alignment: Phase 0 does not answer whether a damaged
  trajectory can recover after compression pressure is lifted. Before
  committing Phase 2 to purely anticipatory prediction, run the
  pre-registered Phase 1 intervention probe in
  `gold/phase-1-intervention-probe.md`. The probe is deliberately
  separate from the fixed-ratio measurement headline.
