# Phase 4 Pareto Pilot Results

Date: 2026-05-06.
Scope: 50 GSM8K prompts x 6 policies x greedy decoding x
`max_new_tokens=512` x `decoding_knorm` x `K=16` x `budgets={64,128,256}`.
Single-threshold pilot at `t=0.0639` (online p75 from
`results/phase4/controller_calibration_smoke/online_distribution.json`).

This is a pilot, not the publishable controller experiment. The
threshold and budget grid were locked before launch and never tuned
after seeing quality outcomes.

## Headline

HERALD beats `random_matched_budget` and `fixed_64` on ROUGE-L
versus the no_compression anchor with paired bootstrap 95% CI above
zero, but does not improve GSM8K exact-match accuracy: every aggressive
policy (`fixed_64`, `herald_t0.0639`, `random_t0.0639`) scores 0/50.
The action vocabulary `{64, 128, 256}` is too compressed for any
operating point in the grid to land on a useful Pareto frontier on
GSM8K + Qwen2.5-7B-Instruct.

Recommendation: rerun the pilot at `budgets={256, 512, 1024, 2048}`
(or `{128, 256, 512, 1024}`) where the cliff actually lives. The
predictor signal is real (HERALD beats random on ROUGE-L; CI excludes
zero), but the current grid sits far below where compression is a
trade-off rather than catastrophic.

## Per-policy summary (n=50 prompts)

| policy | acc | trunc% | tokens | wct/tok | evicted | ROUGE-L | ROUGE-L drop | mean_ret | max_ret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_compression | 0.881 | 2.0% | 302.9 | 0.01442s | 0 | 1.000 | 0.000 | 261.8 | 644 |
| fixed_256 | 0.410 | 10.0% | 324.7 | 0.01445s | 4494 | 0.868 | 0.132 | 218.0 | 256 |
| fixed_128 | 0.075 | 56.0% | 428.9 | 0.01448s | 10648 | 0.521 | 0.479 | 127.4 | 128 |
| **herald_t0.0639** | 0.000 | 96.0% | 508.1 | 0.01446s | 14547 | 0.360 | 0.640 | 69.2 | 160 |
| fixed_64 | 0.000 | 94.0% | 500.4 | 0.01447s | 13126 | 0.322 | 0.678 | 64.0 | 64 |
| random_t0.0639 | 0.000 | 96.0% | 504.5 | 0.01446s | 13133 | 0.322 | 0.678 | 67.3 | 128 |

Accuracy is computed over runs with parsable answers; counts of
unparsable / truncated runs are non-trivial in the aggressive cells
(fixed_64: 26 unscored, HERALD: 22 unscored).

Wall-clock per token is statistically indistinguishable across all
six policies (~14.4-14.5 ms/token; the predictor + controller add no
measurable overhead).

## Paired comparisons (bootstrap 95% CI on mean delta)

| a | b | metric | mean delta | 95% CI | n |
|---|---|---|---:|---|---:|
| HERALD | random_matched | ROUGE-L vs no_comp | **+0.0377** | [+0.0177, +0.0584] | 50 |
| HERALD | fixed_64 | ROUGE-L vs no_comp | **+0.0385** | [+0.0189, +0.0586] | 50 |
| HERALD | random_matched | task_score | 0.000 | [0.000, 0.000] | 18 |
| HERALD | fixed_64 | task_score | 0.000 | [0.000, 0.000] | 17 |
| HERALD | random_matched | total_evicted_tokens | **+1414** | [+1052, +1772] | 50 |
| HERALD | random_matched | wall_clock_per_token | 0.000 | [-0.0, +0.0] | 50 |
| HERALD | no_compression | task_score | -0.920 | [-1.000, -0.800] | 25 |
| HERALD | no_compression | ROUGE-L vs no_comp | -0.640 | [-0.658, -0.622] | 50 |

The first two rows are the load-bearing ones: HERALD beats both
fixed_64 and random_matched on lexical drift versus the
high-quality anchor, with paired CIs above zero. The win is small in
absolute terms (delta ROUGE-L ~0.04) but is not a tie.

The accuracy comparison is degenerate at this budget: 0/0 paired
runs in the parsable cells, so the predictor cannot show task-quality
recovery here.

## Decision-gate analysis

The pre-spec'd "promising" criteria were:

1. HERALD improves quality over fixed_64. **Partial.** ROUGE-L:
   yes (CI > 0). Accuracy: no (both 0).
2. HERALD reduces quality loss vs anchor while preserving
   compression savings. **No.** ROUGE-L recovery
   `(0.360 - 0.322) / (1.000 - 0.322) = 5.6%`, far below the 50%
   bar. Compute is *higher* than fixed_64 and random_matched, not
   lower.
3. HERALD beats random_matched at similar compute. **Yes on
   ROUGE-L** (CI > 0). However, HERALD evicted ~10% more tokens
   than random_matched on average (`+1414`, CI `[+1052, +1772]`),
   so the comparison is not strictly compute-matched even though
   the budget multiset is.

Why HERALD evicts more than random_matched at the same
budget multiset: the multiset is preserved by construction but the
*order* matters. `RiskBudgetStepPolicy(start_index=0)` opens at the
most aggressive budget and only steps up after a high-risk segment,
then steps back down on a low-risk one. This keeps the effective
cache small for longer than a uniform shuffle does, which produces
more decode-time evictions across the run.

## Why the headline result is weak

The action grid `{64, 128, 256}` puts HERALD on the wrong side of the
GSM8K cliff:

- `no_compression` (target=4096, never fires) reaches 88% accuracy.
- `fixed_256` already loses 47 points of accuracy (88% to 41%) and
  47 points of ROUGE-L (1.0 to 0.87).
- `fixed_128` collapses to 7.5%.
- `fixed_64` and everything that relaxes only as far as 64 collapse
  to 0%.

HERALD's two relax steps end at budget 256 in the worst case, which
the data shows is already past the cliff for many prompts. Truncation
is the dominant failure mode at 96% for HERALD and 94% for fixed_64:
the model rambles past 512 tokens without producing a final answer
and gets `stop_reason=max_tokens`. Once truncation has happened, no
amount of late budget relaxation can recover the answer.

The Phase 2 lead-time finding also shows up here: the JS-trained
predictor scores collapsed/looping runs LOW, so it fires *less* on
the most damaged segments, not more. The few HERALD relax steps
(mean 3.8 per run) tend to be at the right tail of well-behaved
runs, not at the moments when the model is starting to derail.

## What this pilot DOES establish

- The infrastructure works end-to-end on Orion: feasibility,
  predictor export, online feature parity, controller smoke,
  matched-random replay, run/segment/event parquet emission, and
  ROUGE-L analysis with paired bootstrap CIs.
- HERALD trajectories are detectably different from `random_matched`
  in evictions and ROUGE-L (CIs above zero, n=50).
- The predictor is not free of signal at this threshold even though
  the budget grid is in the wrong regime: the +0.04 ROUGE-L lift
  over both `fixed_64` and `random_matched` is real, just small.
- `no_compression` (target=4096) behaves correctly as the anchor:
  press fires every 16 tokens but evicts 0 across all 50 runs.

## What this pilot does NOT claim

- No claim that HERALD recovers compression-induced quality loss in
  any meaningful way at this grid.
- No claim of beating fixed_64 on accuracy. They tie at 0.
- No claim of perfect matched-compute parity with random_matched
  (HERALD evicts ~10% more, even at identical budget multiset).
- No quality calibration of the threshold. Threshold was locked at
  0.0639 (online p75) before the pilot launched and is not tuned
  post-hoc.
- No GSM8K saturation diagnostics: 13-26 runs per aggressive cell
  are unparsable, so per-cell accuracy point estimates are noisy
  even before sampling variance.

## Recommended next experiment

Rerun the same pilot, same threshold philosophy (online p75 from a
new short calibration smoke), at `budgets={256, 512, 1024, 2048}`
with `start_index=2` (HERALD opens at budget=1024). This positions
the action grid near and above the GSM8K cliff (~256-1024 from this
pilot), so:

- `fixed_1024` and `fixed_2048` should match `no_compression`
  closely on accuracy.
- `fixed_256` provides the same low-quality anchor we already have.
- HERALD has room both to relax AND to tighten meaningfully.
- The 50% recovery criterion has somewhere to live.

Secondary recommendations:

- Increase `max_new_tokens` to 768 or 1024 for the next pilot, so
  truncation stops dominating the failure mode for non-aggressive
  policies.
- Recalibrate the threshold on a fresh fixed-budget smoke at the
  new grid: the segment-score distribution under `fixed_1024` will
  differ from the distribution under `fixed_64` that produced
  0.0639.
- Keep `max_new_tokens=512` at 50 prompts as a fast feedback loop;
  bump to 100 prompts only after the new grid produces a meaningful
  Pareto delta.
- Consider a `RiskAIMD` policy variant in addition to
  `RiskBudgetStep`. AIMD's multiplicative-decrease may use HERALD's
  per-segment max-risk signal more aggressively than the symmetric
  `+/-1` step does.

## Go / no-go

**Go to Phase 4 v1 publishable run: not yet.** The pilot shows the
plumbing works and that HERALD extracts a small but real signal at
the load-bearing comparison (vs random_matched at matched action
multiset), but the action grid does not allow a useful Pareto
operating point on this task. The next pilot at
`budgets={256, 512, 1024, 2048}` is a precondition for the
publishable run.

**Don't tune threshold post-hoc on this pilot.** The threshold pick
was honest; the bug is in the action vocabulary, not the threshold.
Re-pick the threshold on a fresh calibration smoke at the new grid.

**Strong enough to scale beyond GSM8K? No, not yet.** Single-task
ROUGE-L lift of +0.04 on a degraded grid with no accuracy gain is
not a robust enough signal to justify a multi-task expansion. Move
to the better grid first; only consider HumanEval / IFEval / Qasper
once the controller produces a non-trivial Pareto delta on GSM8K.

**The matched-compute claim is structurally weak.**
`RiskBudgetStepPolicy(start_index=0)` produces a monotone
oscillation pattern (`64 -> 128 -> 64 -> 128`) that evicts on every
down-transition, while a random shuffle of the same multiset
produces fewer monotone-down transitions. Switching `start_index`
inverts the asymmetry but does not eliminate it. For the next
pilot, either (a) instrument compute-equalized ROUGE-L (per unit
eviction or per retained-cache-second), or (b) constrain the
random_matched seed-set so its eviction distribution matches
HERALD's, before claiming "matched compute".

## Artifacts

- `results/phase4/pareto_pilot/controller_runs.parquet` (300 rows)
- `results/phase4/pareto_pilot/controller_segments.parquet` (7970 rows)
- `results/phase4/pareto_pilot/controller_events.parquet` (217 980 rows)
- `results/phase4/pareto_pilot/pilot_manifest.json`
- `results/phase4/pareto_pilot/pareto_summary.json`
- `results/phase4/pareto_pilot/run_log.txt` (full Orion stdout)
- `results/phase4/pareto_pilot_smoke/` (2-prompt smoke artifacts)

Predictor: `models/phase4_lr_all_cheap.json` sha256
`fa0fcda44f410cda51241a9c7a4e8253211c31fb1a9612edf8c4abf25d65b575`
(matches local + Orion). kvpress 0.5.3, torch 2.10.0+cu128.

## Files changed

- `src/herald/phase4_controller.py`: `SegmentEntry` gained
  `segment_score_max`; `attach_cache_size_per_segment` now emits
  `start_token_pos`, `end_token_pos`, `segment_score_max`, and
  `evicted_tokens_in_segment` per segment; `SEGMENT_COLUMNS` updated.
- `tests/test_phase4_controller.py`: added two tests for the new
  fields. 18 phase4_controller tests pass.
- `scripts/run_phase4_pareto_pilot.py` (new): Orion runner.
- `scripts/analyze_phase4_pareto_pilot.py` (new): Mac analyzer with
  ROUGE-L + paired bootstrap.

## Commands run

Sync (Mac -> Orion, no `--delete`, sync only paths actually changed):

```sh
rsync -av --exclude='__pycache__' --exclude='.mypy_cache' \
  src/herald/phase4_controller.py \
  orion:/clustergpu/home/jcampo/herald/src/herald/

rsync -av --exclude='__pycache__' --exclude='.mypy_cache' \
  scripts/run_phase4_pareto_pilot.py \
  scripts/analyze_phase4_pareto_pilot.py \
  orion:/clustergpu/home/jcampo/herald/scripts/

rsync -av --exclude='__pycache__' --exclude='.mypy_cache' \
  tests/test_phase4_controller.py \
  orion:/clustergpu/home/jcampo/herald/tests/
```

2-prompt smoke (Orion):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/run_phase4_pareto_pilot.py \
    --num-prompts 2 --max-new-tokens 512 \
    --output-dir results/phase4/pareto_pilot_smoke
```

50-prompt pilot (Orion, ~31 min on RTX 5090):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  nohup .venv/bin/python scripts/run_phase4_pareto_pilot.py \
    --num-prompts 50 --max-new-tokens 512 \
    --output-dir results/phase4/pareto_pilot \
    > logs/pareto_pilot_50.log 2>&1 &
```

Pull artifacts and analyze (Mac):

```sh
rsync -av orion:/clustergpu/home/jcampo/herald/results/phase4/pareto_pilot/ \
  results/phase4/pareto_pilot/

.venv/bin/python scripts/analyze_phase4_pareto_pilot.py \
  --input-dir results/phase4/pareto_pilot
```
