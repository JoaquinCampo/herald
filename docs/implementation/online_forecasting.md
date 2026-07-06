# Online damage forecasting from the post-switch stream

Status: ifeval pipeline settled (2026-07-06). This documents the
reframe from "predict damage before switching, transfer to unseen
compressors" to "forecast damage online from the model's own
reaction to the compressed cache, per compressor family", and the
deployable pipeline that resulted. Raw numbers and the full
experiment sequence live in
`results/predictor/experiments/experiment_log.md` (entries dated
2026-07-05 night through 2026-07-06).

## Why the reframe

The locked cross-compressor mission (worst-case savings >= 0.10 at
epsilon 0.01 across 3 held-out compressors) was exhausted at ~0.057
model-space, and both follow-up feature families confirmed the
blocker is compressor shift, not missing signal:

- Attention-reliance internals (SPOT-adapted, reference stream):
  no ceiling lift; collapse the knorm held-out cell
  (`feature_extension.md`, Outcome section).
- Post-switch hybrid-stream features: collapse the streaming_llm
  held-out cell at every seed; prompt-local studentization does not
  rescue transfer.

The same hybrid-stream features are strongly predictive WITHIN each
compressor (AUROC 0.65-0.76; oracle bounds 0.23-0.76, clearing the
0.10 rung on all three compressors without shift). The signal is
real; it does not transfer across compressor families. Deployment
does not require that transfer: the operator knows which compressor
they run. The claim becomes per-compressor.

## Data

Full per-token hybrid feature matrices exist for every ifeval cell
(93,692 (compressor, ratio, prompt, switch-point) runs, variable
length x 20 logit stats). Extracted first-16-step blocks plus the
trailing-8 pre-switch reference window (mean and std) into
`results/predictor/hybrid_streams_ifeval.npz`. All features are
deployable at inference: pre-switch history, plus the post-switch
stream's own stats; no reference continuation, no probe forward
pass, no compressor identity input. gsm8k and humaneval sweeps
predate stream capture; extending the claim to them requires
re-running those hybrid sweeps on Orion with capture enabled.

## The deployable pipeline (settled on ifeval)

Per compressor: XGB classifier (damaged = dq > 0), 5-fold
cross-fitted calibration, 3-seed score ensemble, point tau.

1. Split train prompts into 5 folds; each fold's rows are scored by
   a model fit on the other folds (out-of-fold, honest scores).
2. Average OOF scores over XGB seeds 0/1/2; same for the final
   model (fit on all train prompts).
3. Calibrate tau on the pooled OOF group-cost curve: largest
   savings with mean cost <= epsilon, point estimate. No bootstrap
   buffer: with OOF scores and ~590 calibration groups the point
   estimate transfers (a buffer only strands savings).
4. Variant selection (legacy features vs +hybrid-stream) is made on
   OOF savings, never on test.

Lessons locked in along the way:

- Calibrating tau on the model's own training rows is meaningless
  (overconfident scores; test cost landed 6-20x over budget).
- A dedicated calibration split wastes prompts twice; with only
  ~40-60 calibration prompts the tau is high-variance (budget
  violations in 20/36 cells point, still 4/36 at bootstrap-90).
- Cross-fitting fixes both at once: 17/18 then 18/18 cells in
  budget across seeds/methods, no explicit guarantee machinery
  needed.
- XGB model-seed variance is material at this data size (knorm
  deployable savings ranged 0.03-0.09 across single seeds);
  3-seed score ensembling collapses that axis and lifted
  expected_attention above every single seed.

## Results (ifeval, test prompts, budget epsilon = 0.01)

| compressor | selected variant (by OOF) | savings | cost |
| --- | --- | ---: | ---: |
| expected_attention | feat_only | 0.484 | -0.010 |
| knorm | feat+hyb_k16 | 0.058 | 0.005 |
| streaming_llm | feat_only | 0.227 | 0.002 |

All cells in budget (max observed cost 0.0091 across all 18
variant/method cells). OOF savings predicted test savings closely
(0.509 -> 0.484, 0.073 -> 0.058, 0.219 -> 0.227). For scale, the
best deployable cross-compressor worst-case from the exhaustion
report was 0.034.

The post-switch hybrid-stream features are the OOF-selected variant
on knorm (the hardest cell) and directionally positive there across
seeds (mean 0.053 vs 0.042), though inside seed noise; on the other
two compressors the pre-switch features suffice.

## TabFM update (2026-07-06)

Google TabFM 1.0.0 (zero-shot tabular FM, GPU-only in practice) run
through the same crossfit protocol unlocked the binding cell: knorm
0.1285 savings at 0.0091 cost, in budget with a frozen tau, 2.2x the
XGB pipeline and above the 0.10 deployable rung. Identical AUROC to
XGB; the gain is score shape near the threshold. TabFM calibrates
worse elsewhere (ea point overshoots budget, boot90 rescues;
streaming_llm over budget under every method), so the deployable
design is a per-compressor MIXED FLEET: XGB on expected_attention
(0.484) and streaming_llm (0.227), TabFM on knorm (0.128) ->
worst-case 0.128, all in budget. Scorer/variant selection must be
train-side and budget-aware (select by the OOF bootstrap bound, not
OOF savings alone). Full numbers in the experiment log
(tabfm_crossfit entry).

## Open items

- Scorer/variant/method selection rule needs formalizing
  (budget-aware OOF selection) and a locked-protocol confirmation
  run through `herald.controller_metrics` proper.
- Multi-task claim needs gsm8k/humaneval hybrid-stream capture on
  Orion (days of GPU; user decision).
- Detection latency k=16 post-switch tokens is the current operating
  point; the k-sweep showed signal from k=2. The rollback-cost
  accounting (k tokens regenerated on rejection) is not yet in the
  savings metric.
