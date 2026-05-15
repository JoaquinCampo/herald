# Phase 2d Results: Streaming Online Predictor

**Date**: 2026-05-14.
**Goal**: Produce a streaming, online HERALD score (calibrated
probability of catastrophic onset within the next K=16 tokens) on
the pre-onset, onset-anchored task on held-out runs.
**Target**: AUROC >= 0.96 with AUPRC reported alongside.
**Prerequisites**: `gold/phase-2-dataset.md`,
`gold/phase-2-results.md`, `gold/phase-2b-results.md`,
`gold/phase-2c-early-warning-results.md`.

## TL;DR (Verdict: target NOT met; ship 0.948 run-level wrapper)

1. **Streaming wrapper achieves bit-identical scores to the offline
   pipeline** (max abs diff = 0 across 40 segments / 3 held-out
   runs). The online predictor is correct.
2. **Run-level AUROC (deployment-faithful, max-pool over pre-onset
   segments per held-out run)** of the loop-only XGBoost: macro
   mean **0.9481 ± 0.0094** across 5 folds, pooled **0.9472**
   (CI 0.9416-0.9523). AUPRC: macro 0.7792 ± 0.0212, pooled
   0.7760. This is the headline metric per §6.3 (controller acts
   on `max segment_risk` per `gold/phase-4-controller-design.md`
   line 45).
3. **Per-segment AUROC** (intermediate signal): loop-only
   specialist 0.9247 ± 0.0097; unified (loop+NT+clean) 0.9155 ±
   0.0035; unified NT submetric 0.9384 (inflated by
   `position_in_budget` at 4.21% importance).
4. **Sequence-model lever (GRU) spent and exhausted**: causal GRU
   hidden=64 fold-0 smoke per locked §6.3 protocol returned SEG
   AUROC 0.8311 (CI 0.8146-0.8495) with plateau-then-decline
   trajectory at epoch 1. Below the 0.91 smoke gate. Protocol
   adjustment hidden=128 launched but blocked by a CUDA driver
   failure on the cluster (`cudaErrorDevicesUnavailable`); not
   counted as a tested result. §6.3 fallback invoked.
5. **Isotonic calibration**: ECE 0.126 -> 0.0025 (50x reduction)
   on unified; ECE 0.132 -> 0.0017 (78x) on loop-only. AUROC
   preserved (within 0.001).
6. **Gap to target**: macro AUROC 0.9481 is 1.19 pp short of 0.96
   at the deployment-faithful run level (98.8% of bar); pooled
   0.9472 is 1.28 pp short. The remaining gap is concentrated in
   the loop-positive minority class (1.08% per-segment positive
   rate) where the K=16 forecast horizon includes mostly clean-
   looking pre-onset tokens.
7. **Decision**: ship the XGBoost loop-only run-level wrapper as
   HERALD v1's online score (0.948 run-level macro AUROC,
   calibrated). Do NOT claim 0.96 is reached.

## 1. Streaming wrapper (parity)

`src/herald/online_segment_predictor.StreamingHeraldPredictor`
buffers per-token cheap features (12 fields from `TokenSignals`),
fires every K=16 tokens, runs the same per-segment polars
aggregation as the offline pipeline (`aggregate_tokens_to_segments`
+ `surface_features_per_segment` + `extend_phase2_v2_segments`),
appends the press one-hot + `compression_ratio` +
`position_in_budget`, calls `xgb.XGBClassifier.predict_proba`, and
optionally pipes through the per-fold isotonic mapping.

Parity test (`scripts/test_streaming_parity.py`, 3 runs from fold 0):

| Metric | Value |
|---|---|
| Segments compared | 40 |
| Max abs score diff | 0.0 |
| Mean abs score diff | 0.0 |
| 99th percentile diff | 0.0 |

Bit-identical. The online predictor produces exactly the same
score as the offline `train_phase2_v2_xgb` pipeline, given the
same inputs. K=16 anchors to the Phase 4 controller cadence
(`gold/phase-4-controller-design.md`: "Segment size: K=16",
"K=16 segment policy", "K=16 segment cadence as HERALD"); no
K-sweep required.

## 2. Unified model (per-segment XGBoost, all runs)

`scripts/train_phase2_v2_xgb.py` on
`results/phase2_v2/segments_k16_ext.parquet` (352 952 segments,
511 features after one-hot press; pre-onset filter applied so
`seg_end_tok < first_onset` always).

Per-fold AUROC / AUPRC (cluster bootstrap CI in parens):

| Fold | n_test | AUROC | AUPRC |
|------|--------|-------|-------|
| 0 | 70 607 | 0.9163 (0.9068, 0.9254) | 0.5278 (0.4971, 0.5584) |
| 1 | 70 552 | 0.9152 (0.9064, 0.9236) | 0.5274 (0.4954, 0.5550) |
| 2 | 70 588 | 0.9109 (0.8989, 0.9212) | 0.5160 (0.4804, 0.5424) |
| 3 | 70 539 | 0.9215 (0.9128, 0.9297) | 0.5356 (0.5094, 0.5597) |
| 4 | 70 666 | 0.9134 (0.9013, 0.9232) | 0.5546 (0.5174, 0.5821) |
| **Mean** | | **0.9155 ± 0.0035** | **0.5323 ± 0.0128** |

Per-mode breakdown on the same OOF scores (positives split by
which onset class fired in the next K window):

| Mode | n_pos / fold | AUROC mean | AUPRC mean |
|------|--------------|------------|------------|
| Loop | ~430 | 0.8915 | 0.0940 |
| NT   | ~1 470 | 0.9384 | 0.6008 |

The NT > Loop ordering is the same as Phase 2b's run-level result
(NT 0.844, Loop 0.809). NT is structurally easier under this label
because most NT positives end at the budget cap (60% of NT runs
hit max_tokens = 512), so any feature that tracks "how close to
512 tokens are we" is informative. `position_in_budget` is the
top feature in the unified model at **4.21% importance** for
exactly this reason.

## 3. Loop-only specialist (drop NT-first runs)

`scripts/train_phase2_v2_xgb_loop.py` filters runs to:
- **loop-first**: `looping_onset` is the first catastrophe
  (`looping_onset.notna() AND (nt_onset.isna() OR nt_onset >
  looping_onset)`)
- **clean**: both onsets null

NT-first runs are dropped (7 333 runs), since on those the label
"loop in next K" is mechanically zero everywhere they were
truncated by the budget cap. Kept: 2 157 loop-first + 19 424
clean = 21 581 runs.

Per-fold:

| Fold | AUROC | AUPRC |
|------|-------|-------|
| 0 | 0.9253 (0.9150, 0.9363) | 0.1449 |
| 1 | 0.9395 (0.9327, 0.9464) | 0.1745 |
| 2 | 0.9175 (0.9007, 0.9425) | 0.1395 |
| 3 | 0.9115 (0.9033, 0.9237) | 0.1635 |
| 4 | 0.9297 (0.9135, 0.9514) | 0.1867 |
| **Mean** | **0.9247 ± 0.0097** | **0.1618 ± 0.0177** |

AUPRC drops because loop-only segment positive rate is much
lower than the unified positive rate (1.1% vs 2.7%). AUROC is
+0.93 pp over the unified loop submetric (0.8915), confirming
the run filter is doing real work.

Top features (mean importance over 5 folds):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | tail_mass_max_cum_mean | 0.0399 |
| 2 | tail_mass_std_cum_mean | 0.0352 |
| 3 | compression_ratio | 0.0139 |
| 4 | bigram_dup_rate_64_roll4_max | 0.0110 |
| 5 | avg_logp_max_roll4_max | 0.0097 |
| 6 | press_random | 0.0083 |
| 7 | press_tova | 0.0065 |
| ... | ... | ... |
| ?? | position_in_budget | **0.0020** |

`position_in_budget` falls from rank 1 (4.21%) in the unified
model to rank 33+ (0.20%) in the loop-only model. Loop
catastrophes do not depend on hitting the budget cap; they happen
organically when the model gets stuck in repetition. The
loop-only AUROC is therefore an honest measure of HERALD's loop
detection ability, not an artifact of the truncation policy.

## 4. position_in_budget ablation (one fold)

`scripts/ablate_position_in_budget.py`, fold 0, train one variant
with the full feature set and one with `position_in_budget`
removed.

| Variant | Unified AUROC | Unified delta | Loop AUROC | Loop delta |
|---------|--------------|---------------|------------|------------|
| Full feature set | 0.9163 | (baseline) | 0.9253 | (baseline) |
| - position_in_budget | 0.9163 | 0.0000 | 0.9253 | 0.0000 |

Both deltas are exactly zero. This is **not** evidence that
`position_in_budget` carries no information; it is evidence that
the same information is recoverable from `seg_end_tok` (the raw
end-of-segment token index, which is in the feature set), since
`position_in_budget = seg_end_tok / max_budget` is a perfect linear
function of `seg_end_tok`. XGBoost finds the equivalent split on
`seg_end_tok` and the model is unchanged.

The load-bearing evidence that the loop-only specialist does not
lean on budget arithmetic is the loop-only `position_in_budget`
importance of 0.20% (rank 33+, vs 4.21% rank 1 in unified, a 21x
drop). Loop-only top features (`tail_mass_max_cum_mean`,
`tail_mass_std_cum_mean`, `compression_ratio`) are content
signals, not positional ones. A stricter ablation that also drops
`seg_end_tok` and `seg_start_tok` would tighten the unified-model
story but would not change the loop-only conclusion, which is the
shipped model.

## 5. Calibration (isotonic, leave-one-fold-out)

`scripts/calibrate_phase2_v2.py`. For each fold f, fit
`IsotonicRegression(out_of_bounds="clip")` on the pooled OOF
scores from the other 4 folds (disjoint prompt_ids by GroupKFold),
apply to fold f, write `calibrated_fold{f}.parquet` and
`isotonic_fold{f}.json` (knot x/y arrays for the streaming
wrapper).

**Unified model**:

| Metric | Raw mean | Calibrated mean |
|--------|----------|-----------------|
| Brier | 0.0593 | 0.0170 |
| ECE | 0.1264 | 0.0025 |
| AUROC | 0.9155 | 0.9154 |

**Loop-only model**:

| Metric | Raw mean | Calibrated mean |
|--------|----------|-----------------|
| Brier | 0.0676 | 0.0095 |
| ECE | 0.1322 | 0.0017 |
| AUROC | 0.9247 | 0.9244 |

Isotonic preserves AUROC (within 0.0003) and collapses ECE by
50-78x. The streaming wrapper loads
`isotonic_fold{f}.json` (per fold knots) and applies linear
interpolation between knots in `_apply_isotonic`. The shipped
score is the calibrated probability.

## 6. Gap to target and what it would take

Target: AUROC >= 0.96 on pre-onset, K=16 onset-anchored task.
Best honest number: **0.925** (loop-only). Gap: 3.5 pp.

The 3.5 pp gap is structural under the current setup:

1. **Label is union of two failure modes that need different
   models**. Phase 2b (run-level) and this work (segment-level)
   both show NT and loop separate cleanly: NT >> loop. A unified
   model averages discriminability and pays a 4.4 pp penalty.
   Splitting the head into per-mode classifiers is one option;
   the loop-only result above is the upper bound on what that
   buys for the loop side without architectural changes.
2. **Loop label has high segment-level noise**. K=16 forecast
   horizon means a positive segment is "any of the 16 tokens
   immediately after this segment is the loop onset". Most
   pre-onset segments look very similar to clean segments
   because the token-level signal sharpens only in the last few
   tokens before onset. AUPRC of 0.16 at 1.1% base rate is
   already 14x lift over chance; squeezing 3.5 pp more AUROC
   would likely require either a longer-horizon multi-task head
   or token-level features beyond the cheap signal pack (which
   contradicts the "lightweight, zero-cost" premise of HERALD).
3. **Per-press models** are tested below as a single-fold
   diagnostic. Per-press positive counts (per fold, loop-only)
   are estimated to range 30 to 150; cluster-bootstrap CIs would
   be too wide to claim 0.96 honestly even if the point estimate
   moved on a single press, so the headline metric must be a
   macro mean across all six presses, not a cherry-picked best.

### 6.1 Pre-stated bar for per-press loop-specialized diagnostic

**Bar locked here BEFORE running** (per advisor's pre-conditions
to avoid post-hoc gaming):

- **Headline metric**: macro mean AUROC across all 6 presses
  (knorm, snapkv, expected_attention, streaming_llm, tova, random)
  on fold 0 only, loop-only run filter applied per press.
- **Decision rule**: if macro mean AUROC >= 0.96, expand to all
  5 folds and ship per-press wrapper as v1. If macro mean < 0.96,
  the structural ceiling at ~0.92 is confirmed empirically and
  the loop-only specialist (0.9247) is shipped as v1.
- **Forbidden moves**: no per-press hyperparameter tuning after
  seeing results; no high-compression-ratio filtering of the test
  set; no ensembling with the unified model (unified is
  contaminated); no replacing the macro mean with a max if the
  max happens to be >= 0.96.
- **Time-box**: one Orion run, fold 0 only, ~25 min wall.

Result table fills in §6.2 below once
`results/phase2_v2/per_press_loop_fold0.json` lands.

### 6.2 Per-press loop-specialized result (fold 0, locked bar)

`scripts/train_phase2_v2_xgb_loop_per_press.py`, fold 0,
loop-only run filter applied per press.

| Press | n_runs (loop-only) | pos_test | AUROC | 95% CI | AUPRC |
|-------|--------------------|----------|-------|--------|-------|
| expected_attention | 4 504 | 41  | 0.8596 | (0.809, 0.921) | 0.218 |
| knorm              | 2 365 | 85  | 0.8507 | (0.823, 0.886) | 0.126 |
| random             | 3 271 | 156 | 0.8682 | (0.851, 0.889) | 0.127 |
| snapkv             | 3 491 | 80  | 0.9519 | (0.939, 0.965) | 0.213 |
| streaming_llm      | 4 097 | 23  | 0.8551 | (0.771, 0.959) | 0.216 |
| tova               | 3 853 | 55  | 0.8363 | (0.785, 0.897) | 0.103 |
| **Macro mean**     |       |     | **0.8703 ± 0.0377** | | **0.167** |

**Bar**: 0.96 (locked in §6.1 BEFORE running).
**Result**: 0.8703 macro mean (8.97 pp short of bar).
**Target met**: NO.

Even the best single press (snapkv at 0.9519) misses the 0.96 bar
by 0.81 pp at the point estimate, and its CI upper bound (0.9649)
only just clips it. The other five presses range 0.836 to 0.868;
splitting the model per press makes things measurably worse on
average than the pooled loop-only specialist (0.9247), because
each per-press model trains on roughly one sixth the data with
under 200 positives in fold 0's training set.

The structural ceiling for cheap-feature loop detection at K=16
on this dataset is empirically ~0.92 to 0.95 depending on press
mix. Per-press fragmentation is a net regression.

### 6.3 Pre-stated bar for sequence-model run-level predictor

**Bar locked here BEFORE running** (same pre-conditions discipline
as §6.1, written into the gold file before any sequence-model
training launches; advisor-validated 2026-05-14):

**Goal-aligned reframing of the headline metric.** The original
goal text says "AUROC >= 0.96 on the pre-onset, onset-anchored task
on **held-out runs** at a lead time set by the controller's
reaction window". The controller's reaction window is K=16 tokens
(one segment) per `gold/phase-4-controller-design.md` lines 41, 45,
75, 129; the cooldown clause (line 184) governs *consecutive*
actions, not the first reaction, so K is not extensible to 32. The
phrase "held-out runs" naturally maps to a **run-level** metric,
because the controller decision is per-run (acts on `max
segment_risk over completed segments` per design line 45). Per-
segment AUROC is the intermediate signal; run-level AUROC is the
deployment-faithful metric. This is the headline.

**Established baselines (computed BEFORE the sequence model
trains):**

- Per-segment loop-only XGBoost (5-fold GroupKFold, all 6 presses
  pooled): **AUROC 0.9247 ± 0.0097** (§6 above, locked).
- **Run-level max-pool of those XGBoost scores** over pre-onset
  segments per held-out run (script `run_phase2_v2_runlevel_auroc
  .py`, output `runlevel_summary.json`): macro mean AUROC across
  5 folds **0.9481 ± 0.0094**, pooled AUROC **0.9472** (CI
  0.9416-0.9523), AUPRC macro 0.7792, AUPRC pooled 0.7760.
- **Both numbers fail the 0.96 bar.** The +2.34 pp lift from
  per-segment to run-level is the gain from max-pooling pre-onset
  scores; it is not metric shopping, it is the deployment-faithful
  evaluation. The ceiling for cheap-feature XGBoost at the run
  level is empirically 0.948 ± 0.009.

**Sequence-model lever (the only remaining legitimate path).** The
extended XGBoost feature set (§6.2 footprint) already includes
`cum_mean`, `roll8_std`, `prev`, `delta`, `roll4_mean`, `roll4_max`,
`cum_max_window20_repeats`, `cum_max_top1_streak`,
`n_unique_64_below_peak`. Adding more hand-crafted aggregates is
diminishing returns. The remaining lever is a model class that
learns the temporal trajectory natively rather than from
hand-crafted summaries.

**Locked architecture and protocol (no post-hoc tuning):**

- **Model**: 1-layer **causal GRU**, hidden_size = 64,
  dropout = 0.1, input = the same 511-dim per-segment feature
  vector (extended set + press one-hot + `position_in_budget`),
  per-segment sigmoid output. Causal (no bidirectional) so the
  predictor remains streaming-online by construction.
- **Loss**: BCE with `pos_weight = (1 - pos_rate) / pos_rate`
  computed from the training fold y-mean.
- **Optimizer**: AdamW, lr = 1e-3, weight_decay = 1e-4.
- **Training**: batch_size = 32 runs, max_epochs = 20, gradient
  clip 1.0, early-stop on validation AUPRC with patience 3.
- **Cross-validation**: same 5-fold GroupKFold on `prompt_id` as
  XGBoost; same loop-only run filter (drop NT-first runs).
- **Evaluation**: per-segment AUROC + run-level max-pool AUROC
  per fold; macro mean across 5 folds is the headline; cluster
  bootstrap CI by `run_id` with n_boot = 2000.
- **Smoke gate before full 5-fold launch**: train fold 0 only;
  if per-segment AUROC < 0.91 the architecture is broken and we
  try ONE adjustment (hidden = 128). If still < 0.91, we accept
  the structural ceiling at run-level 0.948.

**Decision rule (locked):**

- If macro mean run-level AUROC **>= 0.96** -> ship sequence model
  + max-pool wrapper as HERALD v1's online score; revise §7 to
  reflect target met.
- If macro mean run-level AUROC **< 0.96** -> structural ceiling
  empirically confirmed; ship the XGBoost run-level wrapper at
  0.948 with honest §7 statement that the goal-as-stated was not
  met and explain why (sequence model trained at the deployment-
  faithful target did not exceed the cheap-feature run-level
  ceiling).

**Forbidden moves** (same discipline as §6.1):

- No per-fold or per-press hyperparameter tuning after seeing
  results.
- No replacing the macro mean with a max if a single fold happens
  to clip 0.96.
- No ensembling sequence-model with XGBoost to chase the target.
- No relabeling at H > 16 to expand the look-ahead window.

**Time-box**: one Orion smoke fold (~25 min wall), then if pass,
one full 5-fold run (~2 h wall).

Result table fills in §6.4 below once
`results/phase2_v2/seq_loop/seq_summary.json` and
`runlevel_summary_seq.json` land.

### 6.4 Sequence-model result (causal GRU, fold 0 smoke + retry)

**Run-level baselines (XGBoost, frozen before sequence-model
training).** Computed by `scripts/run_phase2_v2_runlevel_auroc.py`
on `results/phase2_v2/xgb_ext_loop/scores_fold{0..4}.parquet`
(max-pool of per-segment scores over each held-out run's pre-onset
segments; cluster bootstrap CI by `run_id`, `n_boot=2000`):

| Fold | n_runs | n_pos | AUROC | 95% CI | AUPRC |
|------|--------|-------|-------|--------|-------|
| 0 | 4 348 | 426 | 0.9504 | (0.9390, 0.9606) | 0.7689 |
| 1 | 4 283 | 438 | 0.9652 | (0.9566, 0.9727) | 0.8161 |
| 2 | 4 309 | 399 | 0.9385 | (0.9230, 0.9528) | 0.7738 |
| 3 | 4 328 | 519 | 0.9413 | (0.9311, 0.9504) | 0.7523 |
| 4 | 4 313 | 375 | 0.9451 | (0.9309, 0.9581) | 0.7848 |
| **Macro** | | | **0.9481 ± 0.0094** | | **0.7792 ± 0.0212** |
| **Pooled** | 21 581 | 2 157 | 0.9472 | (0.9416, 0.9523) | 0.7760 |

Macro mean run-level AUROC **0.9481** falls 1.19 pp short of the
0.96 bar; pooled run-level AUROC **0.9472** falls 1.28 pp short.
The +2.34 pp lift over per-segment AUROC (0.9247) is the
deployment-faithful gain from max-pooling per-run, not metric
shopping. This is the structural ceiling that the sequence-model
lever has to clear.

**Causal GRU smoke (fold 0, hidden=64).** Run on Orion at
2026-05-14 23:24-23:29; locked architecture per §6.3
(1-layer GRU hidden=64, dropout=0.1, BCE+pos_weight=91.74,
AdamW lr=1e-3 wd=1e-4, batch=32, max_epochs=20, patience=3).

Per-epoch validation trajectory (fold 0):

| Epoch | train_loss | val_AUROC | val_AUPRC |
|-------|-----------|-----------|-----------|
| 0 | 1.1466 | 0.8256 | 0.0645 |
| 1 | 1.0830 | **0.8311** | **0.0736** |
| 2 | 1.1105 | 0.8251 | 0.0600 |
| 3 | 1.0985 | 0.8220 | 0.0702 |
| 4 | 1.1012 | 0.8206 | 0.0681 |

Early-stop fired at epoch 4 (best at epoch 1 with patience 3). The
trajectory is plateau-then-decline starting at epoch 1, not a
slow climb. Final fold-0 metrics:

| Metric | Value | 95% CI |
|--------|-------|--------|
| SEG AUROC | 0.8311 | (0.8146, 0.8495) |
| SEG AUPRC | 0.0736 | (0.0623, 0.0925) |
| RUN AUROC (max-pool) | 0.8593 | (n/a) |
| RUN AUPRC (max-pool) | 0.4943 | (n/a) |

**Smoke gate**: per-segment AUROC must reach 0.91. Result 0.8311.
**Gate failed by 7.9 pp.** Per §6.3 protocol, this triggers ONE
adjustment (hidden=128) before invoking the structural-ceiling
fallback.

**Causal GRU hidden=128 retry: blocked by GPU infrastructure.**
The retry was launched on Orion at 2026-05-14 23:30:51 with the
same script and `--hidden 128`. It failed during model `.to(cuda)`
with `torch.AcceleratorError: CUDA error: CUDA-capable device(s)
is/are busy or unavailable` (`cudaErrorDevicesUnavailable`).
Subsequent `nvidia-smi` calls returned `No devices were found` /
`Unable to determine the device handle for GPU0: Unknown Error`,
indicating the driver itself is in a bad state, not the script.
Recovery requires admin intervention on the cluster (the safety
rule prohibits killing processes outside `/clustergpu/home/jcampo`
or rebooting the node), so the hidden=128 retry could not be
completed in this session.

**This is recorded as infrastructure, not as a result.** The
hidden=128 architecture has not been falsified; it has not been
tested. A future re-run on a healthy GPU should reproduce this
configuration with the same script
(`scripts/train_phase2_v2_seq_loop.py --hidden 128 --folds 0`).

**Verdict: §6.3 fallback invoked.** The fallback decision rule
states: "If still < 0.91, we accept the structural ceiling at
run-level 0.948." We invoke it on the strength of two facts:

1. **Empirical**: the GRU-64 trajectory is plateau-then-decline at
   epoch 1, not under-training; doubling capacity to hidden=128
   would change scale, not the inductive-bias mismatch that the
   trajectory signals (recurrent hidden state on 511-dim per-
   segment vectors loses to gradient-boosted trees that get
   high-order feature interactions natively).
2. **Operational**: the protocol-specified hidden=128 retry cannot
   be completed without admin intervention on the cluster. We do
   not silently substitute another adjustment; we explicitly note
   the gap and accept the ceiling.

The structural ceiling at run-level **macro mean AUROC 0.9481 ±
0.0094 / pooled AUROC 0.9472 (CI 0.9416-0.9523)** stands as the
best honest number for cheap-feature streaming under the locked
protocol. AUPRC at the same ceiling: macro 0.7792 ± 0.0212,
pooled 0.7760.

## 7. Decision

**Goal not met.** Pre-stated bar in §6.3 was AUROC >= 0.96 at the
deployment-faithful run level on held-out runs. Achieved
**run-level macro AUROC 0.9481 ± 0.0094 / pooled 0.9472 (CI
0.9416-0.9523)**, 1.19 pp short of bar at the macro mean and
1.28 pp short pooled. AUPRC reported alongside as required:
macro 0.7792 ± 0.0212, pooled 0.7760.

**Ship the XGBoost loop-only run-level wrapper as HERALD v1's
online score.** This is the per-segment loop-only XGBoost (§3,
SEG AUROC 0.9247 ± 0.0097) wrapped with the run-level max-pool
defined in `scripts/run_phase2_v2_runlevel_auroc.py`. The
controller acts on `max segment_risk over completed segments`
(per `gold/phase-4-controller-design.md` line 45), so the wrapper
matches the deployed decision rule by construction. Calibrated
via the same per-fold isotonic mapping as §5; isotonic preserves
AUROC within 0.0003.

**Sequence-model lever spent honestly.** The causal GRU
(hidden=64) was trained per the locked §6.3 protocol on fold 0;
per-segment AUROC 0.8311 with plateau-then-decline trajectory at
epoch 1 falsified the architecture under the cheap-feature
regime. The protocol-specified hidden=128 retry was launched but
blocked by a CUDA driver failure on the cluster (§6.4) and is
not counted as a tested result. The fallback was invoked on the
GRU-64 trajectory plus the operational unavailability of GPU.

**Why the bar was not met (structural).** The remaining 1.19 pp
gap is concentrated in the loop-positive minority class (1.08%
per-segment positive rate) at K=16 forecast horizon. Most pre-
onset segments look very similar to clean segments because the
token-level signal sharpens only in the last few tokens before
onset. Closing the gap honestly likely requires either a longer-
horizon multi-task head (which contradicts the controller's
single-segment reaction window) or token-level features beyond
the cheap signal pack (which contradicts HERALD's "lightweight,
zero-cost" premise).

**Report the unified model as a single-model baseline** with the
`position_in_budget` caveat made explicit (the NT half of the
score depends on budget-cap arithmetic; this is realistic for the
deployed setting where `max_tokens` is fixed, but it is not a
generic catastrophe detector).

**Per-press fragmentation rejected** (§6.2): macro mean 0.870 vs
pooled 0.925 - splitting per press hurts the macro mean.

**Phase 4 controller** uses the loop-only run-level score for
loop prevention and the unified score (or a separate NT head)
for budget-aware NT prevention; `gold/phase-4-controller-design
.md` already wires the controller to `max segment_risk` per
design line 45, so no design change is required to consume the
v1 score.

## Reproduction

```bash
# Build dataset (already on Orion)
uv run python scripts/build_phase2_v2_segments.py
uv run python scripts/extend_phase2_v2_segments.py

# Train unified
uv run python scripts/train_phase2_v2_xgb.py \
  --input results/phase2_v2/segments_k16_ext.parquet \
  --output-dir results/phase2_v2/xgb_ext

# Train loop-only specialist
uv run python scripts/train_phase2_v2_xgb_loop.py \
  --input results/phase2_v2/segments_k16_ext.parquet \
  --output-dir results/phase2_v2/xgb_ext_loop

# Calibrate (per-fold isotonic, leave-one-fold-out)
uv run python scripts/calibrate_phase2_v2.py \
  --scores-dir results/phase2_v2/xgb_ext
uv run python scripts/calibrate_phase2_v2.py \
  --scores-dir results/phase2_v2/xgb_ext_loop

# Streaming parity
uv run python scripts/test_streaming_parity.py \
  --tokens-root results/phase1/final/tokens \
  --runs results/phase1/final/runs.parquet \
  --scores results/phase2_v2/xgb_ext/scores_fold0.parquet \
  --model results/phase2_v2/xgb_ext/model_fold0.json \
  --summary results/phase2_v2/xgb_ext/summary.json \
  --n-runs 5

# position_in_budget ablation
uv run python scripts/ablate_position_in_budget.py \
  --input results/phase2_v2/segments_k16_ext.parquet \
  --output results/phase2_v2/ablation_position.json --fold 0

# Per-press loop diagnostic (fold 0)
uv run python scripts/train_phase2_v2_xgb_loop_per_press.py \
  --input results/phase2_v2/segments_k16_ext.parquet \
  --output results/phase2_v2/per_press_loop_fold0.json

# Run-level AUROC headline (max-pool over pre-onset segments)
uv run python scripts/run_phase2_v2_runlevel_auroc.py \
  --scores-dir results/phase2_v2/xgb_ext_loop \
  --output results/phase2_v2/xgb_ext_loop/runlevel_summary.json

# Causal GRU smoke fold 0 (locked §6.3 architecture)
uv run python scripts/train_phase2_v2_seq_loop.py \
  --input results/phase2_v2/segments_k16_ext.parquet \
  --output-dir results/phase2_v2/seq_ext_loop_smoke \
  --folds 0
```

## Files

| File | Description |
|------|-------------|
| `src/herald/online_segment_predictor.py` | StreamingHeraldPredictor + isotonic |
| `scripts/test_streaming_parity.py` | Parity assertion vs offline scores |
| `scripts/train_phase2_v2_xgb.py` | Unified per-segment XGBoost |
| `scripts/train_phase2_v2_xgb_loop.py` | Loop-only specialist |
| `scripts/calibrate_phase2_v2.py` | Per-fold isotonic + ECE/Brier reporting |
| `scripts/ablate_position_in_budget.py` | Single-fold ablation |
| `scripts/train_phase2_v2_xgb_loop_per_press.py` | Per-press loop diagnostic |
| `scripts/run_phase2_v2_runlevel_auroc.py` | Run-level AUROC analyzer (max-pool) |
| `scripts/train_phase2_v2_seq_loop.py` | Causal GRU sequence predictor (§6.3, §6.4) |
| `results/phase2_v2/per_press_loop_fold0.json` | Per-press fold 0 results (§6.2) |
| `results/phase2_v2/xgb_ext_loop/runlevel_summary.json` | Run-level XGBoost wrapper headline (§6.4) |
| `results/phase2_v2/seq_ext_loop_smoke/summary.json` | Causal GRU fold-0 smoke (§6.4) |
| `results/phase2_v2/xgb_ext/{summary,by_mode_summary,importance}.json` | Unified results |
| `results/phase2_v2/xgb_ext/calibrated/{calibration_summary,isotonic_fold*}.json` | Unified calibration |
| `results/phase2_v2/xgb_ext_loop/{summary,importance}.json` | Loop-only results |
| `results/phase2_v2/xgb_ext_loop/calibrated/...` | Loop-only calibration |
| `results/phase2_v2/ablation_position.json` | position_in_budget ablation |
