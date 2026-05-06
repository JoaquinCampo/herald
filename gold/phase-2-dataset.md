# Phase 2 Predictor Dataset Spec

**Status:** post-audit. Phase 1 `final/runs.parquet`, `final/tokens/`,
and `final/replay/` were rsynced from Orion on 2026-05-05; the
audit (`scripts/phase2_audit.py`) ran cleanly against the full
32 852-run dataset. All OPEN-1..7 questions are closed in the
"Audit results" section at the end of this document. The builder
implementation in `src/herald/predictor_dataset.py` may proceed.

This document is the contract that the builder, baselines, and
evaluation read from. Do not implement against assumed schemas;
verify each one against the rsynced artifacts first.

## Source artifacts

The Phase 1 sweep produced four families of artifacts. Phase 2 reads
from these only; no GPU regeneration, no re-sweep.

### Run-level (`results/phase1/final/runs.parquet`) — pending rsync

Confirmed schema from Phase 0's identical layout
(`results/phase0/final/runs.parquet`, 28 columns):

```
run_id, prompt_id, prompt_text, prompt_hash,
model, model_revision, tokenizer_revision, dtype, device_class,
task, press, compression_ratio,
max_new_tokens, decoding_config, seed,
baseline_run_id, generated_text, generated_token_ids,
num_tokens_generated, stop_reason,
predicted_answer, ground_truth, correct,
catastrophes, replay_status, replay_error,
created_at, herald_git_sha
```

Phase 1 expected row count: **32 852** ok rows
(`gold/phase-1-results.md` Closeout block).

### Token-level (`results/phase1/final/tokens/`) — pending rsync

Hive-partitioned by `press=*/ratio=*`, one parquet per `run_id`.
Confirmed schema from Phase 0 (18 columns):

```
run_id, token_pos, token_id, token_str,
entropy, top1_prob, top5_prob, top5_logprobs,
h_alts, avg_logp, delta_h, delta_h_valid,
kl_div, top10_jaccard, eff_vocab_size, tail_mass,
logit_range, lookback_ratio
```

One row per generated token. `token_pos` is 0-indexed within the
generated sequence (not within the prompt+generation concatenation).
`delta_h`, `kl_div`, `top10_jaccard` are NaN at `token_pos=0` by
construction — `delta_h_valid` flags this.

### Replay-level (`results/phase1/final/replay/`) — pending rsync

Hive-partitioned by `press=*/ratio=*`, one parquet per `run_id`.
Confirmed schema from Phase 0 (16 columns):

```
run_id, token_pos, realized_token_id, union_top_k_token_ids,
logprobs_compressed, logprobs_uncompressed,
tail_mass_compressed, tail_mass_uncompressed,
realized_logprob_compressed, realized_logprob_uncompressed,
js_full, kl_unc_comp_full, kl_comp_unc_full,
top1_match, top1_rank_comp_under_unc, top1_rank_unc_under_comp
```

**OPEN-1: replay sampling stride.** Resolved by audit
(`results/phase2/audit/phase2_audit.json`): **stride = 1**
(every-token replay), not the every-8 the research plan called for.
516/516 sampled runs across all (task, press, ratio) cells show:

- `global_stride_distribution = {1: 516}`
- `first_replay_position_distribution = {0: 516}` — replay starts at
  token 0
- `last_replay_offset_distribution = {0: 516}` — replay extends to
  the last generated token

This is denser than expected and simplifies the builder: future-
window aggregates can sum every replay row in `(t, t+H]` without
sparsity handling. The disk-size disparity (9.8 GB replay vs 637 MB
tokens) is consistent with full per-token replay — replay rows
carry `union_top_k_token_ids` + dual logprob arrays per token.

### Run-level damage (`results/phase1/metrics/run_damage.parquet`) — local

Confirmed schema (32 088 rows, 39 columns; one row per compressed
run, joined to its uncompressed baseline):

```
Metadata: run_id, prompt_id, task, press, compression_ratio,
          baseline_run_id, catastrophes
Trajectory: sum_kl, sum_js, nll_ratio, first_divergence_point
Sequence: rouge_l_drop, char_edit_ratio, length_diff_ratio,
          embedding_cosine_drop (null until [metrics] extra)
Tags: has_looping, has_non_termination, has_format_break, has_drift
Outcome: baseline_correct_final, compressed_correct_final,
         baseline_quality_score, compressed_quality_score,
         quality_delta, gross_harm_final, gross_help_final,
         quality_label_source
Per-task scoring detail: qasper_em, qasper_f1, qasper_threshold,
                          qasper_n_golds,
                          ifeval_num_constraints,
                          ifeval_num_supported,
                          ifeval_num_unsupported,
                          ifeval_num_satisfied, ifeval_score,
                          ifeval_threshold, ifeval_unsupported_types,
                          baseline_quality_label_source
```

Schema defined in `gold/run-damage-table.md`. `gross_harm_final` is
`null` (not `False`) when either side's correctness is undefined; the
builder must propagate this null, not coerce.

Phase 1 corrected success criterion passes 4 of 4 tasks
(`results/phase1/metrics/phase1_success_criterion.final.json`).
Placeholder-grader artifacts (`phase1_success_criterion.json`,
`alignment_phase1.parquet`, `intrinsic_to_outcome_phase1.parquet`)
are preserved and must not be overwritten.

## Open schema questions — closed by audit

All seven are resolved in the "## Audit results" section at the end
of this document. Summary of resolutions:

| ID | Status | Headline |
|---|---|---|
| OPEN-1 | closed | replay stride = 1 (every token), not every-8 |
| OPEN-2 | closed | replay.token_pos ⊆ tokens.token_pos, 0 misaligned of 516 sampled |
| OPEN-3 | closed | baseline (press=none) replay present (764 runs) |
| OPEN-4 | closed | runs_ok = tokens = replay = 32 852; compressed = run_damage = 32 088; all set diffs zero |
| OPEN-5 | closed | segment metrics not available; segment unit deferred |
| OPEN-6 | closed | nll_ratio sign is negative across all 4 tasks; flip before use |
| OPEN-7 | closed | audit ran clean on streaming polars; builder must stream by (press, ratio) partition |

## Unit of observation

**Primary: token-level.** One row per compressed generated token
where enough future replay positions exist for the largest horizon
in scope. With the actual stride = 1 (OPEN-1 closed), a token at
`token_pos = t` is included iff `t + H_max < num_tokens_generated`
where `H_max = 50`. Practically: drop the last 50 tokens of each
run for the H = 50 label set. Shorter horizons (H = 5, 10, 25)
retain more rows.

**Compressed runs only.** Baseline (press=`none`) runs have nothing
to predict and are excluded from training rows. They remain
relevant as the source of `quality_delta` (already joined into
`run_damage.parquet`).

**Segment-level: deferred.** OPEN-5 must close before the segment
unit is admitted. If segment metrics are missing or partial in Phase
1, the v1 Phase 2 baseline runs token-only; segments are a follow-up.

## Horizons

`H ∈ {5, 10, 25, 50}`. Counted in realized tokens (i.e. positions in
`tokens.parquet`). Since the actual replay stride is 1 (OPEN-1
closed), every horizon window `(t, t+H]` is fully populated for any
`t` with `t + H < num_tokens_generated`. Rows with insufficient
remaining tokens for a given `H` get null labels for that horizon
only and are dropped per-horizon at training time.

## Features

All features must be available **at time t**. Anything that requires
information from `(t, ∞)` is forbidden as a feature (it is reserved
for the label).

**Tier 0 (raw token features, from `tokens/`):**

`entropy`, `top1_prob`, `top5_prob`, `h_alts`, `avg_logp`,
`delta_h`, `kl_div`, `top10_jaccard`, `eff_vocab_size`, `tail_mass`,
`logit_range`, `delta_h_valid`. The five `top5_logprobs` are flattened
to `logprob_0..4` (matching `src/herald/features.py:flatten_signals`).

**Tier 0.5 (rolling, causal):**

EWMA, rolling mean, rolling std, rolling max over each Tier 0 target
in `{entropy, top1_prob, h_alts, delta_h, kl_div, top10_jaccard}`,
windows `{8, 32}`. All windows are causal (right-edge at `t`); the
existing `src/herald/features.py:add_rolling_features` pattern uses
`min_periods=1` for early tokens, which matches.

**Tier 1 (instability):**

`kl_div` between consecutive timestep distributions is already in
Tier 0 (note: this is the model's own `kl(p_{t-1} || p_t)`, not
matched-prefix KL — the latter is a label, not a feature). Distance
from prompt embedding is **out of scope** for v1 unless the
`[metrics]` extra is wired by the time the builder runs.

**Position features:**

- `token_pos`: raw position
- `relative_progress`: `token_pos / max_new_tokens` (matches existing
  `src/herald/features.py` normalization)
- `output_length_so_far`: equal to `token_pos` for now; will diverge
  if early-stopping changes per-run length

**Metadata features (for the metadata-only baseline):**

- `task` (categorical: gsm8k, humaneval, ifeval, longbench_single)
- `press` (categorical: streaming_llm, snapkv, knorm,
  expected_attention, tova, random)
- `compression_ratio` (continuous)

These are conditioning inputs for the predictor and direct features
for the metadata baseline. For cross-press transfer (Phase 2 Task 3,
held-out press split), `press` is dropped from the feature set on
both train and test sides — leaving it in would let the model
specialize on a label that is constant on each side of the split.

## Labels

Computed per token at `token_pos = t`, summing over the replay rows
whose `token_pos` falls in `(t, t+H]`. The replay column
**`js_full`** is the symmetric, bounded matched-prefix divergence;
**`kl_unc_comp_full`** is the directional KL (uncompressed → compressed)
that aligns with the trajectory `sum_kl` defined in
`gold/run-damage-table.md`.

### Continuous labels (regression / threshold)

| name | formula |
|---|---|
| `future_sum_kl_H` | `Σ kl_unc_comp_full[t']` over `t' in (t, t+H] ∩ replay_positions` |
| `future_sum_js_H` | `Σ js_full[t']` over the same set |
| `future_max_js_H` | `max js_full[t']` over the same set |

Empty windows (no replay rows in `(t, t+H]`) yield null. Null rows
are dropped per-horizon at training time, not zero-imputed.

### Binary labels (classification)

For each horizon and each continuous label, threshold from the
**training split only** at the p90 quantile of that horizon's
`future_sum_js_H` over compressed-run training tokens. Specifically:

1. Apply the chosen split (e.g. held-out prompts).
2. Compute `T_H = quantile_0.90(future_sum_js_H | split=train,
   future_sum_js_H is not null)`.
3. Define `binary_future_js_H = (future_sum_js_H >= T_H)`. Same
   recipe with `future_sum_kl_H` and `future_max_js_H`.

Threshold values are recorded in
`results/phase2/dataset/phase2_dataset_summary.json`. They are
**re-computed per split** so each held-out evaluation has its own
threshold derived from its own training data — never from the global
pool, never from a held-out cell. This is the explicit no-leakage
rule.

The p90 quantile is the v1 threshold. A sensitivity analysis at
`{p75, p90, p95}` is reserved for Phase 2 Task 4 follow-up.

### Run-level validators (joined for Task 4 only, never used as token labels)

From `run_damage.parquet`:

- `gross_harm_final` (Boolean, nullable)
- `quality_delta` (Float, nullable)
- `rouge_l_drop`, `char_edit_ratio`, `length_diff_ratio`
- `has_looping`, `has_non_termination`, `has_format_break`,
  `has_drift`

These are **not** training targets. They are joined only for the
Task 4 user-facing-validation step that scores per-run aggregated
predictions against extrinsic damage.

## `nll_ratio` sign

Confirmed footgun (`gold/phase-1-results.md` Open items): on per-task
pooled rows, `nll_ratio` Spearman vs outcome harm is **negative**
where `sum_kl`/`sum_js` are positive. The builder must flip the sign
convention so larger-is-worse for any feature or label that uses
`nll_ratio`. Document the flip explicitly in code comments and in
the dataset summary JSON.

## Splits

All four splits run on the **same row population**. Per-split
threshold recomputation (above) means binary labels are not
identical across splits; continuous labels are.

| split | train | test |
|---|---|---|
| held-out prompts | 80% prompts | 20% prompts (per task) |
| held-out ratios | all but one ratio | one ratio |
| held-out presses | all but one press | one press |
| held-out tasks | all but one task | one task |

For held-out tasks, only 4 tasks → 4 folds, point estimates only,
no within-fold variance. Reported as is, with a sentence noting the
absence of CIs is a coverage limit, not a methodological choice.

For held-out prompts, fold by `prompt_id` (not `run_id`) so all 7
press × ratio replicas of a held-out prompt land on the same side.
Phase 2 baselines use 5-fold GroupKFold over `prompt_id`.

## Outputs

- `results/phase2/dataset/phase2_tokens.parquet` — one row per
  qualifying token, all features + all labels + all run-level
  validators.
- `results/phase2/dataset/phase2_dataset_summary.json` — row counts
  by `(task, press, ratio)`, label positive rates per horizon,
  threshold values per horizon per split, null counts per label,
  `nll_ratio` sign-flip note, `OPEN-*` resolutions.
- `results/phase2/baselines/phase2_baseline_results.parquet` — one
  row per `(model, horizon, label, split, fold)` AUROC/AUPRC.
- `results/phase2/baselines/phase2_baseline_summary.json` — best
  baseline per split, paired-bootstrap deltas vs best
  entropy/EWMA-style baseline, "beats by ≥ 0.05 with CI not crossing
  zero" decision per model.

## Leakage controls

1. Thresholds for binary labels are computed from training split
   only. Recomputed per split.
2. Future-window features are forbidden. Only `tokens.parquet`
   columns at `token_pos ≤ t` and rolling features computed causally
   may enter `X`.
3. `press` is excluded from features for the held-out-press split.
4. Held-out prompts: fold by `prompt_id`, never by `run_id` —
   otherwise the same prompt under different (press, ratio) leaks
   across the split.
5. Tokens past `t > num_tokens_generated - H_max` are dropped to
   keep label horizons fully observed.
6. Run-level validators (`gross_harm_final`, `quality_delta`, etc.)
   are joined onto rows but are **never** included in `X`. They are
   used only at Task 4 (per-run validation).

## Implementation notes for the builder

- Stream by `(press, ratio)` partition. Don't load all `tokens/` and
  `replay/` into memory simultaneously.
- For each compressed run: join `tokens` and `replay` on
  `(run_id, token_pos)` with an outer join (replay is sparse). For
  each horizon `H`, compute `future_sum_kl_H`, `future_sum_js_H`,
  `future_max_js_H` by a forward-window aggregation over the
  replayed positions in `(t, t+H]`.
- Reuse pattern from
  `src/herald/analysis/information_ceiling.py:build_token_dataset` —
  it already handles per-token + per-run join, onset filtering, and
  the position/metadata/online feature partitioning. Phase 2's
  builder is a strict superset.
- CPU only; polars is fine. No CUDA, no model load.

## Pipeline

```sh
# After rsync from Orion has landed Phase 1 final/:
uv run python scripts/phase2_audit.py --root results/phase1     # audit, closes OPEN-1..7
uv run python scripts/build_phase2_dataset.py --root results/phase1 \
    --output-dir results/phase2/dataset
uv run python scripts/run_phase2_baselines.py \
    --dataset results/phase2/dataset/phase2_tokens.parquet \
    --output-dir results/phase2/baselines
```

The audit script does not yet exist; it will be the first thing
implemented after rsync. It writes the answers to OPEN-1..7 into
this document under a "## Audit results" section before the builder
is allowed to run.

## Audit results

Generated by `scripts/phase2_audit.py` on 2026-05-05 against
`results/phase1/final/`. Full JSON at
`results/phase2/audit/phase2_audit.json`.

### runs.parquet

- 32 852 rows, 32 852 unique `run_id`, all `replay_status = ok`
- baseline (press = none): 764 rows
- compressed: 32 088 rows
- by task: gsm8k 8 600, humaneval 7 052, ifeval 8 600,
  longbench_single 8 600
- by press: each compressed press 5 348, baseline 764

### Set diffs (closes OPEN-4)

| set | size |
|---|---:|
| `runs.parquet` (replay_status=ok) | 32 852 |
| `tokens/` unique `run_id` | 32 852 |
| `replay/` unique `run_id` | 32 852 |
| `run_damage.parquet` rows | 32 088 |
| `runs_ok − tokens` | 0 |
| `tokens − runs_ok` | 0 |
| `runs_ok − replay` | 0 |
| `replay − runs_ok` | 0 |
| `compressed_ok − run_damage` | 0 |
| `run_damage − compressed_ok` | 0 |

Every `ok` run has both a `tokens/{run_id}.parquet` and a
`replay/{run_id}.parquet`. Every compressed run is in
`run_damage.parquet`. The baseline runs are correctly absent from
`run_damage.parquet` (it is a per-compressed-run table joined to
its baseline; there is no "baseline harm" cell for a baseline run).

### Replay stride and alignment (closes OPEN-1, OPEN-2)

Sampled 3 runs per `(task, press, ratio)` cell, 516 runs total.

- `global_stride_distribution = {1: 516}` — every sampled run has
  per-token replay (consecutive `token_pos` differences mode = 1).
- `first_replay_position_distribution = {0: 516}` — every sampled
  run replays from position 0.
- `last_replay_offset_distribution = {0: 516}` — every sampled run
  replays to its last generated token.
- `n_misaligned_runs = 0` — `replay.token_pos ⊆ tokens.token_pos`
  on every inspected run.

This contradicts `gold/research-plan.md` Phase 1 Scope, which
committed to "Uniform matched-prefix replay every 8 tokens" with the
sub-sampling decision recorded in `gold/phase-0-results.md`. The
on-disk reality is full per-token replay. Implications for Phase 2:

- Future-window aggregates `future_sum_kl_H`, `future_sum_js_H`,
  `future_max_js_H` over horizon `H` realized tokens equal the sum
  over `H` replay rows (no sparsity gap, no null windows from stride).
- The `(t+1, t+H]` window is dense; positive-rate calibration of the
  threshold-based binary labels does not need stride-aware adjustment.
- Disk cost (9.8 GB replay vs 637 MB tokens) is the only downside —
  the builder streams per (press, ratio) partition to keep RSS bounded.

### Baseline (press=none) replay (closes OPEN-3)

- `replay/press=none/ratio=0.0000/` exists, 764 parquets.

These exist for completeness — replaying an uncompressed run on
itself should give `js_full ≈ 0` at every position. Useful as a
sanity floor when validating intrinsic damage signals; not used as
training rows in Phase 2.

### Segment metrics (closes OPEN-5)

- `metrics/segments.parquet`: not present
- `metrics/segment_metrics.parquet`: not present
- `final/segments/`: not present

`gold/phase-1-results.md` confirms the segment build aborted at cell
13/172 (`gsm8k/knorm/ratio=0.9375`) during the Phase 1 metrics pass.
**v1 Phase 2 is token-only.** Segment-level features are deferred
until either the segment build is rerun successfully or a
vectorized segment builder is written. This is consistent with the
research plan: "Segment-level: deferred. OPEN-5 must close before
the segment unit is admitted."

### nll_ratio sign convention (closes OPEN-6)

Spearman correlation per task on the corrected `run_damage.parquet`:

| task | ρ(sum_kl, nll_ratio) | ρ(sum_js, nll_ratio) | n |
|---|---:|---:|---:|
| gsm8k | -0.827 | -0.794 | 8 400 |
| humaneval | -0.940 | -0.937 | 6 888 |
| ifeval | -0.823 | -0.757 | 8 400 |
| longbench_single | -0.891 | -0.887 | 8 400 |

Sign is negative across all four tasks with high magnitude. The
builder **must flip nll_ratio sign** (`flipped_nll_ratio =
-nll_ratio`) for any feature or label use, so that "larger value =
worse damage" is consistent across the trajectory aggregate trio.
This flip is recorded in
`results/phase2/dataset/phase2_dataset_summary.json` and in the
column docs of `phase2_tokens.parquet`.

### Memory and I/O (closes OPEN-7)

The audit completed on a single 16 GiB Mac without OOM, scanning
the full 32 852-run hive partition for set-diffs (one `pl.scan_parquet
("**/*.parquet")` over each tree → `unique` → `collect`) and reading
3 token + 3 replay parquets per cell directly by hive path for stride
inspection. The builder follows the same shape: stream by
`(press, ratio)` partition, never load the full 9.8 GB replay tree at
once. The earlier audit version that scanned the full tree per
sampled run_id was ~100× slower and was killed; the patched version
reads files directly via `_direct_run_path`.

### What changes in the dataset spec

- "Future-window aggregates" become straightforwardly dense — null
  rows arise only at the right boundary of each run, not from stride.
- "Predictor reads tokens from `tokens.parquet`, labels from
  `replay.parquet`, joined on `(run_id, token_pos)`" is a true join
  (not a left-outer with sparse replay).
- The right-censoring rule remains: drop the last `H_max - 1` tokens
  of each run so every retained row has all `H` future replay
  positions defined for the largest `H = 50`. (Exception: runs whose
  `num_tokens_generated < H_max` contribute no rows to the H = 50
  label set — they remain in the dataset for shorter horizons.)
