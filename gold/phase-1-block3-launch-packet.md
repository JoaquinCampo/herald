# Phase 1 Block 3 launch packet

**Status**: drafted; LongBench failure cleared pending validation rerun
(see `gold/phase-1-longbench-failure-diagnosis.md`). Do not launch
until that validation reports 12/12 ok.

## Scope

The Phase 1 fixed-ratio measurement sweep:

- 1 model (`Qwen/Qwen2.5-7B-Instruct`).
- 4 tasks (gsm8k, humaneval, ifeval, longbench_single).
- Press matrix: `streaming_llm`, `snapkv`, `knorm`, `expected_attention`,
  `tova`, `random` (6 presses) plus a baseline `none` cell per task.
- Ratio grid: `{0.25, 0.375, 0.5, 0.75, 0.875, 0.9375, 0.96875}` for
  compressed cells; `0.0` for the per-task baseline.
- 200 prompts per task, seed 42, greedy decoding,
  `max_new_tokens = 512`.
- Matched-prefix replay every 8 tokens (Phase 0 decision; see
  `gold/phase-0-results.md`).

Total cells: `4 tasks × 6 presses × 7 compressed ratios = 168`
compressed + `4` baseline = **172 cells**, **34 400 runs**.

## Exact launch command

Block 3 launch is one wrapper script per task, sequential, sharing the
loaded model across cells in a task. No such wrapper exists yet
(`scripts/run_sweep.py` runs a single task with a fixed press list);
the launcher is the engineering item to land before launch. The
intended invocation is:

    cd /clustergpu/home/jcampo/herald
    nohup .venv/bin/python scripts/phase1_sweep.py \
        --tasks gsm8k,humaneval,ifeval,longbench_single \
        --presses streaming_llm,snapkv,knorm,expected_attention,tova,random \
        --ratios 0.25,0.375,0.5,0.75,0.875,0.9375,0.96875 \
        --include-baseline \
        --num-prompts 200 \
        --max-new-tokens 512 \
        --seed 42 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output-root results/phase1 \
        --cost-budget gold/phase-1-cost-budget.json \
        --watchdog-tolerance 1.25 \
        --watchdog-consecutive-breaches 3 \
        --skip-existing \
        > results/phase1_sweep.log 2>&1 &

`scripts/phase1_sweep.py` is to be implemented by lifting the
per-cell loop out of `phase1_profile.py` and adding:

- `CostWatchdog` per cell (predicted_s_per_token from
  `phase-1-cost-budget.json`, tolerance 1.25, 3 consecutive breaches).
- Skip-existing: on a per-cell basis, a prompt is considered done
  when `paths.all_exist()` returns true and the saved
  `replay_status == "ok"`.
- Truncation-sidecar enforcement: if any prompt's run record contains
  generation against a chat-templated length implying truncation but
  the sidecar is missing, abort.
- Per-task baseline cells run before any compressed cell for that
  task so `baseline_run_id` linkage is valid.

If a launcher script is not landed in time, the fallback is a shell
loop calling `phase1_profile.py` per task, with `--num-prompts 200`
and the same press/ratio lists, redirecting log per task. That path
loses the watchdog and is therefore *not* a Block 3 launch — it is a
Block 2-shaped sweep at full N.

## Estimated wall-clock

From `gold/phase-1-cost-budget.json` (Block 2, 50 prompts, Orion
RTX 5090). All numbers are per-cell and extrapolate linearly to 200
prompts.

| task | s/token median | tokens median | per-prompt s | per-cell s @ 200 | per-cell min |
| --- | --- | --- | --- | --- | --- |
| gsm8k / none@0 | 0.0127 | 277 | 3.75 | 750 | 12.5 |
| gsm8k / stream@0.5 | 0.0136 | 328 | 4.71 | 942 | 15.7 |
| gsm8k / stream@0.875 | 0.0137 | 227 | 3.32 | 664 | 11.1 |
| gsm8k / stream@0.9375 | 0.0138 | 426 | 6.20 | 1240 | 20.7 |
| humaneval / none@0 | 0.0129 | 47 | 0.67 | 134 | 2.2 |
| humaneval / stream@0.875 | 0.0136 | 88 | 1.29 | 257 | 4.3 |
| ifeval / none@0 | 0.0128 | 187 | 2.57 | 514 | 8.6 |
| ifeval / stream@0.9375 | 0.0137 | 512 | 7.36 | 1473 | 24.5 |
| longbench / none@0 | 0.0559 | 53.5 | 6.03 | 1206 | 20.1 |
| longbench / stream@0.9375 | 0.0546 | 57 | 6.13 | 1226 | 20.4 |

Cross-task averages from these representative cells (gsm8k ~15 min,
humaneval ~3 min, ifeval ~12 min, longbench ~20 min). With 6 presses
× 7 ratios = 42 compressed cells per task plus 1 baseline:

| task | cells | est min/cell | task min | task hours |
| --- | --- | --- | --- | --- |
| gsm8k | 43 | 15 | 645 | 10.8 |
| humaneval | 43 | 3 | 129 | 2.2 |
| ifeval | 43 | 12 | 516 | 8.6 |
| longbench_single | 43 | 20 | 860 | 14.3 |
| **total** | **172** | — | **2150** | **35.8** |

This is a point estimate against the *post-fix* LongBench cost
profile. The current `phase-1-cost-budget.json` was computed on the
44 surviving LongBench prompts at full length; per-run cost will fall
slightly under truncation, so 35.8 GPU-hours is a soft upper bound.
The user's earlier `~42 GPU-hours` estimate carried more safety; the
budget JSON should be re-profiled on a 50-prompt LongBench slice
post-fix before relying on the tighter number.

## Estimated storage

Block 2 measured `parquet_bytes_per_run` from 540 KB (gsm8k
baseline) to 928 KB (ifeval stream@0.9375); the median across all
profile cells is ~440 KB. At 200 prompts per cell × 172 cells:

- Median: 200 × 172 × 440 KB ≈ **15.1 GiB raw parquet**.
- Upper bound at the 950 KB/run cell: 200 × 172 × 950 KB ≈
  **31.2 GiB raw parquet**.

Plus finalized concatenated parquet under `results/phase1/final/`
(roughly the same size). Plan for **~60 GiB** of free disk on Orion
before launch.

## Watchdog rules

The sweep aborts if any of the following triggers fires.

1. **Cost drift per cell.** `CostWatchdog` (`herald.experiment`,
   already shipped) consumes each prompt's `wall_clock_per_token`. If
   the rolling value exceeds
   `predicted_s_per_token * 1.25` for **3 consecutive prompts**, the
   cell is aborted and the launcher records the abort.
2. **Failure rate per cell.** If any cell records
   `n_failed / n_attempted > 0.02` (2%), the launcher aborts the
   sweep, not just the cell. The launcher tracks `n_attempted` and
   `n_failed` (where "failed" includes both generation exceptions and
   `replay_status != "ok"`).
3. **Repeated replay-status non-ok with the same failure mode.** If
   the last 3 non-ok runs share the same `replay_error` class, the
   sweep aborts. This guards against a regressed code path that
   produces e.g. repeated CUDA OOM without crossing the 2% threshold.
4. **Peak memory.** If any prompt records
   `peak_memory_mb > 28 * 1024 = 28 672 MiB`, the sweep aborts.
   Headroom under the 32 GiB GPU is small enough that crossing 28 GiB
   means the next prompt will likely OOM.
5. **Truncation metadata missing.** For LongBench prompts, the
   sidecar JSONL must contain a record matching the run_id. If the
   record is missing for a run whose chat-templated prompt was longer
   than `LONGBENCH_PROMPT_TOKEN_BUDGET`, the sweep aborts. If the
   sidecar reports `loop_exhausted = true`, the sweep aborts (the
   truncation loop did not converge).

The watchdog state is checkpointed alongside results so a relaunch
can re-arm without losing earlier runs.

## Resume / skip-existing

`run_prompts` already supports per-prompt JSON-line checkpointing via
`_load_checkpoint` / `_append_checkpoint`. The Block 3 launcher
extends this with cell-level skip-existing semantics:

- A cell is skipped if every prompt under it has all three parquets
  written (`paths.all_exist()`) and `replay_status == "ok"` for each.
- A prompt is skipped if its `paths.all_exist()` is true and
  `replay_status == "ok"`.
- A prompt is *re-run* if `replay_status != "ok"` OR any of the three
  parquets is missing. Old artifacts are overwritten in place.
- The truncation sidecar is appended (one line per attempt) so a
  re-run produces an additional sidecar entry rather than replacing
  the existing one. The launcher's missing-metadata check uses the
  most recent record per `run_id`.

This matches the behavior already used by `phase1_profile.py`
(`_maybe_skip`).

## Logs and results

- `results/phase1/<task>/<press>/ratio=<r:.4f>/raw/{runs,tokens,replay,truncation.jsonl}/...`
  per cell, identical to the Block 2 layout.
- `results/phase1/<task>/<press>/ratio=<r:.4f>/cell_summary.json` —
  per-cell summary written at end-of-cell with completion stats and
  watchdog state.
- `results/phase1/phase1_sweep_summary.json` — overall summary.
- `results/phase1_sweep.log` — combined stdout/stderr from
  `phase1_sweep.py`.
- `gold/phase-1-cost-budget.json` is read-only input.
- After the sweep, `finalize_dataset` (already in
  `herald.metrics.io`) builds `results/phase1/final/{runs,tokens,replay}.parquet`
  for downstream analysis.

## Pre-flight checklist

Before issuing the launch command above, the operator confirms:

- [ ] The validation rerun in
      `gold/phase-1-longbench-failure-diagnosis.md` reports 12/12 ok
      and shows the truncation sidecar populated.
- [ ] `nvidia-smi` shows the GPU clean (no stale processes).
- [ ] `~/.cache/huggingface/datasets/` contains gsm8k, humaneval,
      ifeval, and THUDM/LongBench/narrativeqa.
- [ ] `phase-1-cost-budget.json` has been re-profiled on a 50-prompt
      LongBench slice with the truncation fix (the existing values
      stay safe but tighter values give the watchdog more margin).
- [ ] `scripts/phase1_sweep.py` exists and ships the watchdog rules
      above.
- [ ] Free disk on Orion ≥ 60 GiB.
- [ ] `tmux`/`screen`/`nohup` is used so the sweep survives an SSH
      disconnect.

## Outstanding items before launch

1. Implement `scripts/phase1_sweep.py` (lifted from
   `phase1_profile.py` plus the watchdog rules above).
2. Re-profile LongBench cells with the truncation fix and update the
   cost-budget JSON.
3. Decide on the final `--ratios` list. The grid above is the
   research-plan grid; the cost-budget JSON currently only has 0.5,
   0.875, 0.9375 entries. Either expand the budget profile to cover
   0.25, 0.375, 0.75, 0.96875 too, or relax watchdog rule 1 to use
   the nearest-profiled-ratio prediction.
