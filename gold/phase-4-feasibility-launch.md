# Phase 4 Feasibility Launch Packet

Date: 2026-05-05.
Status: ready for the 3-prompt GSM8K smoke on Orion, gated on
kvpress >= 0.5.3.

This packet contains the minimum information needed to launch the
decode-time press feasibility gate on Orion. It does NOT cover the
HERALD controller, AIMD, or LoopGuard; those land after this gate
passes.

## Pre-launch Checklist

1. SSH into Orion: `ssh orion`.
2. `cd` to the synced HERALD checkout.
3. Sync source tree: rsync `src/`, `tests/`, `scripts/` from Mac.
   Do NOT sync `.venv/`, `results/`, or `models/`.
4. Verify GPU is clean: `nvidia-smi` (no zombie CUDA processes).
   `pkill -f` any stale jobs from a prior crash.
5. Verify kvpress version:
   `uv run python -c "import kvpress; print(kvpress.__version__)"`
   Required: >= 0.5.3.
   If 0.5.1 (current): STOP. Upgrade with
   `uv pip install -U "kvpress>=0.5.3"` and re-verify before
   proceeding. The launch script also enforces this gate at runtime
   and refuses to run on older versions.
6. Confirm DecodingPress import:
   `uv run python -c "from kvpress import DecodingPress, DMSPress, KnormPress; print('ok')"`.

## Smoke Command (3 prompts, GSM8K, fixed budgets)

Primary candidate, `DecodingPress(KnormPress)`:

```
nohup uv run python scripts/run_phase4_decoding_press_feasibility.py \
    --press decoding_knorm \
    --task gsm8k \
    --num-prompts 3 \
    --max-new-tokens 256 \
    --target-sizes 256 512 1024 \
    --compression-interval 16 \
    --output-dir results/phase4/feasibility/decoding_knorm \
    > logs/phase4_feas_decoding_knorm.log 2>&1 &
```

Backup candidate, only if `decoding_knorm` fails the acceptance
gate:

```
nohup uv run python scripts/run_phase4_decoding_press_feasibility.py \
    --press dms_knorm \
    --task gsm8k \
    --num-prompts 3 \
    --max-new-tokens 256 \
    --target-sizes 1024 \
    --compression-interval 16 \
    --output-dir results/phase4/feasibility/dms_knorm \
    > logs/phase4_feas_dms_knorm.log 2>&1 &
```

DMS feasibility passes a single placeholder `--target-sizes 1024` so
the runs.parquet schema stays joinable, but DMS is threshold-based
and the script's v1 `build_press` constructs `DMSPress(KnormPress(),
decoding=True)` without honoring target_size. Do NOT pass multiple
sizes here, or the parquet will look like a sweep that wasn't run.
A per-threshold DMS sweep is a follow-up: extend the script with a
`--dms-thresholds` knob before that.

## Mandatory 10s Smoke Check After Launch

Per the saved Orion launch policy, immediately after backgrounding:

```
sleep 10
ps -p $! && tail -n 50 logs/phase4_feas_decoding_knorm.log
```

Expect:

- process is alive,
- log shows past argparse and into kvpress version check,
- no proxy / network hang.

If the log shows `kvpress 0.5.1 installed` -> STOP and upgrade.
If the log shows a CUDA OOM or model-load failure -> STOP and triage.
Do not schedule any away-check until smoke is green.

## Expected Runtime

- 3 prompts x 3 target_sizes = 9 runs.
- Qwen2.5-7B-Instruct, fp16, RTX 5090, max 256 new tokens.
- Phase 1 fixed-StreamingLLM baseline runs ~3-5s per generation on
  this prompt budget; decode-time `DecodingPress` is expected to be
  1-3x slower (per-step compress() at every 16 steps).
- Total wall-clock budget: under 10 minutes including model load.
- If wall-clock exceeds 30 minutes -> STOP, treat as feasibility
  failure for overhead.

## Expected Outputs

Under `results/phase4/feasibility/<press>/`:

- `decoding_press_report.json` (summary including
  `decode_compression_observed` and `pass_gate` boolean),
- `runs.parquet` (one row per (target_size, prompt) — 9 rows for
  the smoke),
- `segments.parquet` (K=16 aggregates),
- `events.parquet` (per-compression-event log; expected to be
  non-empty with `event_type == "decode"` rows).

## Pass / Fail Interpretation

Pass conditions (all must hold):

- exit code 0 from the script,
- `decoding_press_report.json["decode_compression_observed"]` is
  `true`,
- `events.parquet` contains rows with `event_type == "decode"`,
- `runs.parquet` has 9 rows with `stop_reason in {eos, max_tokens}`,
- mean `wall_clock_per_token` is within 3x of the existing
  fixed-compression generation path.

If pass: proceed to Step 2 of the implementation plan (predictor
export). HumanEval feasibility extension can run next under the
same gate.

Soft fail (compression happens but overhead pathological):

- `decode_compression_observed` true but per-token wall clock > 3x.
- Action: record overhead in `gold/phase-4-feasibility-results.md`,
  re-evaluate `compression_interval` (32, 64), re-test once.

Hard fail (no decode compression observed) for `decoding_knorm`:

- Action: launch the `dms_knorm` backup smoke. Document
  `decoding_knorm` failure mode in
  `gold/phase-4-feasibility-results.md`.

Hard fail for both:

- Action: STOP. Do NOT invent a custom compressor in this task.
  Escalate; the implementation plan documents the reprefill /
  prompt-level fallback as the next branch.

## Stop Conditions (do not continue, escalate)

- kvpress < 0.5.3 detected at any point.
- CUDA OOM at fp16 on RTX 5090 with `--num-prompts 3`.
- Generation hangs past 30 minutes without producing logs.
- `events.parquet` is empty (zero compression events) for both
  presses.
- `runs.parquet` shows `stop_reason == "other"` for >50% of runs
  (suggests a stop-criterion plumbing bug, not a feasibility result).

## What the Smoke Does NOT Cover

- HERALD risk policy.
- AIMD or `RiskBudgetStep` controller.
- LoopGuard baseline.
- Random matched-budget controller.
- Predictor export, online feature parity, calibration.
- HumanEval (gated behind GSM8K passing first).
- Per-prompt baseline-uncompressed pairing for quality_delta.

These are the next implementation steps and are explicitly out of
scope for this launch.

## Caveats Carried Into Orion

- The `EventRecorder` wrapper assumes kvpress's `BasePress.compress`
  signature is
  `compress(module, hidden_states, keys, values, attentions, kwargs)`.
  If upstream changed it, the recorder still passes the call
  through (via `__getattr__`) but per-event before/after retained
  cache lengths may be missing. Confirm on the first Orion run that
  `events.parquet` contains non-zero
  `retained_cache_len_before/after`. If not, fix the wrapper before
  reading further smoke results.
- The decode/prefill classification is heuristic: the first event
  per layer is labelled `prefill`, subsequent events `decode`. This
  is good enough for the pass/fail boolean but not for fine-grained
  event accounting. Refine after the first Orion run by comparing
  to step indices logged by the model's forward.
- `wall_clock_seconds` per token in `TokenLog` is currently NaN —
  per-step timing requires an additional logits-callback hook that
  is not in scope for the substrate gate. Whole-run wall clock and
  `peak_memory_mb` are populated and sufficient for the overhead
  check.
