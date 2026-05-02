# Phase 1 LongBench failure diagnosis

**Status as of 2026-05-02**: fix landed locally; tiny Orion validation rerun
pending. Block 3 cleared from the LongBench failure perspective only after
validation reports clean.

## Context

Phase 1 Block 2 Option B per-task profiling on Orion (RTX 5090, 32 GiB,
Qwen2.5-7B-Instruct fp16, 50 prompts × 4 LongBench cells) reported
`n_failed = 6` for every LongBench-Single cell:

| task | press | ratio | n_runs | n_failed |
| --- | --- | --- | --- | --- |
| longbench_single | none | 0.0 | 44 | 6 |
| longbench_single | streaming_llm | 0.5 | 44 | 6 |
| longbench_single | streaming_llm | 0.875 | 44 | 6 |
| longbench_single | streaming_llm | 0.9375 | 44 | 6 |

GSM8K, HumanEval, IFEval reported `n_failed = 0` across all cells.

Source: `results/phase1_profile/phase1_profile_summary.json` (Orion).

## Failed prompt IDs

The same 6 NarrativeQA prompts deterministically failed in every
LongBench cell:

- `longbench_1842b0ff1882e545a6d41d5caf67bba5312872423fa48e74`
- `longbench_32e116c58a3c59fc170aa5f4e1dde414c8f3881872889826`
- `longbench_7570a52d69ab93c5f54eba4c45d44a3411650c1e4694760a`
- `longbench_b03244c8cc2681df1008d27c974d81415336396dff81f06d`
- `longbench_df6c6350671baab25c635bfa495eea90c69a7d201b5fe460`
- `longbench_fbeb825de92309788269da33aa6bd189c7b1d46b997746f4`

These IDs are also pinned as a fixture in
`tests/test_tasks.py::TestLongBenchFormatPrompt::test_failing_prompt_ids_fixture`.

## Per-failure data (none / ratio=0.0 cell)

Pulled from `results/phase1_profile/longbench_single/none/ratio=0.0000/raw/runs/*.parquet`.

| prompt_id (truncated) | prompt_chars | num_tokens_generated | replay_status | failure stage |
| --- | --- | --- | --- | --- |
| `longbench_1842b0ff…` | 202268 | 63 | failed | replay |
| `longbench_7570a52d…` | 202270 | 48 | failed | replay |
| `longbench_b03244c8…` | 199497 | 26 | failed | replay |
| `longbench_df6c6350…` | 199514 | 65 | failed | replay |
| `longbench_fbeb825d…` | 199486 | 56 | failed | replay |
| `longbench_32e116c5…` | 199503 | 11 | failed | replay |

Generation completed in every case (11–65 generated tokens, parquet
`runs/` and `tokens/` written; `replay/` parquet missing). The
exception captured in `replay_error` is `torch.cuda.OutOfMemoryError`
attempting to allocate 17–18 GiB for an attention activation while
~14–16 GiB of GPU memory was free after the model load.

Concretely, `_generate_compressed` populated:
- `replay_status = "failed"`
- `replay_error = OutOfMemoryError('CUDA out of memory. Tried to allocate 17.10–18.20 GiB...')`

Failure stage: **during replay**, after generation. `phase1_profile.py`
counts this as `n_failed += 1` because `PerRunPaths.all_exist()` is False
when `replay/` parquet is missing.

## Root cause

Matched-prefix replay does a single full forward pass on the
*uncompressed* model with `use_cache=False` over `[input_ids; gen_t]`
(see `src/herald/metrics/replay.py:replay_forward`). At sequence length
~50 000 tokens the attention activation tensor demands 17–18 GiB of GPU
memory, exceeding what the 32 GiB RTX 5090 can satisfy after the
~14.7 GiB Qwen2.5-7B fp16 weights.

The 6 failing NarrativeQA prompts contain entire short stories
(199 486–202 270 characters; ~50 000 tokens after chat templating).
Generation succeeds because it uses an incremental KV cache (memory
proportional to current length, not its square). Replay does not.

The binding constraint is **replay activation memory**, not the
positional limit. Qwen2.5-7B-Instruct's `max_position_embeddings` is
32 768, so prompts longer than that are also out-of-spec for the
positional embeddings; both constraints point in the same direction
("the prompt is too long"), but the replay-activation constraint is
tighter and is what we must budget against.

This is **not** experimental noise: every cell sees the same 6 prompt
IDs fail with the same error class, deterministically. The 12% loss is
an artifact of a fixed-budget infrastructure path, not a sampling
issue.

## Fix

Implemented in `src/herald/tasks.py` and wired into
`src/herald/experiment.py`.

**Approach.** Add a `format_prompt(prompt_data, tokenizer, system_prompt,
max_new_tokens) -> (question_str, meta)` hook to the `Task` ABC. The
default returns `prompt_data["question"]` unchanged with empty
metadata; GSM8K, HumanEval, and IFEval inherit this default and are
behaviorally untouched (covered by
`TestNonLongBenchFormatPromptUnchanged`).

`LongBenchSingleTask.format_prompt` deterministically truncates the
context so the chat-templated prompt plus generated tokens fit inside
a fixed budget:

- `LONGBENCH_PROMPT_TOKEN_BUDGET = 16384` (chat-templated tokens).
- Plus a `LONGBENCH_TRUNCATION_SAFETY_TOKENS = 64` margin.
- Plus a `chat_overhead = 96` initial estimate for ChatML role tags
  and the generation prompt suffix.
- Question/instruction text is preserved verbatim; only the *context*
  block (`Context:\n{ctx}\n\nQuestion: {inp}\n\nAnswer:`) is
  shortened.
- Truncation pattern follows the BPE-safe form recommended by the
  advisor: `encode → slice → decode` once; **do not re-encode after
  decode** for the kept slice (BPE round-trip is not identity).
- After truncation, the chat template is applied and the templated
  length is verified; if it overshoots (chat overhead exceeded the
  estimate), the slice is shrunk and the verification repeats. Loop
  bounded at 8 iterations.

Why 16 384? Three reasons: (1) Qwen2.5-7B-Instruct has a 32 768
positional limit, so 16k + 512 max_new + chat overhead leaves
~16k headroom under the positional ceiling; (2) at sequence length 16k
the attention activation tensor is ~6.6× smaller than at 50k, well
inside the ~17 GiB free after model load; (3) every prompt that
succeeded in Block 2 was below this length, so dropping to 16k
preserves the existing behavior on the 44 surviving prompts and only
truncates the 6 that previously failed.

The same prompt always produces the same formatted input
(`test_truncation_is_deterministic`).

**Truncation metadata sidecar.** Each run writes one JSONL line to
`<output_root>/raw/truncation.jsonl` from `_generate_compressed` →
`_append_truncation_sidecar`. The line carries `run_id`, `prompt_id`,
`task`, `truncated`, `original_context_tokens`,
`truncated_context_tokens`, `chat_templated_tokens`, `budget`,
`max_new_tokens`, `safety_tokens`. The Block 3 watchdog reads this to
enforce "stop if LongBench truncation metadata is missing for
truncated prompts" without a parquet schema migration.

**Files changed.**

- `src/herald/tasks.py`: added `Task.format_prompt` ABC default;
  added `LongBenchSingleTask.format_prompt` truncation; added
  `LONGBENCH_PROMPT_TOKEN_BUDGET` and `LONGBENCH_TRUNCATION_SAFETY_TOKENS`
  constants; `LongBenchSingleTask.load` now stores raw `context` and
  `input` fields alongside the pre-built `question`.
- `src/herald/experiment.py`: `_generate_compressed` now takes `task`
  and uses `task.format_prompt(...)` in place of
  `prompt_data["question"]`; both call sites updated;
  `_append_truncation_sidecar` writes the sidecar per run from
  `run_single_with_replay`.
- `tests/test_tasks.py`: `TestLongBenchFormatPrompt`,
  `TestNonLongBenchFormatPromptUnchanged`,
  `TestLongBenchBudgetConstant`, plus a `_StubTokenizer` for offline
  CPU tests; failing prompt IDs pinned as a fixture.

**Hard rules respected.** Failed prompts are not silently dropped: all
50 are retained, the 6 long ones get a deterministic, documented
truncation. GSM8K / HumanEval / IFEval behavior is not modified —
their `format_prompt` is the inherited no-op (regression-tested).

## Tests

Run:

    uv run poe check

Result on the working tree at the time of writing:

- `poe lint` (ruff): all checks passed.
- `poe typecheck` (mypy strict): no issues found in 42 source files.
- `poe test` (pytest): **321 passed, 2 skipped, 1 deselected** in
  ~57 s on the Mac. New tests under `TestLongBenchFormatPrompt`,
  `TestNonLongBenchFormatPromptUnchanged`, `TestLongBenchBudgetConstant`,
  including determinism, budget enforcement, question preservation,
  short-context no-op, no-tokenizer fallback, missing-raw-fields
  fallback, and per-task no-op regressions.

CPU tests use a stub tokenizer (whitespace tokens, fake ChatML
overhead). The fix is exercised against the real Qwen tokenizer only
through the Orion validation rerun below.

## Tiny Orion validation rerun

Script: `scripts/phase1_longbench_validate.py` (added).

Purpose: prove the fix on the 6 previously failing prompts under both
a baseline and one compressed cell, write to a separate output path,
do not overwrite the profile data.

Validation command (on Orion):

    cd /clustergpu/home/jcampo/herald
    .venv/bin/python scripts/phase1_longbench_validate.py \
        --output-root results/phase1_validate \
        --presses none,streaming_llm \
        --ratios 0.0,0.875 \
        --max-new-tokens 512

Validation success criteria:

- All 6 baseline runs (`press=none, ratio=0.0`) produce
  `replay_status = "ok"` and an existing `replay/<run_id>.parquet`.
- All 6 compressed runs (`press=streaming_llm, ratio=0.875`) likewise
  produce `replay_status = "ok"`.
- No schema/parquet failures.
- `truncation.jsonl` sidecar contains a `truncated=True` line for each
  of the 6 prompts in each cell.
- `peak_memory_mb` stays below 24 576 (24 GiB).

Result: **pending — to be filled in after the Orion run.**

## Remaining risk

- The 16 384-token budget is conservative and should fit; the loop
  inside `format_prompt` re-shrinks if the chat-template overhead is
  larger than expected. If the loop exhausts, the metadata records
  `loop_exhausted=true`; the watchdog should surface this.
- Some other LongBench subtasks (qasper, multifieldqa_en) have
  different length distributions; the fix applies uniformly because
  the budget is on chat-templated tokens, not on subtask identity.
  Phase 1 currently uses NarrativeQA only.
- The cost-budget JSON
  (`gold/phase-1-cost-budget.json`) was computed on the surviving 44
  LongBench prompts at full length. Post-fix, all 50 will run; per-run
  cost will *decrease* because prefill and replay are smaller. The
  current entries are an upper bound and stay safe for the watchdog;
  updating them is deferred until validation gives actual numbers.

## Block 3 readiness from the LongBench failure perspective

- Diagnosis: complete.
- Fix: implemented and CPU-tested.
- Validation: **pending the Orion rerun.** Block 3 is **not yet cleared**.
  Once the validation rerun reports 12/12 ok and the truncation sidecar
  shows the expected entries, Block 3 is cleared from this failure
  perspective and the launch packet (separate document /
  `gold/phase-1-block3-launch-packet.md`) governs the rest.
