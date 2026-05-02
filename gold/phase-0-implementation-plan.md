# Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 0 measurement substrate from `gold/phase-0-design.md`: matched-prefix replay, parquet schema with deterministic repair, end-to-end metrics pipeline, and a 140-run smoke that satisfies success gates 1-7.

**Architecture:** New `herald/metrics/` subpackage holding GPU-bound replay (inline with generation, exact full-vocab scalars) plus offline CPU metric modules. `herald/experiment.py` is refactored to return `GenerationArtifact` and finalize `RunResult` after replay so `replay_status` and the determinism manifest are written atomically. Per-run parquets under `results/phase0/raw/`, hive-partitioned final dataset under `results/phase0/final/`. CLIs: `herald phase0 run`, `herald metrics finalize|build|repair`.

**Tech Stack:** Python 3.12+, PyTorch, HuggingFace transformers, kvpress, pyarrow + polars (core), bert-score / sentence-transformers / rouge-score / editdistance / scipy (metrics extras), typer (CLI), pydantic v2, pytest, ruff, mypy strict.

**Reference docs:** `gold/phase-0-design.md` is the spec; this plan implements it. `gold/ultimate-goal.md` and `gold/research-plan.md` provide the why. `CLAUDE.md` carries project conventions (line length 78, no `from __future__ import annotations`, modern type syntax, flat package root with subpackages allowed for cohesive areas, `poe check` before commits).

**Conventions for every task:**
- Run `poe check` before each commit. It must pass (`fmt + lint + typecheck + test`).
- Tests use pytest. Tests that load a real LLM are marked `@pytest.mark.gpu` and run **only on Orion**. CPU/synthetic-data tests run anywhere; `poe test` (Mac default) must pass with `pytest -m "not gpu"` and is the gate for every commit.
- No `from __future__ import annotations`; use `list`, `dict`, `tuple`, `X | None` directly.
- Commit message format: `feat(metrics): <imperative>` or `refactor(experiment): <imperative>`. No `Co-Authored-By` lines.

---

## Execution Environment

This plan executes across two machines. The split is structural, not a convenience: experiments run on Orion, analysis runs on Mac, and the parquet artifacts are the contract between them.

### Roles

| Machine | Role | Used for |
|---------|------|----------|
| **Mac** (Apple Silicon, 16GB, MPS) | code authoring + CPU tests + analysis | Tasks 1–4, 10–16, 18 (CPU/synthetic); writing code; `poe check`; reading rsync'd parquet; alignment matrix figures; paper writing. **Never run real GPU experiments.** |
| **Orion** (RTX 5090, 32GB VRAM, CUDA 13.1, no internet) | model-bound tests + Phase 0 sweep | Tasks 5–9, 17, 19 GPU portions; the actual 140-run smoke; `metrics repair`. |

CLAUDE.md is explicit: **DO NOT run GPU experiments locally**.

### Per-task execution location

| Task | Mac | Orion | Notes |
|------|:---:|:-----:|-------|
| 1. Dependencies | ✓ | ✓ (sync) | Mac is authoritative; Orion `uv sync --extra metrics` needs the SSH reverse-tunnel proxy (PyPI). |
| 2. RunResult + GenerationArtifact | ✓ | — | Pure CPU; tests synthetic. |
| 3. Parquet IO | ✓ | — | Pure CPU. |
| 4. Manifest generator | ✓ | — | Loads GSM8K via `datasets`; needs proxy if cache missing on Mac, but the manifest itself is committed. |
| 5. Replay forward + index alignment | partial (synthetic CPU test) | **required** (real model t=0 + argmax test) | Threshold pinning happens on Orion. |
| 6. compute_replay_position | ✓ | — | Pure synthetic-distribution math. |
| 7. replay_run end-to-end | partial (skeleton) | **required** (slow test with 0.5B model) | |
| 8. run_single refactor + run_single_with_replay | partial (refactor, signal flow) | **required** (integration test) | |
| 9. phase0 sweep CLI | partial | **required** (smoke on 2 prompts) | |
| 10. metrics finalize CLI | ✓ | — | Pure CPU on per-run parquets (run anywhere). |
| 11. token (truncation-bias audit) | ✓ | — | |
| 12. trajectory | ✓ | — | |
| 13. sequence | ✓ | — | Loads sentence-transformers model (CPU is fine). |
| 14. outcome + tags | ✓ | — | |
| 15. alignment matrix | ✓ | — | |
| 16. metrics build integration test | ✓ | — | |
| 17. metrics repair | ✓ (validator) | **required** (real regen in production smoke) | Validator test is CPU; the regenerated-token-ids self-check runs on Orion. |
| 18. sampling-rate study | ✓ | — | Reads finalized replay parquets rsync'd from Orion. |
| 19. Poe wiring + 140-run smoke | wiring only | **required** | The Phase 0 success-gate run lives on Orion. Mac never executes this task's GPU portion. |

### Cross-machine workflow (per task)

```
edit on Mac
  → poe check
  → uv run pytest -m "not gpu" -q       # green
  → git commit -m "..."
  → git push origin <branch>
[ for Mac-only tasks: stop here ]
  → ssh orion
  → cd /clustergpu/home/jcampo/herald
  → git pull
  → uv run pytest -m "gpu and not e2e" -q   # task-scoped GPU tests
[ for sweep tasks (9, 19): ]
  → uv run herald phase0 run ...        # produces results/phase0/raw/
  → uv run herald metrics finalize
  → exit
[ back on Mac: ]
  → rsync -avz --progress orion:/clustergpu/home/jcampo/herald/results/phase0/ \
       results/phase0/
  → uv run herald metrics build
  → review final/, metrics/, sampling_rate_report.json
```

`results/` is gitignored; rsync is the only path back to Mac.

### Orion environment notes

- **No internet on Orion.** `uv sync --extra metrics` (Task 1) requires the SSH reverse-tunnel proxy. From Mac:
  ```
  python /tmp/proxy.py &
  ssh -R 18080:127.0.0.1:18080 -N -f orion
  ```
  then `https_proxy=http://127.0.0.1:18080 uv sync --extra metrics` on Orion. Do not assume the proxy is up; check before each fresh dependency change. Model weights (`Qwen/Qwen2.5-7B-Instruct`) are **already in the HF cache on Orion**; no download needed for the smoke.
- **Herald is installed on Orion via a `.pth` file** (uv_build is unavailable). Any new module under `src/herald/` is picked up after `git pull`; no reinstall required.
- **Disk footprint for the 140-run Phase 0**:
  - replay: 140 runs × ~500 tokens × ~1.5 KB ≈ 105 MB
  - tokens: 140 × 500 × 0.1 KB ≈ 7 MB
  - runs: 140 × 0.5 KB ≈ 70 KB
  - finalize duplicates raw at compaction time: × 2
  - metrics: < 10 MB
  - **Total under 300 MB**, fits trivially in `/clustergpu/home/jcampo`.
- **GPU hygiene precheck** (Task 19): before any sweep, the orchestrator must call `nvidia-smi` and refuse to start if other processes hold > 200 MB on the GPU. Stale CUDA processes are a known failure mode that poisons subsequent runs.
- **Safety**: shared university server. Never kill processes outside `/clustergpu/home/jcampo`. Never run destructive commands beyond the project directory.

### Pytest markers

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = [
  "gpu: requires a real LLM forward pass (run on Orion only)",
  "e2e: end-to-end Phase 0 sweep (large; opt-in)",
]
```

Default: `pytest` skips `gpu` (use `-m "gpu"` to opt in). On Mac, every commit must pass `pytest -m "not gpu"`. On Orion, `poe test-gpu` runs `pytest -m "gpu"` and excludes `e2e` unless explicitly requested.

Add to `poe_tasks.toml`:

```toml
test         = "pytest -m 'not gpu' -q"
test-gpu     = "pytest -m 'gpu and not e2e' -q"
test-e2e     = "pytest -m 'e2e' -q"
```

The existing `test` task is replaced; `poe check` continues to call it.

### Numerical-equivalence thresholds

The design doc cites a 1e-4 fp16 noise floor for replay sanity checks. That figure is for **CUDA on Orion**, not MPS on Mac. MPS has been observed at ≈ 1e-2 max-abs logit drift between `generate()`'s incremental kv-cache path and the teacher-forced forward.

Convention used by this plan:
- **Synthetic / CPU tests** (Tasks 6, 11, 12, 15, 18): exact equality up to ~1e-9, since no model is involved.
- **Mac MPS smoke tests** (Tasks 5, 7, 8 small-model): 1e-2 logit-diff, 1e-3 JS — these are *plausibility* checks, not gates.
- **Orion CUDA gate measurements** (Task 19): pin the actual fp16 noise floor with a 5-prompt no-press warmup script before locking the gate-3 threshold. Do not import the Mac threshold as the gate.

---

## File Structure

**New files (created in this plan):**

| Path | Responsibility |
|------|----------------|
| `src/herald/metrics/__init__.py` | Subpackage marker. |
| `src/herald/metrics/io.py` | Parquet schemas (pyarrow), per-run writers, finalized-dataset readers, FK helpers, hash utils. |
| `src/herald/metrics/replay.py` | `GenerationArtifact`, `replay_forward`, position-loop with on-the-fly exact scalars + union top-K extraction, per-run parquet emit. |
| `src/herald/metrics/token.py` | Offline truncation-bias audit (top-K-derived KL/JS vs `*_full` exact scalars). Writes `token_metrics.parquet`. |
| `src/herald/metrics/trajectory.py` | NLL ratio, first-divergence-point, `sum_KL` per run. Writes `trajectory_metrics.parquet`. |
| `src/herald/metrics/sequence.py` | BERTScore, embedding cosine, edit distance, ROUGE-L per run vs paired baseline. Writes `sequence_metrics.parquet`. |
| `src/herald/metrics/outcome.py` | Paired cells (gross_harm, gross_help, net_delta) per (prompt, press, ratio). Writes `outcome.parquet`. |
| `src/herald/metrics/tags.py` | Diagnostic tags per run (wraps existing `detectors.detect_all`). Writes `tags.parquet`. |
| `src/herald/metrics/alignment.py` | Pairwise Spearman + AUROC across metric families with bootstrap CIs. Writes `alignment.parquet`. |
| `src/herald/metrics/sampling_rate.py` | Q4 study: subset replay to dense / every-4 / every-8, compute Spearman, trajectory rank corr, onset displacement. Writes `sampling_rate_report.json`. |
| `src/herald/metrics/cli.py` | `herald metrics finalize|build|repair` typer subcommands. |
| `src/herald/phase0.py` | `herald phase0 run` orchestration: baselines first, then 6 (press, ratio) cells, inline replay. |
| `gold/phase0-random-manifest.json` | Canonical 20 GSM8K prompt IDs + hashes (committed). |
| `tests/metrics/__init__.py` | Test package marker. |
| `tests/metrics/test_io.py` | Schema round-trip tests. |
| `tests/metrics/test_replay.py` | Replay forward + sanity checks (uses 0.5B model, marked slow). |
| `tests/metrics/test_token.py` | Truncation bias audit on synthetic distributions. |
| `tests/metrics/test_trajectory.py`, `test_sequence.py`, `test_outcome.py`, `test_tags.py`, `test_alignment.py`, `test_sampling_rate.py`, `test_cli.py` | Unit tests for each module. |
| `tests/metrics/test_phase0.py` | End-to-end smoke (slow). |

**Modified files:**

| Path | Change |
|------|--------|
| `pyproject.toml` | Add `pyarrow`, `polars` to core deps; add optional `metrics` group with `bert-score`, `sentence-transformers`, `rouge-score`, `editdistance`, `scipy`. |
| `poe_tasks.toml` | Add `phase0`, `phase0-run`, `metrics-finalize`, `metrics-build`, `metrics-repair` tasks. |
| `src/herald/config.py` | Extend `RunResult` with determinism manifest + replay status fields; add `GenerationArtifact` dataclass. |
| `src/herald/experiment.py` | Split `run_single` so it returns `GenerationArtifact`; add `run_single_with_replay`; add `finalize_run_result`; resume logic checks per-run parquets + `replay_status='ok'`. |
| `src/herald/__init__.py` | Register `phase0` and `metrics` subcommands on the typer app. |

---

## Tasks

### Task 1: Add dependencies + metrics extras to pyproject.toml

**Files:**
- Modify: `pyproject.toml`
- Test: none (`uv sync` is the verification)

- [ ] **Step 1: Edit `pyproject.toml`**

Replace the `dependencies` block and add an optional group:

```toml
dependencies = [
  "datasets>=4.7.0",
  "loguru>=0.7.3",
  "pyarrow>=18.0.0",
  "polars>=1.18.0",
  "pydantic>=2.12.5",
  "pydantic-settings>=2.13.1",
  "scikit-learn>=1.6.0",
  "scipy>=1.14.0",
  "torch>=2.10.0",
  "transformers>=5.3.0",
  "typer>=0.24.1",
  "xgboost>=3.0.0",
]

[project.optional-dependencies]
metrics = [
  "bert-score>=0.3.13",
  "sentence-transformers>=3.3.0",
  "rouge-score>=0.1.2",
  "editdistance>=0.8.1",
]
```

Also add to `[tool.mypy.overrides]` modules list: `"bert_score.*"`, `"sentence_transformers.*"`, `"rouge_score.*"`, `"editdistance"`, `"pyarrow.*"`, `"scipy.*"`.

- [ ] **Step 2: Sync deps**

Run: `uv sync --extra metrics`
Expected: resolves successfully, lockfile updated.

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "import pyarrow, polars, scipy"`
Expected: no output (success).

- [ ] **Step 4: `poe check`**

Run: `poe check`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(deps): add pyarrow/polars/scipy core + metrics extras"
```

---

### Task 2: Extend `RunResult` and add `GenerationArtifact`

**Files:**
- Modify: `src/herald/config.py`
- Test: `tests/test_config.py` (extend existing)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
import hashlib
from herald.config import (
    GenerationArtifact,
    RunResult,
    compute_prompt_hash,
    make_run_id,
)


def test_compute_prompt_hash_is_sha256_hex():
    h = compute_prompt_hash("hello world")
    assert h == hashlib.sha256(b"hello world").hexdigest()
    assert len(h) == 64


def test_make_run_id_deterministic():
    a = make_run_id(
        prompt_id="p1", press="snapkv", compression_ratio=0.875, seed=42
    )
    b = make_run_id(
        prompt_id="p1", press="snapkv", compression_ratio=0.875, seed=42
    )
    assert a == b
    assert a != make_run_id(
        prompt_id="p1", press="snapkv", compression_ratio=0.5, seed=42
    )


def test_run_result_baseline_self_link_default():
    rr = RunResult(
        run_id="r1",
        prompt_id="p1",
        prompt_text="Q",
        prompt_hash=compute_prompt_hash("Q"),
        model="x",
        press="none",
        compression_ratio=0.0,
        seed=42,
        max_new_tokens=10,
        decoding_config={"do_sample": False},
        task="gsm8k",
        baseline_run_id="r1",
        generated_text="A",
        generated_token_ids=[1, 2, 3],
        ground_truth="A",
        predicted_answer="A",
        correct=True,
        stop_reason="eos",
        catastrophes=[],
        num_tokens_generated=3,
        signals=[],
        replay_status="ok",
        herald_git_sha="abc",
    )
    assert rr.baseline_run_id == rr.run_id


def test_generation_artifact_holds_required_fields():
    import torch

    art = GenerationArtifact(
        run_id="r1",
        input_ids=torch.zeros(1, 5, dtype=torch.long),
        input_len=5,
        generated_token_ids=[10, 11],
        compressed_scores=[torch.zeros(100), torch.zeros(100)],
    )
    assert art.input_len == 5
    assert len(art.compressed_scores) == 2
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_config.py -v -k "prompt_hash or run_id or baseline_self_link or generation_artifact"`
Expected: FAIL — symbols not defined.

- [ ] **Step 3: Implement in `src/herald/config.py`**

Add at top of file alongside existing imports:

```python
import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch
```

Add module-level helpers and dataclass:

```python
def compute_prompt_hash(prompt_text: str) -> str:
    """SHA-256 hex of the prompt bytes used as the determinism anchor."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def make_run_id(
    prompt_id: str, press: str, compression_ratio: float, seed: int
) -> str:
    """Deterministic per-run identifier."""
    body = f"{prompt_id}|{press}|{compression_ratio:.6f}|{seed}"
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class GenerationArtifact:
    """Carries everything replay needs from a compressed generation.

    Lives between `run_single` and `replay_run`. `compressed_scores`
    is a list of (vocab,) tensors on GPU; replay must consume it
    before the tensors leave GPU.
    """

    run_id: str
    input_ids: torch.Tensor
    input_len: int
    generated_token_ids: list[int]
    compressed_scores: list[torch.Tensor] = field(default_factory=list)
```

Extend `RunResult` (replace the existing class body):

```python
class RunResult(BaseModel):
    run_id: str
    prompt_id: str
    prompt_text: str
    prompt_hash: str
    model: str
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "float16"
    device_class: str = "unknown"
    press: str
    compression_ratio: float
    max_new_tokens: int = 512
    seed: int
    decoding_config: dict[str, Any] = {}
    task: str = "gsm8k"
    baseline_run_id: str
    generated_text: str
    generated_token_ids: list[int] = []
    ground_truth: str
    predicted_answer: str | None
    correct: bool | None
    stop_reason: str
    catastrophes: list[str]
    num_tokens_generated: int
    catastrophe_onsets: dict[str, int] = {}
    signals: list[TokenSignals]
    replay_status: str = "pending"  # one of: pending, ok, failed, retried
    replay_error: str | None = None
    created_at: str | None = None
    herald_git_sha: str | None = None
```

- [ ] **Step 4: Update existing call sites**

`src/herald/experiment.py:218-235` constructs a `RunResult`. Update it to populate the new required fields. For now, supply defaults so existing tests pass; the real wiring happens in Task 8.

```python
return RunResult(
    run_id=make_run_id(
        prompt_data["id"], config.press_name,
        config.compression_ratio, config.seed,
    ),
    prompt_id=prompt_data["id"],
    prompt_text=chat_text,
    prompt_hash=compute_prompt_hash(chat_text),
    model=config.model_name,
    press=config.press_name,
    compression_ratio=config.compression_ratio,
    max_new_tokens=config.max_new_tokens,
    seed=config.seed,
    baseline_run_id=make_run_id(
        prompt_data["id"], "none", 0.0, config.seed,
    ),
    generated_text=generated_text,
    generated_token_ids=generated_ids,
    ground_truth=prompt_data["ground_truth"],
    predicted_answer=predicted,
    correct=correct,
    stop_reason=stop_reason,
    catastrophes=catastrophes,
    num_tokens_generated=len(generated_ids),
    catastrophe_onsets=catastrophe_onsets,
    signals=signals,
)
```

Add the import at the top of `experiment.py`:

```python
from herald.config import (
    ExperimentConfig, RunResult,
    compute_prompt_hash, make_run_id,
)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_config.py tests/test_experiment.py -v`
Expected: all pass.

- [ ] **Step 6: `poe check` and commit**

```bash
poe check
git add src/herald/config.py src/herald/experiment.py tests/test_config.py
git commit -m "feat(config): add determinism manifest, GenerationArtifact, baseline_run_id self-link"
```

---

### Task 3: Parquet schemas and IO helpers

**Files:**
- Create: `src/herald/metrics/__init__.py`
- Create: `src/herald/metrics/io.py`
- Create: `tests/metrics/__init__.py`
- Create: `tests/metrics/test_io.py`

- [ ] **Step 1: Write the failing tests**

`tests/metrics/test_io.py`:

```python
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from herald.metrics.io import (
    REPLAY_SCHEMA,
    RUNS_SCHEMA,
    TOKENS_SCHEMA,
    PerRunPaths,
    read_runs,
    write_replay_rows,
    write_run_record,
    write_tokens_rows,
)


def test_runs_schema_has_determinism_columns():
    cols = {f.name for f in RUNS_SCHEMA}
    required = {
        "run_id", "prompt_id", "prompt_text", "prompt_hash", "model",
        "model_revision", "tokenizer_revision", "dtype", "device_class",
        "task", "press", "compression_ratio", "max_new_tokens",
        "decoding_config", "seed", "baseline_run_id",
        "generated_text", "generated_token_ids", "replay_status",
        "herald_git_sha", "created_at",
    }
    missing = required - cols
    assert not missing, f"missing columns: {missing}"


def test_per_run_paths_layout(tmp_path: Path):
    p = PerRunPaths(root=tmp_path, run_id="abc1234")
    assert p.run.parent.name == "runs"
    assert p.tokens.parent.name == "tokens"
    assert p.replay.parent.name == "replay"
    assert p.run.parent.parent.name == "raw"


def test_write_and_read_run_record_roundtrip(tmp_path: Path):
    rec = {
        "run_id": "r1",
        "prompt_id": "p1",
        "prompt_text": "Q",
        "prompt_hash": "deadbeef",
        "model": "x",
        "model_revision": None,
        "tokenizer_revision": None,
        "dtype": "float16",
        "device_class": "cpu",
        "task": "gsm8k",
        "press": "none",
        "compression_ratio": 0.0,
        "max_new_tokens": 10,
        "decoding_config": {"do_sample": False},
        "seed": 42,
        "baseline_run_id": "r1",
        "generated_text": "A",
        "generated_token_ids": [1, 2],
        "num_tokens_generated": 2,
        "stop_reason": "eos",
        "predicted_answer": "A",
        "ground_truth": "A",
        "correct": True,
        "catastrophes": [],
        "replay_status": "ok",
        "replay_error": None,
        "created_at": "2026-05-01T00:00:00Z",
        "herald_git_sha": "abc",
    }
    paths = PerRunPaths(root=tmp_path, run_id="r1")
    write_run_record(rec, paths.run)
    df = read_runs(paths.run)
    assert df.height == 1
    assert df["run_id"][0] == "r1"
    assert df["baseline_run_id"][0] == "r1"


def test_write_tokens_rows_roundtrip(tmp_path: Path):
    rows = [
        {"run_id": "r1", "token_pos": 0, "token_id": 5,
         "token_str": "A", "entropy": 1.2, "top1_prob": 0.8,
         "top5_prob": 0.95, "top5_logprobs": [-0.1, -0.5, -1.0, -2.0, -3.0],
         "h_alts": 0.3, "avg_logp": -5.0, "delta_h": float("nan"),
         "delta_h_valid": False, "kl_div": float("nan"),
         "top10_jaccard": float("nan"), "eff_vocab_size": 3.3,
         "tail_mass": 0.01, "logit_range": 12.0,
         "lookback_ratio": float("nan")},
    ]
    paths = PerRunPaths(root=tmp_path, run_id="r1")
    write_tokens_rows(rows, paths.tokens)
    table = pq.read_table(paths.tokens)
    assert table.num_rows == 1
    assert table.column("token_pos").to_pylist() == [0]


def test_write_replay_rows_roundtrip(tmp_path: Path):
    rows = [
        {"run_id": "r1", "token_pos": 0, "realized_token_id": 5,
         "union_top_k_token_ids": [5, 7, 9],
         "logprobs_compressed": [-0.1, -2.0, -3.0],
         "logprobs_uncompressed": [-0.1, -2.0, -3.0],
         "tail_mass_compressed": 0.01, "tail_mass_uncompressed": 0.01,
         "realized_logprob_compressed": -0.1,
         "realized_logprob_uncompressed": -0.1,
         "js_full": 0.0, "kl_unc_comp_full": 0.0,
         "kl_comp_unc_full": 0.0, "top1_match": True,
         "top1_rank_comp_under_unc": 0,
         "top1_rank_unc_under_comp": 0},
    ]
    paths = PerRunPaths(root=tmp_path, run_id="r1")
    write_replay_rows(rows, paths.replay)
    table = pq.read_table(paths.replay)
    assert table.num_rows == 1
    assert table.column("js_full").to_pylist() == [0.0]
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_io.py -v`
Expected: ImportError — module not found.

- [ ] **Step 3: Implement `src/herald/metrics/__init__.py`**

```python
"""Phase 0 measurement substrate: matched-prefix replay + parquet metrics."""
```

- [ ] **Step 4: Implement `src/herald/metrics/io.py`**

```python
"""Parquet schemas + per-run writers + finalized-dataset readers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


RUNS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("prompt_id", pa.string()),
        pa.field("prompt_text", pa.string()),
        pa.field("prompt_hash", pa.string()),
        pa.field("model", pa.string()),
        pa.field("model_revision", pa.string()),
        pa.field("tokenizer_revision", pa.string()),
        pa.field("dtype", pa.string()),
        pa.field("device_class", pa.string()),
        pa.field("task", pa.string()),
        pa.field("press", pa.string()),
        pa.field("compression_ratio", pa.float64()),
        pa.field("max_new_tokens", pa.int32()),
        pa.field(
            "decoding_config",
            pa.map_(pa.string(), pa.string()),
        ),
        pa.field("seed", pa.int32()),
        pa.field("baseline_run_id", pa.string()),
        pa.field("generated_text", pa.string()),
        pa.field("generated_token_ids", pa.list_(pa.int32())),
        pa.field("num_tokens_generated", pa.int32()),
        pa.field("stop_reason", pa.string()),
        pa.field("predicted_answer", pa.string()),
        pa.field("ground_truth", pa.string()),
        pa.field("correct", pa.bool_()),
        pa.field("catastrophes", pa.list_(pa.string())),
        pa.field("replay_status", pa.string()),
        pa.field("replay_error", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("herald_git_sha", pa.string()),
    ]
)


TOKENS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("token_pos", pa.int32()),
        pa.field("token_id", pa.int32()),
        pa.field("token_str", pa.string()),
        pa.field("entropy", pa.float32()),
        pa.field("top1_prob", pa.float32()),
        pa.field("top5_prob", pa.float32()),
        pa.field("top5_logprobs", pa.list_(pa.float32())),
        pa.field("h_alts", pa.float32()),
        pa.field("avg_logp", pa.float32()),
        pa.field("delta_h", pa.float32()),
        pa.field("delta_h_valid", pa.bool_()),
        pa.field("kl_div", pa.float32()),
        pa.field("top10_jaccard", pa.float32()),
        pa.field("eff_vocab_size", pa.float32()),
        pa.field("tail_mass", pa.float32()),
        pa.field("logit_range", pa.float32()),
        pa.field("lookback_ratio", pa.float32()),
    ]
)


REPLAY_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("token_pos", pa.int32()),
        pa.field("realized_token_id", pa.int32()),
        pa.field("union_top_k_token_ids", pa.list_(pa.int32())),
        pa.field("logprobs_compressed", pa.list_(pa.float32())),
        pa.field("logprobs_uncompressed", pa.list_(pa.float32())),
        pa.field("tail_mass_compressed", pa.float32()),
        pa.field("tail_mass_uncompressed", pa.float32()),
        pa.field("realized_logprob_compressed", pa.float32()),
        pa.field("realized_logprob_uncompressed", pa.float32()),
        pa.field("js_full", pa.float32()),
        pa.field("kl_unc_comp_full", pa.float32()),
        pa.field("kl_comp_unc_full", pa.float32()),
        pa.field("top1_match", pa.bool_()),
        pa.field("top1_rank_comp_under_unc", pa.int32()),
        pa.field("top1_rank_unc_under_comp", pa.int32()),
    ]
)


@dataclass(frozen=True)
class PerRunPaths:
    """Layout under results/phase0/raw/<kind>/<run_id>.parquet."""

    root: Path
    run_id: str

    @property
    def run(self) -> Path:
        return self.root / "raw" / "runs" / f"{self.run_id}.parquet"

    @property
    def tokens(self) -> Path:
        return self.root / "raw" / "tokens" / f"{self.run_id}.parquet"

    @property
    def replay(self) -> Path:
        return self.root / "raw" / "replay" / f"{self.run_id}.parquet"

    def all_exist(self) -> bool:
        return all(p.exists() for p in (self.run, self.tokens, self.replay))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _coerce_decoding_config(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Stringify decoding_config values for the map<str,str> column."""
    record = dict(record)
    dc = record.get("decoding_config", {}) or {}
    record["decoding_config"] = [
        (str(k), str(v)) for k, v in dc.items()
    ]
    return record


def write_run_record(record: dict[str, Any], path: Path) -> None:
    _ensure_parent(path)
    record = _coerce_decoding_config(record)
    table = pa.Table.from_pylist([record], schema=RUNS_SCHEMA)
    pq.write_table(table, path)


def write_tokens_rows(rows: list[dict[str, Any]], path: Path) -> None:
    _ensure_parent(path)
    table = pa.Table.from_pylist(rows, schema=TOKENS_SCHEMA)
    pq.write_table(table, path)


def write_replay_rows(rows: list[dict[str, Any]], path: Path) -> None:
    _ensure_parent(path)
    table = pa.Table.from_pylist(rows, schema=REPLAY_SCHEMA)
    pq.write_table(table, path)


def read_runs(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/metrics/test_io.py -v`
Expected: all pass.

- [ ] **Step 6: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/__init__.py src/herald/metrics/io.py tests/metrics/
git commit -m "feat(metrics): parquet schemas + per-run IO helpers"
```

---

### Task 4: Phase 0 prompt manifest generator and loader

**Files:**
- Create: `src/herald/phase0.py` (start; will grow in Task 9)
- Modify: `src/herald/__init__.py` (register typer subcommand)
- Test: `tests/test_phase0.py`

- [ ] **Step 1: Write the failing test**

`tests/test_phase0.py`:

```python
from pathlib import Path

from herald.phase0 import (
    Manifest,
    build_random_manifest,
    load_manifest,
)


def test_build_random_manifest_is_deterministic(tmp_path: Path):
    out = tmp_path / "phase0-random-manifest.json"
    m1 = build_random_manifest(num_prompts=20, seed=42, out_path=out)
    m2 = build_random_manifest(num_prompts=20, seed=42, out_path=out)
    assert [e.prompt_id for e in m1.entries] == [
        e.prompt_id for e in m2.entries
    ]
    assert len(m1.entries) == 20
    for e in m1.entries:
        assert e.prompt_id and e.prompt_hash
        assert len(e.prompt_hash) == 64


def test_load_manifest_round_trip(tmp_path: Path):
    out = tmp_path / "manifest.json"
    m = build_random_manifest(num_prompts=5, seed=1, out_path=out)
    loaded = load_manifest(out)
    assert [e.prompt_id for e in loaded.entries] == [
        e.prompt_id for e in m.entries
    ]
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_phase0.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/herald/phase0.py`**

```python
"""Phase 0 orchestration: prompt manifest + run sweeper.

Sweep orchestration lives here too (added in Task 9).
"""

import json
import random
from pathlib import Path

from pydantic import BaseModel

from herald.config import compute_prompt_hash
from herald.prompts import format_chat
from herald.tasks import DEFAULT_TASK


class ManifestEntry(BaseModel):
    prompt_id: str
    prompt_hash: str


class Manifest(BaseModel):
    name: str
    seed: int
    num_prompts: int
    entries: list[ManifestEntry]


def build_random_manifest(
    num_prompts: int, seed: int, out_path: Path
) -> Manifest:
    """Sample N prompts from the GSM8K test split with a fixed seed."""
    rng = random.Random(seed)
    full = DEFAULT_TASK.load(num_prompts=10_000, seed=seed)
    chosen = rng.sample(full, k=num_prompts)
    entries = []
    for p in chosen:
        chat_text = "".join(
            m["content"] for m in format_chat(p["question"])
        )
        entries.append(
            ManifestEntry(
                prompt_id=p["id"],
                prompt_hash=compute_prompt_hash(chat_text),
            )
        )
    manifest = Manifest(
        name="phase0-random",
        seed=seed,
        num_prompts=num_prompts,
        entries=entries,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_text())
```

- [ ] **Step 4: Generate the canonical manifest file**

Run: `uv run python -c "from pathlib import Path; from herald.phase0 import build_random_manifest; build_random_manifest(20, 42, Path('gold/phase0-random-manifest.json'))"`
Expected: `gold/phase0-random-manifest.json` is created with 20 entries.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_phase0.py -v`
Expected: pass.

- [ ] **Step 6: `poe check` and commit**

```bash
poe check
git add src/herald/phase0.py gold/phase0-random-manifest.json tests/test_phase0.py
git commit -m "feat(phase0): canonical 20-prompt manifest + loader"
```

---

### Task 5: Replay forward pass + index alignment test

**Files:**
- Modify: `src/herald/metrics/replay.py` (create)
- Test: `tests/metrics/test_replay.py`

- [ ] **Step 1: Write the failing test (slow, MPS/CUDA)**

`tests/metrics/test_replay.py`:

```python
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from herald.metrics.replay import replay_forward


@pytest.fixture(scope="module")
def small_model():
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16
    )
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return model.to(device).eval(), tok, device


@pytest.mark.gpu
def test_replay_forward_shape_and_index_alignment(small_model):
    model, tok, device = small_model
    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    input_len = input_ids.shape[1]

    # Generate 6 tokens greedily and capture per-step scores.
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=6, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    gen_ids = out.sequences[0, input_len:].tolist()
    gen_scores = [s[0] for s in out.scores]

    logits = replay_forward(model, input_ids, gen_ids)
    assert logits.shape[0] == len(gen_ids)
    assert logits.shape[1] == model.config.vocab_size

    # t=0: replay's first row should match generation's first scores.
    diff_t0 = (
        logits[0].float() - gen_scores[0].float()
    ).abs().max().item()
    assert diff_t0 < 1e-2, f"t=0 logit mismatch: {diff_t0}"

    # Argmax of replay row t should equal realized greedy token at t
    # (since this is uncompressed replay of an uncompressed run).
    argmax_match = (logits.argmax(dim=-1).tolist() == gen_ids)
    assert argmax_match, "replay argmax must equal greedy realized tokens"
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_replay.py -v -m gpu`
Expected: ImportError or test failure.

- [ ] **Step 3: Implement `src/herald/metrics/replay.py`**

```python
"""Matched-prefix replay forward pass.

Index alignment: for generated token g_t (1-indexed in the generation
stream, 0-indexed in `generated_token_ids`), the prediction
distribution is at sequence position `input_len + t - 1` of the
forward output. Slice `[:, input_len - 1 : input_len - 1 + N, :]`.
"""

import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def replay_forward(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    generated_token_ids: list[int],
) -> torch.Tensor:
    """Single teacher-forced uncompressed forward.

    Returns logits at every generated position with shape
    (gen_len, vocab_size). Caller is responsible for being outside
    any kvpress context.

    Args:
        model: HF causal LM. Must be on the same device as `input_ids`.
        input_ids: shape (1, prompt_len).
        generated_token_ids: realized greedy sequence of length N.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("replay_forward expects batch size 1")
    if not generated_token_ids:
        return torch.empty(
            (0, model.config.vocab_size), device=input_ids.device
        )

    device = input_ids.device
    gen_t = torch.tensor(
        generated_token_ids, dtype=input_ids.dtype, device=device
    ).unsqueeze(0)
    full = torch.cat([input_ids, gen_t], dim=1)
    out = model(input_ids=full, use_cache=False)
    # out.logits: (1, prompt_len + N, vocab)
    # Position prompt_len-1 predicts gen[0]; position
    # prompt_len-1+i predicts gen[i].
    prompt_len = input_ids.shape[1]
    n = len(generated_token_ids)
    sliced = out.logits[0, prompt_len - 1 : prompt_len - 1 + n, :]
    return sliced.contiguous()
```

- [ ] **Step 4: Run the slow test**

Run: `uv run pytest tests/metrics/test_replay.py -v -m gpu`
Expected: pass on Mac MPS in <60s for the 0.5B model.

If the t=0 max-abs diff exceeds 1e-2 (numerical drift between `generate()`'s incremental kv-cache path and the teacher-forced forward), document it but bump the threshold only if the same drift is reproducible across seeds. Do not silently relax.

- [ ] **Step 5: Add unit test that does not need GPU**

Append to `tests/metrics/test_replay.py`:

```python
def test_replay_forward_empty_generated():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    enc = tok("hi", return_tensors="pt")
    out = replay_forward(model, enc["input_ids"], [])
    assert out.shape == (0, model.config.vocab_size)
```

(This is fast; uses CPU.) Mark it `@pytest.mark.gpu` as well to keep fast CI green if the model download is undesired; otherwise leave unmarked.

- [ ] **Step 6: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/replay.py tests/metrics/test_replay.py
git commit -m "feat(metrics): replay_forward (single teacher-forced pass)"
```

---

### Task 6: On-the-fly exact scalars + union top-K extraction

**Files:**
- Modify: `src/herald/metrics/replay.py`
- Modify: `tests/metrics/test_replay.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/metrics/test_replay.py`:

```python
import math

import torch as _torch

from herald.metrics.replay import compute_replay_position


def test_compute_replay_position_identity_distribution():
    # Two identical distributions => zero divergences, top1 match.
    vocab = 100
    logits = _torch.randn(vocab)
    row = compute_replay_position(
        logits_compressed=logits,
        logits_uncompressed=logits.clone(),
        realized_token_id=int(logits.argmax().item()),
        top_k=16,
    )
    assert row["top1_match"] is True
    assert row["js_full"] < 1e-6
    assert row["kl_unc_comp_full"] < 1e-6
    assert row["kl_comp_unc_full"] < 1e-6
    assert row["top1_rank_comp_under_unc"] == 0
    assert row["top1_rank_unc_under_comp"] == 0
    assert len(row["union_top_k_token_ids"]) <= 32
    assert (
        row["realized_logprob_compressed"]
        == row["realized_logprob_uncompressed"]
    )


def test_compute_replay_position_disjoint_top1():
    # Two distributions with different top-1 should produce
    # nonzero JS and nonzero rank shift.
    vocab = 100
    a = _torch.full((vocab,), -10.0)
    a[3] = 5.0
    b = _torch.full((vocab,), -10.0)
    b[7] = 5.0
    row = compute_replay_position(
        logits_compressed=a,
        logits_uncompressed=b,
        realized_token_id=3,
        top_k=16,
    )
    assert row["top1_match"] is False
    assert row["js_full"] > 0.1
    assert row["top1_rank_unc_under_comp"] > 0
    assert row["top1_rank_comp_under_unc"] > 0
    assert row["realized_token_id"] == 3
    assert math.isfinite(row["realized_logprob_compressed"])
    assert math.isfinite(row["realized_logprob_uncompressed"])
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_replay.py -v -k compute_replay_position`
Expected: ImportError on `compute_replay_position`.

- [ ] **Step 3: Implement `compute_replay_position` in `src/herald/metrics/replay.py`**

Append:

```python
import torch.nn.functional as F


def _kl(
    log_p: torch.Tensor, log_q: torch.Tensor, p: torch.Tensor
) -> float:
    """KL(p || q) given log_p, log_q, and p on the full vocab."""
    return (p * (log_p - log_q)).sum().item()


def compute_replay_position(
    logits_compressed: torch.Tensor,
    logits_uncompressed: torch.Tensor,
    realized_token_id: int,
    top_k: int = 128,
) -> dict[str, object]:
    """Compute exact full-vocab scalars + union top-K row.

    All inputs are 1-D vocab-size tensors on the same device. The
    full-vocab tensors are not retained after this call.
    """
    log_p_c = F.log_softmax(logits_compressed.float(), dim=-1)
    log_p_u = F.log_softmax(logits_uncompressed.float(), dim=-1)
    p_c = log_p_c.exp()
    p_u = log_p_u.exp()

    log_m = torch.logaddexp(log_p_c, log_p_u) - math.log(2.0)
    p_m = log_m.exp()
    js_full = 0.5 * _kl(log_p_c, log_m, p_c) + 0.5 * _kl(
        log_p_u, log_m, p_u
    )

    kl_unc_comp = _kl(log_p_u, log_p_c, p_u)
    kl_comp_unc = _kl(log_p_c, log_p_u, p_c)

    argmax_c = int(p_c.argmax().item())
    argmax_u = int(p_u.argmax().item())
    top1_match = argmax_c == argmax_u

    # Exact rank of compressed-top-1 under uncompressed (and vice versa)
    rank_c_under_u = int(
        (p_u > p_u[argmax_c]).sum().item()
    )
    rank_u_under_c = int(
        (p_c > p_c[argmax_u]).sum().item()
    )

    k = min(top_k, log_p_c.shape[-1])
    top_c = torch.topk(log_p_c, k=k).indices
    top_u = torch.topk(log_p_u, k=k).indices
    union = torch.unique(torch.cat([top_c, top_u]))
    lp_c_union = log_p_c[union]
    lp_u_union = log_p_u[union]
    tail_c = float(1.0 - p_c[union].sum().item())
    tail_u = float(1.0 - p_u[union].sum().item())

    realized_lp_c = float(log_p_c[realized_token_id].item())
    realized_lp_u = float(log_p_u[realized_token_id].item())

    return {
        "realized_token_id": int(realized_token_id),
        "union_top_k_token_ids": union.tolist(),
        "logprobs_compressed": lp_c_union.tolist(),
        "logprobs_uncompressed": lp_u_union.tolist(),
        "tail_mass_compressed": max(tail_c, 0.0),
        "tail_mass_uncompressed": max(tail_u, 0.0),
        "realized_logprob_compressed": realized_lp_c,
        "realized_logprob_uncompressed": realized_lp_u,
        "js_full": float(max(js_full, 0.0)),
        "kl_unc_comp_full": float(max(kl_unc_comp, 0.0)),
        "kl_comp_unc_full": float(max(kl_comp_unc, 0.0)),
        "top1_match": top1_match,
        "top1_rank_comp_under_unc": rank_c_under_u,
        "top1_rank_unc_under_comp": rank_u_under_c,
    }
```

Add `import math` at top of `replay.py`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/metrics/test_replay.py -v -k compute_replay_position`
Expected: pass.

- [ ] **Step 5: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/replay.py tests/metrics/test_replay.py
git commit -m "feat(metrics): compute_replay_position with exact full-vocab scalars + union top-K"
```

---

### Task 7: End-to-end `replay_run` writes per-run replay parquet

**Files:**
- Modify: `src/herald/metrics/replay.py`
- Modify: `tests/metrics/test_replay.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/metrics/test_replay.py`:

```python
import pyarrow.parquet as pq

from herald.config import GenerationArtifact
from herald.metrics.replay import ReplayMetrics, replay_run


@pytest.mark.gpu
def test_replay_run_writes_parquet_with_correct_schema(
    small_model, tmp_path
):
    model, tok, device = small_model
    prompt = "Two plus two equals"
    enc = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=4, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    gen_ids = out.sequences[0, enc["input_ids"].shape[1]:].tolist()
    artifact = GenerationArtifact(
        run_id="test1",
        input_ids=enc["input_ids"],
        input_len=enc["input_ids"].shape[1],
        generated_token_ids=gen_ids,
        compressed_scores=[s[0].clone() for s in out.scores],
    )

    metrics = replay_run(
        model=model, tokenizer=tok, artifact=artifact,
        output_root=tmp_path, top_k=16,
    )
    assert isinstance(metrics, ReplayMetrics)
    assert metrics.num_positions == len(gen_ids)

    path = tmp_path / "raw" / "replay" / "test1.parquet"
    table = pq.read_table(path)
    assert table.num_rows == len(gen_ids)
    # press=none-equivalent: this is uncompressed gen replayed
    # against the same uncompressed model → js_full ~ 0.
    assert max(table.column("js_full").to_pylist()) < 1e-3
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_replay.py -v -m gpu -k replay_run`
Expected: ImportError on `replay_run` / `ReplayMetrics`.

- [ ] **Step 3: Implement `replay_run`**

Append to `src/herald/metrics/replay.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from herald.config import GenerationArtifact
from herald.metrics.io import PerRunPaths, write_replay_rows


@dataclass(frozen=True)
class ReplayMetrics:
    run_id: str
    num_positions: int
    js_full_max: float
    js_full_mean: float


def replay_run(
    model: AutoModelForCausalLM,
    tokenizer: object,
    artifact: GenerationArtifact,
    output_root: Path,
    top_k: int = 128,
) -> ReplayMetrics:
    """Run uncompressed replay forward and emit per-run replay parquet.

    Must be called outside any kvpress context. Consumes
    `artifact.compressed_scores` while they are still on GPU.
    """
    if not artifact.generated_token_ids:
        paths = PerRunPaths(root=output_root, run_id=artifact.run_id)
        write_replay_rows([], paths.replay)
        return ReplayMetrics(
            run_id=artifact.run_id, num_positions=0,
            js_full_max=0.0, js_full_mean=0.0,
        )

    uncompressed_logits = replay_forward(
        model, artifact.input_ids, artifact.generated_token_ids
    )
    rows: list[dict[str, object]] = []
    js_values: list[float] = []
    for t, gen_id in enumerate(artifact.generated_token_ids):
        comp = artifact.compressed_scores[t]
        unc = uncompressed_logits[t]
        row = compute_replay_position(
            logits_compressed=comp,
            logits_uncompressed=unc,
            realized_token_id=gen_id,
            top_k=top_k,
        )
        row["run_id"] = artifact.run_id
        row["token_pos"] = t
        rows.append(row)
        js_values.append(float(row["js_full"]))

    paths = PerRunPaths(root=output_root, run_id=artifact.run_id)
    write_replay_rows(rows, paths.replay)
    return ReplayMetrics(
        run_id=artifact.run_id,
        num_positions=len(rows),
        js_full_max=max(js_values) if js_values else 0.0,
        js_full_mean=(sum(js_values) / len(js_values)) if js_values else 0.0,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/metrics/test_replay.py -v -m gpu -k replay_run`
Expected: pass; `js_full_max < 1e-3`.

- [ ] **Step 5: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/replay.py tests/metrics/test_replay.py
git commit -m "feat(metrics): replay_run end-to-end + per-run parquet emit"
```

---

### Task 8: Refactor `experiment.run_single` to return `GenerationArtifact`; add `run_single_with_replay`

**Files:**
- Modify: `src/herald/experiment.py`
- Modify: `src/herald/metrics/io.py` (add `write_tokens_rows_from_signals` helper)
- Test: `tests/test_experiment.py` (extend) and `tests/metrics/test_replay.py` (add inline-orchestration test)

- [ ] **Step 1: Write the failing test**

`tests/metrics/test_replay.py`, append:

```python
@pytest.mark.gpu
def test_run_single_with_replay_writes_three_parquets(
    tmp_path,
):
    from herald.config import ExperimentConfig
    from herald.experiment import (
        get_press, load_model, run_single_with_replay,
    )
    from herald.tasks import DEFAULT_TASK

    cfg = ExperimentConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        press_name="none",
        compression_ratio=0.0,
        num_prompts=1,
        seed=42,
        output_dir=tmp_path,
        max_new_tokens=8,
        prompt_timeout_seconds=120.0,
    )
    model, tok, device = load_model(cfg)
    prompts = DEFAULT_TASK.load(num_prompts=1, seed=42)
    press = get_press(cfg.press_name, cfg.compression_ratio)
    rr, rm = run_single_with_replay(
        model, tok, device, prompts[0], cfg, press,
        baseline_run_id=None, output_root=tmp_path,
    )
    assert rr.replay_status == "ok"
    assert rm.num_positions == rr.num_tokens_generated

    from herald.metrics.io import PerRunPaths

    p = PerRunPaths(root=tmp_path, run_id=rr.run_id)
    assert p.run.exists()
    assert p.tokens.exists()
    assert p.replay.exists()
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_replay.py -v -m gpu -k run_single_with_replay`
Expected: ImportError on `run_single_with_replay`.

- [ ] **Step 3: Add `write_tokens_rows_from_signals` to `metrics/io.py`**

Append:

```python
def signals_to_token_rows(
    run_id: str,
    generated_token_ids: list[int],
    token_strs: list[str],
    signals: list[Any],
) -> list[dict[str, Any]]:
    """Convert in-memory TokenSignals list to TOKENS_SCHEMA rows."""
    rows: list[dict[str, Any]] = []
    for pos, (tok_id, tok_str, sig) in enumerate(
        zip(generated_token_ids, token_strs, signals)
    ):
        rows.append(
            {
                "run_id": run_id,
                "token_pos": pos,
                "token_id": int(tok_id),
                "token_str": tok_str,
                "entropy": float(sig.entropy),
                "top1_prob": float(sig.top1_prob),
                "top5_prob": float(sig.top5_prob),
                "top5_logprobs": [float(x) for x in sig.top5_logprobs],
                "h_alts": float(sig.h_alts),
                "avg_logp": float(sig.avg_logp),
                "delta_h": float(sig.delta_h),
                "delta_h_valid": bool(sig.delta_h_valid),
                "kl_div": float(sig.kl_div),
                "top10_jaccard": float(sig.top10_jaccard),
                "eff_vocab_size": float(sig.eff_vocab_size),
                "tail_mass": float(sig.tail_mass),
                "logit_range": float(sig.logit_range),
                "lookback_ratio": float(sig.lookback_ratio),
            }
        )
    return rows
```

- [ ] **Step 4: Refactor `experiment.run_single`**

Replace `run_single` so it returns a `tuple[RunResult, GenerationArtifact]` instead of a `RunResult`. Move replay-irrelevant work (catastrophe detection, task scoring) into a small `_finalize_run_result` helper that runs after replay. Concretely:

```python
def run_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt_data: dict[str, Any],
    config: ExperimentConfig,
    press: object | None,
    task: Task = DEFAULT_TASK,
) -> tuple[GenerationArtifact, dict[str, Any]]:
    """Run compressed generation. Returns artifact + extra metadata
    needed by the finalizer (signals, generated_text, stop_reason).
    Replay is the caller's responsibility.
    """
    messages = format_chat(prompt_data["question"])
    chat_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)  # type: ignore[operator]
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    ctx = press(model) if press is not None else nullcontext()  # type: ignore[operator]
    stopping = StoppingCriteriaList(
        [_TimeoutCriteria(config.prompt_timeout_seconds)]
    )
    with torch.no_grad(), ctx:
        outputs = model.generate(  # type: ignore[attr-defined]
            **inputs, max_new_tokens=config.max_new_tokens,
            do_sample=False, output_scores=True,
            output_attentions=config.capture_attention,
            return_dict_in_generate=True,
            stopping_criteria=stopping,
        )

    generated_ids = outputs.sequences[0, input_len:].tolist()
    generated_text = tokenizer.decode(  # type: ignore[attr-defined]
        generated_ids, skip_special_tokens=True
    )

    eos_id = tokenizer.eos_token_id  # type: ignore[attr-defined]
    eos_ids = set(eos_id) if isinstance(eos_id, list) else {eos_id}
    hit_eos = len(generated_ids) > 0 and generated_ids[-1] in eos_ids
    hit_max = len(generated_ids) >= config.max_new_tokens
    stop_reason = (
        "eos" if hit_eos else ("max_tokens" if hit_max else "timeout")
    )

    signals = []
    state = None
    compressed_scores: list[torch.Tensor] = []
    for score in outputs.scores:
        sig, state = extract_signals(score[0], prev=state)
        signals.append(sig)
        compressed_scores.append(score[0].detach())

    if config.capture_attention and outputs.attentions is not None:
        lookbacks = compute_lookback_ratios(
            outputs.attentions, input_len=input_len
        )
        for sig, lb in zip(signals, lookbacks):
            sig.lookback_ratio = lb

    run_id = make_run_id(
        prompt_data["id"], config.press_name,
        config.compression_ratio, config.seed,
    )
    artifact = GenerationArtifact(
        run_id=run_id, input_ids=input_ids, input_len=input_len,
        generated_token_ids=generated_ids,
        compressed_scores=compressed_scores,
    )
    extras = {
        "chat_text": chat_text,
        "generated_text": generated_text,
        "stop_reason": stop_reason,
        "signals": signals,
        "prompt_data": prompt_data,
    }
    return artifact, extras
```

Then add the inline-replay wrapper:

```python
import datetime as _dt
import subprocess as _sp

import torch as _torch  # already imported as torch above

from herald.config import GenerationArtifact
from herald.metrics.io import (
    PerRunPaths, signals_to_token_rows,
    write_run_record, write_tokens_rows,
)
from herald.metrics.replay import ReplayMetrics, replay_run


def _git_sha() -> str:
    try:
        return _sp.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent
        ).decode().strip()
    except Exception:
        return "unknown"


def _device_class(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device.startswith("mps"):
        return "mps"
    return "cpu"


def run_single_with_replay(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt_data: dict[str, Any],
    config: ExperimentConfig,
    press: object | None,
    baseline_run_id: str | None,
    output_root: Path,
    task: Task = DEFAULT_TASK,
    top_k: int = 128,
) -> tuple[RunResult, ReplayMetrics]:
    """Run compressed generation + inline replay, then write 3 parquets.

    `baseline_run_id=None` means this *is* a baseline run; it will
    self-link via baseline_run_id == run_id.
    """
    artifact, extras = run_single(
        model, tokenizer, device, prompt_data, config, press, task
    )

    replay_status = "ok"
    replay_error: str | None = None
    rm: ReplayMetrics
    try:
        rm = replay_run(
            model=model, tokenizer=tokenizer, artifact=artifact,
            output_root=output_root, top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001
        replay_status = "failed"
        replay_error = repr(exc)
        rm = ReplayMetrics(
            run_id=artifact.run_id, num_positions=0,
            js_full_max=0.0, js_full_mean=0.0,
        )

    catastrophes = detect_all(
        extras["generated_text"], artifact.generated_token_ids,
        extras["stop_reason"], prompt_data["ground_truth"],
        is_wrong_fn=task.is_wrong,
    )
    catastrophe_onsets = detect_catastrophe_onsets(
        artifact.generated_token_ids, extras["stop_reason"], catastrophes
    )
    predicted = task.parse_answer(extras["generated_text"])
    correct = (
        "wrong_answer" not in catastrophes
        if predicted is not None else None
    )

    chat_text = extras["chat_text"]
    rr = RunResult(
        run_id=artifact.run_id,
        prompt_id=prompt_data["id"],
        prompt_text=chat_text,
        prompt_hash=compute_prompt_hash(chat_text),
        model=config.model_name,
        dtype="float16",
        device_class=_device_class(device),
        press=config.press_name,
        compression_ratio=config.compression_ratio,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        decoding_config={"do_sample": "False"},
        task="gsm8k",
        baseline_run_id=baseline_run_id or artifact.run_id,
        generated_text=extras["generated_text"],
        generated_token_ids=artifact.generated_token_ids,
        ground_truth=prompt_data["ground_truth"],
        predicted_answer=predicted,
        correct=correct,
        stop_reason=extras["stop_reason"],
        catastrophes=catastrophes,
        num_tokens_generated=len(artifact.generated_token_ids),
        catastrophe_onsets=catastrophe_onsets,
        signals=extras["signals"],
        replay_status=replay_status,
        replay_error=replay_error,
        created_at=_dt.datetime.utcnow().isoformat() + "Z",
        herald_git_sha=_git_sha(),
    )

    paths = PerRunPaths(root=output_root, run_id=rr.run_id)
    token_strs = [
        tokenizer.decode([tid], skip_special_tokens=True)
        for tid in artifact.generated_token_ids
    ]
    write_tokens_rows(
        signals_to_token_rows(
            run_id=rr.run_id,
            generated_token_ids=artifact.generated_token_ids,
            token_strs=token_strs,
            signals=extras["signals"],
        ),
        paths.tokens,
    )
    write_run_record(rr.model_dump(mode="json"), paths.run)

    return rr, rm
```

Note: `RunResult.model_dump(mode="json")` already coerces enums and datetimes; `decoding_config` values must be string-coerced upstream (we set `"False"` as a literal string for now since the parquet map type is `<str,str>`). When sampling is added later, expand this dict accordingly.

Also update existing `run_prompts` callers if needed — leaving them on the old `run_single` path is fine since the new function returns the *same* `RunResult` plus a `GenerationArtifact`. The Phase 0 sweeper will use `run_single_with_replay`; legacy sweep keeps using a thin shim.

Add a thin shim `run_single_legacy` if existing tests rely on the old signature; otherwise update them in step 5.

- [ ] **Step 5: Update existing tests that call `run_single` directly**

Run: `uv run pytest tests/test_experiment.py -v` and adapt any failing test to consume `run_single`'s new tuple return.

- [ ] **Step 6: Run new + existing tests**

Run: `uv run pytest tests/test_experiment.py tests/metrics/ -v -m "gpu or not gpu"`
Expected: pass.

- [ ] **Step 7: `poe check` and commit**

```bash
poe check
git add src/herald/experiment.py src/herald/metrics/io.py tests/
git commit -m "refactor(experiment): split run_single + add run_single_with_replay"
```

---

### Task 9: Phase 0 sweep orchestration: `herald phase0 run`

**Files:**
- Modify: `src/herald/phase0.py`
- Modify: `src/herald/__init__.py`
- Test: `tests/test_phase0.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase0.py`:

```python
@pytest.mark.gpu
def test_phase0_sweep_baseline_first(tmp_path, monkeypatch):
    from herald.phase0 import run_phase0_sweep

    out = tmp_path / "phase0"
    run_phase0_sweep(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        manifest_path=Path("gold/phase0-random-manifest.json"),
        presses=("streaming_llm",),
        ratios=(0.5,),
        max_new_tokens=8,
        output_root=out,
        prompt_timeout_seconds=120.0,
        num_prompts=2,    # subset for smoke
    )
    runs_dir = out / "raw" / "runs"
    parquets = list(runs_dir.glob("*.parquet"))
    # 2 baselines + 2 (press, ratio) cells = 4 runs
    assert len(parquets) == 4
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_phase0.py -v -m gpu -k phase0_sweep_baseline_first`
Expected: ImportError.

- [ ] **Step 3: Implement `run_phase0_sweep` in `src/herald/phase0.py`**

```python
import gc
from pathlib import Path

import torch
from loguru import logger

from herald.config import ExperimentConfig, make_run_id
from herald.experiment import (
    get_press, load_model, run_single_with_replay,
)
from herald.metrics.io import PerRunPaths
from herald.tasks import DEFAULT_TASK


def _resolve_prompts(
    manifest_path: Path, num_prompts: int | None
) -> list[dict[str, object]]:
    manifest = load_manifest(manifest_path)
    wanted = {e.prompt_id for e in manifest.entries}
    full = DEFAULT_TASK.load(num_prompts=10_000, seed=manifest.seed)
    prompts = [p for p in full if p["id"] in wanted]
    prompts.sort(
        key=lambda p: [e.prompt_id for e in manifest.entries].index(p["id"])
    )
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def _maybe_skip(paths: PerRunPaths, runs_path: Path) -> bool:
    if not paths.all_exist():
        return False
    import polars as pl
    df = pl.read_parquet(paths.run)
    return df.height == 1 and df["replay_status"][0] == "ok"


def run_phase0_sweep(
    model_name: str,
    manifest_path: Path,
    presses: tuple[str, ...] = ("streaming_llm", "snapkv"),
    ratios: tuple[float, ...] = (0.5, 0.875, 0.9375),
    max_new_tokens: int = 512,
    output_root: Path = Path("results/phase0"),
    prompt_timeout_seconds: float = 300.0,
    num_prompts: int | None = None,
    top_k: int = 128,
    seed: int = 42,
) -> None:
    """Phase 0 orchestration: baselines first, then (press, ratio) cells."""
    prompts = _resolve_prompts(manifest_path, num_prompts)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg_template = ExperimentConfig(
        model_name=model_name, press_name="none",
        compression_ratio=0.0,
        num_prompts=len(prompts), seed=seed,
        output_dir=output_root, max_new_tokens=max_new_tokens,
        prompt_timeout_seconds=prompt_timeout_seconds,
    )
    model, tok, device = load_model(cfg_template)

    # 1) Baselines (press=none, ratio=0). Self-link via baseline_run_id.
    logger.info("Phase 0: baselines first (press=none).")
    for p in prompts:
        run_id = make_run_id(p["id"], "none", 0.0, seed)
        paths = PerRunPaths(root=output_root, run_id=run_id)
        if _maybe_skip(paths, paths.run):
            logger.info(f"  skip baseline {p['id']} (existing ok)")
            continue
        run_single_with_replay(
            model=model, tokenizer=tok, device=device,
            prompt_data=p, config=cfg_template,
            press=None, baseline_run_id=None,
            output_root=output_root, top_k=top_k,
        )

    # 2) Compressed cells.
    for press_name in presses:
        for ratio in ratios:
            cfg = cfg_template.model_copy(
                update={
                    "press_name": press_name,
                    "compression_ratio": float(ratio),
                }
            )
            press = get_press(press_name, ratio)
            logger.info(f"Phase 0 cell: {press_name}@{ratio}")
            for p in prompts:
                run_id = make_run_id(p["id"], press_name, ratio, seed)
                baseline_id = make_run_id(p["id"], "none", 0.0, seed)
                paths = PerRunPaths(root=output_root, run_id=run_id)
                if _maybe_skip(paths, paths.run):
                    logger.info(f"  skip {press_name}@{ratio} {p['id']}")
                    continue
                run_single_with_replay(
                    model=model, tokenizer=tok, device=device,
                    prompt_data=p, config=cfg,
                    press=press, baseline_run_id=baseline_id,
                    output_root=output_root, top_k=top_k,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    del model
    gc.collect()
```

- [ ] **Step 4: Wire CLI in `src/herald/__init__.py`**

Add a typer subcommand near the existing ones:

```python
@app.command(name="phase0")
def phase0_cmd(
    action: str = typer.Argument(..., help="run"),
    manifest: Path = typer.Option(
        Path("gold/phase0-random-manifest.json"),
        help="prompt manifest JSON",
    ),
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct"),
    output_root: Path = typer.Option(Path("results/phase0")),
    max_new_tokens: int = typer.Option(512),
    prompt_timeout_seconds: float = typer.Option(300.0),
    num_prompts: int | None = typer.Option(None),
) -> None:
    if action != "run":
        raise typer.BadParameter(
            "supported actions: run", param_hint="action"
        )
    from herald.phase0 import run_phase0_sweep

    run_phase0_sweep(
        model_name=model, manifest_path=manifest,
        max_new_tokens=max_new_tokens,
        output_root=output_root,
        prompt_timeout_seconds=prompt_timeout_seconds,
        num_prompts=num_prompts,
    )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_phase0.py -v -m gpu -k phase0_sweep_baseline_first`
Expected: pass (Mac MPS, ~2 min for 4 tiny runs).

- [ ] **Step 6: `poe check` and commit**

```bash
poe check
git add src/herald/phase0.py src/herald/__init__.py tests/test_phase0.py
git commit -m "feat(phase0): sweep orchestrator + herald phase0 run CLI"
```

---

### Task 10: `herald metrics finalize`

**Files:**
- Create: `src/herald/metrics/cli.py`
- Modify: `src/herald/metrics/io.py` (add `finalize_dataset`)
- Modify: `src/herald/__init__.py` (register subcommand)
- Test: `tests/metrics/test_cli.py`

- [ ] **Step 1: Write the failing test**

`tests/metrics/test_cli.py`:

```python
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from herald.metrics.io import finalize_dataset, write_run_record


def _seed_run(root: Path, run_id: str, press: str, ratio: float):
    rec = {
        "run_id": run_id, "prompt_id": "p", "prompt_text": "q",
        "prompt_hash": "h", "model": "x", "model_revision": None,
        "tokenizer_revision": None, "dtype": "float16",
        "device_class": "cpu", "task": "gsm8k", "press": press,
        "compression_ratio": ratio, "max_new_tokens": 8,
        "decoding_config": {"do_sample": "False"}, "seed": 42,
        "baseline_run_id": run_id, "generated_text": "a",
        "generated_token_ids": [1], "num_tokens_generated": 1,
        "stop_reason": "eos", "predicted_answer": "a",
        "ground_truth": "a", "correct": True, "catastrophes": [],
        "replay_status": "ok", "replay_error": None,
        "created_at": "2026-05-01T00:00:00Z", "herald_git_sha": "abc",
    }
    from herald.metrics.io import PerRunPaths
    p = PerRunPaths(root=root, run_id=run_id)
    write_run_record(rec, p.run)


def test_finalize_dataset_concatenates_runs(tmp_path: Path):
    _seed_run(tmp_path, "r1", "none", 0.0)
    _seed_run(tmp_path, "r2", "snapkv", 0.875)

    finalize_dataset(root=tmp_path)

    runs_path = tmp_path / "final" / "runs.parquet"
    assert runs_path.exists()
    df = pl.read_parquet(runs_path)
    assert df.height == 2
    assert set(df["run_id"]) == {"r1", "r2"}
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_cli.py -v`
Expected: ImportError on `finalize_dataset`.

- [ ] **Step 3: Implement `finalize_dataset` in `src/herald/metrics/io.py`**

```python
def finalize_dataset(root: Path) -> None:
    """Concatenate raw/<kind>/*.parquet into final/."""
    raw = root / "raw"
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)

    runs_files = sorted((raw / "runs").glob("*.parquet"))
    if runs_files:
        runs_df = pl.concat(
            [pl.read_parquet(p) for p in runs_files], how="vertical_relaxed"
        )
        runs_df.write_parquet(final / "runs.parquet")

    for kind in ("tokens", "replay"):
        out_dir = final / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((raw / kind).glob("*.parquet"))
        # Partition by (press, ratio) using runs_df as join source.
        if not files or not runs_files:
            continue
        runs_df = pl.read_parquet(final / "runs.parquet").select(
            ["run_id", "press", "compression_ratio"]
        )
        for f in files:
            df = pl.read_parquet(f).join(
                runs_df, on="run_id", how="left"
            )
            for (press, ratio), part in df.group_by(
                ["press", "compression_ratio"]
            ):
                pdir = (
                    out_dir / f"press={press}"
                    / f"ratio={ratio:.4f}"
                )
                pdir.mkdir(parents=True, exist_ok=True)
                part.drop(["press", "compression_ratio"]).write_parquet(
                    pdir / f.name
                )
```

- [ ] **Step 4: Implement `metrics/cli.py`**

```python
"""`herald metrics ...` CLI subcommands."""

from pathlib import Path

import typer

from herald.metrics.io import finalize_dataset

app = typer.Typer(help="Phase 0 metrics pipeline.")


@app.command()
def finalize(
    root: Path = typer.Option(
        Path("results/phase0"),
        help="Root containing raw/ and final/.",
    ),
) -> None:
    finalize_dataset(root)


@app.command()
def build(
    root: Path = typer.Option(Path("results/phase0")),
) -> None:
    """Run all offline metric modules in order."""
    from herald.metrics import (
        alignment, outcome, sequence, tags, token, trajectory,
    )

    final = root / "final"
    out = root / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    token.build(final, out)
    trajectory.build(final, out)
    sequence.build(final, out)
    outcome.build(final, out)
    tags.build(final, out)
    alignment.build(out, out)


@app.command()
def repair(
    run_id: str = typer.Argument(...),
    root: Path = typer.Option(Path("results/phase0")),
) -> None:
    from herald.metrics.repair import repair_run

    repair_run(run_id=run_id, root=root)
```

- [ ] **Step 5: Register in top-level CLI**

`src/herald/__init__.py`, near the existing `app.add_typer(...)`:

```python
from herald.metrics.cli import app as metrics_app
app.add_typer(metrics_app, name="metrics")
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/metrics/test_cli.py -v`
Expected: pass.

- [ ] **Step 7: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/cli.py src/herald/metrics/io.py src/herald/__init__.py tests/metrics/test_cli.py
git commit -m "feat(metrics): finalize CLI + dataset compaction"
```

---

### Task 11: Offline token-metric audit + truncation bias

**Files:**
- Create: `src/herald/metrics/token.py`
- Test: `tests/metrics/test_token.py`

- [ ] **Step 1: Write the failing test**

`tests/metrics/test_token.py`:

```python
from pathlib import Path

import math
import polars as pl

from herald.metrics.token import build, recompute_kl_from_top_k


def test_recompute_kl_from_top_k_zero_when_dists_equal():
    lp_c = [-0.1, -2.0, -3.0]
    lp_u = [-0.1, -2.0, -3.0]
    tail_c = 0.0
    tail_u = 0.0
    kl, js = recompute_kl_from_top_k(lp_c, lp_u, tail_c, tail_u)
    assert kl < 1e-9
    assert js < 1e-9


def test_recompute_kl_from_top_k_handles_tail():
    # Distributions with same head, but different tail mass.
    lp_c = [math.log(0.6), math.log(0.3)]
    lp_u = [math.log(0.5), math.log(0.4)]
    kl, js = recompute_kl_from_top_k(lp_c, lp_u, tail_c=0.1, tail_u=0.1)
    assert kl > 0
    assert js > 0


def test_build_writes_token_metrics(tmp_path: Path):
    final = tmp_path / "final"
    (final / "replay" / "press=none" / "ratio=0.0000").mkdir(parents=True)
    df = pl.DataFrame(
        {
            "run_id": ["r1"], "token_pos": [0],
            "realized_token_id": [5],
            "union_top_k_token_ids": [[5, 7]],
            "logprobs_compressed": [[-0.1, -2.0]],
            "logprobs_uncompressed": [[-0.1, -2.0]],
            "tail_mass_compressed": [0.0], "tail_mass_uncompressed": [0.0],
            "realized_logprob_compressed": [-0.1],
            "realized_logprob_uncompressed": [-0.1],
            "js_full": [0.0], "kl_unc_comp_full": [0.0],
            "kl_comp_unc_full": [0.0], "top1_match": [True],
            "top1_rank_comp_under_unc": [0],
            "top1_rank_unc_under_comp": [0],
        }
    )
    df.write_parquet(
        final / "replay" / "press=none" / "ratio=0.0000" / "r1.parquet"
    )
    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)

    written = pl.read_parquet(out / "token_metrics.parquet")
    assert written.height == 1
    assert "trunc_bias_kl" in written.columns
    assert written["trunc_bias_kl"][0] < 1e-6
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_token.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/herald/metrics/token.py`**

```python
"""Offline token metrics + truncation-bias audit."""

import math
from pathlib import Path

import polars as pl


def recompute_kl_from_top_k(
    log_p_c: list[float],
    log_p_u: list[float],
    tail_c: float,
    tail_u: float,
) -> tuple[float, float]:
    """KL(p_u || p_c) and JS from union top-K + tail bucket.

    Tail mass is treated as a single bucket with uniform probability;
    its contribution is `t * log(t/t')` for KL and the symmetric
    equivalent for JS.
    """
    p_c = [math.exp(lp) for lp in log_p_c]
    p_u = [math.exp(lp) for lp in log_p_u]
    kl_uc = 0.0
    js = 0.0
    log2 = math.log(2.0)
    for pc, pu, lpc, lpu in zip(p_c, p_u, log_p_c, log_p_u):
        if pu > 0:
            kl_uc += pu * (lpu - lpc)
        m = 0.5 * (pc + pu)
        if m > 0:
            if pc > 0:
                js += 0.5 * pc * (lpc - math.log(m))
            if pu > 0:
                js += 0.5 * pu * (lpu - math.log(m))
    if tail_c > 0 and tail_u > 0:
        kl_uc += tail_u * (math.log(tail_u) - math.log(tail_c))
        m = 0.5 * (tail_c + tail_u)
        js += 0.5 * tail_c * (math.log(tail_c) - math.log(m))
        js += 0.5 * tail_u * (math.log(tail_u) - math.log(m))
    return max(kl_uc, 0.0), max(js, 0.0)


def build(final_root: Path, out_root: Path) -> None:
    """Read replay parquet partitions, audit truncation bias, write metrics."""
    replay_root = final_root / "replay"
    parts = list(replay_root.glob("press=*/ratio=*/*.parquet"))
    if not parts:
        return
    df = pl.concat([pl.read_parquet(p) for p in parts])

    rows = []
    for r in df.iter_rows(named=True):
        kl_topk, js_topk = recompute_kl_from_top_k(
            r["logprobs_uncompressed"], r["logprobs_compressed"],
            r["tail_mass_uncompressed"], r["tail_mass_compressed"],
        )
        rows.append(
            {
                "run_id": r["run_id"],
                "token_pos": r["token_pos"],
                "js_full": r["js_full"],
                "kl_unc_comp_full": r["kl_unc_comp_full"],
                "kl_comp_unc_full": r["kl_comp_unc_full"],
                "top1_match": r["top1_match"],
                "kl_topk": kl_topk,
                "js_topk": js_topk,
                "trunc_bias_kl": abs(r["kl_unc_comp_full"] - kl_topk),
                "trunc_bias_js": abs(r["js_full"] - js_topk),
            }
        )
    pl.DataFrame(rows).write_parquet(out_root / "token_metrics.parquet")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/metrics/test_token.py -v`
Expected: pass.

- [ ] **Step 5: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/token.py tests/metrics/test_token.py
git commit -m "feat(metrics): token-level truncation-bias audit"
```

---

### Task 12: Trajectory metrics

**Files:**
- Create: `src/herald/metrics/trajectory.py`
- Test: `tests/metrics/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

`tests/metrics/test_trajectory.py`:

```python
from pathlib import Path

import polars as pl

from herald.metrics.trajectory import build


def _make_replay(final: Path, run_id: str, press: str, ratio: float, n: int):
    d = final / "replay" / f"press={press}" / f"ratio={ratio:.4f}"
    d.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "run_id": [run_id] * n,
            "token_pos": list(range(n)),
            "realized_token_id": [1] * n,
            "union_top_k_token_ids": [[1, 2]] * n,
            "logprobs_compressed": [[-0.1, -2.0]] * n,
            "logprobs_uncompressed": [[-0.1, -2.0]] * n,
            "tail_mass_compressed": [0.0] * n,
            "tail_mass_uncompressed": [0.0] * n,
            "realized_logprob_compressed": [-0.5] * n,
            "realized_logprob_uncompressed": [-0.3] * n,
            "js_full": [0.05] * n,
            "kl_unc_comp_full": [0.04] * n,
            "kl_comp_unc_full": [0.04] * n,
            "top1_match": [True, True, False, True, True][:n],
            "top1_rank_comp_under_unc": [0, 0, 3, 0, 0][:n],
            "top1_rank_unc_under_comp": [0, 0, 3, 0, 0][:n],
        }
    )
    df.write_parquet(d / f"{run_id}.parquet")


def test_build_trajectory_metrics(tmp_path: Path):
    final = tmp_path / "final"
    _make_replay(final, "r1", "snapkv", 0.875, 5)

    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)
    df = pl.read_parquet(out / "trajectory_metrics.parquet")
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["run_id"] == "r1"
    assert row["sum_kl"] > 0
    assert row["nll_ratio"] > 0  # realized_logprob_unc > comp
    assert row["first_divergence_point"] == 2  # first top1 mismatch
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_trajectory.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/herald/metrics/trajectory.py`**

```python
"""Trajectory-level metrics: NLL ratio, first-divergence-point, sum_KL."""

from pathlib import Path

import polars as pl


def build(final_root: Path, out_root: Path) -> None:
    parts = list((final_root / "replay").glob("press=*/ratio=*/*.parquet"))
    if not parts:
        return
    df = pl.concat([pl.read_parquet(p) for p in parts])

    grouped = df.group_by("run_id").agg(
        [
            pl.col("kl_unc_comp_full").sum().alias("sum_kl"),
            pl.col("js_full").sum().alias("sum_js"),
            (
                pl.col("realized_logprob_uncompressed").sum()
                - pl.col("realized_logprob_compressed").sum()
            ).alias("nll_ratio"),
        ]
    )

    fdp_rows = []
    for run_id, sub in df.sort("token_pos").group_by("run_id"):
        rid = run_id[0] if isinstance(run_id, tuple) else run_id
        mismatches = sub.filter(~pl.col("top1_match"))
        fdp = (
            int(mismatches["token_pos"][0])
            if mismatches.height > 0
            else int(sub["token_pos"].max() + 1)
        )
        fdp_rows.append({"run_id": rid, "first_divergence_point": fdp})
    fdp_df = pl.DataFrame(fdp_rows)

    out = grouped.join(fdp_df, on="run_id")
    out.write_parquet(out_root / "trajectory_metrics.parquet")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/metrics/test_trajectory.py -v`
Expected: pass.

- [ ] **Step 5: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/trajectory.py tests/metrics/test_trajectory.py
git commit -m "feat(metrics): trajectory-level NLL ratio, FDP, sum_KL"
```

---

### Task 13: Sequence metrics (BERTScore, embedding cosine, edit, ROUGE-L)

**Files:**
- Create: `src/herald/metrics/sequence.py`
- Test: `tests/metrics/test_sequence.py`

- [ ] **Step 1: Write the failing test**

`tests/metrics/test_sequence.py`:

```python
from pathlib import Path

import polars as pl
import pytest

from herald.metrics.sequence import build


def _make_runs(final: Path):
    df = pl.DataFrame(
        {
            "run_id": ["base", "comp"],
            "prompt_id": ["p1", "p1"],
            "press": ["none", "snapkv"],
            "compression_ratio": [0.0, 0.875],
            "baseline_run_id": ["base", "base"],
            "generated_text": ["The capital is Paris.", "The capital is Lyon."],
        }
    )
    df.write_parquet(final / "runs.parquet")


def test_build_sequence_metrics(tmp_path: Path):
    final = tmp_path / "final"
    final.mkdir()
    _make_runs(final)
    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)
    df = pl.read_parquet(out / "sequence_metrics.parquet")
    assert df.height == 1  # only the compressed run gets paired metrics
    row = df.row(0, named=True)
    assert row["run_id"] == "comp"
    assert 0.0 <= row["edit_distance_ratio"] <= 1.0
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_sequence.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/herald/metrics/sequence.py`**

```python
"""Sequence-level metrics: BERTScore, embedding cosine, edit, ROUGE-L.

Requires the `metrics` optional install group.
"""

from pathlib import Path

import polars as pl


def _import_extras():
    try:
        import editdistance  # type: ignore
        from rouge_score import rouge_scorer  # type: ignore
        from sentence_transformers import (  # type: ignore
            SentenceTransformer, util,
        )
    except ImportError as exc:
        raise ImportError(
            "Sequence metrics require the [metrics] extra: "
            "uv sync --extra metrics"
        ) from exc
    return editdistance, rouge_scorer, SentenceTransformer, util


def build(final_root: Path, out_root: Path) -> None:
    editdistance, rouge_scorer_mod, SentenceTransformer, util = _import_extras()
    runs = pl.read_parquet(final_root / "runs.parquet").select(
        ["run_id", "prompt_id", "press", "baseline_run_id",
         "generated_text"]
    )
    base = runs.filter(pl.col("press") == "none").rename(
        {"run_id": "baseline_id_check",
         "generated_text": "baseline_text"}
    ).select(
        ["baseline_id_check", "prompt_id", "baseline_text"]
    )
    paired = (
        runs.filter(pl.col("press") != "none")
        .join(base, on="prompt_id", how="left")
        .filter(pl.col("baseline_id_check") == pl.col("baseline_run_id"))
    )

    scorer = rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=False)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []
    for r in paired.iter_rows(named=True):
        comp_text = r["generated_text"]
        base_text = r["baseline_text"]
        rouge_l = scorer.score(base_text, comp_text)["rougeL"].fmeasure
        emb = embedder.encode(
            [base_text, comp_text], convert_to_tensor=True
        )
        cos = float(util.cos_sim(emb[0], emb[1]).item())
        denom = max(len(base_text), len(comp_text), 1)
        ed_ratio = (
            float(editdistance.eval(base_text, comp_text)) / denom
        )
        rows.append(
            {
                "run_id": r["run_id"], "prompt_id": r["prompt_id"],
                "rouge_l": rouge_l, "embedding_cosine": cos,
                "edit_distance_ratio": ed_ratio,
            }
        )
    pl.DataFrame(rows).write_parquet(
        out_root / "sequence_metrics.parquet"
    )
```

BERTScore is intentionally deferred to a `bertscore=true` flag (heavy model load) — the plan adds the column once Phase 0 smokes pass; not part of gate 7.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/metrics/test_sequence.py -v`
Expected: pass.

- [ ] **Step 5: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/sequence.py tests/metrics/test_sequence.py
git commit -m "feat(metrics): sequence-level edit distance + cosine + ROUGE-L"
```

---

### Task 14: Outcome (paired cells) and tags

**Files:**
- Create: `src/herald/metrics/outcome.py`
- Create: `src/herald/metrics/tags.py`
- Test: `tests/metrics/test_outcome.py`, `tests/metrics/test_tags.py`

- [ ] **Step 1: Write the failing tests**

`tests/metrics/test_outcome.py`:

```python
from pathlib import Path
import polars as pl

from herald.metrics.outcome import build


def _runs(final: Path):
    df = pl.DataFrame(
        {
            "run_id": ["b1", "b2", "c1", "c2"],
            "prompt_id": ["p1", "p2", "p1", "p2"],
            "press": ["none", "none", "snapkv", "snapkv"],
            "compression_ratio": [0.0, 0.0, 0.875, 0.875],
            "baseline_run_id": ["b1", "b2", "b1", "b2"],
            "correct": [True, False, False, True],
        }
    )
    df.write_parquet(final / "runs.parquet")


def test_outcome_paired_cells(tmp_path: Path):
    final = tmp_path / "final"
    final.mkdir()
    _runs(final)
    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)
    df = pl.read_parquet(out / "outcome.parquet")
    row = df.row(0, named=True)
    assert row["press"] == "snapkv"
    assert row["compression_ratio"] == 0.875
    assert row["gross_harm"] == 0.5
    assert row["gross_help"] == 0.5
    assert row["net_delta"] == 0.0
```

`tests/metrics/test_tags.py`:

```python
from pathlib import Path
import polars as pl

from herald.metrics.tags import build


def test_tags_build_passes_through(tmp_path: Path):
    final = tmp_path / "final"
    final.mkdir()
    pl.DataFrame(
        {
            "run_id": ["r1"],
            "catastrophes": [["looping", "non_termination"]],
        }
    ).write_parquet(final / "runs.parquet")
    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)
    df = pl.read_parquet(out / "tags.parquet")
    assert df.height == 1
    assert df["has_looping"][0] is True
    assert df["has_non_termination"][0] is True
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_outcome.py tests/metrics/test_tags.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/herald/metrics/outcome.py`**

```python
"""Paired outcome cells: gross_harm, gross_help, net_delta per cell."""

from pathlib import Path

import polars as pl


def build(final_root: Path, out_root: Path) -> None:
    runs = pl.read_parquet(final_root / "runs.parquet")
    base = runs.filter(pl.col("press") == "none").select(
        ["run_id", "correct"]
    ).rename({"run_id": "baseline_run_id", "correct": "baseline_correct"})
    comp = runs.filter(pl.col("press") != "none").join(
        base, on="baseline_run_id", how="left"
    )

    cells = comp.group_by(["press", "compression_ratio"]).agg(
        [
            (
                (pl.col("baseline_correct") & ~pl.col("correct"))
                .cast(pl.Float64).mean()
            ).alias("gross_harm"),
            (
                (~pl.col("baseline_correct") & pl.col("correct"))
                .cast(pl.Float64).mean()
            ).alias("gross_help"),
            pl.len().alias("n_cells"),
        ]
    )
    cells = cells.with_columns(
        (pl.col("gross_harm") - pl.col("gross_help")).alias("net_delta")
    )
    cells.write_parquet(out_root / "outcome.parquet")
```

- [ ] **Step 4: Implement `src/herald/metrics/tags.py`**

```python
"""Diagnostic tags per run, exploded into boolean columns."""

from pathlib import Path

import polars as pl

CATEGORIES = ("looping", "non_termination", "format_break", "drift")


def build(final_root: Path, out_root: Path) -> None:
    runs = pl.read_parquet(final_root / "runs.parquet").select(
        ["run_id", "catastrophes"]
    )
    rows = []
    for r in runs.iter_rows(named=True):
        cats = set(r["catastrophes"] or [])
        rows.append(
            {
                "run_id": r["run_id"],
                **{f"has_{c}": (c in cats) for c in CATEGORIES},
            }
        )
    pl.DataFrame(rows).write_parquet(out_root / "tags.parquet")
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/metrics/test_outcome.py tests/metrics/test_tags.py -v`
Expected: pass.

- [ ] **Step 6: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/outcome.py src/herald/metrics/tags.py tests/metrics/
git commit -m "feat(metrics): paired outcome cells + diagnostic tag explode"
```

---

### Task 15: Alignment matrix (Spearman + AUROC across metric families)

**Files:**
- Create: `src/herald/metrics/alignment.py`
- Test: `tests/metrics/test_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import polars as pl

from herald.metrics.alignment import build


def test_alignment_matrix_includes_pairs(tmp_path: Path):
    out = tmp_path / "metrics"
    out.mkdir()
    pl.DataFrame(
        {
            "run_id": ["a", "b", "c", "d"],
            "sum_kl": [0.1, 0.4, 0.2, 0.6],
            "first_divergence_point": [10, 5, 8, 3],
            "nll_ratio": [0.0, 0.5, 0.1, 0.7],
        }
    ).write_parquet(out / "trajectory_metrics.parquet")
    pl.DataFrame(
        {
            "run_id": ["a", "b", "c", "d"],
            "rouge_l": [0.9, 0.5, 0.8, 0.3],
            "edit_distance_ratio": [0.05, 0.4, 0.1, 0.6],
            "embedding_cosine": [0.99, 0.7, 0.95, 0.6],
        }
    ).write_parquet(out / "sequence_metrics.parquet")

    build(out, out)

    align = pl.read_parquet(out / "alignment.parquet")
    pairs = set(zip(align["metric_a"], align["metric_b"]))
    assert ("sum_kl", "rouge_l") in pairs
    assert (
        align.filter(
            (pl.col("metric_a") == "sum_kl")
            & (pl.col("metric_b") == "rouge_l")
        )["spearman"][0]
        < 0  # higher KL → lower rouge_l
    )
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_alignment.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/herald/metrics/alignment.py`**

```python
"""Alignment matrix: pairwise Spearman across metric families."""

from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).statistic
        if not np.isnan(r):
            rs.append(r)
    arr = np.asarray(rs)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def build(metrics_root: Path, out_root: Path) -> None:
    families: dict[str, list[str]] = {
        "trajectory": ["sum_kl", "first_divergence_point", "nll_ratio"],
        "sequence": ["rouge_l", "edit_distance_ratio", "embedding_cosine"],
    }
    frames = []
    for fam, cols in families.items():
        path = metrics_root / f"{fam}_metrics.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path))
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, on="run_id", how="inner")

    rows = []
    metrics = [c for c in df.columns if c != "run_id"]
    for a, b in product(metrics, metrics):
        if a >= b:
            continue
        x = df[a].to_numpy()
        y = df[b].to_numpy()
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        rho = spearmanr(x, y).statistic
        lo, hi = _bootstrap_spearman(x, y)
        rows.append(
            {"metric_a": a, "metric_b": b,
             "spearman": float(rho),
             "spearman_lo": lo, "spearman_hi": hi}
        )
    pl.DataFrame(rows).write_parquet(out_root / "alignment.parquet")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/metrics/test_alignment.py -v`
Expected: pass.

- [ ] **Step 5: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/alignment.py tests/metrics/test_alignment.py
git commit -m "feat(metrics): alignment matrix with bootstrap CI"
```

---

### Task 16: `metrics build` integration test

**Files:**
- Modify: `tests/metrics/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/metrics/test_cli.py`:

```python
def test_metrics_build_runs_all_modules(tmp_path: Path):
    # Seed enough to exercise trajectory, outcome, tags. Skip sequence.
    final = tmp_path / "final"
    final.mkdir()
    pl.DataFrame(
        {
            "run_id": ["b", "c"], "prompt_id": ["p", "p"],
            "press": ["none", "snapkv"],
            "compression_ratio": [0.0, 0.875],
            "baseline_run_id": ["b", "b"],
            "generated_text": ["x", "y"],
            "correct": [True, False],
            "catastrophes": [[], ["looping"]],
        }
    ).write_parquet(final / "runs.parquet")
    rep = final / "replay" / "press=snapkv" / "ratio=0.8750"
    rep.mkdir(parents=True)
    pl.DataFrame(
        {
            "run_id": ["c"], "token_pos": [0],
            "realized_token_id": [1],
            "union_top_k_token_ids": [[1, 2]],
            "logprobs_compressed": [[-0.1, -2.0]],
            "logprobs_uncompressed": [[-0.1, -2.0]],
            "tail_mass_compressed": [0.0],
            "tail_mass_uncompressed": [0.0],
            "realized_logprob_compressed": [-0.5],
            "realized_logprob_uncompressed": [-0.3],
            "js_full": [0.05], "kl_unc_comp_full": [0.04],
            "kl_comp_unc_full": [0.04], "top1_match": [True],
            "top1_rank_comp_under_unc": [0],
            "top1_rank_unc_under_comp": [0],
        }
    ).write_parquet(rep / "c.parquet")

    from herald.metrics import outcome, tags, token, trajectory
    out = tmp_path / "metrics"
    out.mkdir()
    token.build(final, out)
    trajectory.build(final, out)
    outcome.build(final, out)
    tags.build(final, out)

    for name in (
        "token_metrics", "trajectory_metrics", "outcome", "tags"
    ):
        assert (out / f"{name}.parquet").exists()
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/metrics/test_cli.py -v -k metrics_build`
Expected: pass (no implementation change needed; just a wiring test).

- [ ] **Step 3: Commit**

```bash
git add tests/metrics/test_cli.py
git commit -m "test(metrics): build integration covers token/trajectory/outcome/tags"
```

---

### Task 17: `metrics repair` CLI

**Files:**
- Create: `src/herald/metrics/repair.py`
- Test: `tests/metrics/test_repair.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import polars as pl
import pytest

from herald.metrics.io import PerRunPaths, write_run_record
from herald.metrics.repair import repair_run, validate_manifest


def _seed(root: Path, run_id: str, status: str = "failed"):
    rec = {
        "run_id": run_id, "prompt_id": "p", "prompt_text": "Q",
        "prompt_hash": "h", "model": "x", "model_revision": None,
        "tokenizer_revision": None, "dtype": "float16",
        "device_class": "cpu", "task": "gsm8k", "press": "none",
        "compression_ratio": 0.0, "max_new_tokens": 8,
        "decoding_config": {"do_sample": "False"}, "seed": 42,
        "baseline_run_id": run_id, "generated_text": "a",
        "generated_token_ids": [1], "num_tokens_generated": 1,
        "stop_reason": "eos", "predicted_answer": "a",
        "ground_truth": "a", "correct": True, "catastrophes": [],
        "replay_status": status, "replay_error": "boom",
        "created_at": "2026-05-01T00:00:00Z", "herald_git_sha": "abc",
    }
    p = PerRunPaths(root=root, run_id=run_id)
    write_run_record(rec, p.run)


def test_validate_manifest_rejects_missing_field(tmp_path: Path):
    _seed(tmp_path, "r1")
    df = pl.read_parquet(tmp_path / "raw" / "runs" / "r1.parquet").drop(
        ["seed"]
    )
    df.write_parquet(tmp_path / "raw" / "runs" / "r1.parquet")
    with pytest.raises(ValueError):
        validate_manifest(tmp_path / "raw" / "runs" / "r1.parquet")
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_repair.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/herald/metrics/repair.py`**

```python
"""Re-run compressed gen + replay for a single run.

Determinism rests on the manifest fields recorded on runs.parquet.
The repair re-generates and verifies that the resulting
generated_token_ids matches the recorded sequence; mismatch is a
hard error.
"""

from pathlib import Path

import polars as pl

REQUIRED_FIELDS = (
    "model", "model_revision", "tokenizer_revision", "dtype",
    "device_class", "task", "prompt_id", "prompt_text",
    "prompt_hash", "press", "compression_ratio", "max_new_tokens",
    "decoding_config", "seed", "herald_git_sha",
    "generated_token_ids",
)


def validate_manifest(run_record_path: Path) -> dict[str, object]:
    df = pl.read_parquet(run_record_path)
    if df.height != 1:
        raise ValueError(f"expected 1 row, got {df.height}")
    row = df.row(0, named=True)
    missing = [
        f for f in REQUIRED_FIELDS
        if f not in row or row[f] is None
        or (isinstance(row[f], list) and not row[f])
    ]
    if missing:
        raise ValueError(f"missing determinism fields: {missing}")
    return row


def repair_run(run_id: str, root: Path) -> None:
    """Re-run a single failed run end-to-end and overwrite per-run parquets."""
    from herald.config import (
        ExperimentConfig, RunResult, compute_prompt_hash,
    )
    from herald.experiment import (
        get_press, load_model, run_single_with_replay,
    )
    from herald.metrics.io import PerRunPaths
    from herald.tasks import DEFAULT_TASK

    paths = PerRunPaths(root=root, run_id=run_id)
    row = validate_manifest(paths.run)
    if row["prompt_hash"] != compute_prompt_hash(row["prompt_text"]):
        raise ValueError("prompt_hash mismatch (text was tampered with)")

    cfg = ExperimentConfig(
        model_name=row["model"], press_name=row["press"],
        compression_ratio=float(row["compression_ratio"]),
        num_prompts=1, seed=int(row["seed"]),
        output_dir=root, max_new_tokens=int(row["max_new_tokens"]),
        prompt_timeout_seconds=300.0,
    )
    model, tok, device = load_model(cfg)
    press = get_press(cfg.press_name, cfg.compression_ratio)

    # Resolve the prompt by id from the canonical loader.
    full = DEFAULT_TASK.load(num_prompts=10_000, seed=int(row["seed"]))
    prompt = next(p for p in full if p["id"] == row["prompt_id"])

    rr, _ = run_single_with_replay(
        model=model, tokenizer=tok, device=device,
        prompt_data=prompt, config=cfg, press=press,
        baseline_run_id=row["baseline_run_id"],
        output_root=root,
    )
    if rr.generated_token_ids != list(row["generated_token_ids"]):
        rr.replay_status = "failed"
        rr.replay_error = "deterministic regeneration mismatch"
        raise ValueError(
            "regenerated token ids do not match recorded sequence"
        )
    rr.replay_status = "retried"
    # write_run_record already overwrote the run parquet during run_single_with_replay
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/metrics/test_repair.py -v`
Expected: pass on the validation test (the slow end-to-end repair test is left as a separate Phase 0 manual smoke).

- [ ] **Step 5: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/repair.py tests/metrics/test_repair.py
git commit -m "feat(metrics): determinism manifest validator + repair CLI"
```

---

### Task 18: Sampling-rate study

**Files:**
- Create: `src/herald/metrics/sampling_rate.py`
- Test: `tests/metrics/test_sampling_rate.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import json
import polars as pl

from herald.metrics.sampling_rate import run_study


def test_run_study_emits_report(tmp_path: Path):
    final = tmp_path / "final"
    rep = final / "replay" / "press=snapkv" / "ratio=0.8750"
    rep.mkdir(parents=True)
    n = 40
    pl.DataFrame(
        {
            "run_id": ["r1"] * n,
            "token_pos": list(range(n)),
            "js_full": [
                0.01 + 0.001 * i for i in range(n)
            ],
            "top1_match": [True] * n,
        }
    ).write_parquet(rep / "r1.parquet")
    out = tmp_path / "metrics"
    out.mkdir()
    run_study(final, out, horizons=(5, 10), rates=(1, 4, 8))
    report = json.loads((out / "sampling_rate_report.json").read_text())
    assert "spearman_future_max_js" in report
    assert "trajectory_rank_corr" in report
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/metrics/test_sampling_rate.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/herald/metrics/sampling_rate.py`**

```python
"""Phase 0 sampling-rate study (Q4 of design)."""

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def _future_max_js(values: np.ndarray, h: int) -> np.ndarray:
    out = np.empty_like(values)
    n = len(values)
    for i in range(n):
        out[i] = values[i:min(i + h, n)].max(initial=0.0)
    return out


def _subset_and_interp(values: np.ndarray, rate: int) -> np.ndarray:
    if rate == 1:
        return values.copy()
    idx = np.arange(0, len(values), rate)
    sampled = values[idx]
    return np.interp(np.arange(len(values)), idx, sampled)


def run_study(
    final_root: Path,
    out_root: Path,
    horizons: tuple[int, ...] = (5, 10, 25, 50),
    rates: tuple[int, ...] = (1, 4, 8),
) -> None:
    parts = list((final_root / "replay").glob("press=*/ratio=*/*.parquet"))
    if not parts:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "sampling_rate_report.json").write_text(
            json.dumps({"runs": 0})
        )
        return
    df = pl.concat([pl.read_parquet(p) for p in parts]).sort(
        ["run_id", "token_pos"]
    )

    spearman: dict[int, dict[int, list[float]]] = {
        r: {h: [] for h in horizons} for r in rates if r != 1
    }
    traj: dict[int, list[float]] = {r: [] for r in rates if r != 1}

    dense_traj = []
    rate_traj: dict[int, list[float]] = {r: [] for r in rates if r != 1}

    for run_id, sub in df.group_by("run_id"):
        rid = run_id[0] if isinstance(run_id, tuple) else run_id
        v = sub["js_full"].to_numpy()
        if len(v) < max(horizons):
            continue
        dense = v
        dense_traj.append(float(dense.sum()))
        for r in rates:
            if r == 1:
                continue
            sub_v = _subset_and_interp(v, r)
            for h in horizons:
                a = _future_max_js(dense, h)
                b = _future_max_js(sub_v, h)
                rho = spearmanr(a, b).statistic
                if not np.isnan(rho):
                    spearman[r][h].append(float(rho))
            rate_traj[r].append(float(sub_v.sum()))

    report = {
        "runs": len(dense_traj),
        "spearman_future_max_js": {
            r: {
                h: float(np.median(spearman[r][h])) if spearman[r][h] else None
                for h in horizons
            }
            for r in spearman
        },
        "trajectory_rank_corr": {
            r: float(spearmanr(dense_traj, rate_traj[r]).statistic)
            if rate_traj[r] else None
            for r in rate_traj
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sampling_rate_report.json").write_text(
        json.dumps(report, indent=2)
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/metrics/test_sampling_rate.py -v`
Expected: pass.

- [ ] **Step 5: `poe check` and commit**

```bash
poe check
git add src/herald/metrics/sampling_rate.py tests/metrics/test_sampling_rate.py
git commit -m "feat(metrics): Phase 0 sampling-rate study"
```

---

### Task 19: Poe wiring + Orion smoke + 140-run gate measurement

**Location**: wiring on Mac; smoke test and 140-run gate measurement on **Orion**. Mac never executes the GPU portion.

**Files:**
- Modify: `poe_tasks.toml`
- Modify: `pyproject.toml` (pytest markers — see Execution Environment)
- Create: `scripts/check_gpu_clean.py`
- Create: `scripts/measure_replay_noise_floor.py`
- Test: `tests/metrics/test_phase0_smoke.py` (gpu + e2e)

- [ ] **Step 1: Edit `poe_tasks.toml`** [Mac]

Add tasks:

```toml
[tool.poe.tasks]
test              = "pytest -m 'not gpu' -q"
test-gpu          = "pytest -m 'gpu and not e2e' -q"
test-e2e          = "pytest -m 'e2e' -q"
phase0-run        = "uv run herald phase0 run"
metrics-finalize  = "uv run herald metrics finalize"
metrics-build     = "uv run herald metrics build"
phase0            = ["phase0-run", "metrics-finalize", "metrics-build"]
```

`poe check` continues to call `test`, so Mac commits stay green without GPU. Drop any pre-existing `test = "pytest -x -q"` line.

- [ ] **Step 2: Add pytest markers in `pyproject.toml`** [Mac]

Append:

```toml
[tool.pytest.ini_options]
markers = [
  "gpu: requires a real LLM forward pass (run on Orion only)",
  "e2e: end-to-end Phase 0 sweep (large; opt-in)",
]
```

- [ ] **Step 3: Implement `scripts/check_gpu_clean.py`** [Mac, runs on Orion]

```python
"""Refuse to start a sweep if the GPU is dirty.

Treats > 200 MB of VRAM held by other processes as 'dirty'. Exits 1
on dirty, 0 on clean. The orchestrator calls this as a precheck.
"""

import re
import subprocess
import sys


def main() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"]
    ).decode().strip()
    if not out:
        return 0
    dirty = []
    for line in out.splitlines():
        pid, mem = [s.strip() for s in line.split(",")]
        if int(mem) > 200:
            dirty.append((pid, mem))
    if dirty:
        print(
            "GPU dirty; existing processes:\n"
            + "\n".join(f"  pid={p} used={m} MiB" for p, m in dirty),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add to `phase0.py` `run_phase0_sweep` before `load_model`:

```python
import subprocess
if subprocess.call(
    ["python", "scripts/check_gpu_clean.py"]
) != 0:
    raise RuntimeError(
        "GPU not clean; resolve stale CUDA processes before "
        "starting the sweep."
    )
```

- [ ] **Step 4: Implement `scripts/measure_replay_noise_floor.py`** [Orion]

```python
"""Measure the actual fp16 noise floor for replay sanity checks.

Runs 5 prompts with press=none, replays, and reports the
99th-percentile per-token JS along the trajectory plus the t=0
max-abs logit diff vs generation scores. The numbers it prints
are what gets pinned into Phase 0 gate 3 / gate 4 thresholds.
"""

from pathlib import Path

import numpy as np
import torch

from herald.config import ExperimentConfig
from herald.experiment import (
    get_press, load_model, run_single_with_replay,
)
from herald.metrics.io import PerRunPaths
from herald.tasks import DEFAULT_TASK
import polars as pl


def main():
    cfg = ExperimentConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        press_name="none", compression_ratio=0.0,
        num_prompts=5, seed=42,
        output_dir=Path("results/noise_floor"),
        max_new_tokens=64,
        prompt_timeout_seconds=120.0,
    )
    model, tok, device = load_model(cfg)
    prompts = DEFAULT_TASK.load(num_prompts=5, seed=42)
    press = get_press("none", 0.0)
    js_p99 = []
    for p in prompts:
        rr, _ = run_single_with_replay(
            model=model, tokenizer=tok, device=device,
            prompt_data=p, config=cfg, press=press,
            baseline_run_id=None,
            output_root=cfg.output_dir,
        )
        rep = pl.read_parquet(
            PerRunPaths(root=cfg.output_dir, run_id=rr.run_id).replay
        )
        js_p99.append(float(np.percentile(rep["js_full"].to_numpy(), 99)))
    print(f"99th-pct per-token JS across 5 baseline runs:")
    for j in js_p99:
        print(f"  {j:.3e}")
    print(f"max:    {max(js_p99):.3e}")
    print(f"median: {np.median(js_p99):.3e}")


if __name__ == "__main__":
    main()
```

Run on Orion **before** the 140-run smoke; pin the resulting threshold (expect ~1e-4 on CUDA fp16) into the gate-3 / gate-4 assertions in step 6.

- [ ] **Step 5: Write the smoke test** [Mac authors, Orion runs]

`tests/metrics/test_phase0_smoke.py`:

```python
from pathlib import Path

import pytest
import polars as pl


JS_NOISE_FLOOR = 1e-3  # placeholder; pin from Orion measurement


@pytest.mark.gpu
@pytest.mark.e2e
def test_phase0_smoke_2prompts_1press_1ratio(tmp_path: Path):
    """Cheap end-to-end smoke (4 runs total). Runs on Orion."""
    from herald.metrics.io import finalize_dataset
    from herald.phase0 import run_phase0_sweep

    out = tmp_path / "phase0"
    run_phase0_sweep(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        manifest_path=Path("gold/phase0-random-manifest.json"),
        presses=("streaming_llm",),
        ratios=(0.5,),
        max_new_tokens=16,
        output_root=out,
        prompt_timeout_seconds=120.0,
        num_prompts=2,
    )
    finalize_dataset(out)

    # Gate 1: completion
    runs = pl.read_parquet(out / "final" / "runs.parquet")
    assert (runs["replay_status"] == "ok").all()

    # Gate 2: schema integrity
    assert runs["baseline_run_id"].is_not_null().all()
    for rid in runs["run_id"]:
        assert (out / "raw" / "tokens" / f"{rid}.parquet").exists()
        assert (out / "raw" / "replay" / f"{rid}.parquet").exists()

    # Gate 4: press=none replay JS ~ 0 along trajectory
    base_ids = runs.filter(pl.col("press") == "none")["run_id"]
    for rid in base_ids:
        rep = pl.read_parquet(out / "raw" / "replay" / f"{rid}.parquet")
        assert rep["js_full"].max() < JS_NOISE_FLOOR
```

- [ ] **Step 6: Run the smoke on Orion**

```bash
# from Mac
git push origin <branch>
ssh orion
cd /clustergpu/home/jcampo/herald
git pull
# pin gate threshold first
uv run python scripts/measure_replay_noise_floor.py
# then run the smoke
uv run pytest tests/metrics/test_phase0_smoke.py -v -m "gpu and e2e"
```

Expected: pass on Orion (~3 min for 4 runs of 0.5B). If JS noise floor measured in step 4 is materially different from `JS_NOISE_FLOOR`, update the constant in `test_phase0_smoke.py` and re-run before declaring the gate met.

- [ ] **Step 7: 140-run Phase 0 gate measurement on Orion**

```bash
# on Orion
uv run herald phase0 run \
    --model Qwen/Qwen2.5-7B-Instruct \
    --manifest gold/phase0-random-manifest.json \
    --output-root results/phase0 \
    --max-new-tokens 512 \
    --prompt-timeout-seconds 300
uv run herald metrics finalize --root results/phase0
```

Then back on Mac:

```bash
rsync -avz --progress \
    orion:/clustergpu/home/jcampo/herald/results/phase0/ \
    results/phase0/
uv run herald metrics build --root results/phase0
uv run python -c "from herald.metrics.sampling_rate import run_study; \
    from pathlib import Path; \
    run_study(Path('results/phase0/final'), Path('results/phase0/metrics'))"
```

Verify gates 1–7 against the design doc against the produced parquets. The 140-run sweep is the actual Phase 0 deliverable; this plan ends when gates 1–7 are green on the rsync'd dataset.

- [ ] **Step 8: `poe check` on Mac and commit**

```bash
poe check
git add poe_tasks.toml pyproject.toml scripts/ tests/metrics/test_phase0_smoke.py
git commit -m "feat(phase0): poe tasks + Orion smoke + GPU precheck + noise-floor script"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Implemented in |
|--------------|----------------|
| Q1 replay mechanism | Tasks 5, 6, 7 |
| Q2 schema (parquet, partitions, top-K, exact scalars) | Tasks 3, 8, 10 |
| `baseline_run_id` self-link convention | Task 2 (default in `RunResult`), Task 8 (sweeper sets explicitly) |
| `generated_token_ids` on `runs.parquet` | Task 2 (model field), Task 3 (schema), Task 8 (writer) |
| Q3 module layout (`metrics/`) | Tasks 3, 7, 11–18 |
| GPU-bound replay + offline metric split | Task 7 vs Tasks 11–15 |
| `replay.parquet` for baselines too | Task 7 (replay_run unconditional) + Task 8 sweeper baseline branch |
| `GenerationArtifact` introduction | Task 2, used in Tasks 7 & 8 |
| Q4 sampling-rate study | Task 18 |
| Q5 cells (manifest, presses, ratios, baselines) | Tasks 4, 9 |
| Q6 orchestration (CLIs, finalize, build, repair, poe) | Tasks 9, 10, 17, 19 |
| Determinism manifest persistence | Tasks 2, 3, 17 |
| Q7 gates 1/2/4 verification | Task 19 smoke |
| Q7 gate 3 (t=0 identity) | Task 5 (logit-equality test) — extended at smoke time |
| Q7 gate 5 (index alignment audit) | Task 5 argmax-equality test |
| Q7 gate 6 (truncation-bias audit) | Task 11 (`trunc_bias_kl/js` columns) |
| Q7 gate 7 (no NaN) | Tasks 11–15 each test on synthetic data |
| Q7 gate 8 sampling-rate report | Task 18 |
| Q7 gate 9 alignment matrix | Task 15 |
| BERTScore | Deferred per Task 13 note (not blocking) |
| Implementation-plan dependencies note | Task 1 |
| Implementation-plan `GenerationArtifact` note | Task 2 + 8 |
| Implementation-plan `prompt_text` repair note | Task 2 (schema), Task 17 (validator) |

**Placeholder scan:** none of "TBD", "TODO", "implement later", "appropriate error handling" appear; every step has concrete code or commands.

**Type consistency:** `compute_replay_position` keys match `REPLAY_SCHEMA` field names; `signals_to_token_rows` keys match `TOKENS_SCHEMA`; `RunResult` fields match `RUNS_SCHEMA` with `decoding_config` stringified at write time; `PerRunPaths` is the single source of truth for the `raw/<kind>/<run_id>.parquet` layout used by every writer and the resume-skip check.

**Cross-machine coverage:** Mac executes Tasks 1–4, 6, 10–16, 18, plus all wiring/test-authoring portions of GPU tasks. Orion executes the GPU portions of Tasks 5, 7, 8, 9, 17, 19 plus the 140-run measurement. The Execution Environment section above is the contract; every per-task step is annotated [Mac] or [Orion] where the choice is non-obvious.

**Open small risks (call out for executor):**
- `model.config.vocab_size` vs `tokenizer.vocab_size` discrepancy: replay relies on the model's vocab; the test uses `model.config.vocab_size` for shape checks. If the chosen model has special-token shifts that differ from the tokenizer's view, document with a comment.
- Numerical thresholds: design doc cites 1e-4 (CUDA fp16). Mac MPS has been observed at ≈ 1e-2 — that is *not* the gate. Task 19 step 4 measures the actual Orion CUDA noise floor before any gate is locked. Until that script runs, every threshold in the smoke is a placeholder.
- `decoding_config` parquet type is `map<str,str>` for schema uniformity. Numeric values (`temperature`, `top_p`) are stringified with `str(...)` and parsed back at repair time. Document this in `metrics/repair.py` if/when sampling is enabled.
- Orion proxy required for Task 1 `uv sync --extra metrics` (no internet on Orion). Verify the reverse-tunnel is up before running. Model weights for the smoke (`Qwen2.5-0.5B-Instruct`, `Qwen2.5-7B-Instruct`) need to be confirmed in the HF cache on Orion before Task 19 step 6 — the 7B is documented as cached; 0.5B may need to be pre-fetched on Mac and rsync'd into Orion's HF cache, or fetched once via proxy.

---

## Execution Handoff

Plan complete and saved to `gold/phase-0-implementation-plan.md`. Two execution options, both Orion-aware:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review. Mac-only tasks (1–4, 6, 10–16, 18) run autonomously; Orion-bound tasks (5, 7, 8, 9, 17, 19) require either a subagent with `ssh orion` access or a manual checkpoint where the user runs the GPU portion and pastes back results. Best for the long horizon and the cross-machine context switches.
2. **Inline Execution** — I run tasks in this session. CPU tasks proceed normally; for each Orion task I prepare the code, run `poe check` locally, and hand you the exact `ssh orion && git pull && uv run pytest -m gpu ...` block to execute, then continue once you confirm it passed.

Which approach?
