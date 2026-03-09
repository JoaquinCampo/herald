# HERALD

**H**azard **E**stimation via **R**eal-time **A**nalysis of **L**ogit **D**istributions.

Predicts catastrophic failures (looping, non-termination, instruction amnesia) in KV-cache compressed LLM generation from per-token logit signals — before they manifest.

**Thesis**: Heavy KV-cache compression causes dramatic, detectable failures. A lightweight XGBoost trained on zero-cost logit features can forecast these H tokens ahead, enabling any downstream system to intervene.

## Architecture

Single flat package in `src/herald/`.

**Key modules**: config.py (Pydantic models) -> prompts.py (GSM8K loading) -> signals.py (per-token extraction) -> detectors.py (catastrophe detection) -> experiment.py (generation + sweep runner) -> features.py (ML feature engineering) -> labeling.py (hazard labels) -> train.py (XGBoost training + evaluation)

**Data flow**: GSM8K -> format_prompt -> model.generate(output_scores=True) inside press(model) context -> per-token extract_signals -> detect_all -> RunResult -> JSON -> build_dataset -> train_predictor -> evaluate

## Commands

Uses `poethepoet` as task runner.

```bash
poe check               # REQUIRED before commits: fmt + lint + typecheck + test
poe fmt                 # ruff format src/ tests/
poe lint                # ruff check src/ tests/
poe typecheck           # mypy src/ (strict mode)
poe test                # pytest -x -q
poe sweep               # Run full compression sweep (GPU)
poe train               # Train hazard predictor
poe figures             # Generate paper figures
poe setup               # Sync deps + install pre-commit hooks
```

**Single experiment**: `uv run herald run --press streaming_llm --compression-ratio 0.875 --num-prompts 10`
**Full sweep**: `uv run herald sweep --num-prompts 500 --model "Qwen/Qwen2.5-7B-Instruct"`
**Train predictor**: `uv run herald train --results-dir results --output-dir models`

## Conventions

- Python 3.12+. No `from __future__ import annotations`.
- `list`, `dict`, `tuple` — not `typing.List`, `typing.Dict`, `typing.Tuple`.
- `X | None` — not `Optional[X]`.
- Pydantic models for structured data. Keep flat and simple.
- Functions over classes. Classes only when state management is genuinely needed.
- Line length: 100 characters (ruff enforces).
- Imports sorted by ruff.
- Flat package structure — no subpackages under `src/herald/`.

## Boundaries

### Always safe (no approval needed)
- Run `poe check`, `pytest`, `ruff`, `mypy`
- Read any file in the repo
- Edit source files in `src/` and `tests/`
- Run `uv run herald` commands

### Ask first
- Modify `pyproject.toml` (dependency changes)
- Push to `main`
- Add new dependencies

### Never
- Commit `results/` data (gitignored for a reason)
- Create subpackages under `src/herald/`
- Add `from __future__ import annotations`

