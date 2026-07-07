---
name: code-quality-review
description: "Default Python development and review standard for this repo. Use whenever writing, changing, refactoring, debugging, testing, or reviewing Python code; when asked for modern Python, best practices, clean code, or Zen of Python guidance; and as the final self-review rubric before declaring Python work complete. Targets Python 3.12."
---

# Python Development Standard (Python 3.12)

This is the go-to Python skill for this repo. Use it before writing code,
while implementing, and when reviewing the result. The old role of this
skill, code-quality review, is now one workflow inside a broader development
playbook.

Do not duplicate Ruff. Ruff catches many mechanical issues. This skill covers
human judgment, API design, testing discipline, Python 3.12 subtleties, and
repo conventions that tools cannot reliably infer.

**Full cited catalogue:** `references/` contains 97 source-cited entries
(PEPs, docs.python.org, Ruff, pytest, library docs), indexed in
`references/README.md`. Open only the topic needed for the task. Each entry has
a rule, why it matters, bad-to-good examples, and a citation.

## When to use this skill

Use this for any Python work in `src/`, `tests/`, `scripts/`, or project
configuration that affects Python behavior:

- design or implementation of a feature
- bug fixes and debugging
- refactors and cleanup
- tests, fixtures, mocks, or evaluation harnesses
- reviews of diffs, plans, generated code, or PRs
- requests for best practices, modern Python, or less ugly code

If the task touches Pydantic, Hugging Face generation, kvpress, survival
modeling, GSM8K, HPC, or browser/deployment work, also load the relevant
specialized skill. This skill remains the baseline Python standard.

## Sources of truth

Follow these in order:

1. The user's explicit request.
2. Project `AGENTS.md` files and `pyproject.toml`.
3. This skill.
4. The cited topic references under `references/`.
5. General Python practice.

For this repo, `pyproject.toml` currently sets Python 3.12, Ruff line length
78, Ruff lint selection `E, F, I, UP, B, SIM`, strict mypy over `src` and
`tests`, and `pytest` over `tests`.

## Non-negotiable repo rules

- Use Python 3.12+.
- Do not add `from __future__ import annotations` unless there is a real,
  documented need for postponed annotation evaluation. Modern `list[int]` and
  `X | None` syntax already works without it.
- Use modern typing: `list`, `dict`, `tuple`, `set`, `X | None`, `X | Y`.
  Never use `List`, `Dict`, `Tuple`, `Optional`, or `Union` for new code.
- Prefer functions over classes. Add a class only when it owns real state,
  invariants, or polymorphic behavior.
- Keep `src/herald/` flat by default. Add a subpackage only when several
  modules form one cohesive concern.
- Keep Pydantic simple. Use it for structured config and data validation, not
  for clever business logic.
- Use `pathlib.Path` for filesystem paths. Use explicit UTF-8 encodings for
  text IO.
- Use `loguru` for logging, not `print` or stdlib `logging` in project code.
- Use `typer` for CLIs.
- Use `uv run`, never bare `python`, for project commands.
- Use Ruff for format and lint, mypy for types, pytest for tests.
- Before any commit, run `uv run poe check` unless the user explicitly scopes
  the task to a smaller verification surface.

## Default development loop

Use this loop for every nontrivial Python change.

1. **Orient**
   - Inspect the nearest existing code and tests before designing new shapes.
   - Read `pyproject.toml` when command, lint, type, or test behavior matters.
   - Open the relevant topic file under `references/` when the task touches a
     specialized concern.

2. **Define the contract**
   - Name the inputs, outputs, side effects, failure modes, and test surface.
   - Prefer the smallest API that current callers need. Do not add knobs,
     hooks, strategies, or future-proof parameters without a real caller.
   - Decide the data shape deliberately: scalar, dataclass, Pydantic model,
     `TypedDict`, enum, protocol, or plain collection.

3. **Test first when behavior changes**
   - Add or update a focused test before implementation when practical.
   - Run the focused test and observe the failure when the failure mode is easy
     to check.
   - Use literal expectations and observable outcomes. Avoid tests that only
     assert a mock was called.

4. **Implement simply**
   - Keep functions single-purpose and at one abstraction level.
   - Use guard clauses to avoid deep nesting.
   - Prefer explicit errors over `None`, `-1`, empty strings, or silent fallbacks
     as failure sentinels.
   - Keep IO, parsing, modeling, and orchestration separable enough to test.
   - Avoid speculative abstraction. Delete unused flexibility.

5. **Validate**
   - Run the narrowest useful test first, for example
     `uv run pytest tests/test_file.py -q`.
   - Run type and lint checks relevant to the changed surface.
   - Before declaring substantial work complete, run `uv run poe check` or state
     exactly why it was not run.

6. **Self-review**
   - Check the code against the high-signal rules below.
   - Fix high-severity issues immediately.
   - Summarize changed files and verification evidence.
   - If behavior changed, mention the test that protects it.

## API and data design

Prefer boring, typed, testable APIs.

- Make options and flags keyword-only with `*`.
- Use positional parameters only when order is obvious and stable.
- Use a private sentinel object when `None` is a valid argument value.
- Return a concrete `list` when callers need reuse, length, indexing, or a
  second pass. Return an iterator only when one-shot streaming is the contract.
- Return a small frozen dataclass from new APIs with multiple related result
  fields. Avoid bare tuples and avoid `NamedTuple` unless tuple semantics are
  truly part of the API.
- Use `TypedDict` for external JSON/config shapes when you need typed dict
  access without runtime model behavior.
- Use Pydantic models for runtime validation or settings, kept flat and simple.
- Use `Protocol` for dependency boundaries you do not own. Do not rely on
  `@runtime_checkable` for real structural validation.
- Use `Enum` or `StrEnum` for named modes. Spell serialized `StrEnum` values
  explicitly when they cross a wire, file, or database boundary.
- Use constants for magic numbers and strings whose meaning matters.

## Errors, logging, and control flow

- Catch the narrowest exception type. Avoid blanket `except Exception`; never
  catch `BaseException` for normal errors.
- Keep try bodies minimal so handlers cannot hide unrelated bugs.
- Re-raise with `raise NewError(...) from exc` by default. Use `from None` only
  when hiding implementation noise is intentional.
- Raise `TypeError` for wrong types and `ValueError` for unacceptable values of
  the right type.
- Never use `assert` for input validation or runtime invariants that must hold
  in optimized Python.
- Prefer real exceptions over error-sentinel returns.
- Log with enough context to debug, but do not add noisy logs. In hot paths,
  avoid eager f-string formatting for disabled log levels.

## Testing standard

Tests should make behavior hard to break without making refactors painful.

- Assert observable behavior, outputs, state, files, or raised errors.
- Use `pytest.raises(..., match=...)` and escape regex metacharacters when
  matching exact text.
- Use `tmp_path` and `monkeypatch` for filesystem and environment isolation.
- Use `pytest.mark.parametrize` instead of loops and conditional logic inside
  test bodies.
- Keep expected values literal unless computing them is the point of the test.
- Mock only at boundaries. Prefer fakes for collaborators you own.
- When mocking is needed, patch where the name is looked up and use autospec or
  a spec.
- Do not mock third-party internals directly. Wrap them in a small adapter you
  own, then fake that adapter.
- Use property-based tests for pure logic with clear invariants.
- Register custom markers in `pyproject.toml` before using them.

## Review mode

When reviewing code, separate tool findings from judgment findings.

1. Assume Ruff, mypy, and pytest will run. Do not spend findings on issues the
   enforced tools already catch unless the tool is not being run for this task.
2. Optionally run a local expanded Ruff sweep on a target file without changing
   config:

   ```bash
   uv run ruff check --select N,C4,RET,PTH,RUF,PERF,PIE,TC,ARG,FBT --no-cache path/to/file.py
   ```

3. Focus review attention on API design, failure modes, data modeling, test
   quality, hidden side effects, unnecessary abstraction, Python 3.12 subtleties,
   and repo conventions.

### Severity policy

- **high:** correctness bug, silent failure, API misuse, data loss, broken test
  signal, or runtime behavior that can fail in production. Blocks completion.
- **medium:** maintainability, unclear API, wrong abstraction, fragile test, or
  design debt likely to slow near-term work. Fix now or state a deliberate
  deferral.
- **low:** local readability improvement. Include only when it genuinely helps;
  do not bikeshed.

Every finding must include a concrete fix and why it matters. Prefer the
smallest behavior-preserving change. If a fix changes behavior, say so and add
or request a test.

### Finding schema

Return findings as structured records when reviewing:

```json
{
  "file": "src/herald/example.py",
  "line": 42,
  "category": "repo|subtle|design|testing|errors|typing|io|perf|library",
  "severity": "high|medium|low",
  "title": "Short problem statement",
  "why": "Concrete risk or violated rule",
  "fix": "Specific smallest useful change"
}
```

Rules for findings:

- One issue per finding.
- No raw file dumps back to the caller.
- No invented issues to look thorough.
- Locate by content when line numbers are approximate.
- If there are no findings, say what was checked and what evidence supports the
  clean result.

## High-signal review checklist

Use this checklist after implementation and during reviews.

### Python 3.12 and typing

- No unnecessary future annotations import.
- Modern builtins and unions, no legacy `typing` aliases.
- `Self`, `Protocol`, `TypedDict`, and overloads used only where they simplify
  real type relationships.
- Accepted parameter types are as broad as useful, return types are concrete
  enough for callers.

### Functions and structure

- Functions do one thing and stay at one abstraction level.
- Orchestrators read like intent, with low-level work in named helpers.
- Deep nesting is replaced with guard clauses where clearer.
- No class exists only to wrap one method or a bag of static methods.
- No speculative parameters, hooks, or strategy objects with one caller.

### IO and resources

- Path handling uses `Path`.
- Text IO has explicit encoding.
- Files, network handles, and temp resources are managed with context managers
  or pytest fixtures.
- Streaming is used for large data when callers do not need reuse.

### Data, libraries, and ML code

- Pydantic models are simple and validate only what needs runtime validation.
- NumPy and torch code is explicit about device, dtype, and randomness.
- Inference paths use the appropriate no-grad or inference context.
- Transformers, kvpress, and tokenizer behavior is checked against the relevant
  specialized skill before changing generation or compression internals.

### Design and readability

- Names reveal intent and do not shadow builtins.
- Comments explain why, not what the code already says.
- Dead code and commented-out code are removed.
- Side effects are visible from the function name or call site.
- Hard-to-explain code is simplified before it is documented.

## Reference catalogue

Open topic files only when relevant:

- `typing.md`: annotation semantics, modern syntax, Protocol, TypedDict, Self.
- `errors.md`: exception design, chaining, assert caveats, silent failures.
- `functions.md`: keyword-only APIs, sentinels, generators, overloads.
- `classes.md`: dataclass options, frozen models, derived fields.
- `io.md`: pathlib, encoding, temp files, JSON, CSV, packaged data.
- `imports.md`: import hygiene, `TYPE_CHECKING`, lazy exposure.
- `perf.md`: hot-loop traps, caching, comprehensions, batching.
- `naming.md`: naming, docstrings, comments, builtin shadowing.
- `numpy-torch.md`: RNG, views, inference mode, device and dtype.
- `libraries.md`: Pydantic v2, loguru, transformers, qdrant-client.
- `testing.md`: pytest, mocks, fixtures, property-based tests.
- `design.md`: dataclass returns, YAGNI, enums, properties, stateless classes.
