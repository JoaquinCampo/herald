---
name: python-development
description: "General Python development skill for designing, implementing, refactoring, debugging, testing, validating, and reviewing Python code. Use for Python 3.11+ projects, especially modern pyproject-based repos using uv, Ruff, pytest, mypy or pyright, Pydantic v2, Typer, or Loguru. Adapt to the target repository before applying defaults."
---

# Python Development

Use this as the default Python development playbook. It is intentionally broader
than a review rubric: it guides how to understand a repository, design the
smallest useful API, write the code, test the behavior, validate the change, and
review the result.

The skill is opinionated but not blind. First discover the target project's
Python version, tools, commands, and local conventions. Then apply the defaults
below only where they fit.

## When to use this skill

Use this for Python work involving:

- feature implementation
- bug fixes and debugging
- refactoring and cleanup
- test design and test repair
- API, data model, CLI, or package design
- code review and self-review
- migration to modern Python tooling
- questions about Python best practices, clean code, or maintainability

If a domain-specific skill is also relevant, such as web deployment, browser
automation, ML inference, data science, or HPC, use it as an overlay. This skill
remains the baseline for Python quality.

## Operating rule

Do not start by coding. Start by discovering the local contract.

1. Read local instructions and the user request.
2. Inspect project metadata and configuration.
3. Inspect nearby code and tests.
4. Identify the exact behavior or design contract.
5. Choose the narrowest useful implementation and validation path.

When the user asks for planning, advice, review, assessment, explanation, or
any other no-edit task, do not edit files unless they explicitly ask for
changes.

## Sources of truth

Follow these in order:

1. The user's explicit request.
2. Local agent or repository instructions.
3. Project configuration, such as `pyproject.toml`, `setup.cfg`, `tox.ini`,
   `noxfile.py`, `pytest.ini`, `.pre-commit-config.yaml`, CI workflows, and lock
   files.
4. Existing code and tests.
5. This skill and its references.
6. General ecosystem practice.

If project configuration conflicts with this skill, follow the project and note
the tradeoff only when it matters.

## Modern Python defaults

Use these as defaults when the project does not say otherwise:

- Python: target the version declared by the project. For new code, prefer
  Python 3.12+ when the project allows it.
- Packaging and metadata: prefer `pyproject.toml` as the central configuration
  file.
- Environment and package manager: prefer `uv` for new projects or projects
  already using it. Otherwise follow the existing tool, such as pip, pip-tools,
  Poetry, Hatch, PDM, tox, or nox.
- Formatting and linting: prefer Ruff formatter plus Ruff linting when the
  project has not standardized elsewhere.
- Type checking: prefer strict or gradually strict mypy or pyright. Follow the
  checker already configured in the repo.
- Testing: prefer pytest for new test suites unless the project uses unittest or
  another runner.
- Validation: run focused checks first, then broader checks before declaring a
  substantial change complete.

Never hardcode commands from this skill if the repository defines its own.
Discover and use the repo's actual commands.

## Development loop

### 1. Orient

- Find the package layout, source roots, tests, and configured tools.
- Read the nearest existing code before inventing new patterns.
- Read tests around the behavior before changing behavior.
- For large or unfamiliar areas, build a short map: inputs, outputs, side
  effects, dependencies, callers, tests, and failure modes.

### 2. Define the contract

Before implementation, state or infer:

- what the function, module, CLI, or service accepts
- what it returns or produces
- what side effects it has
- what errors it raises
- what invariants must hold
- what tests or checks will prove it works

Prefer the smallest API current callers need. Do not add hooks, modes, flags, or
configuration for hypothetical futures.

### 3. Test the behavior

When behavior changes, add or update tests before implementation when practical.
At minimum, know what focused validation will fail before the fix or protect the
new behavior after the change.

Good tests assert observable behavior. Weak tests assert only that an internal
method was called.

### 4. Implement simply

- Keep functions small and single-purpose.
- Keep abstraction levels separate: orchestration reads like intent, helpers do
  low-level work.
- Prefer guard clauses over deep nesting.
- Prefer explicit errors over `None`, `-1`, empty strings, or silent fallbacks as
  failure signals.
- Use plain data structures until a named model improves clarity or validation.
- Prefer the standard library or an existing dependency over custom code when it
  cleanly solves the problem.
- Add abstractions only when there are real current call sites or invariants.

### 5. Validate

Use the narrowest useful command first, for example one changed test module or
one type-check target. Then run the broader project check before declaring
substantial work complete, or state why it was not run.

Validation evidence should include the command, result, and any known limitation.

### 6. Self-review

Before completion, review the diff for:

- correctness and failure modes
- API clarity
- typing and data model fit
- test signal quality
- hidden side effects
- unnecessary abstraction
- IO and resource safety
- performance and concurrency hazards in hot paths
- consistency with project conventions

## Mode playbooks

### Feature work

- Identify the public contract and callers first.
- Add focused tests for the new behavior.
- Implement the smallest useful path.
- Prefer obvious data flow over clever reuse.
- Validate focused tests, then broader checks.

### Bug fixing

- Reproduce the bug or locate the failing path with evidence.
- Add a regression test when practical.
- Fix the root cause, not only the symptom.
- Verify the original path and the regression test.
- Note any adjacent risks not fixed.

### Refactoring

- Separate mechanical moves from behavior changes.
- Preserve behavior with existing tests or characterization tests.
- Keep the public API stable unless the task is an API migration.
- Rename for clarity when it reduces future mistakes.
- Delete dead code instead of routing around it.

### Debugging

- Capture the exact error, input, environment, and stack trace.
- Reduce to the smallest reproducer.
- Inspect runtime state when source reading is insufficient.
- Form one hypothesis at a time and test it.
- Remove temporary probes before finishing.

### Testing

- Prefer tests that exercise real behavior through public or stable seams.
- Mock at boundaries, not inside the code being tested.
- Use fixtures to remove setup noise, not to hide essential inputs.
- Use parametrization for variations.
- Use property-based tests for pure logic with clear invariants.

### Review

- Triage first. Route depth by risk, not habit. A tiny isolated change needs a
  different review than an auth, data, dependency, migration, concurrency, or
  public API change.
- Form expectations before reading the diff. Compare the stated goal with what
  changed, what did not change, and what tests prove.
- Check the premise before polishing implementation. Ask whether the change
  should exist, whether existing code already solves it, and whether a standard
  library or current dependency should replace custom code.
- Read the relevant code, tests, and configuration before judging.
- Separate configured tool failures from human judgment findings.
- Report only actionable issues with a concrete fix.
- Do not invent findings to appear thorough.
- End substantial reviews with what you did not verify, such as commands not
  run, call sites not checked, or integration paths not exercised.
- If there are no findings, say what was checked, what was not checked, and why
  the result is acceptable.

## Review finding schema

Use this JSON-friendly shape for review mode. The canonical schema lives in
`references/review-schema.md`; keep this summary and the examples in sync with
that file.

```json
{
  "file": "src/package/module.py",
  "line": 42,
  "symbol": "optional function or class name",
  "category": "correctness|api|design|typing|errors|testing|io|performance|concurrency|security|packaging|maintainability|style|tooling",
  "severity": "high|medium|low",
  "title": "Short problem statement",
  "why": "Concrete risk, broken contract, or violated rule",
  "fix": "Smallest useful change",
  "validation": "Test, check, or manual verification that should cover the fix"
}
```

Severity:

- **high:** correctness bug, silent failure, data loss, security issue, broken
  test signal, API misuse, or likely production failure.
- **medium:** unclear API, fragile design, weak test, wrong abstraction, or
  maintainability issue likely to affect near-term work.
- **low:** local readability or consistency issue that materially improves
  comprehension.

Rules:

- One issue per record.
- Include why it matters and how to fix it.
- Do not report issues already enforced by configured tools unless those tools
  are not being run.
- Locate by content when line numbers are approximate.

## High-signal checklist

### API design

- Ambiguous options and flags are keyword-only.
- Return types match consumption: reusable collection versus one-shot iterator.
- `None` is not overloaded when it is a valid input or output.
- Data crossing boundaries has a named shape when that improves clarity.
- Public APIs avoid speculative parameters.

### Typing

- Type syntax matches the project's supported Python version.
- Use built-in generics and `|` unions when supported.
- Use `Protocol` for structural dependency boundaries.
- Use overloads only when they make caller types more precise.
- Avoid type tricks that make runtime behavior hard to understand.

### Errors and logging

- Exceptions are specific and preserve useful causes.
- Runtime validation uses explicit raises, not `assert`.
- Failure modes do not pass silently.
- Logs add diagnostic context without noise.
- Hot paths avoid expensive disabled log formatting.

### Imports and packaging

- Imports are absolute inside packages. Do not add relative imports.
- Runtime imports are not hidden under `TYPE_CHECKING`.
- Packaging changes are validated with the project's configured build or import
  smoke checks.

### IO and resources

- Paths use `pathlib` unless the project has a reason not to.
- Text IO uses explicit encodings.
- Files, network handles, locks, tasks, and subprocesses are cleaned up.
- Serialization formats and schemas are explicit at boundaries.

### Tests

- Tests assert outcomes, not implementation trivia.
- Exception tests match useful messages.
- Filesystem and environment changes are isolated.
- Mocks have specs when possible.
- Third-party libraries are wrapped behind seams before heavy mocking.

### Security basics

- Inputs are validated at trust boundaries.
- Secrets are never logged or committed.
- Subprocesses avoid shell strings when argument lists work.
- Archive extraction and user-provided paths cannot write outside intended
  directories.

### Maintainability

- Names reveal intent.
- Public functions, classes, modules, and non-obvious helpers have useful
  docstrings when the project expects them.
- Comments explain why, not what the code already says.
- Dead code is removed.
- Side effects are visible at the call site.
- Hard-to-explain code is simplified before it is documented.

## Reference catalogue

Open only the files relevant to the task. When editing this skill itself, first
read `references/maintenance.md` and `quality/README.md`, then run the quality
harness before declaring the edit complete.

- `references/project-discovery.md`: how to detect Python version, tools,
  commands, package layout, and local conventions.
- `references/tooling.md`: formatters, linters, type checkers, pre-commit, and
  CI.
- `references/development-loop.md`: detailed workflow for writing code.
- `references/validation.md`: choosing checks by work mode and risk.
- `references/api-design.md`: functions, parameters, return values, and public
  contracts.
- `references/typing.md`: modern typing and version-aware annotation choices.
- `references/data-modeling.md`: dataclasses, Pydantic, TypedDict, enums, and
  plain collections.
- `references/configuration.md`: settings precedence, environment, files, and
  secrets.
- `references/cli-apps.md`: command shape, exit codes, stdout, stderr, and CLI
  tests.
- `references/boundaries.md`: web, database, external service, and adapter
  boundaries.
- `references/errors-logging.md`: exceptions, validation, chaining, and logs.
- `references/observability.md`: logs, metrics, tracing, health, and diagnostics.
- `references/testing.md`: pytest-oriented testing guidance and mocking seams.
- `references/debugging.md`: reproducing, isolating, and fixing bugs.
- `references/refactoring.md`: safe behavior-preserving change.
- `references/review-schema.md`: structured findings and severity policy.
- `references/anti-patterns.md`: common process, design, testing, and safety
  smells.
- `references/io-filesystem.md`: pathlib, encodings, temp files, JSON, CSV, and
  resources.
- `references/imports-packaging.md`: imports, pyproject, source roots, and build
  behavior.
- `references/dependency-management.md`: lockfiles, dependency groups, extras,
  and upgrades.
- `references/ci-release.md`: CI commands, builds, entry points, and package
  smoke tests.
- `references/generated-code-and-migrations.md`: generated outputs, schema
  changes, and data migrations.
- `references/performance.md`: profiling, batching, caching, and hot paths.
- `references/concurrency.md`: async, threads, subprocesses, cancellation, and
  cleanup.
- `references/security.md`: inputs, secrets, subprocesses, and unsafe
  deserialization.
- `references/documentation.md`: docstrings, comments, TODOs, and reader
  contracts.
- `references/style-naming.md`: names, builtin shadowing, and style review
  boundaries.
- `references/notebooks.md`: notebook reproducibility, review, and test
  extraction.
- `references/maintenance.md`: consistency checks when editing this skill.
- `references/scientific-python.md`: optional NumPy, pandas, and torch guidance.
- `references/libraries.md`: optional Pydantic, Typer, Loguru, FastAPI, and
  other ecosystem notes.

Examples live in `examples/` and show review findings and validation summaries.
