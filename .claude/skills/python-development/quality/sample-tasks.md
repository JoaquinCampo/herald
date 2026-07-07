# Sample task suite

Use these tasks to evaluate whether an agent applies the skill in realistic
situations. The machine-readable source is `sample_tasks.json`.

## Tasks

- `plan-no-edit`: verifies that planning and review prompts do not trigger file
  edits.
- `feature-cli`: checks CLI design, config validation, command discovery, and
  stdout or stderr behavior.
- `bug-config-cause`: checks exception chaining, regression tests, and focused
  validation.
- `review-risky-diff`: checks risk triage, premise review, migrations, and
  verification-gap reporting.
- `typing-forward-refs`: checks Python-version-aware typing and annotation
  guidance.
- `io-security`: checks path traversal and archive extraction safety.
- `async-cleanup`: checks async debugging, cancellation, timeouts, and cleanup.
- `dependency-upgrade`: checks lockfile discipline, dependency boundaries, and
  release smoke validation.
- `data-model-choice`: checks whether agents choose appropriate named data
  shapes without overusing validation frameworks.
- `observability-boundary`: checks logs, metrics, tracing, secret handling, and
  external service adapter boundaries.
- `docs-style-cleanup`: checks behavior-preserving cleanup of names, comments,
  and docstrings without formatter bikeshedding.

## Common failure modes

- Editing files on no-edit prompts.
- Using generic commands before discovering project commands.
- Applying optional library guidance as a universal rule.
- Reporting tests as passed without running them.
- Returning review findings without concrete fixes or validation.
- Missing security implications for paths, archives, subprocesses, or secrets.

## How to use

1. Pick one sample task.
2. Give the prompt and fixture to an agent with the skill available.
3. Score the transcript with `scorecard.md`.
4. Record structured results with `scorecard.schema.json` if desired.
