# Tooling

Tooling should make the easy checks automatic so human review can focus on
contracts, correctness, and design.

## Formatting and linting

Discover configured tools before running them. Ruff can act as formatter,
import sorter, and linter, but repositories may use Black, isort, flake8,
Pylint, or pre-commit instead.

Guidance:

- Use the project's configured command and rule set.
- Do not argue with formatter-owned style in human review.
- Treat unsafe auto-fixes as code changes that need review.
- Treat preview rules or formatter behavior as opt-in and potentially unstable.
- Prefer adding or adjusting tooling only when repeated human comments show a
  rule should be automated.

## Type checking

Use the checker configured by the project, usually mypy, pyright, basedpyright,
or pyre. Do not mix checker-specific advice without checking which one is in
use.

Good type-checking practice:

- Keep annotations useful at boundaries and public APIs.
- Avoid broad `Any` leaks unless there is a boundary or gradual-typing reason.
- Add ignores narrowly, with a reason when project style expects it.
- Prefer simplifying code over fighting the checker with casts.

## Pre-commit and CI

If a repo uses pre-commit, prefer its configured hooks for local validation. If
CI is the source of truth, identify the closest local command and state any gap
between local and CI coverage.

## Sources

- Ruff docs: https://docs.astral.sh/ruff/
- Ruff fixes: https://docs.astral.sh/ruff/linter/#fixes
- Ruff preview: https://docs.astral.sh/ruff/preview/
- mypy docs: https://mypy.readthedocs.io/
- pyright docs: https://microsoft.github.io/pyright/
- pre-commit docs: https://pre-commit.com/
