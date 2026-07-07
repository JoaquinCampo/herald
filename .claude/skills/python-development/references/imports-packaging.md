# Imports and packaging

Packaging and imports define how code is discovered, installed, tested, and
reused. Follow the project's existing layout unless you are intentionally
migrating it.

## Imports

- Keep imports at module top unless a deferred import has a measured startup,
  optional dependency, or cycle reason.
- Use absolute imports inside packages. Do not add relative imports in new code.
  If a project already has relative imports, leave broad migration for an
  explicit cleanup task rather than mixing it into unrelated work.
- Do not hide runtime-needed imports under `TYPE_CHECKING`.
- Move type-only imports under `TYPE_CHECKING` only when postponed annotations or
  string annotations make that safe for the project.

## pyproject

For modern projects, `pyproject.toml` should carry standardized project metadata
and tool configuration where supported. Do not duplicate the same setting across
several files unless the tools require it.

Common fields to inspect:

- `[project]` name, version, dependencies, optional dependencies, scripts.
- `[dependency-groups]` for standardized local dependency groups such as dev,
  test, lint, or docs groups when the project uses the PyPA dependency-groups
  specification.
- `[build-system]` backend and requirements.
- tool tables such as `[tool.ruff]`, `[tool.pytest.ini_options]`, `[tool.mypy]`,
  `[tool.pyright]`, `[tool.uv]`, `[tool.hatch]`, or `[tool.pdm]`.

## Source layout

A `src/` layout helps catch import mistakes because tests import the installed
package rather than the working directory by accident. Flat layouts are also
valid, especially for small apps. Follow the repo unless changing layout is the
explicit task.

## Build validation

For packaging changes, validate the affected behavior:

- import the package
- run configured tests
- build a wheel or sdist when relevant
- verify CLI entry points or package data if changed

## Sources

- Python Packaging User Guide: https://packaging.python.org/
- pyproject guide: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- Dependency groups specification: https://packaging.python.org/en/latest/specifications/dependency-groups/
- src layout discussion: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- import system docs: https://docs.python.org/3/reference/import.html
