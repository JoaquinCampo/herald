# Project discovery

A general Python skill must adapt to the repository. Do not assume package
manager, Python version, linter, type checker, line length, source layout, or
commands before checking local files.

## Discovery order

1. Read local agent or contributor instructions.
2. Inspect `pyproject.toml` first when present.
3. Check secondary config: `setup.cfg`, `setup.py`, `tox.ini`, `noxfile.py`,
   `pytest.ini`, `.pre-commit-config.yaml`, `ruff.toml`, `.ruff.toml`,
   `mypy.ini`, `pyrightconfig.json`, CI workflows, Dockerfiles, and Makefiles.
4. Check human command surfaces: `README*`, `CONTRIBUTING*`, `Justfile`,
   `Taskfile.yml`, `Poe` or `poethepoet` tasks, `Makefile`, `package.json`
   scripts for mixed repos, `scripts/`, and CI job commands.
5. Identify lock files: `uv.lock`, `poetry.lock`, `pdm.lock`,
   `requirements*.txt`, `Pipfile.lock`, or conda environment files.
6. Inspect nearby code and tests for established patterns.
7. Prefer existing commands over invented commands.

## What to detect

- Supported Python versions, from `requires-python`, CI, Docker, tox, nox, or
  docs.
- Package manager and environment command.
- Source roots, such as `src/`, flat package layout, app directories, or
  notebooks.
- Test runner and markers.
- Formatter, linter, import sorter, and selected rule families.
- Type checker and strictness.
- Runtime framework, such as FastAPI, Django, Flask, Typer, or batch scripts.
- Public API surface, CLI entry points, package data, and plugin hooks.

## Tool defaults when absent

When starting a new Python project or filling an unspecified gap, these are good
modern defaults:

- `pyproject.toml` for project metadata and tool config.
- `uv` for environment, dependency, lock, and run workflows.
- Ruff formatter and Ruff linter for formatting, import sorting, and linting.
- pytest for tests.
- mypy or pyright for type checking. Choose one primary checker unless the repo
  already uses both.

## Command selection

Prefer commands defined by the project. Check task runners and scripts before
inventing a generic command. Common examples include:

- `uv run pytest`, `pytest`, `tox`, `nox`, or a documented task for tests.
- `uv run ruff check`, `ruff check`, pre-commit, or a documented task for
  linting.
- `uv run mypy`, `mypy`, `pyright`, `basedpyright`, or a documented task for
  types.
- `uv build`, `python -m build`, Hatch, PDM, Poetry, or a documented task for
  builds.

State the command you ran and the result. If you cannot run the full project
check, run focused checks and explain the limitation.

## Sources

- Python Packaging User Guide, pyproject metadata:
  https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- uv project docs: https://docs.astral.sh/uv/concepts/projects/
- Ruff configuration docs: https://docs.astral.sh/ruff/configuration/
- pytest configuration docs: https://docs.pytest.org/en/stable/reference/customize.html
- mypy configuration docs: https://mypy.readthedocs.io/en/stable/config_file.html
- pyright configuration docs: https://microsoft.github.io/pyright/#/configuration
