# Library notes

This file is optional guidance. Do not require a project to adopt these
libraries. Apply the relevant section only when the project already uses the
library or the user asks for it.

## uv

Use `uv` for project workflows when present or when starting a new modern repo:

- `uv run` to run commands in the project environment.
- `uv sync` to synchronize the environment from the lockfile.
- `uv lock` to update the lockfile.
- `uv add` and `uv remove` for dependencies when dependency changes are in
  scope.

## Ruff

Ruff can format, lint, organize imports, and enforce many modernization rules.
Use the repository's configured selection. Do not assume every Ruff rule is
enabled.

Treat fixes according to Ruff's safety model. Safe fixes are intended to preserve
behavior; unsafe fixes may change behavior and should be reviewed like code
changes. Preview rules and preview formatter behavior can change, so do not make
preview-only guidance universal unless the repository opted in.

Sources:

- https://docs.astral.sh/ruff/
- https://docs.astral.sh/ruff/linter/#fixes
- https://docs.astral.sh/ruff/preview/

## Pydantic v2

Use Pydantic for runtime validation and serialization. Prefer simple models,
plain fields, `Field`, and clear validators. Remember that v2 validator APIs and
serialization semantics differ from v1.

## Typer

Use Typer for CLIs when the project wants type-hint-driven command definitions.
Keep command functions thin and delegate business logic to testable helpers.

## Loguru

Use Loguru when the project has standardized on it. Prefer structured messages
with contextual values. Do not log secrets. Avoid noisy logs in tight loops.

## FastAPI

Use Pydantic models at request and response boundaries. Keep route handlers thin
and move business logic into testable functions or services.

## Sources

- uv docs: https://docs.astral.sh/uv/
- Ruff docs: https://docs.astral.sh/ruff/
- Ruff fixes: https://docs.astral.sh/ruff/linter/#fixes
- Ruff preview: https://docs.astral.sh/ruff/preview/
- Pydantic docs: https://docs.pydantic.dev/latest/
- Typer docs: https://typer.tiangolo.com/
- Loguru docs: https://loguru.readthedocs.io/
- FastAPI docs: https://fastapi.tiangolo.com/
