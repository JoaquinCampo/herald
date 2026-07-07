# Source audit

This file records the primary sources used to keep the skill current. It is not
a substitute for checking the docs again when making strong future claims.

## Python language and standard library

- Python 3.14 What's New, annotation semantics, multiprocessing defaults,
  asyncio changes, debugger updates:
  https://docs.python.org/3.14/whatsnew/3.14.html
- Python typing docs:
  https://docs.python.org/3/library/typing.html
- `annotationlib` docs:
  https://docs.python.org/3/library/annotationlib.html
- `pathlib`, `subprocess`, `tarfile`, `argparse`, `logging`, `asyncio`,
  `multiprocessing`, and `concurrent.futures` docs under:
  https://docs.python.org/3/library/

## Packaging and environments

- Python Packaging User Guide:
  https://packaging.python.org/
- pyproject guide:
  https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- Dependency groups specification:
  https://packaging.python.org/en/latest/specifications/dependency-groups/
- uv docs:
  https://docs.astral.sh/uv/

## Tooling

- Ruff docs:
  https://docs.astral.sh/ruff/
- Ruff fixes:
  https://docs.astral.sh/ruff/linter/#fixes
- Ruff preview:
  https://docs.astral.sh/ruff/preview/
- pytest docs:
  https://docs.pytest.org/
- mypy docs:
  https://mypy.readthedocs.io/
- pyright docs:
  https://microsoft.github.io/pyright/

## Optional ecosystem

- Pydantic docs:
  https://docs.pydantic.dev/latest/
- Typer docs:
  https://typer.tiangolo.com/
- Loguru docs:
  https://loguru.readthedocs.io/
- FastAPI docs:
  https://fastapi.tiangolo.com/
- OpenTelemetry Python docs:
  https://opentelemetry.io/docs/languages/python/

## Scientific Python

- NumPy docs:
  https://numpy.org/doc/stable/
- pandas docs:
  https://pandas.pydata.org/docs/
- PyTorch docs:
  https://pytorch.org/docs/stable/

## Review cadence

For technical edits, re-check the relevant source category above. For Python
version guidance, check the target version's docs and porting notes. For tool
behavior, check the exact tool's current docs and the repository's configured
version when possible.
