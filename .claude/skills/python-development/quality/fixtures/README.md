# Fixture catalogue

Fixtures are compact project shapes used by the sample task suite. Some are
runnable tiny repositories; others are descriptive fixtures for manual review.

## library-service

A pyproject-based library with `src/package/`, `tests/`, pytest, Ruff, and a
JSON config loader. It exposes a small public API and a service adapter.

## cli-config-app

Runnable fixture under `cli-config-app/`. A small application with one CLI entry
point, config files, environment variable overrides, and tests around stdout,
stderr, and exit codes.

## migration-api-change

A service package with a public Python API, a database migration directory, and
integration tests. The review should treat migration changes as high risk.

## typed-library

A reusable package supporting Python 3.10 through 3.14, with runtime code that
inspects annotations for registration.

## file-processing-app

Runnable fixture under `file-processing-app/`. An app that accepts uploaded
archives, extracts selected files, and writes them under a configured output
directory.

## async-worker

An asyncio worker with background tasks, cancellation behavior, and flaky tests
that sometimes leave tasks pending.

## packaged-library

A library with dependency metadata, a lockfile, CI build jobs, and an adapter for
an external HTTP client.
