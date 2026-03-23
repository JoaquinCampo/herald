# Learnings

## Tools

- When adding dependencies to a uv-managed Python project, always use `uv add <package>` instead of manually editing pyproject.toml — ensures proper version resolution and lockfile updates — discovered 2026-03-09

## Conventions

- Always use conventional commits: `type(scope): description` — types: feat, fix, docs, test, chore, refactor, style, ci, build, perf — e.g. `feat(detectors): add coherence collapse detection` — discovered 2026-03-09
