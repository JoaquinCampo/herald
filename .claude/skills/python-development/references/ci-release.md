# CI and release

CI and release behavior are part of the development contract. Discover them
before changing packaging, commands, tests, or deployment-sensitive code.

## CI discovery

Inspect workflow files and scripts for:

- supported Python versions
- OS matrix
- install command
- cache behavior
- lint, type, and test commands
- coverage thresholds
- integration service setup
- release or publish jobs

Do not assume local commands are the same as CI commands. Prefer the documented
local equivalent when present.

## Release-sensitive changes

For packaging, CLI entry points, public APIs, or dependency changes, consider:

- build succeeds
- package imports after installation
- CLI scripts execute
- package data is included
- version metadata is correct
- optional extras or dependency groups still resolve

## Reporting

When you cannot run CI-equivalent checks locally, say which CI job would provide
coverage and what remains unverified.

## Sources

- Python Packaging User Guide: https://packaging.python.org/
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- uv build docs: https://docs.astral.sh/uv/guides/package/
