# cli-config-app fixture

A tiny argparse app for evaluating CLI, configuration, and validation behavior.

Run, from this directory:

```bash
PYTHONPATH=src uv run --with pytest pytest -q
```

Expected focus:

- command handler delegates to testable helpers
- stdout is machine-readable output
- stderr is diagnostics
- malformed config preserves useful error context
