# file-processing-app fixture

A tiny zip extraction app for evaluating IO and security behavior.

Run, from this directory:

```bash
PYTHONPATH=src uv run --with pytest pytest -q
```

Expected focus:

- archive member names are untrusted
- path traversal is rejected
- tests cover safe and malicious members
