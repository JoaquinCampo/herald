# Observability

Observability is how future maintainers understand behavior in production,
tests, and local runs. Keep it useful, low-noise, and safe.

## Logging

- Use the project's logging library.
- Log at boundaries, retries, long-running operations, and unexpected failures.
- Include stable identifiers and context that help debug.
- Do not log secrets or full sensitive payloads.
- Avoid noisy per-item logs in hot loops unless sampled or debug-only.

## Metrics and tracing

When a project already has metrics or tracing, preserve its naming and tag
conventions. Useful signals include latency, error counts, queue depth, retry
counts, cache hit rates, and external dependency status.

Do not introduce a new observability stack as part of ordinary feature work
unless requested. Add seams or hooks that fit the existing stack.

## Health and diagnostics

For services and long-running jobs, prefer explicit health or diagnostic paths
over relying on log scraping alone. Diagnostic output should not expose secrets.

## Sources

- Python logging docs: https://docs.python.org/3/library/logging.html
- Loguru docs: https://loguru.readthedocs.io/
- OpenTelemetry Python docs: https://opentelemetry.io/docs/languages/python/
