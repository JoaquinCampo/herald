# Errors and logging

Errors should be explicit, typed, and diagnosable. Logs should explain what
happened without becoming noise.

## Exceptions

- Catch the narrowest exception type that the protected operation can raise.
- Avoid blanket `except Exception` except at deliberate isolation boundaries,
  where you log or record and then continue intentionally.
- Never catch `BaseException` for normal errors because it includes interrupts
  and interpreter shutdown signals.
- Keep try blocks small so handlers do not catch unrelated bugs.
- Prefer `raise NewError(...) from exc` when wrapping an exception.
- Use `from None` only to hide implementation details deliberately.

## Validation

Use explicit exceptions for runtime validation. Do not use `assert` for checks
that must run in production, because optimized Python can remove asserts.

Raise:

- `TypeError` for the wrong type.
- `ValueError` for an unacceptable value of the right type.
- A project-specific exception subclass when callers need to catch a family of
  domain errors.

## Failure signals

Do not return `None`, `False`, `-1`, or empty strings to mean failure when those
values could also be valid results. Raise an exception or return a named result
that encodes success and failure explicitly.

## Logging

- Prefer the logging library already used by the repo.
- Log at boundaries, retries, unexpected exceptions, and long-running operations.
- Include useful context, not secrets.
- Avoid `print` in library code.
- In hot paths, avoid eager formatting for disabled log levels.

## Sources

- Python tutorial on exceptions: https://docs.python.org/3/tutorial/errors.html
- Python exceptions docs: https://docs.python.org/3/library/exceptions.html
- Google Python style guide, exceptions: https://google.github.io/styleguide/pyguide.html#24-exceptions
- Loguru docs: https://loguru.readthedocs.io/
