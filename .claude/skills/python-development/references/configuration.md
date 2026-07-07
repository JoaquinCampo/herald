# Configuration and settings

Configuration is a boundary. Treat it like input from users, files, the
environment, deployment systems, or command lines.

## Precedence

Discover the project's precedence before changing settings behavior. When the
project does not define one, prefer a simple documented order such as:

1. explicit function or CLI arguments
2. environment variables
3. config files
4. application defaults

Do not silently let two sources fight. If precedence is ambiguous, document it
or raise a clear error for conflicting values.

## Validation

Validate configuration at the boundary and expose typed values internally.
Useful choices include:

- plain parser functions for small apps
- dataclasses for internal settings already parsed elsewhere
- Pydantic settings when runtime parsing, environment integration, and type
  validation are worth the dependency

Keep validators simple. Avoid hiding business logic inside settings models.

## Secrets

- Do not print or log raw secrets.
- Keep secret values out of repr strings and exception messages.
- Prefer references to secret names or sources over secret contents.
- Treat config dumps as potentially sensitive.

## Errors

Configuration errors should name the setting, source, and expected shape when
safe. They should not require reading a traceback to understand what to fix.

## Sources

- Pydantic settings docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
- argparse docs: https://docs.python.org/3/library/argparse.html
- Python os.environ docs: https://docs.python.org/3/library/os.html#os.environ
