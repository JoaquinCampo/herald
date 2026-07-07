# CLI applications

Command-line interfaces are user interfaces. Keep command functions thin,
validated, and easy to test.

## Command shape

- Parse arguments at the boundary.
- Convert arguments into typed values early.
- Delegate business logic to functions that do not depend on CLI framework
  objects.
- Return meaningful exit codes.
- Write machine-readable output to stdout when promised.
- Write diagnostics, progress, and errors to stderr.

## Framework choice

Use the framework already present. For new CLIs:

- `argparse` is a solid standard-library default.
- Typer is useful for type-hint-driven CLIs when adding a dependency is
  acceptable.

Do not force Typer onto a project that already has a simple argparse CLI unless
the task is a CLI migration.

## Testing

Test CLI behavior through the command entry point or framework test runner when
possible. Also test the underlying business logic directly without CLI parsing.

Check:

- argument validation
- exit code
- stdout and stderr separation
- config precedence
- error messages

## Sources

- argparse docs: https://docs.python.org/3/library/argparse.html
- Typer docs: https://typer.tiangolo.com/
- pytest output capture: https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html
