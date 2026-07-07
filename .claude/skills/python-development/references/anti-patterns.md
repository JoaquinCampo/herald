# Anti-patterns

Use this as a quick smell list during implementation and review. A smell is not
a finding by itself. It becomes a finding only when you can name the concrete
risk and the smallest useful fix.

## Process anti-patterns

- Editing files when the user asked only for advice, planning, explanation, or
  review.
- Running broad formatters before understanding the diff.
- Inventing commands instead of discovering project commands.
- Claiming tests or validation without running them.
- Fixing nearby unrelated problems in the same change.

## Design anti-patterns

- A function that mixes orchestration, parsing, IO, validation, and formatting.
- A class that wraps one stateless method.
- A parameter or hook with no current caller.
- A tuple return whose positions become public API.
- A framework handler containing business logic that could be tested separately.

## Testing anti-patterns

- Tests that assert only a mock call.
- Tests with expected values computed by the same logic as the implementation.
- Tests that depend on execution order or leaked environment variables.
- Async tests that use arbitrary sleeps as synchronization.
- Exception tests that assert only exception type when many code paths can raise
  that type.

## Safety anti-patterns

- Broad exception handlers that continue silently.
- Path joins with untrusted absolute or traversal components.
- `shell=True` with user-controlled input.
- Logs that include secrets, tokens, cookies, or full sensitive payloads.
- Unsafe deserialization for untrusted data.

## Sources

- Google Python style guide: https://google.github.io/styleguide/pyguide.html
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- Python subprocess security considerations: https://docs.python.org/3/library/subprocess.html#security-considerations
- OWASP Top Ten: https://owasp.org/www-project-top-ten/
