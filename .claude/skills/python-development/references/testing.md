# Testing

Tests should protect behavior while allowing implementation to improve.

## Test observable behavior

Prefer assertions on outputs, state, files, emitted events, database rows, or
raised errors. Avoid tests whose only assertion is that a mock method was called.

## Exception tests

Use `pytest.raises(..., match=...)` when testing errors. Remember that `match`
is a regex search, so escape or anchor when you mean exact text.

## Parametrize cases

Use `pytest.mark.parametrize` for input variations. Avoid loops, branching, and
computed expectations inside test bodies unless the computation is the point of
the test.

## Fixtures

Fixtures should reduce noise without hiding important inputs. Use factory
fixtures when tests need several related instances.

Use:

- `tmp_path` for filesystem isolation.
- `monkeypatch` for environment variables and attribute patches.
- `capsys` or `capfd` for stdout and stderr behavior.
- `caplog` for stdlib logging behavior when the project uses stdlib logging.
- registered markers for categories like slow, integration, network, or gpu.

## Mocking

- Patch where the name is looked up, not where it was originally defined.
- Use `autospec`, `spec`, or fakes so tests fail when interfaces drift.
- Do not mock third-party internals throughout the suite. Wrap third-party APIs
  behind small adapters you own, then fake those adapters.
- Prefer fakes over mocks when behavior is simple enough to model.

## Property-based testing

Use Hypothesis or another property-based tool for pure logic with clear
invariants, such as parsers, normalizers, encoders, routing functions, and
round-trips.

## Test quality checklist

- Would this test fail for the bug it claims to prevent?
- Does it verify behavior rather than implementation trivia?
- Is the expected value independent of the implementation?
- Is state isolated from other tests?
- Does the failure message point to the problem?

## Sources

- pytest assertions: https://docs.pytest.org/en/stable/how-to/assert.html
- pytest fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html
- pytest monkeypatch: https://docs.pytest.org/en/stable/how-to/monkeypatch.html
- pytest parametrization: https://docs.pytest.org/en/stable/how-to/parametrize.html
- pytest logging capture: https://docs.pytest.org/en/stable/how-to/logging.html
- pytest output capture: https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html
- unittest.mock where to patch: https://docs.python.org/3/library/unittest.mock.html#where-to-patch
- Hypothesis quickstart: https://hypothesis.readthedocs.io/en/latest/quickstart.html
- Hynek Schlawack on mocking: https://hynek.me/articles/what-to-mock-in-5-mins/
