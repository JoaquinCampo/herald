# Validation report examples

## Feature change

Changed files:

- `src/package/parser.py`
- `tests/test_parser.py`

Validation:

- `pytest tests/test_parser.py -q`, passed
- `ruff check src/package/parser.py tests/test_parser.py`, passed
- `mypy src/package/parser.py`, passed

Notes:

- Full suite was not run because the user requested a narrow patch. The touched
  parser behavior is covered by focused tests.

## Bug fix

Root cause:

- `load_config` swallowed `JSONDecodeError` context while wrapping it in a
  domain error.

Validation:

- Added regression test for malformed JSON.
- Confirmed the test failed before the fix.
- `pytest tests/test_config.py::test_malformed_json_preserves_cause -q`, passed
  after the fix.

Residual risk:

- Other config loaders were not audited.

## Review only

Scope reviewed:

- Current diff for `src/package/client.py` and `tests/test_client.py`.

Evidence:

- Inspected public API, retry behavior, timeout handling, and test assertions.
- No commands were run because review was requested as read-only.

Findings:

- None.

Residual risk:

- Integration behavior against the live service was not exercised.
