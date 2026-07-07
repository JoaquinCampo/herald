# Development loop

Use this loop for implementation work. It keeps changes small, testable, and
aligned with the repository instead of with generic preferences.

## 1. Orient

Before coding, identify:

- the caller or user-visible behavior
- nearby implementation patterns
- existing tests
- relevant config and validation commands
- dependencies and side effects
- likely failure modes

For broad changes, write a short scratch map before editing. For small changes,
keep the map in your response or task notes.

## 2. Define the contract

A useful contract answers:

- What inputs are valid?
- What outputs are promised?
- What exceptions can happen?
- What state, files, network calls, subprocesses, or environment variables are
  touched?
- What behavior must remain backward compatible?
- What tests prove success?

If you cannot state the contract, pause and inspect more code or ask a narrow
question.

## 3. Test deliberately

For behavior changes, prefer test-first:

1. Add or update a focused test.
2. Run it and confirm the expected failure when practical.
3. Implement the fix or feature.
4. Re-run the focused test.
5. Run related and broader checks.

If test-first is impractical, state why and add protective tests before
completion.

## 4. Implement the smallest useful change

- Prefer simple functions and clear data flow.
- Avoid speculative flags and generic abstractions.
- Keep IO and pure logic separable.
- Keep orchestration separate from low-level helpers.
- Preserve public behavior unless the task is an intentional migration.

## 5. Validate in layers

Recommended order:

1. Focused unit or regression test.
2. Related test module or package.
3. Static checks on changed code.
4. Full project check when appropriate.
5. Manual smoke test for CLIs, apps, IO flows, or integrations.

## 6. Report evidence

A completion summary should include:

- changed files
- behavior changed or preserved
- validation commands and results
- known limitations or skipped checks
- follow-up risks, if any

## Sources

- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- Python Packaging User Guide: https://packaging.python.org/
- Ruff docs: https://docs.astral.sh/ruff/
