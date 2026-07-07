# Debugging

Debugging is evidence work. Do not guess past missing evidence.

## Workflow

1. Capture the exact symptom: command, input, traceback, log, response, or wrong
   output.
2. Reproduce it in the smallest available path.
3. Identify expected versus actual behavior.
4. Inspect configuration and environment differences.
5. Form one hypothesis and test it.
6. Add a regression test or clear verification path.
7. Fix the root cause.
8. Re-run the reproducer and related checks.

## When source reading is not enough

Use a debugger, probes, or logging when the source does not reveal runtime state.
Inspect values, types, call order, environment, current working directory,
configured paths, and external service responses.

Remove temporary probes before finishing.

## Common Python debugging traps

- Import shadowing from local files named like packages.
- Different Python interpreter or virtual environment than expected.
- Stale editable install or old bytecode.
- Working directory assumptions.
- Environment variables leaking between tests.
- Async tasks swallowed or not awaited.
- Mutable default arguments sharing state.
- Time, randomness, or global caches causing nondeterminism.

## Completion standard

A debugging fix is complete when you can state:

- root cause
- evidence
- fix
- validation command or manual check
- regression test, or reason one was not practical

## Sources

- Python pdb docs: https://docs.python.org/3/library/pdb.html
- pytest failure debugging: https://docs.pytest.org/en/stable/how-to/failures.html
