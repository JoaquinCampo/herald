# Validation matrix

Choose validation from the work mode and project configuration. Run focused
checks first, then broader checks when the change is substantial or risky.

| Work mode | Focused validation | Broader validation | Completion signal |
| --- | --- | --- | --- |
| Review only | Read target files, tests, config, and callers needed for judgment | Run read-only checks if allowed and useful | Findings are actionable, or clean result states evidence and gaps |
| Feature work | New or updated behavior tests | Full test suite, lint, type check, smoke test | New behavior is covered and checks pass or limits are explicit |
| Bug fix | Reproducer or regression test | Related test module, full suite for risky fixes | Regression fails before the fix when practical and passes after |
| Refactor | Existing focused tests or characterization tests | Static checks and wider suite around moved code | Public behavior is unchanged and tests still protect it |
| Test-only change | Changed tests and intentional failure check when practical | Related test package | Tests assert observable behavior and fail for the intended bug |
| Debugging | Minimal reproducer and captured runtime evidence | Regression test or smoke path | Root cause, evidence, fix path, and verification are stated |
| Packaging or config | Command affected by config, import package, entry point smoke | Build wheel or CI-equivalent command | Config behavior works in the configured toolchain |
| Performance | Baseline measurement and focused benchmark | Profile representative workload | Improvement is measured and correctness did not regress |
| Concurrency or async | Tests with timeouts and cleanup checks | Stress or race-focused tests when available | No leaked tasks, deadlocks, unhandled exceptions, or flaky outcomes observed |
| Security-sensitive change | Input validation tests, unsafe path checks, secret/log review | Dedicated security review or threat model | Obvious abuse cases are covered and residual risks are named |

## Reporting validation

Always report:

- command or manual check performed
- pass, fail, or not run
- scope covered
- known gaps

Do not claim full verification when only focused checks ran.

## Sources

- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
- Python Packaging User Guide: https://packaging.python.org/
- Ruff docs: https://docs.astral.sh/ruff/
