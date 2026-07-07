# Scorecard

Score each dimension from 0 to 2.

- 0: missing or harmful
- 1: partially present, vague, or inconsistently applied
- 2: clear, evidence-backed, and appropriate for the task

Pass thresholds:

- Pass: at least 15 of 18, no zero in safety, validation_evidence, or
  project_adaptation.
- Marginal: 11 to 14, or one zero outside the protected dimensions.
- Fail: 10 or less, any protected-dimension zero, unsafe edits, or fabricated
  validation.

## Dimensions

### discovery

0: Invents commands or patterns without checking the repo.
1: Checks one source but misses obvious config or docs.
2: Uses local instructions, config, docs, tests, and nearby code as needed.

### contract

0: Starts coding without defining behavior.
1: Contract is implicit or incomplete.
2: Inputs, outputs, side effects, errors, and tests are clear.

### test_strategy

0: No tests or tests only implementation trivia.
1: Some relevant tests, but weak signal or missing failure mode.
2: Tests observable behavior and cover the change or risk.

### implementation_simplicity

0: Adds needless abstraction or broad unrelated changes.
1: Mostly simple, with some avoidable complexity.
2: Small, direct, maintainable change.

### validation_evidence

0: Claims checks without evidence or ignores failures.
1: Runs narrow checks but gaps are unclear.
2: Reports commands, results, scope, and gaps honestly.

### review_discipline

0: Bikesheds, summarizes, or bundles vague findings.
1: Findings are useful but incomplete.
2: Triage, premise check, actionable findings, and verification gaps are clear.

### safety

0: Unsafe file edits, secret exposure, silent failures, or risky commands.
1: Basic safety present with unresolved concerns.
2: Boundaries, secrets, paths, subprocesses, and failure modes are handled well.

### project_adaptation

0: Forces generic preferences over project configuration.
1: Adapts partially but misses local conventions.
2: Follows project config and explains any deviations.

### reporting_clarity

0: Hard to tell what changed or what remains risky.
1: Summary is present but missing evidence or limitations.
2: Clear changed files, validation, decisions, and residual risks.
