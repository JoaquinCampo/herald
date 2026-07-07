# Review schema

Use structured findings for review output so results are actionable and easy to
route into follow-up work.

## Finding schema

```json
{
  "file": "src/package/module.py",
  "line": 42,
  "symbol": "optional function or class name",
  "category": "correctness|api|design|typing|errors|testing|io|performance|concurrency|security|packaging|maintainability|style|tooling",
  "severity": "high|medium|low",
  "title": "Short problem statement",
  "why": "Concrete risk, broken contract, or violated rule",
  "fix": "Smallest useful change",
  "validation": "Test, check, or manual verification that should cover the fix"
}
```

## Severity

- `high`: correctness bug, data loss, security issue, silent failure, API misuse,
  broken test signal, or likely production failure.
- `medium`: unclear API, fragile design, weak test, wrong abstraction, or
  maintainability issue likely to affect near-term work.
- `low`: local readability or consistency issue that materially improves
  comprehension.

## Review discipline

- Triage by risk before reading line by line.
- Form an expectation from the stated goal, issue, or caller need, then compare
  that expectation to the diff.
- Check the premise before implementation details: should this exist, is it the
  right scope, and does the codebase or standard ecosystem already solve it?
- One issue per finding.
- Include a concrete fix.
- Explain why the issue matters.
- Do not repeat configured linter findings unless the tool is not being run.
- Do not bikeshed.
- Do not invent problems to look thorough.
- If a fix changes behavior, say so and include validation.
- End substantial reviews with a verification gap statement.

## Clean review result

When there are no findings, say:

- what files or diff were reviewed
- what evidence was checked, such as tests, config, callers, or command output
- what was not verified
- any residual uncertainty

## Sources

- Google Python style guide: https://google.github.io/styleguide/pyguide.html
- Python Packaging User Guide: https://packaging.python.org/
- pytest good practices: https://docs.pytest.org/en/stable/explanation/goodpractices.html
