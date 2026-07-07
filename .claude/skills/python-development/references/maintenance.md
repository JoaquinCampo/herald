# Skill maintenance and consistency

Use this when editing this skill. The skill is only useful if the playbook,
references, examples, and schemas stay consistent.

## Consistency checks

Before declaring edits complete, check:

- Every file under `references/` is listed in `references/README.md`.
- Every file listed in `references/README.md` exists.
- `SKILL.md` links every major topic that an agent is expected to open.
- The review schema in `SKILL.md`, `references/review-schema.md`, and
  `examples/review-findings.json` uses the same fields.
- Category names in examples are members of the documented category vocabulary.
- Severity meanings are consistent everywhere.
- Optional ecosystem guidance stays conditional and does not become a universal
  rule.
- Tool commands are presented as discovered examples, not requirements, unless
  the project config says so.
- Version-specific advice names the Python version it depends on.
- Source links still point to official or authoritative material.
- No project-specific names, paths, environment variables, or internal commands
  leaked into the general skill.

## Drift hazards

- Python annotation semantics changed in Python 3.14. Do not copy older
  guidance about eager evaluation or future annotations without a version gate.
- Tooling changes quickly. Ruff, uv, pytest, mypy, pyright, and Pydantic advice
  should be checked against their docs when making strong claims.
- Review schemas tend to drift when duplicated. Keep `references/review-schema.md`
  canonical and make `SKILL.md` a summary.
- Examples train behavior. Keep examples complete and schema-valid.

## Suggested validation command

From the skill root, run:

```bash
python3 quality/validate_skill.py
```

From `quality/`, run:

```bash
python3 validate_skill.py
```

The harness checks reference inventory, schema consistency, JSON examples,
forbidden leakage terms, forbidden punctuation, source sections, sample tasks,
and scorecard dimensions. Manual review is still required for advice quality,
technical freshness, and whether optional ecosystem guidance is properly framed.

## Sources

- Python 3.14 annotation changes: https://docs.python.org/3.14/whatsnew/3.14.html#pep-649-pep-749-deferred-evaluation-of-annotations
- Python Packaging User Guide: https://packaging.python.org/
- Ruff docs: https://docs.astral.sh/ruff/
