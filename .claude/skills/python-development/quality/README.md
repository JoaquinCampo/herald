# Python development skill quality harness

This directory helps maintain `python-development` as a durable skill instead of
a one-off document.

## Automated check

Run from the skill root:

```bash
python3 quality/validate_skill.py
```

Or run from this `quality/` directory:

```bash
python3 validate_skill.py
```

The validator uses only the Python standard library. It checks:

- required files and directories
- skill front matter
- reference inventory and index coverage
- source sections in reference files
- forbidden project-specific leakage terms
- forbidden punctuation configured for this skill
- JSON example parsing
- review finding schema fields, categories, and severities
- sample task records
- scorecard schema dimensions

## Runnable fixture smoke tests

Two fixtures include tiny runnable test suites:

```bash
cd quality/fixtures/cli-config-app && PYTHONPATH=src uv run --with pytest pytest -q
cd ../file-processing-app && PYTHONPATH=src uv run --with pytest pytest -q
```

Remove generated `.venv`, `uv.lock`, `.pytest_cache`, and `__pycache__` files
after ad hoc fixture smoke runs if you do not intend to keep them:

```bash
find quality/fixtures -name .venv -type d -prune -exec rm -rf {} +
find quality/fixtures -name .pytest_cache -type d -prune -exec rm -rf {} +
find quality/fixtures -name __pycache__ -type d -prune -exec rm -rf {} +
find quality/fixtures -name uv.lock -type f -delete
```

## Manual review assets

- `sample_tasks.json`: machine-readable sample tasks.
- `sample-tasks.md`: human guide to the task suite.
- `fixtures/README.md`: fixture catalogue used by sample tasks.
- `manual-review.md`: prompts for reviewing a transcript that used the skill.
- `scorecard.md`: scoring rubric.
- `scorecard.schema.json`: JSON shape for storing scorecard results.
- `decision-log.md`: durable design decisions behind the skill.
- `source-audit.md`: primary source map for future technical refreshes.
- `update-checklist.md`: checklist for future edits.

## Maintenance cadence

Run the automated check after any edit to `SKILL.md`, `references/`, `examples/`,
or `quality/`. For substantive edits, also run at least one manual sample task
review with `scorecard.md`.

## What automation does not prove

The validator does not prove that advice is complete, current, or good. It only
catches drift and obvious structural failures. Use reviewer agents and official
docs for technical accuracy checks.
