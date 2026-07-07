# Update checklist

Use this checklist when changing the skill.

## New reference file

- Add the file under `references/`.
- Add it to `references/README.md`.
- Add it to the `SKILL.md` reference catalogue if agents should discover it
  directly.
- Include a `## Sources` section.
- Run `python3 quality/validate_skill.py`.

## Schema change

- Update `references/review-schema.md` first.
- Update the summary in `SKILL.md`.
- Update `examples/review-findings.json`.
- Update `quality/validate_skill.py` if fields, categories, or severities
  changed.
- Run the validator.

## Example change

- Keep JSON examples parseable.
- Keep every finding schema-complete.
- Avoid private paths and project-specific names.
- Run the validator.

## Source or technical guidance change

- Prefer official docs or stable authoritative sources.
- Check `quality/source-audit.md` for the current source map and refresh links
  when needed.
- Add version gates for Python-version-specific advice.
- Mark optional ecosystem guidance as optional.
- Ask a reviewer to check for stale or over-strong claims.

## Major behavior change

- Add or update a sample task.
- Update manual review prompts if criteria changed.
- Score at least one transcript or dry run with `scorecard.md`.
