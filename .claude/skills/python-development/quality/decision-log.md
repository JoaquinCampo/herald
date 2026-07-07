# Decision log

This log records durable design decisions for the `python-development` skill.

## New skill instead of mutating the baseline

Decision: create `python-development` as a new general skill and leave the
review-oriented baseline as source material.

Reason: the baseline contained project-specific framing and review-only shape.
A new skill avoids compatibility breakage and keeps the general artifact clean.

## Discovery before defaults

Decision: make project discovery the first operating rule.

Reason: Python projects differ in supported versions, package managers, test
runners, linters, type checkers, source layouts, and CI commands. A general
skill should adapt rather than impose one stack.

## Defaults are conditional

Decision: describe uv, Ruff, pytest, mypy or pyright, Pydantic, Typer, Loguru,
FastAPI, and scientific Python as defaults or optional ecosystem guidance, not
universal requirements.

Reason: these are strong modern tools, but forcing them into existing projects
would create project-specific behavior and unnecessary churn.

## References are modular

Decision: keep `SKILL.md` operational and move details into topic references.

Reason: agents need a fast playbook first. Deep citations and examples should be
loaded only when relevant to avoid context bloat.

## Review schema has a canonical home

Decision: `references/review-schema.md` is canonical for review finding fields,
categories, severity, and clean-review output.

Reason: duplicated schemas drift. `SKILL.md` keeps a summary, and the quality
harness checks alignment.

## Quality harness is stdlib-only

Decision: `quality/validate_skill.py` uses only the Python standard library.

Reason: the skill lives as a user-level artifact, not a package with managed
dependencies. Validation should be runnable on a normal Python install.

## Runnable fixtures are tiny and optional

Decision: include small runnable fixtures for high-risk sample tasks, while
keeping most fixtures descriptive.

Reason: runnable fixtures improve reproducibility, but a full fake ecosystem
would add maintenance cost and distract from the skill itself.

## Technical freshness needs source audits

Decision: add `quality/source-audit.md` and require source checks for strong
technical edits.

Reason: Python and tooling move quickly. The validator can catch structure and
drift, but it cannot prove that external advice remains current.
