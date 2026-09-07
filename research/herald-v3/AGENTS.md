# Working in HERALD v3

## Research posture

Read `docs/research-brief.md` first. Treat it as an open question, not a
settled method. Choose the estimand and a small realistic acceptance
experiment before choosing a predictor or building a broad pipeline.

Carry forward engineering knowledge; do not inherit a preferred solution
or a success/failure verdict from v2. Consult `docs/prior-work.md` when a
specific question warrants it. Prior evidence is scoped to its actual
intervention, features, data, and evaluation. A fresh folder does not make
previously inspected data an untouched confirmation set.

Distinguish final task-quality loss from distributional divergence,
ordinary model error, and catastrophic behavior. Define the reference,
action, timing, horizon, and quality metric explicitly. Validate that
paired continuations start from equivalent independent state.

Keep all variants of a prompt in one split. Fit transforms, calibration,
and stacked models strictly inside the appropriate training folds.
Evaluate against matched information baselines. Record exploratory versus
confirmatory use and measure runtime overhead before making deployment claims.

## Engineering

- Prefer the smallest clear implementation and mature library primitives.
- Use uv, Ruff, MyPy, and pytest. Add Typer, Loguru, Pydantic, and
  pydantic-settings where an actual CLI, logging, schema, or configuration
  boundary needs them, rather than scaffolding unused abstractions.
- Use built-in generic types, no `from __future__ import annotations`.
- Preserve an exact real reproduction before debugging. Compare competing
  explanations with the cheapest distinguishing test. Rerun the original
  failure immediately after a fix, then broader checks.
- Tests must exercise meaningful behavior; passing types alone proves no
  runtime or scientific claim. Keep verification commands fail-fast.
- Record environment, seeds, data provenance, and intervention semantics
  with each experiment. Never silently drop failed or unscorable outcomes.
- Do not commit datasets, results, models, credentials, or virtualenvs.
- Delegate only bounded independent work with explicit ownership when
  useful. Keep final research decisions with the main thread and user.
  No project-specific model-routing policy is enabled here.

## Boundaries

Local work is for implementation, CPU checks, and analysis. Do not launch
GPU experiments on this Mac. Before configuring remote experiments,
verify the intended host, project path, hardware availability, and ownership
of any processes. Do not terminate unrelated jobs or reuse a v2 remote path.

Make reversible in-scope changes autonomously. Ask before publishing,
pushing, using credentials, or making hardware/admin changes. Read v2 only
when needed for the authorized reference review; do not edit it or other
research projects. Do not copy its data or models without assessing scope
and provenance first.

## Communication

Use short, clear prose, no em dashes. Explain what evidence establishes and
what remains uncertain. Respect the user's pace during discussion.
