# Mission: deployable controller, rung 1

Written 2026-07-05. One active mission at a time; when this one
completes, raise the rung and rewrite this file. Arm it by passing
the text below to `/goal`.

---

# Goal
A switch predictor whose controller policy is deployable on an unseen
compressor: budget respected on all 3 held-out compressors with
worst-case savings-at-budget of at least 0.10, beating every
budget-respecting locked reference. The oracle gap (0.67 to 0.95
savings) is the direction; the rung is the finish line.

## Proof
`uv run python scripts/check_mission.py --candidate <summary.json>`
prints PASS, where the candidate summary is produced through the
unchanged locked evaluator (`herald.controller_metrics`) on the
canonical splits. Source of truth:
`docs/implementation/controller_metrics.md` and
`controller_metric_lock.json`. Every report also states the gap to
oracle per compressor.

## Limits
- Do not change: `controller_metrics.md`,
  `controller_metric_lock.json`, `switch_baseline_lock.json`,
  `scripts/check_mission.py`, the metric semantics in
  `src/herald/controller_metrics.py`, splits, epsilon, or the tau
  rule.
- Forbidden model inputs unchanged: compressor, prompt_id, q_ref,
  q_hybrid, damaged, major_damage, raw s, relative_s, ref_len.
- `random` and `snapkv` rows are training-only donors; the 3 primary
  compressors remain the only held-out test sets.
- Local Mac experiments and dev-group deps are free; any Orion GPU
  job or new data generation: ask first.

## Stop
- Done when the Proof passes.
- Unreachable is claimable only via an exhaustion report (at least 3
  hypothesis families with canonical results and falsifying evidence,
  a ceiling argument, and the resource that would unblock), presented
  to the user; only the user clears the goal on that basis.
- After 5 consecutive canonical experiments without improving the
  verified best worst-case, either commit in writing to a new
  hypothesis family or begin the exhaustion report.

> Method per PRINCIPLES.md, boundaries per CLAUDE.md. Choose your own
> path, experiments, and tools within the limits. If well-designed
> attempts show the goal is unreachable as stated, reporting that
> evidence with a written analysis IS completing the goal; do not
> grind and do not game the proof. Stop and ask for anything
> irreversible or outward-facing beyond the limits.
