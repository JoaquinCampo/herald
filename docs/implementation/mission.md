# Mission: live grace-window controller, rung 2

Written 2026-07-06, superseding the rung-1 cross-compressor mission
(exhausted; per-compressor reframe and full post-mortem in
`online_forecasting.md` and the experiment log entries
`fleet_robustness`, `epsilon_frontier`, `aimd_feasibility`). One
active mission at a time; when this one completes, raise the rung
and rewrite this file.

Context the mission stands on: the AIMD feasibility replay showed
that a grace-window recovery policy (attempt compression, watch
k=2 post-switch tokens, roll back on alarm) dominates or matches
the static frontier on every compressor at 4-13% token overhead,
with a perfect-alarm ceiling of 0.73-0.96 savings. All of that is
simulation from recorded single-switch runs. This mission makes it
real.

---

# Goal
A live controller running on Orion that executes the grace-window
policy during actual generation on ifeval, with three verified
properties:

1. REPLAY FIDELITY: on held-out test prompts (canonical split 0),
   per compressor, live mean savings and mean cost fall within the
   replay's cluster-bootstrap 95% CI for the same frozen alarm and
   theta. This validates the zero-GPU replay methodology itself.
2. FRONTIER DOMINANCE: at epsilon 0.03, per compressor, live
   savings meet or exceed the static point-rule result on the same
   split at a realized cost within budget.
3. MEASURED LEDGERS: wall-clock decode overhead <= 15% vs an
   uncompressed run, and peak KV memory reported, converting the
   token-overhead ledger into GPU-seconds on real hardware.

## Proof
A results summary produced by the live harness on the canonical
test prompts, evaluated through the unchanged locked evaluator
semantics (`herald.controller_metrics.evaluate_frozen_tau` applied
to the live outcomes), showing 1-3 above per compressor, plus the
per-compressor gap to the perfect-alarm oracle. Alarm and theta
frozen train-side before any live test run, exactly as in the
replay (crossfit OOF, point method, eps 0.03, k=2).

## Limits
- Do not change: `controller_metrics.md`,
  `controller_metric_lock.json`, `switch_baseline_lock.json`, the
  metric semantics in `src/herald/controller_metrics.py`, splits,
  or the tau/theta calibration rules.
- Grace-window semantics are the replay spec and are not
  negotiable within the mission: hold the uncompressed cache, k=2
  detection window, on alarm discard the k tokens and continue
  from the held cache (greedy), commit frees the held cache.
  Deviations (larger k, no rollback) are new experiments, not this
  mission.
- Hybrid generation remains batch=1 (left-padding corrupts
  compression for 4/5 presses).
- ifeval-only. gsm8k/humaneval sweeps, trajectory campaigns beyond
  what fidelity validation needs, and progressive-deepening runs
  are Phase B scope: ask first.
- Orion single-run jobs for this mission are in scope without
  asking; every nohup launch gets the 10-second smoke check;
  keepalive stays alive.

## Stop
- Done when the Proof passes.
- If live results violate replay fidelity, that is a FINDING, not
  a failure: diagnose (nondeterminism, cache-restore mechanics,
  stream mismatch), document, and present to the user before
  patching the policy. The replay methodology's validity is itself
  a paper claim; do not tune the live system toward the replay.
- Unreachable is claimable only via an exhaustion report presented
  to the user; only the user clears the goal on that basis.
- After 5 consecutive live-harness iterations without passing a
  numbered property, stop and write up the blocker.

> Method per PRINCIPLES.md, boundaries per CLAUDE.md. Choose your
> own path, experiments, and tools within the limits. If
> well-designed attempts show the goal is unreachable as stated,
> reporting that evidence with a written analysis IS completing
> the goal; do not grind and do not game the proof. Stop and ask
> for anything irreversible or outward-facing beyond the limits.
