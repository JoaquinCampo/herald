# Archived mission: live grace-window controller, rung 2

Written 2026-07-06 and archived when the project adopted the deployment
contract in `../deployment_contract.md`.

## Goal

A live controller running on Orion that executes the grace-window policy
during actual generation on IFEval, with three verified properties:

1. Replay fidelity: per compressor, live mean savings and mean cost fall
   within the replay's cluster-bootstrap 95% interval for the same frozen
   alarm and threshold.
2. Frontier dominance: at epsilon 0.03, per compressor, live exposure
   savings meet or exceed the static point-rule result at a realized cost
   within budget.
3. Measured ledgers: wall-clock decode overhead at most 15% versus an
   uncompressed run, with peak allocator memory reported.

## Frozen semantics

- Alarm and threshold frozen train-side before live evaluation.
- Hold the uncompressed cache during a two-token detection window.
- On alarm, discard those tokens and continue from the held cache.
- On commit, release the held cache.
- Batch size one for hybrid generation.
- Canonical IFEval split 0 only.

## Historical outcome

The v3 live artifact completed 552 episodes. Replay exposure savings were
close to their targets, but the stricter deployment interpretation did not
pass:

- Total wall overhead was 16.2% for expected-attention and Knorm, and
  11.9% for StreamingLLM.
- Recorded-reference quality missed the epsilon 0.03 target for Knorm and
  StreamingLLM.
- Peak memory was total allocator memory, not isolated KV-cache storage.
- The evaluator printed failures but returned a successful process status.

The mission therefore remains useful as historical mechanism validation,
but it is not evidence that HERALD meets the current quality, speed, and
memory objective.
