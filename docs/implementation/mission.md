# Mission: maximize deployable KV-cache savings

Status: active. This supersedes the archived rung-2 live-controller
mission in `archive/rung2_live_grace_window.md`.

## Goal

Find and validate the HERALD configuration with the greatest real
KV-cache memory savings that satisfies the executable deployment contract:

- statistically non-inferior paired output quality;
- no unacceptable concentration of major quality failures;
- at most 5% end-to-end inference slowdown; and
- positive isolated KV-cache savings.

The exact thresholds, baselines, statistics, and forbidden proxies are in
`deployment_contract.md` and `herald.deployment_contract`.

## Proof

A structured report from `scripts/evaluate_deployment.py` over a tracked,
reproducible experiment manifest. The report must identify at least one
feasible configuration and rank feasible configurations by the lower 95%
confidence bound of isolated peak KV savings.

The implementation, tests, report, evidence manifest, and documentation
must agree. Missing model/task coverage is named as a limit, not implied by
the headline.

## Work program

1. Instrument the real runtime path for paired quality, total wall time,
   and isolated retained KV tensors.
2. Reproduce the strongest existing controller under that contract.
3. Remove avoidable controller overhead without changing token semantics.
4. Search compressor, ratio, switch, gate, alarm, and sampling choices in a
   prompt-disjoint protocol.
5. Validate the strongest feasible candidate on long-context workloads and
   additional model/task scopes relevant to the final claim.
6. Publish an evidence manifest with hashes and reproduction commands.

## Integrity rules

- Never substitute compression exposure or allocator peak for KV bytes.
- Never use a recorded reference when a paired live baseline is required.
- Never tune on the held-out evaluation prompts.
- Never weaken a threshold because a candidate misses it.
- A failed hypothesis is retained as evidence and changes the next test.
- Historical missions and results remain immutable records.

## Stop

Finish when the highest-performing configuration found in the declared
search space passes every hard constraint, broader validation is complete
for the scope claimed, and remaining gaps are explicit.

If no configuration passes, finish only with a reproducible exhaustion
report that identifies the binding constraint and demonstrates that the
declared search space was actually tested.
