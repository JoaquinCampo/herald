# Mission: maximize deployable KV-cache savings

Status: superseded 2026-09-02 by the quality-risk goal spec built on
`quality_risk_protocol.md`; retained as a record. This superseded the archived rung-2 live-controller
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

A target-specific structured report from `scripts/evaluate_deployment.py`
over a tracked, reproducible experiment manifest. Each report must bind one
exact candidate and contain its complete manifest prompt coverage. Compare
only passing target reports, then rank them by the lower 95% confidence bound
of isolated peak KV savings.

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

## Completion audit checklist

Do not declare a configuration deployable until every item has direct,
current evidence.

1. **Raw sweep lineage.** Preserve a completed root `config.json`, the
   matching frozen-statistics digest, exact reference and hybrid coverage,
   canonical feature artifacts, and a successful
   `validate_sweep_completeness` invocation.
2. **Frozen controller.** Preserve a new atomic alarm bundle,
   `fidelity_targets.json`, source parquet, hybrid streams, and their
   mutually verified hashes. The bundle must use prompt-disjoint train and
   test identities.
3. **Prompt-disjoint triage.** Preserve separate fresh five-prompt live
   directories for each one-shot or sustained candidate. A triage result is
   only a rejection or expansion decision, never a configuration claim.
4. **Paired deployment evidence.** For any triage survivor, preserve a
   fresh complete live manifest, one live uncompressed baseline and one
   target-specific deployed episode per frozen test prompt, at least 30
   unique paired prompts, and `evaluate_deployment.py` output. It must pass
   the quality, major-damage, end-to-end slowdown, and isolated peak-KV
   lower-bound gates.
5. **Selection.** Compare only passing target-specific reports and rank
   them by the lower 95% bound of isolated peak retained-KV savings. Neither
   compression exposure, replay, analytical estimates, nor allocator peaks
   may enter this comparison.
6. **Broader claim.** Preserve prompt-disjoint live evidence for each
   additional claimed task, model family, and long-context regime, or state
   those scopes as exclusions.
7. **Reproduction and integrity.** Verify the evidence manifest hashes,
   commands, runtime environment, rollback tag, and absence of results or
   models in Git before publishing a result.

## Stop

Finish when the highest-performing configuration found in the declared
search space passes every hard constraint, broader validation is complete
for the scope claimed, and remaining gaps are explicit.

If no configuration passes, finish only with a reproducible exhaustion
report that identifies the binding constraint and demonstrates that the
declared search space was actually tested.
