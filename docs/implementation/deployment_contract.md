# Deployment contract

Status: active, version 1.

HERALD's objective is to maximize real KV-cache memory savings under two
hard constraints: output quality remains non-inferior to an uncompressed
run, and end-to-end inference slowdown remains at or below 5%.

This contract is executable in `herald.deployment_contract` and the
default CLI is `scripts/evaluate_deployment.py`. A result is not a HERALD
success unless that evaluator accepts it.

## Paired measurement unit

Each observation is one prompt evaluated twice with the same model,
weights, tokenizer, decoding configuration, output cap, device, and
software environment.

- The baseline is plain uncompressed generation. It does not run HERALD
  feature collection, gates, alarms, retries, or rollback logic.
- The candidate is the complete deployed HERALD path. All monitoring,
  compression attempts, reverted tokens, and continuation work count.
- Quality uses the live baseline output from the same run, never a stale
  recorded reference from another environment.
- Prompt IDs are the bootstrap clusters. At least 30 unique paired prompts
  are required for a configuration-level claim. Duplicate evidence is a hard
  error, never a way to increase N.

## Hard constraints

### Quality non-inferiority

For each pair, damage is

`quality(uncompressed) - quality(HERALD)`.

The upper endpoint of the prompt-cluster bootstrap 95% interval for mean
damage must be at most 0.01 on the task's 0-1 quality scale.

Tail failures are separate. A pair with damage at least 0.5 is a major
failure; the upper endpoint of its bootstrap rate must be at most 0.01.
Mean improvements cannot cancel a concentration of severe regressions.

### End-to-end speed

For each pair, slowdown is

`HERALD wall seconds / uncompressed wall seconds - 1`.

The upper endpoint of the prompt-cluster bootstrap 95% interval must be
at most 0.05. Candidate output tokens per second is reported as a
diagnostic, but it cannot replace total user-visible latency.

### Real KV memory

Memory is counted from retained key/value tensor storages. Total CUDA
allocator peaks are not an acceptable substitute because they include
model weights, workspaces, fragmentation, and unrelated tensors.

The lower endpoint of the bootstrap 95% interval for isolated peak KV
savings must be positive. The candidate measurement is the actual live
retained cache peak, including concurrent reference and grace-fork caches,
not allocator accounting. Among configurations that pass quality and speed,
selection maximizes that lower confidence bound. KV-byte-token area is
reported when available to capture time-averaged residency.

Compression exposure (`1 - switch_position / output_length`) is a policy
diagnostic only. It is not a memory-savings result.

## Evaluator behavior

The evaluator accepts one explicit compressor, ratio, and sustain mode at
a time. Its live manifest binds the exact candidate, frozen bundle and gate
directory digests, test prompts, and runtime configuration. It rejects partial
coverage, stale or mixed run IDs, mode mixing, duplicate evidence, missing
isolated KV measurements, allocator peaks, and analytical exposure proxies.
It writes structured JSON and exits nonzero when its named target is infeasible.

The current ignored `live_controller_v3` artifact predates this contract.
It has no isolated KV measurements, and no ratio clears the 5% slowdown
confidence bound. It is historical evidence, not a passing baseline.

## Expansion and stopping

IFEval on Llama is the first mechanism-validation workload, not the final
claim. A final configuration must be checked on long-context prompts where
KV memory is material, on more than one task, and on every model family for
which transfer is claimed.

Search stops only after the practical, preregistered configuration space
has been evaluated and further memory savings cross a quality or speed
constraint. Unsupported generalization is recorded as a remaining gap,
not converted into a success claim.
