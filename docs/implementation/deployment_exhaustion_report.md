# StreamingLLM cache-fork deployment branch exhaustion

Date: 2026-07-10. Mission: `mission.md`. This report closes the
declared, direct-cache-fork StreamingLLM branch, not the broader
five-compressor research program.

## Scope and integrity

The practical branch was defined by the live controller's zero-reprefill
cache-fork path, the frozen StreamingLLM alarm, stride 16, paired live
uncompressed baselines, and the executable thresholds in
`deployment_contract.md`. Its ratio grid was 0.10, 0.125, 0.20, and 0.25.
The five IFEval triage prompts are evaluation-only. They did not select a
ratio, threshold, gate, or switch position.

The branch tested these independently motivated variants:

1. Sustained decode pruning every 32 tokens.
2. Sustained decode pruning every token, which bounds regrowth between
   pruning events.
3. The frozen training-only StreamingLLM gate at ratio 0.25.
4. A read-only training-only position-policy preflight. The existing
   stricter per-cell policy reached only 1.5% to 2.1% replay savings, and
   replay is not a deployment-memory or latency proxy.

The 46-pair predecessor remains a direct live reference point. Its closest
cell, StreamingLLM ratio 0.25, missed quality and positive lower-bound
isolated KV savings. The new variants may only expand to N at least 30 after
clearing every active five-prompt triage gate. None did.

## Falsified mechanisms

- **Interval 32, ratio 0.10.** Quality upper 20.0%, slowdown upper 15.1%,
  and peak-KV lower -17.4%. It fails quality, speed, and isolated KV.
- **Interval 1, ratio 0.10.** Quality upper 20.0%, slowdown upper 16.1%,
  and peak-KV lower -17.4%. It fails quality, speed, and isolated KV.
- **Interval 32, frozen gate, ratio 0.25.** Quality upper 80.0%,
  major-damage upper 80.0%, slowdown upper 11.6%, and peak-KV lower -4.3%.
  It fails quality, tail quality, speed, and isolated KV.
- **Earlier N=46 cache fork, ratio 0.25.** Quality upper 1.449%, slowdown
  upper 4.899%, and peak-KV lower -3.643%. It fails quality and isolated KV.

All triage reports use 2,000 prompt-cluster bootstrap resamples and exact
retained KV tensor bytes. They are not deployment claims. Their purpose is
to reject cells before prohibited, uninformative N at least 30 expansions.
The detailed commands and all grid results are in
`deployment_experiment_log.md`.

## Last runtime-mechanism preflight

A storage-reusing fork is not a distinct practical implementation under the
current Transformers cache interface and exact rollback requirement.
`_fork_score_cache` in `src/herald/live_controller.py` must keep the complete
reference cache until the grace decision. StreamingLLM retains a sink and a
trailing token range, but a standard cache layer exposes one contiguous key
and value tensor. `StreamingLLMPress.compress` therefore materializes a new,
contiguous candidate tensor. During grace the exact retained-KV measurement
correctly charges the reference plus candidate tensors.

Mutating the reference cache in place would overwrite its middle tokens. A
rollback shadow buffer must retain those overwritten values, with the same
order of storage as the candidate. It cannot improve the measured peak.
Sharing the two discontiguous ranges would require a segmented-cache
abstraction and Llama attention changes that consume segments without
materializing a concatenation. That is a new model-runtime design, not a
cache-fork implementation or a legal configuration change. It cannot be
represented by this branch's frozen controller and has no current
correctness proof for the cache API.

## Verdict and limits

No StreamingLLM direct-cache-fork configuration in this declared practical
branch satisfies the deployment contract. The binding constraints are prompt
quality and the lower confidence bound of isolated peak-KV savings; frequent
pruning also exceeds the end-to-end latency limit. No full expansion was
permitted because no triage cell survived the frozen protocol.

This is not a claim that every registered compressor, task, model, or
long-context workload was exhaustively evaluated. Those scopes require a
feasible candidate first. The next technically distinct unblockers are
hybrid-stream evidence that changes the frozen controller's information, or
a model-native segmented cache with an independent correctness and memory
proof. Both are outside this branch and require a new preregistered design.
