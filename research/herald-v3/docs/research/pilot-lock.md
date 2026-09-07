# HERALD v3 pilot lock

Status: frozen roster, pending owner launch, 2026-09-05. This lock contains
selection and protocol decisions only. The freeze script reads the official
IFEval JSONL, the final exposure ledger, and the accepted engineering prompt
manifest. It does not read outputs, scores, features, or GPU state.

## Roster

The exposure ledger contains 541 official IFEval rows, 321 rows in the
recorded exposed union, and 220 rows outside that union. The four owner review
flags, keys 288, 2337, 3224, and 3750, are removed before ranking. The other
216 remaining rows have unique normalized prompt hashes and no remaining to
remaining near duplicate candidate, so they form the singleton candidate
population.

The 216 candidates are ranked by the hexadecimal SHA-256 digest of
`herald-v3-js-pilot-v1:0|normalized_prompt_utf8_sha256`, with the official key
as a deterministic tie breaker. The first 160 are the fixed roster. The first
120 are the primary target, the next 40 are roster reserve, and the remaining
56 are an unselected holdout. No ranking step uses outcomes or observed
damage. The selected prompt manifest records the exact Qwen system and user
messages, chat-template bytes, official instruction IDs, scorer kwargs, and
source hashes.

The pinned official JSONL is authoritative for prompt text and scorer
metadata. A provenance comparison found one upstream text correction at key
2785: the cached Arrow text says “at least one placeholder”, while the pinned
JSONL says “at least 3 placeholders”; both carry the same instruction IDs and
`num_placeholders: 3` kwargs. Key 2785 remains selected because it is exactly
unexposed and the correction is part of the authoritative source. The older
Arrow path and hash remain in the lock only as engineering-template
provenance, and never define pilot prompt bytes.

Each singleton prompt is assigned to one of five groups using
`sha256('fold|' + normalized_prompt_utf8_sha256)[:8]` interpreted as an
integer modulo five. Every action for a prompt stays in its prompt group.

The generated artifacts are under `data/pilot-v1/`, which is ignored because
it contains local provenance data:

* `exposure-review.json` records the source counts, exclusions, deterministic
  ranking rule, selected keys, and 56-key holdout.
* `prompts.json` is the portable 160-prompt `PromptManifest` used by the
  launcher.
* `lock.json` is the complete protocol and source hash lock.

Recreate them with:

```sh
uv run python scripts/freeze_pilot.py
```

The script fails if source counts drift, an outcome-contaminated ledger is
provided, official rows are missing, a selected prompt is exposed or excluded,
the singleton condition fails, the roster is not exactly 160 prompts, or the
manifest does not cover all five groups. It also round trips the generated
manifest and checks its fingerprint.

## Frozen generation and action boundary

The owner launches Qwen/Qwen2.5-7B-Instruct from snapshot
`a09a35458c702b33eeacc393d103063234e8bc28`, with BF16 weights, SDPA
attention, greedy decoding, seed 0, EOS IDs 151643 and 151645, and a total
new-token budget of 1,024. The accepted Qwen `im_start` and `im_end` template
is preserved with its system message and each official user prompt.

At the decision boundary, 32 output tokens are committed. The cache contains
the prompt and output indices through 30, while output index 31 is the pending
token. The one-time action is applied to an independent live-cache clone just
before forwarding that pending token, so the first affected distribution
predicts output index 32. The reference arm receives no action. Candidate arms
apply the verified direct live-cache Knorm transform once, removing 0.25 or
0.50 of eligible entries, then continue normally until EOS or the budget.

## Frozen target and analysis

For each action, retain the signed loose IFEval fraction difference
`d = q0 - qa`, including positive, zero, and negative values. Repeat the
analysis under strict scoring as a prespecified sensitivity analysis. Scores
are prompt-equal and action paired. The no-op has a known zero label and is
excluded from prediction metrics. No future token, generated answer, final
score, or scorer annotation enters a predictor.

The fixed feature sets are cumulative. B0 contains action fraction, prompt
token count, decision index, and pre-action cache size. B1 adds uncompressed
next-distribution entropy and top-two margin. B2 adds compressed entropy and
margin deltas plus argmax match. B3 adds one full-vocabulary Jensen-Shannon
divergence. Each five-fold model fits `StandardScaler` on the training fold
only, then `Ridge(alpha=1)` with no tuning or calibration, and clips
predictions to [-1, 1]. The action-wise training mean is a fixed sanity
baseline. Primary error is prompt-equal out-of-fold MSE; MAE, signed bias,
action-specific errors, and positive, zero, and negative subsets are
diagnostics.

Use 2,000 prompt-cluster bootstrap replicates, preserving repeated cluster
multiplicity. The pilot is information-bearing only if at least 20 distinct
prompts have nonzero `d` for either action. A later independent replication
would require B3 to improve MSE by at least 5 percent against every comparator
and its paired intervals to exclude zero improvement. This is a research
triage rule, not a deployment guarantee.

No pilot result, model fit, freshness guarantee beyond the recorded exposure
sources, or deployment claim follows from this lock. The owner must run the
GPU collection and then verify per-prompt state parity, scorer parity, action
evidence, timing, memory, complete outcomes, and artifact hashes.
