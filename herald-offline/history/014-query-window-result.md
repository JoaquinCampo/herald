# Fixed query-window diagnostic result

Using the final 32 prefill rows for EA query statistics increased score mass on
known needle spans, but did not establish a quality-loss predictor. The primary
public feature, all-head mean removed-mass excess, shifted similarly in all four
cases. The already-correct case still had much more oracle removed-span mass
than any failed case. These are exposed-case diagnostic observations only.

| Case | Full needle mass | Tail 32 needle mass | Full excess | Tail 32 excess |
|---|---:|---:|---:|---:|
| 000, failed | .003386 | .035464 | -.034701 | -.052616 |
| 004, failed | .006235 | .046250 | -.030885 | -.050364 |
| 011, failed | .005972 | .041147 | -.030932 | -.050274 |
| 001, correct | .064004 | .113781 | -.033972 | -.051108 |

The tail/full needle-mass ratios were about 10.47, 7.42, 6.89 and 1.78 respectively.
Tail 32 oracle removed-span mass was .000255, .000336, .000284 and .005153.
Oracle spans were used only to evaluate where the score concentrates, never as
inputs to query statistics or to fit a model. No statistical separation claim,
head/window search or predictor fitting was performed.

## Measurement evidence

All four cases passed 25 real GPU controls, including exact plain/full/tail cache
and source equality, storage independence, preserved original RoPE q_len,
full-stat identity, 112 head rows per window, finite values, identical native
Knorm masks, unchanged sources and exact prior reference/standard continuations.

The adapter inherits ExpectedAttentionPress.score unchanged. Only
get_query_statistics selects the last 32 actual hidden-state rows, without a
second sink drop, then applies average RoPE at the original full prompt length.
Its setting remains covariance-free, value-weighted, horizon 128, removal .10.
A direct short-input call would change those semantics and was not used.

Frozen source: results/query-window-v1-launch/frozen-measurement.py,
SHA256 937df6121ef5aaf530f51f1d5324e8eecc96f6070c3f79035499ae227a1661cf.
Tiny CPU original reproduction passed after final finite/head-count and failure
persistence corrections; results/query-window-v1-cpu. Real first/rest exit 0,
raw artifacts in results/query-window-v1-first/ and query-window-v1-rest/.
Compact arithmetic: results/query-window-v1-owner-summary.json.
Independent audit PASSED: results/query-window-v1-audit.json.

## Decision

Stop expanding EA window or head aggregation variants. The causal rescue result
shows that specific information matters; this diagnostic shows that improved
score concentration alone is insufficient. Return to the relationship between
decision-time observations and final task scoring before another fit or larger
collection. No confirmation population was opened, and no predictor is validated.
