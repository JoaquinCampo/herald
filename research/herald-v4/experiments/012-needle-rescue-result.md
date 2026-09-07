# Needle rescue diagnostic result

At the same cache budget, oracle needle-span rescue recovered the answer on all
three failed retrieval cases. A control with the same victims and swap count
recovered none. The fourth case was already correct and remained correct in both
branches. This supports a causal role for preserving the injected information
in these three failures. It does not validate a predictor or a practical policy.

| Exposed case | Reference | Standard .10 | Oracle rescue | Matched control |
|---|---:|---:|---:|---:|
| NIAH000 | 1 | 0 | 1 | 0 |
| NIAH004 | 1 | 0 | 1 | 0 |
| NIAH011 | 1 | 0 | 1 | 0 |
| NIAH001 | 1 | 1 | 1 | 1 |

All four passed 28 state, replay, mask and prior-result checks. Reference and
standard tokens exactly reproduced the previous development experiment.
Repeated standard continuations and native-mask manual replay were exact,
including final cache fingerprints. Both custom branches preserved source
caches, the baseline sink-token retention and unaffected cache-slot ordering.

The full injected sentence spans 21 or 22 tokens. For each layer/head, rescue
adds its missing sentence tokens and control adds the same number of nearby
excluded non-needle tokens. Both evict exactly the same lowest-priority retained
non-needle, non-sink tokens and use the same vacated slots. Cache lengths match.
The branches use oracle information only to diagnose the failure mechanism.

Across the 112 layer/KV-head pairs, rescue changed 26, 39, 44 and 32 pairs for
cases 000,004,011,001 respectively, adding an average 0.545,0.866,1.000 and 1.205
positions per pair. This does not identify which pairs caused the recovery.
The already-correct case had the most missing span positions in aggregate,
so raw missing-span count alone is not sufficient to order these four losses.

## Evidence and scope

Frozen source: results/needle-rescue-v1-launch/frozen-diagnostic.py,
SHA256 ace1754ac20955b5e292c76cb557c5abdee388bfb64810a32002131af77ede2a.
Real first/rest runs exited 0; results are in results/needle-rescue-v1-first/
and results/needle-rescue-v1-rest/. Summary: needle-rescue-v1-owner-summary.json.
Independent audit PASSED: results/needle-rescue-v1-audit.json, all 28 scores,
2,240 branch mask checks and 448 victim-set comparisons independently verified.

The cases were selected by generator metadata after development exposure.
They are not a fresh evaluation population. The result does not distinguish
answer-token damage from surrounding entity/association damage, nor identify
causal heads. Exact replay weakens implementation instability on these cases,
but it does not address statistical uncertainty in the earlier fitted predictor.

Stop the rescue experiment at four cases as planned. Next examine whether
query-conditioned decision-time evidence can recognize important retained or
removed content without access to the answer span. No confirmation data opened.
