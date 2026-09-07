# Causal value-token partition result

All eight failed answers were recovered by restoring only oracle value-token
entries. Restoring all remaining sentence-token entries recovered none. Neither
count-matched non-needle control recovered any failure. The prior full-sentence
rescue again recovered all eight. Healthy001 remained correct in every branch.
The fixed recurrent value-copy mechanism gate passes on the recorded outcomes;
the surrounding-context gate does not.

Every case replayed original reference, ordinary Knorm and full-sentence rescue
IDs, termination and scores exactly. All41recorded controls passed per case.
Root aggregation is scripts/analyze_span_partition.py and
results/span-partition-summary.json. Raw files are span-partition-v1-first/rest.
Independent audit passed all nine cases, all masks/partitions,63official scores
and27historical branch replays. See results/span-partition-owner-audit.json.
An initial audit failure was an auditor ordering error: it used distance order
when assigning selected control additions to slots, while033requires sorted
token-position order. Failed report retained in its audit_history. The corrected
audit reconstructed the frozen producer rule, without changing GPU artifacts.

Value rescue exchanged34to44entries across layer/KV heads in failed cases;
healthy001 exchanged50. Context rescue exchanged27to83in failures and85in the
guard. Physical cache budgets are identical for all compressed arms, and each
control uses its own rescue's exact victims and swap counts. Value and context
rescue counts differ, so this is not an equal-swap efficiency comparison. More
value entries removed is not a supported monotonic explanation: the healthy
case lost more than each failed case.

The result identifies a consistent sufficient repair under this fixed action:
restore the original value K/V entries. It does not identify which heads or
which individual values are necessary, establish a cheap locator, or predict
loss on unseen prompts. Oracle answer annotations remain inadmissible inputs
for an eventual predictor. The surrounding partition includes all non-value
sentence tokens, so it cannot isolate a pure entity or binding effect.

This experiment is deliberately selected causal development on eight known
failures plus one healthy guard. No predictor, new population or confirmation
was evaluated. Freeze source861a95d8 and exact owner CPU reproduction are in033.
