# Fixed causal partition of the successful oracle rescue

The exact needed-sentence mean eviction statistic is weak even with oracle
locations (032). Test what information the successful full-sentence rescue
restores, without proposing another scalar predictor. Competing explanations:
answer-value copying is the bottleneck; surrounding binding/context is the
bottleneck; both parts are jointly necessary; mechanisms differ across prompts.
No layer/head selection, feature fitting or action/timing sweep.

Use the eight known failed B0 .10 NIAH cases000,002,003,004,005,008,010,011,
plus healthy001 as a guard. These are deliberately exposed diagnostic cases.
Same original shared B0, Qwen checkpoint, greedy continuation cap128 and score.
Retain all failures. Exact original reference, standard action and earlier full
oracle-rescue token IDs, termination and score must replay before interpretation.

Partition the old oracle sentence's cached token positions into two disjoint
sets: tokens whose character offsets overlap the unique answer value occurrence
inside the sentence, and all remaining sentence tokens. The latter is surrounding
context, not a pure entity or relation intervention. Their union must equal the
old full sentence. Answer annotations are permitted only because this is explicitly
an oracle causal diagnostic; neither partition is an admissible predictor input.

Restore all evicted positions of each partition separately. Protect the entire
original sentence from being chosen as donor victims. For each head, use the first
k victims in the old full-sentence rescue's deterministic victim ordering, where k
is that partition's missing-token count. That ordering already excludes the whole
sentence and the first four sink tokens. Replace the corresponding sorted physical
slots with the sorted missing partition positions, leaving all other slots fixed.

Each partition has its own count-matched control: same victim positions and slots,
but restore k originally evicted positions outside the entire original sentence
and outside sinks, sorted by distance to the partition then token position. Every
arm has the same physical cache budget. Swap counts can differ between partitions;
report them and do not claim equal-swap efficiency or a pure value-versus-context
comparison. Complete partition restoration avoids an infeasible forced matching
that would leave arbitrary target tokens unrestored.

Run reference, standard native action, prior full-sentence oracle replay, value
rescue and its control, surrounding-context rescue and its control. Require source
cache immutability, independent equal branch clones, physical length/byte parity,
exact unchanged slots, unique retained indices, full target retention and exact
per-head control swap counts. The healthy case must remain correct for any branch
used to support a beneficial mechanism. Tiny real CPU reproduction precedes first
real-model case000; exact prior replays gate remaining eight cases.

Interpret once after all integrity checks. At least six of eight failed cases
rescued by value alone with zero value-control rescues supports a recurrent
value-copy bottleneck. The analogous result for surrounding context supports
context/binding involvement. Both can pass, indicating alternate sufficient
restorations. Neither passing while full rescue still recovers all eight supports
joint or heterogeneous mechanisms; distinguish per-case both-fail from mixed
outcomes rather than conflating them. Any matched control rescue prevents
attributing that case's effect specifically to its intended partition.

This is causal development evidence only, no predictive accuracy, confirmation,
compression controller or generality claim. Do not alter the partition after
results. Reuse the existing engine, scoring and branch primitives with a thin
script; no broad new runner or production tooling.

Execution freeze before GPU: source SHA256 861a95d8989959596afd924863e4370100a9ad96b8d36fa9da38c1ba1382cf87.
Owner reran original tiny CPU fixture against results/needle-rescue-v1-cpu;
all controls and exact three historical branch replays passed. Evidence:
results/span-partition-cpu-owner. Earlier invalid check failures remain saved.
The source hash was printed before launch; its comparison against CPU metadata
used the corrected field diagnostic_source_sha256 after a field-name lookup error.
No scientific CPU check failed in the owner rerun.
