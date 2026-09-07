# Direct-query salience, small mechanism slice

All12 oracle rescues support content-specific eviction damage (024), while
mean EA, fixed-prefix and immediate-JS models did not predict signed loss well.
Competing explanations are sparse head-specific content needs, delayed query
needs unavailable at this boundary, value contributions poorly represented by
attention alone, and distributed sentence information requiring multiple tokens.
The next test distinguishes whether a sparse direct-query proxy identifies
causally useful content now. It is not a controller or predictor experiment.

Freeze four already exposed cases000/004/011/001 at .10, same paired-state
protocol, model, seed, max generation and scoring. Query comes from processing
the pending last prompt token in one full-cache independent sandbox forward.
For each layer/KV head, softmax actual RoPE query-key scores over boundary cache
positions only, multiplying each position's probability by its value L2 norm.
Take the maximum over grouped Q heads, then normalize within each KV head.
Exclude sinks from selection. Nominate at most one highest-salience evicted
position per KV head only if it exceeds the highest-salience retained position.
The fixed observation z is the sum of these positive normalized salience gaps.
No answer spans, benchmark answers, generated tokens, or future reference
continuations may influence the feature or nominated positions.

Compare standard Knorm, nominated-position rescue, and lowest-salience removed
non-candidate control. Each changed head restores one position; proxy and
control use identical victims, slots, lengths and byte budgets. Victim selection
uses a fixed baseline priority rule among retained non-sink positions. If no
position is nominated, leave that head unchanged. No threshold/head/window/
horizon search, and no fitting. Details must match actual helper semantics
before launch, including deterministic ties and no accidental sink replacement.

Acceptance: exact reference/noop and prior standard replay, source independence
and immutability, identical masks from instrumented/uninstrumented states,
matched swap counts/victims, finite normalized salience and valid physical
lengths. Proxy must recover at least2/3 failures, control none, and001staycorrect.
A failure closes this fixed proxy. A pass supports only this sparse mechanism;
a separate fixed signed-loss predictor evaluation would still be required.

Implementation authorized after helper feasibility review. Capture actual Q
from an independent FULL-CACHE pending-token probe, never from compressed
standard replay. Hook q_proj and capture exact supplied RoPE cos/sin; use the
installed Qwen2 rotary helper. Verify instrumented and uninstrumented logits
exactly. Feature math float32, no attention-backend change. Ties choose lowest
position; exclude first4sink positions from selection/comparison/victims.
Victim is lowest native priority retained nonsink, with nativepriority=-Knorm;
control is lowest salience evicted nonsink excludingcandidate. Freeze candidates
before all continuation/scoring calls. Save salience tensors and masks compactly
for independent arithmetic. Record full-reference probe plus feature cost
separately from verification costs.

Luna query_salience_runner owns only new diagnose_query_salience.py and its
CPU proof artifacts. No GPU until root verifies tiny real model proof and source.
Reuse existing clone/mask/continuation/scoring helpers, no broad harness copy.
