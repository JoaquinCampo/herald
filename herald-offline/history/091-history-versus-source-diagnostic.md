# One causal diagnostic: source eviction versus accumulated cache history

Selected from the missing link in090, not a predictor study. User prioritizes
understanding internals; no new population, fit, feature ranking, layer sweep or
controller. Use only exposed071 discovery000, whose first wrong token is period
instead of final0 at generated index21. Keep pinned Qwen, B16, Knorm.1 and native
mask fixed. Old outputs select the explanatory example, so no prospective claim.

Replay independent full-reference and native-action caches from the verified
B16 state. Require exact old token IDs through index21. Each pending input before
that step is shared because generated tokens through index20 agree. Preserve
logical positions. The B16 original cache has N entries and the native kept
indices S are frozen per layer/KV head, in their original physical order.

At the initial step16 and target step21, construct a third independent cache:
from the current full-reference cache gather original entries S, then append
all reference-written entries at original positions N..N+t-1. Do not recompute
norm scores or prune these newly written entries. This hybrid has the same
physical length, slot order and original retained K/V as the actual compressed
cache. Only the post-B16 generated K/V history can differ. At step21 there are
five such cached pending-token writes; token20 remains the current pending input.

Probe full, actual and hybrid with that same input/logical position on clones.
Require bitwise equality of all original retained K/V entries between hybrid and
actual, and report differences of newly written entries. At step16 the entire
hybrid must equal actual cache and raw logits exactly. Full/noop probe parity,
source immutability, independent cache storage, native mask hash, full/action
first logits matching087 and historical greedy tokens are acceptance controls.
Save raw three-arm logits, hashes, costs and source provenance. Stop at index21;
do not generate a hybrid answer or label it a final-quality rescue.

The diagnostic is margin m = logit('0') - logit('.') at that step, with token IDs
from the old fixed continuations. Report full vocabulary argmax as well. Exact
identity: m_full-m_actual = (m_full-m_hybrid)+(m_hybrid-m_actual).
The first term measures original eviction and its within-current-forward
consequences with reference-generated history. The second measures replacing
compressed-generated history with reference-generated history, conditional on
original eviction. This is an ordered intervention contrast, not an additive
mechanistic attribution valid across all paths or a head-local explanation.

Hypotheses: (1) hybrid selects0, supporting history replacement as sufficient to
repair this next-token choice; (2) hybrid still selects period, showing original
eviction/current-forward dynamics can produce the error despite reference
history; (3) hybrid selects a third token, showing interaction and no simple
binary explanation; (4) exact controls fail, making interpretation invalid.
Report margin magnitudes in every case, even if argmax remains unchanged. None
proves history irrelevant, direct missing-value causation, final-answer rescue,
a deployable observation or generality beyond this chosen case. Hybrid uses
counterfactual history and is explanatory only.
