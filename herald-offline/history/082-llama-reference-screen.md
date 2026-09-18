# Llama reference-only competence screen

Selected before any Llama generation. Use the same16 exposed development prompts,
with key1174 relation corrected to the literal request as documented081. Preserve
original frozen080 scores; corrected Qwen diagnostic reference score is11/16.
New manifest: `data/ifeval-llama-reference-v1.json`, recording exact lineage.
No fresh/unseen evaluation claim is available for these cases.

Use already-cached Llama3.1-8B-Instruct snapshot
0e9e39f249a16976918f6564b8830bc894c89659, bfloat16 SDPA, greedy seed0, its
native chat template, cap2048. Reuse unchanged run_pair_pilot.py with
shared_boundary and actions0 only. No actual compression in this screen.
Require all16 cases complete, all independent clone/source/no-op checks pass,
and reference strict compliance>=12/16, with>=15/16 terminating before cap.
Full-prefill parity is recorded separately, as in079. Score with pinned official
strict scorer, random/langdetect0 per arm. Empty response scores0.

Hypotheses: (1) model-specific competence limits Qwen, so Llama clears the gate;
(2) task composition remains difficult, so neither clears it; (3) scoring defects
still dominate, requiring source/response audit rather than changing labels to
pass; (4) runtime/no-op incompatibility makes this model unusable in the assay.
This comparison isolates reference model while holding prompt text and cap fixed;
use corrected scoring for both models when comparing, with080 unchanged.

Passing permits consideration of a separate compression feasibility test, not
predictor fitting. Failing triggers a return to task/measurement assumptions.
Keep all cases regardless of outcome. No dependency or credential changes.
