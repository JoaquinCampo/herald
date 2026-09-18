# Fresh B16 replication

Selected September 15 before generating new prompts or outcomes. The earlier
12-prompt study030 failed its gates. Its AUC .8125 and MSE gain 8.21% are only
exploratory motivation, strengthened by independently reproduced susceptibility
in068. Study070 shows that one old structural model was weak, not that all
structural information fails. This is one explicitly reopened prospective
replication, not revision of the historical verdict or reuse of closed labels.

Five competing explanations: the fixed B16 scalar transfers; structural variables
explain it; the earlier estimate was sampling chance; this action lacks sufficient
outcome variation; any signal is specific to this synthetic template. A fresh
single-task comparison isolates the first three without changing the feature.

Use pinned official RULER niah_single_2, original template and essay corpus,
4096 total-token setting, Qwen2.5-7B revision
a09a35458c702b33eeacc393d103063234e8bc28, greedy seed0, total128 output ceiling.
Generate32 discovery and48 locked evaluation prompts, generator seeds2026091571
and2026091572. Record full prompt/context hashes and reject overlap with every
existing v4 manifest and across splits before any new outcome. One context is
one group; any variants stay together. No collision replacement or outcome filtering.
The source corpus is shared historical public material, not claimed unseen text
to the language model. Novel generated contexts are the generalization unit.

B16 follows exactly16 shared uncompressed greedy tokens from the original split
B0. Tokens1..15 are cached, token16 pending. Native Knorm removes .10 of the
whole cache. The target is signed full reference score minus compressed score
under the pinned official scorer, with a total128-token ceiling. Preserve
improvements and imperfect references. EOS at or before16 is recorded and blocks
this fixed study rather than silently excluding or replacing a prompt.

Observe the unchanged029 scalar: actual pending-token Q with native RoPE,
attention times value norm, maximum over grouped query heads, normalization
over cached positions, then average native removed mass over all112 KV heads.
No cue, answer field, future output, head selection or revised reduction.
Persist features and their hashes before paired outcomes; evaluation predictions
must also be persisted before each evaluation continuation or score.

Structural columns are cache length, mean normalized removed position,
fraction of removed positions in last128, fraction in first4 sink positions,
removed K-norm mass and removed V-norm mass. Use the original measure_ea
definitions, averaging equally over all layer/KV heads. Task, action and clock
are constants in this study. The candidate uses all six structural columns
followed by z, the fixed B16 scalar, for exactly seven inputs. It is not a
scalar-only regression.
Fit once on discovery: StandardScaler then Ridge(alpha=1), intercept enabled.
Baselines are discovery signed-loss mean, the same structural Ridge, and
structural StandardScaler plus ordinary LinearRegression. No clipping, tuning,
feature selection or alternate model fits after viewing evaluation outcomes.

Before fitting require32 complete eligible discovery cases, at least8 positive
and8 nonpositive losses, reference mean at least .95, and all integrity checks.
Freeze training IDs, transforms, coefficients and complete source hashes before
generating evaluation outcomes. Evaluation requires48 complete eligible cases,
at least12 positive and12 nonpositive losses, and reference mean at least .95.
Failure of feasibility stops interpretation, not eligibility-based filtering.

Evaluation passes only if candidate MSE is at least10% lower than EACH baseline,
with at least32/48 strict squared-error wins against EACH, raw positive-direction
scalar AUC at least .80, and a positive lower95% paired-prompt bootstrap bound
for relative MSE gain against EACH baseline (10000 draws, seed2026091573).
Use simultaneous Bonferroni bounds across three comparisons, percentile endpoints
at .025/3 and1-.025/3. Report ordinary95% intervals too. These are conditional on
the frozen fit; they do not include training-set variability. Report all signed
outcomes, reference quality, MSE, MAE, bias and prediction ranges.

Verify original exposed replay before collection, instrumented/plain logits,
independent source states, exact reference/noop/uninterrupted replay, native mask
identity and physical effect, complete coverage and signed scoring. Record full
synchronized observation wall time from prepared B16 through feature readiness,
including masks, clones, probe, transfers and reduction; report diagnostic and
artifact-writing overhead separately where possible. Time prediction and ordinary
prefix advancement separately, with actual emitted tokens and peak memory.

A pass earns independent audit and separately frozen genuinely unseen confirmation,
not completion. A miss closes this exact replication without changing learner,
sign, horizon, action or subgroup. After failure, return to understanding with
the fresh negative evidence rather than refitting these exposed outcomes.
