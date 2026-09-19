# How KV-cache compression changes computation and answer quality

**Current state: mediation and continuation follow-up.** Start with [record 14](14_mediation_context_and_quality.md). [Record 11](11_gpu_demand_conditioned_damage.md) contains the prior exported-array reanalysis and anatomical/MLP observations. The original record is preserved in [the historical archive](archive/7_compression_damage_mechanism_20260918.md); its blanket value-inertness and onset-intactness claims are not current knowledge.

## Scope

One Qwen2.5-7B-Instruct checkpoint; eight prompts/four paired families. Original80 arms; earlier GPU72 non-excision arms and406 paired comparisons (297 contexts). New mediation:30 native-prefix decisions,390 patches,60 baseline outputs and75 continuations at five first errors in three families. Repeated conditions and duplicate trajectories are not independent examples. No predictor/controller/compressor is being built.

## Current account

Compression changes source access unevenly across grouped-query attention. The significance depends on the demanded computation, transformed values and recipient state. Earlier arrays localized large signed-margin changes to late MLP residual updates. The completed interventions now show that allowing MLP25-27 to respond to restored readers adds5.625-10.375 logits in all five sampled first errors. Freezing them removes binary correction in only3/5; the remaining two small positive margins do not imply absent mediation.

That is mediation of a defined oracle patch, not a unique natural cause or a fraction of compression damage. The same captured compressed MLP outputs coexist with0/5,2/5 or3/5 correct decisions in different recipient settings. Reference backgrounds resist reverse-reader patches partly through lower susceptibility in the four truncations, not merely greater starting reserve. Comparable reader/MLP responsiveness occurs in preserved cases.

Joint-reader binary rescue5/5 is compatible with subadditive fixed-margin interactions5/5. L23G2 alone repairs0/5 despite its earlier anatomical prominence; alternative L23G1 repairs4/5. It is an active comparator, not an inert sham. Correct continuation does not require every late module to have a positive or reference-like projection.

## Separate token, state and utility

The supplied continuations remove hooks after one forward but retain both its selected token and returned K/V. Earlier-layer patches can change the newly written input-token K/V in higher layers; old slots are not restored. In the pinned decoder, MLP27 comes after all K/V writes. A final-MLP-only patch that leaves the greedy token unchanged is therefore predicted to leave the whole future unchanged, under identical persistent state and removed hooks. This is a formal conditional deduction with exact GPU gates staged, not an additional measurement.

At four actual truncation errors, MLP27-only patches change the last digit and repair the first answer without needing prior-cache repair. All five intervention sites are the last independent truth/distractor distinction (the swap has shared suffix04), so0/5 tests another independent retrieval choice after repair. Do not infer restored source-reading capacity from completion alone.

The75 continuations contain9 distinct full token sequences (11 arm/sequence pairs). All44 immediately correct decisions give correct first numbers; all31 wrong decisions give wrong first numbers. Any-exact-number credit is45/75, not44/75: reverseMLPs in01-planted produce an incorrect first number followed by a contradictory explanation mentioning truth. This is a utility difference, not demonstrated first-answer repair.

## Signals: changed computation is not anticipated quality loss

Earlier onset tail deficits/Jacobian changes precede errors but also occur in survivors. A new gold-free readout statistic projects the final three raw MLP outputs onto current winner minus current runner-up. Negative support occurs before3/5 native failures and in0/3 preserved arms at the sampled checkpoints; misses00-base/02-planted. It becomes positive at the five wrong decisions when the winner changes. This transient pattern is exploratory, not held-out validation. Under intervention it is negative in five correct and five wrong answers.

The broader older-array stored-increment variant is not the same feature: with0/4/6 sampling and a smaller candidate set it flags2/22 failures by error, only1/22 early, with50/50 preserved unflagged. Full-vocabulary runner-up rows are missing at72 onset exports. Do not treat missing observations as safe negatives or extrapolate the small3/5 result.

Detecting a changed computation, an approaching token boundary and loss of a declared utility require different information. Own-stream logits, oracle source spans, reference donors and future truth directions have different costs and privileges. No continuous severity scale follows from exact-number recall.

## Reproduce and continue

[Record14](14_mediation_context_and_quality.md) records counts, exceptions, fixed-contrast calculations and the focused next handoff. `herald-offline/scripts/mediation_packet_core.py` checks the core results directly from the readable packet without model imports. The companion `herald-mediation-reanalysis.zip` preserves the full analysis, ledgers, tests and staged token/cache cross runner.

Next:60 token-by-cache crossed continuations and125 same-token early continuations, with dense conflict observations and exact cache/trajectory gates. These scripts are CPU-tested and compiled, not GPU-executed. All older research records are retained unchanged; raw GPU arrays and user packets remain outside Git.
