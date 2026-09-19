# How KV-cache compression damages NIAH recall: current mechanism record

**Current state: GPU follow-up reanalysis, 2026-09-18.** Start here; read [record 11](11_gpu_demand_conditioned_damage.md) for methods, counts, exceptions and remaining causal questions. The original 2026-09-18 record is preserved unchanged in [the historical archive](archive/7_compression_damage_mechanism_20260918.md). Its value-inertness and onset-intactness conclusions must not be treated as current knowledge.

## Scope

Qwen2.5-7B-Instruct, eight NIAH prompts in four base/planted families. The original corpus has 80 repeated-treatment arms. The GPU follow-up has 72 non-excision arms: 22 damaged and 50 preserved. There are 406 paired comparisons at offsets 0/4/6, representing 297 distinct arm/prefix/position contexts, not 406 independent examples. Same emitted tokens do not imply identical generated K/V. Task damage means original free-running exact-number recall loss, not arbitrary next-token disagreement or continuous prose quality.

## Current account

Compression removes source-reading paths unevenly across grouped-query attention. The significance of a removed path depends on query, cache history, transformed values and downstream computation. At the sampled wrong decisions under correct reference prefixes, large losses of signed decision support are expressed through late MLP residual updates. Whether a decision survives depends on remaining signed margin, not missing attention mass alone. Wrong digits, closure tokens and still-correct digits then lead to different trajectories.

This is a supported working account, not a fully established causal circuit. In particular, the proposed path from the identified reader groups to the late MLP response still needs interchange/mediation tests. Readout bookkeeping is not causal attribution.

## What the new evidence changes

**Values are not ruled out.** Correctly mapped mean-V interventions change the selected token in 4/7 compressed conditions; legacy targeting changed 0/7. In the clean competitor swap, replacing distractor values changes `7` to the correct `2`, while replacing both sources produces a period. The correct/distractor logit margin is -4.75 at baseline, -7.125 with needle replacement, +15.875 with distractor replacement and -1 with both: a -14.5-unit factorial interaction. These are one-step interventions, not verified full-answer rescues.

**Early anatomical damage exists, but is nonspecific.** At native onset, 26/32 nonzero Knorm arms lose over 80% of L23 last-two-digit attention. This includes 21 failures and five successes. In all 26, L23 KV group 2 (query heads 14–20) loses its tail mass already at fixed-query deletion. That group originally carries 83.08–93.00% of the layer's tail attention. Unweighted source-survival counts conceal the concentration.

**Anatomy is not fixed signed support.** For those 26 arms, the recorded FP32 deletion output projected on the same final-truth-digit/period direction is positive at onset and negative at final-digit demand, 26/26. Query and generated history both change between those states; tiny signs lack native per-contrast error bounds. This is not a query-only causal test.

**The late decision is constructed differently.** Under correct reference prefixes there are 28 sampled wrong decisions across 21 damaged arms; `00-planted/knorm:0.05` fails at an unsampled time. In 28/28, the summed reference MLP updates support the correct token and the compressed updates oppose it. In 28/28, the MLP contribution change exceeds the attention contribution change. L25–27 MLP increments account for 60.0–96.5%, median 78.6%, of the signed margin loss under common-scale stored-update accounting. This does not mean those layers uniquely cause that percentage of the error. All 150 reference-owned sampled decisions of preserved arms remain correct.

**Wrong emitted history is not necessary for the sampled first errors.** At all 16 native snapshots exactly at the first error, a full-reference forward using the same emitted prefix selects the correct token while the compressed forward does not. Generated cache states may already differ.

## Three-family terminology

Tail collapse and competitor surge remain useful access/readout descriptions. Cascade is a temporal behavior, not a demonstrated disjoint circuit. Do not identify attention with factual evidence and MLPs with an unchanged prior: in these records reference MLP updates provide substantial correct support. The exact gates and features underlying their changed response remain unmeasured.

Pinned and streaming successes still show that retained-position identity matters, but they retain K/V jointly and do not establish routing-only rescue. The old excision deletes the entire 22-token needle union across all heads; it is not a sparse global anchor-necessity experiment. The decisive-digit-position association remains a four-family confounded observation, not a tested protection law.

## Early signals

On the 72-arm GPU population, onset L23 tail ratio <0.2 has sensitivity 21/22 and specificity 45/50; within the ten preserved nonzero Knorm arms, specificity is only 5/10. Its detected failures are 2–6 token decisions ahead of their first error. Its failure miss is `02-planted/knorm:0.1`; preserved positives are `02-base/knorm:0.25`, `02-base/knorm:0.5`, `03-base/knorm:0.1`, `03-base/knorm:0.25`, and `03-planted/knorm:0.1`.

An onset relative query-Jacobian norm threshold >0.25 also detects 21/22, but specificity is 43/50, or 3/10 within ordinary Knorm survivors. Functional change is not the same as failed quality.

The own-stream digit-gap threshold <ln(10,000) detects 21/22 overall, 14/22 on an earlier decision and 5/22 at least two decisions early; pooled specificity is 44/50, within ordinary Knorm survivors 4/10. Digit entropy >0.02 detects 18/22 overall but only 7/22 early. These are descriptive thresholds, not fitted or held-out validated predictors. Internal reference-access/oracle measurements must not be presented as cheap operational warnings.

Negative summed MLP support at the sampled 0/4/6 positions detects 16/22 by the first error decision, with specificity 48/50, but 0/22 with a positive observed token lead. It is an oracle same-step diagnostic. Missing intermediate checkpoints leave its earlier timing unresolved.

## Reproduction and next evidence

`herald-offline/scripts/gpu_followup_core.py` independently reproduces the primary counts from the returned three-ZIP package, streaming its large trace and loading the exported NPZs. It uses NumPy and makes no model forwards. The complete companion analysis bundle, `herald-gpu-reanalysis.zip`, contains the full cell/exception/signal ledgers, a second analysis implementation, tests, and the staged head/MLP mediation script. The raw GPU archives remain outside Git.

The next causal test restores reference head outputs in compression, then repeats that intervention while freezing late MLP outputs at their original compressed values. Reverse patches, self-patch parity and a same-sized different-head-group control are included. The staged script is CPU-tested and compiled, not GPU-executed. Immediate corrections must not be called repaired recall until full continuations are measured.

All 1,246 input payload hashes were checked; 812 exported array hashes, 24 no-op equalities and the separate exported-array audit pass. There were no new model forwards. The analysis outputs regenerate byte-identically and 69 targeted tests pass. Full-repository testing, Ruff and mypy are not certified.
