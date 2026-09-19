# How KV-cache compression damages NIAH recall: current mechanism record

**Current state: completed reader/MLP intervention follow-up.** Read [record 14](14_mediation_state_summary.md) for the new evidence, deductions, counterexamples, and remaining tests. The complete companion report, effect/exception ledgers, tests and staged token/cache factorial are in `herald-mediation-reanalysis.zip`, delivered with this investigation. The preceding state of this entrypoint is preserved unchanged in [the pre-mediation archive](archive/7_compression_damage_mechanism_before_mediation.md). [Record 11](11_gpu_demand_conditioned_damage.md) retains the earlier array analysis; its proposed intervention has now been run, but its natural-path interpretation remains revisable.

## Scope and evidence units

One Qwen2.5-7B-Instruct checkpoint, eight NIAH prompts in four paired families. The latest collection is Knorm0.1: 30 eligible decisions, 390 patches plus 60 reference/compressed output records, and 75 full continuations at five first errors. There are 13 earlier decisions in eventually failed arms and 12 decisions in preserved arms. Conditions and repeated continuations are not independent prompts. Four first errors are final-digit truncations; only one is a middle-digit swap.

## Current account

Compression can alter source access and the state from which a decision is constructed. Its consequence is conditional on the recipient computation and the particular content/continuation competition. The intervention evidence supports a nonexclusive path through late computation, not one uniquely necessary reader-to-MLP chain. Repairing output does not require reconstructing the original internal computation.

A forward also has two distinct consequences: its chosen token and the K/V written for subsequent steps. These must be separated when explaining future quality. A final-MLP-only intervention follows every cache write and therefore cannot repair that forward's cache. In this deterministic decoder, if it also leaves the token unchanged, it cannot change the subsequent trajectory. This is a source-grounded deduction, with a staged exact cache/continuation test; those earlier continuations were not collected.

## What is now established

**Conditional mediation, not a binary3/5 story.** Both-reader restoration corrects all five first errors. Freezing MLP25/26/27 outputs to their compressed baseline values removes three corrections, but reduces the fixed-target/rival margin benefit in all five. Positive reader benefit remains under the freeze in all five. Controlled contrast arithmetic is not a percentage of natural damage or an exclusive path attribution.

**Joint success is not positive synergy.** Neither reader alone corrects01-base, while both do. Yet the fixed-margin joint interaction is negative in all five first errors. Accumulation across a boundary explains this binary complementarity without a positive continuous interaction. L23G2 alone corrects0/5 but improves all five error margins. The alternative L23G1 reference patch corrects4/5; it is an active comparator, not an established irrelevant sham.

**Context and reserve matter.** Reverse reader patches leave5/5 reference first answers correct, despite a17-logit loss in the swap. Reverse late-MLP patches leave3/5 first answers correct. The selected vectors alone do not determine the answer independently of the surrounding residual/cache computation. Active compensation is possible but not uniquely established.

**Repair is not restoration.** Successful reader patches recover only17.58–60.63% of the lost reference-to-compressed late-MLP support along the fixed contrast. This is one projected gap, not a causal fraction or vector-recovery fraction. MLP27 still opposes the correct token in two successful reader repairs. Component support and final correctness are not equivalent.

**Content versus closure.** In all60 conditions at the four truncation cells, the correct digit remains the highest-logit digit; period sometimes wins globally. The swap loses the competing-digit contest instead. These are distinct output competitions, not proof of dedicated modules or conscious knowledge.

**Quality is scorer-dependent.** All44 correct-first-number continuations match the corresponding reference token sequence exactly. There are45 successes under the original any-exact-number score. The exception is01-planted with compressed MLPs in reference: the first answer is wrong, followed later by the truth in an explanation. Preserve both scores rather than silently changing the task.

## Early signals

A new reference-free quantity projects the current run's last three MLP outputs onto current-winner minus runner-up. The gold pivot in the supplied table cancels. Negative support flags3/5 failed arms early and0/3 preserved arms, but adds no lead beyond existing logit softness in this sample. It first fires at00-planted/d4,01-base/d5,01-planted/d5; misses00-base and02-planted. At all five actual errors it is positive for the newly winning wrong token. It is neither a persistent error flag nor a held-out warning result.

A task-aware period-runner marker flags the four upcoming truncations on this small slice. In the earlier72-arm archive it gives only7/22 early detections and49/50 preserved negatives. Its false positive is02-base/Knorm0.5. It needs known answer length; it is not a mode-general predictor.

## Reproduction and next evidence

`herald-offline/scripts/mediation_core.py` independently checks the five fixed-margin contrasts, projected recovery and continuation counts from `HERALD_MEDIATION_TEXT_PACKET.md`. The full companion bundle includes a separate parser/auditor, complete source locators,28 passing targeted CPU tests, and a staged GPU runner. No new model forward pass was made here. Original full-logit/cache execution gates remain producer evidence; the new independent audit covers exported tables and selected prior NPZs.

The focused next test independently crosses the patched versus unpatched returned cache with the reference versus native current token. It also continues earlier token-preserving reader/final-MLP patches. This distinguishes output redirection from persistent-state repair. The staged code is CPU-tested and not GPU-executed. Raw evidence and large arrays remain outside Git.
