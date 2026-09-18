# Fixed IFEval paired feasibility pilot

Freeze before generation. This is a feasibility test, not predictor evaluation.
The goal remains signed final task-quality loss, here explicitly strict
instruction-compliance success, not semantic correctness of an underlying essay.

Use the16 prompts in `data/ifeval-feasibility-v1.json`, SHA256
`b8a2a25b4a99017542fcfe11aba6546e30ce0e3831012dd6a971f02f86496895`.
Select8 single-rule and8 multi-rule prompts by increasing SHA256 of
`herald-078-feasibility:KEY` within strata, excluding only invalid keys1122/1129
identified in077 before outcomes. All16 become exposed feasibility cases and
cannot later enter a sealed evaluation. The remaining public population is not
yet certified to contain independent semantic groups. This pilot has no split.
Sampled topics are distinct; repeated instruction templates are intentional.

## Intervention and measurement

Reuse the existing `run_pair_pilot.py` and archived pinned engine without edits.
Qwen2.5-7B-Instruct snapshot a09a35458c702b33eeacc393d103063234e8bc28,
CUDA bfloat16, SDPA attention, greedy seed0, existing chat template.
Reference path `shared_boundary`. Boundary is after prefill of all but the final
prompt token, which is pending; no response tokens have been generated.
Reference and actions start from independent equivalent cache clones. Actions
are Knorm0 (control) and Knorm.5 (candidate), compressing the whole cached prompt.
This does not reproduce system-only compression in the Pitfalls paper.
Full-prefill/reference/control token parity and unchanged source-cache checks
must hold. Keep every failure and verify physical eviction on every case.

Generate through EOS or2048 new tokens for every arm, including long requests.
Score the full decoded continuation with the pinned official strict scorer and
the exact original instruction IDs/kwargs. Seed Python random and langdetect
to0 separately before each arm score. Empty output is task failure (score0),
not missing data or an incomplete run if generation completed normally.
Record termination and do not silently truncate or filter difficult prompts.
Loss = int(reference follows all constraints) minus int(action follows all),
including -1 improvements. Record per-instruction results as diagnostics.
Retain generation, cloning and compression costs from the runner. There is no
predictor yet, so do not invent a prediction-cost result.

## Prespecified gates and decision

Feasible only if all16 pairs complete with all integrity and no-op checks passing,
reference strict accuracy is at least12/16, at least4 cases have positive loss,
and at least4 have nonpositive loss at Knorm.5. At least15/16 reference and15/16
action responses must terminate before the cap. No case is excluded after
generation. These gates screen an assay, not establish a predictive effect.

Hypotheses: (1) reference competence is inadequate, expected low reference score;
(2) compliance survives.5 compression, expected too few damaged cases;
(3) damage saturates, expected too few nonpositive cases;
(4) effects vary across prompts despite usable reference competence, supporting
a prospective predictor study with prompt-only and action-aware baselines;
(5) output cap or scorer behavior dominates, invalidating the intended assay.

If feasible, inspect mechanisms on these exposed cases and freeze a predictor
study with independently grouped unseen evaluation and numerical uncertainty
gates. If infeasible, inspect the failure mechanism before choosing any second
experiment. Do not weaken these gates or present this pilot as goal completion.

Prelaunch Luna review caught an incorrect draft claim of eager attention. The
unchanged runner uses SDPA on CUDA; corrected here before any generation.
Strict scoring and full-prefill parity enforcement are performed by the separate
`score_ifeval_feasibility.py`, not by the raw collection runner.
