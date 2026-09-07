# HERALD v3 pilot proposal

Status: first-milestone draft for owner review, 2026-09-05. No experiment has run in v3. The owner selects the architecture and integrates this proposal.

## Question and proposed measurement

Can a candidate action's immediate effect on the full next-token distribution improve prediction of its signed final instruction-compliance loss beyond action, timing, and cheap scalar observations? Distribution change is an input hypothesis, never a quality label. This tests a specific addition to prior measurements, not a claim that probes were invented here.

For prompt i and action a, define d(i,a) = q(y(i,none)) - q(y(i,a)). Retain improvements (negative values), zeros, and losses, including cases where the reference already fails instructions. Predict E[d | X_decision,a] over the specified prompt population. Greedy decoding makes each paired outcome deterministic conditional on the full state and runtime; the expectation is over prompts with the observed information, not over repeatedly sampling answers. A later stochastic-decoding experiment would be a different estimand.

Use the per-prompt fraction of IFEval instructions passed under loose evaluation as q, with strict fractions as a prespecified sensitivity analysis. This is bounded graded compliance, not semantic correctness or general answer quality. Prompt-equal averaging of these fractions is our target population weighting, distinct from pooling all instructions across prompts. IFEval provides executable checks for verifiable instructions, avoiding a paid or model-based judge. [Zhou et al., IFEval](https://arxiv.org/abs/2311.07911), [official scorer](https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_lib.py).

## Proposed initial roster

| Component | Pilot choice and boundary |
|---|---|
| Task | IFEval, all supported instruction types; no filtering on reference correctness or observed damage |
| Model | Qwen/Qwen2.5-7B-Instruct, for continuity with prior engineering; owner must pin the actual cached weight/tokenizer revision and chat template |
| Generation | Batch 1, greedy, frozen dtype/backend/environment; EOS or 1,024 total new tokens, including the common prefix |
| State | Uncompressed reference trajectory after 32 committed output tokens; KV holds prompt and first 31 output tokens, token 32 is the shared pending input |
| Reference | Continue independently without compression, from that exact boundary |
| Actions | One-time direct live-cache Knorm transform removing 0.25 or 0.50 of eligible entries, plus no-op for controls; freeze implementation and record actual kept entries per layer before the pending-token forward |
| After action | Ordinary cache growth until EOS/cap, no repeated eviction, restoration, controller, or action switching |
| Deferred | StreamingLLM replication, ExpectedAttention re-prefill, quantization, additional models/tasks/checkpoints |

This model is an instruction-tuned causal language model with an official chat-template usage path. Availability on Orion is not verified by this worker. [Official model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

A reference ending before the boundary is an explicitly recorded ineligible decision, never a failed or zero-damage action row. Enumerate all selected prompts and report this coverage. Score capped outputs as produced and record each arm's termination reason. The target is quality under the 1,024-token budget; it does not establish unconstrained final-answer quality. The longer budget than the historical 512 cap is proposed to reduce truncation pressure, not justified by measured v3 outcomes.

## Information and hypothesis

Start with B0: action removal fraction, prompt token count, decision index, and pre-action cache size. B1 adds uncompressed next-distribution entropy and top-two margin. B2 adds the earlier probe family: compressed entropy/margin deltas and argmax-match. The candidate B3 adds one prespecified full-vocabulary Jensen-Shannon divergence to B2. All models get identical metadata, folds, regularization, and prompt weighting. This makes B3 versus B2 the incremental full-distribution test; comparisons with B0/B1 show the cost of acquiring any probe. Benchmark-provided instruction counts/types are excluded from all primary predictors; using them later would require a separately named annotation-assisted diagnostic.

Compute p0 and pa by feeding the same pending token into independent sandbox copies of the boundary state, with a applied only to the second. Use stable full-vocabulary log probabilities and JS(p0,pa) with natural logarithms. The next distributions concern token 33, not a previously selected token 32. In zero-based output indices, KV contains indices 0 through 30, pending is index 31, and the first affected distribution predicts index 32. Do not pass `s=32` to an inherited helper without adapting its convention; the old helper instead makes token 33 pending. Save the precise input position, action, state fingerprint, and computation timestamp. No reference or treatment future token, final score, or instruction-pass vector enters X. Scorer annotation fields are excluded as above.

These sandbox measurements are available before committing a live action but cost computation and temporary memory. They differ from free observations of an already-compressed trajectory, which answer a different question. With two candidates, the simplest transparent implementation uses three sandbox forwards, one reference plus one per candidate, and then separate outcome continuations. Charge clone, eviction, forward, distribution reduction, and cleanup costs; an optimized reuse of the reference forward must be separately demonstrated. Report synchronized latency by component and total, peak memory, and measured retained KV bytes. Do not inherit the historical estimate of roughly 6 percent as a v3 result.

## Pilot evaluation, after engineering acceptance

Propose 120 eligible development prompts, with a fixed candidate roster of up to 160 to account for early EOS. Select by a seeded hash before generating outcomes. Five prompt-group folds keep all actions and duplicate/variant prompts together. Fit StandardScaler plus Ridge(alpha=1) only on each training fold, with no tuning or calibration; clip every model's predictions to [-1,1]. Equal weight each prompt and average its two action errors. The no-op has a known zero label and is excluded from prediction metrics. An action-wise training mean is an additional sanity baseline.

MSE is primary because the requested functional is the conditional mean. MAE, mean signed bias, action-specific errors, and error on positive/zero/negative d subsets are diagnostics, not substitutes selected after inspection. Report each baseline separately. The strongest comparator means the lowest aggregate out-of-fold MSE among B0, B1, B2 and the action-wise mean, not a row-wise oracle. [Gneiting, Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902).

Use 2,000 paired prompt-cluster bootstrap replicates on out-of-fold errors, preserving repeated cluster multiplicities. Intervals condition on the fitted fold models and do not include feature-selection or training uncertainty. Do not present them as confirmatory. Report MSE difference and relative skill, with the latter undefined if the comparator MSE is zero.

Before prediction, report action-effect counts and score distributions separately for each removal fraction as well as jointly. A pragmatic pilot information floor is at least 20 distinct prompts with nonzero d for either action. Below that, the assay is inconclusive for prediction; do not launch a model sweep. If variation occurs only at one fraction, any positive conclusion is limited to that fraction. Negative effects need not occur, but absence of improvements limits evidence about predicting their magnitude.

Provisional go criterion for a later independent replication: B3 improves prompt-equal MSE by at least 5 percent against every comparator and its paired interval versus each excludes zero improvement. This is a research triage rule, not a user tolerance or deployment gate. If the point gain clears 5 percent but intervals do not, classify as inconclusive and estimate the additional independent sample requirement before extending. If it fails the point criterion, report no support for this specification. Never tune it on the same results and relabel the analysis as fresh. A reversal under strict scoring limits any positive conclusion to loose compliance.

## Exposure and sequence

The historical 160 prompts, prediction rows, and resulting hypotheses are exposed development evidence. A new model run on those prompts does not make them unseen. Before calling any v3 prompt fresh, build the union of known v2 prompt IDs and normalized-content hashes across the old switch corpus and newer collections, then group near-duplicate variants. An unused legacy confirmation split is not automatically untouched by prior selection. No fresh roster is established in this milestone.

If enough clearly unexposed IFEval prompts remain, lock the engineering and pilot lists separately; otherwise use labeled exposed development for feasibility and require a separately sourced, documented instruction-compliance set for later confirmation. Do not silently invent benchmark-equivalent prompts or promise 120 fresh prompts. Public benchmark exposure in the model's training is a separate unresolved issue.

Sequence: owner reviews this plan; implementation adds only the acceptance slice; owner runs it on the intended model/runtime; freeze exposure roster and experiment lock; collect paired outcomes; verify integrity; evaluate the fixed comparison. A useful predictor or a well-scoped negative/inconclusive result is a legitimate outcome. Controller behavior and deployment gains require separate evidence.
