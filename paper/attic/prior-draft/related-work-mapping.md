# Related Work Mapping

How the SOTA in `research/sota/latex/main.tex` (29 pages, Feb 2026) ports into this paper.

## Scope

The HERALD first paper is the **predictor paper**: zero-cost logit features → XGBoost → catastrophe forecasting → runtime-intervention substrate. Per `structure.md`, no standalone Related Work section; citations live in:

- the intro (positioning / gap),
- §2 (failure characterization),
- §3 (predictor framing).

The Phase 1 measurement-methodology contribution (matched-prefix replay, alignment matrix) is **not in this paper** and not in this mapping. It will need its own related-work pass.

The paper uses six presses: StreamingLLM, SnapKV, ExpectedAttention, Knorm, TOVA, Random. The SOTA covers 30+ — most are not load-bearing here.

## Where citations land

### Intro — gap section

**Claim 1.** "KV-press literature evaluates with task accuracy averages, which hide catastrophic tails."

- Cite: §9.3 of SOTA (Metrics Landscape) verbatim — the CFR vs. average-case framing is exactly our point. Two sentences max in intro; one citation footprint.
- Anchor papers: "What Must We Give in Return?" (2407.01527), "Hold Onto That Thought" (2512.12008). Both directly evaluate trade-offs and miss the catastrophic-tail framing.

**Claim 2.** "Existing output-level quality monitors detect failure after bad text has already been produced."

- Cite: HALT (2602.02888), ERGO (2510.14077), Lookback Lens (2407.07071), Semantic Entropy Probes (2406.15927).
- Reframe SOTA §5 as: these all detect *during* generation but only after the symptom (hallucination, repetition, attention drift) is visible. None predicts *compression-induced* failure pre-onset. HALT is the closest in modality (logit-only, zero-cost) — most important to differentiate.

**Claim 3.** "No prior work asks whether next-token distributions reveal an impending catastrophe before it manifests."

- Cite: Limits of Learned Importance (2601.14279). This paper says token-importance prediction from KV reps is bandwidth-limited (0.12 bits MI vs 0.31 from position). HERALD's pivot — predict *system failure* from logits rather than *which tokens are important* from KV reps — turns that result into a positioning lever.
- Cite: ForesightKV (2602.03203) for the +147% loss on low-entropy eviction finding. We can frame this as evidence the logit distribution carries actionable information about future damage that no current method exploits.
- Cite: ASR-KF-EGR (2512.11221) as the only existing *proposal* for entropy-guided recovery. **Conceptual only, no implementation.** Differentiate on implemented-vs-proposed and on horizon (theirs is reactive recovery, ours is forecasting).

### §2.1 — defining catastrophic

**Looping mechanism.** Cite "Closing the Curious Case" (2310.01693) for the softmax-bottleneck explanation, and the original Holtzman et al. (1904.09751) for degeneration foundation. SOTA §4.1 paragraph on looping ports almost verbatim.

**Why these two failure modes.** Cite ThinKV (2510.01290) and "Pitfalls of KV Cache Compression" (2510.00231) for prior identification of compression-induced loops and instruction amnesia. SOTA §4.1 covers this.

**Out-of-scope failures (instruction amnesia, structured collapse, reasoning-path corruption).** One line, cite "Pitfalls" (2510.00231) and ThinKV. Mention these are real failure modes we don't formalize because we lack robust automatic detectors.

### §2.3 — compressor-specific catastrophic rates

**Per-press characterizations.** Each press cited with a one-line characterization from SOTA §1:
- StreamingLLM (2309.17453) — sink-and-window static eviction
- SnapKV (2404.14469) — voting-based prompt-time eviction
- ExpectedAttention (2510.00636) — predictive query-aware eviction
- Knorm — norm-based eviction (kvpress library; not in SOTA, needs added background)
- TOVA — token-omission via attention (kvpress library; light treatment in SOTA)
- Random — uniform random eviction baseline; cite as standard control.

**Surprising finding ("Knorm/SnapKV worse than random").** Frame as evidence for SOTA §4.2's "stability assumption is fragile" claim — DefensiveKV (2510.13334) shows retained importance drops to 0.34. Knorm and SnapKV both rely on stationary importance; their fragility in our results is consistent.

### §2.4 — ratio scaling

**The cliff is real and known.** Cite "Quantization Hurts Reasoning?" (2504.04823) for the discontinuous-cliff motivation (different compression axis, same shape). SOTA §4.2 paragraph on discontinuity ports.

**Cross-press spread at high ratios.** Cite "Hold Onto That Thought" (2512.12008) and "Key, Value, Compress" (2503.11816) for prior systematic per-task-per-press characterization.

### §3 — HERALD predictor framing

**Logit-derived signals precedent.** HALT (2602.02888) is *the* closest prior work. Same input modality (top-k log-probs), same zero-cost framing, similar feature shape. Differentiate on:
- Target: HALT predicts hallucination on standard inference; HERALD predicts compression-induced catastrophe.
- Architecture: HALT uses a bidirectional GRU (offline classifier); HERALD uses XGBoost (causal, online).
- Application: HALT is a quality monitor; HERALD is a runtime-intervention substrate.

**Entropy-based monitoring.** ERGO (2510.14077) provides operational template: $\Delta H$ thresholds trigger corrective action. Cite as motivation for using entropy-derived signals; differentiate on horizon (ERGO triggers on observed degradation, HERALD forecasts ahead) and on action (ERGO does full context reset, HERALD's signal is consumed by downstream system).

**Why XGBoost.** Brief; cite Chen & Guestrin (2016) and note tree models handle the heterogeneous, non-linear feature interactions in our 13-dim per-token vector well at sub-millisecond inference cost.

**Why predict failure, not importance.** Limits of Learned Importance (2601.14279) again — its result is the strongest argument for our framing. Worth two sentences in §3.

## What's missing from the SOTA that this paper still needs

1. **Distillation as origin of teacher-forced logit comparison.** Hinton et al. 2015 ("Distilling the Knowledge in a Neural Network"). One sentence in §3 if we want historical grounding for using the model's own next-token distribution as a signal. Probably skip for the predictor paper; relevant when the matched-prefix replay measurement paper happens.

2. **Calibration / reliability for sequence models.** Guo et al. 2017 ("On Calibration of Modern Neural Networks"). Needed if §6 reports a calibration plot and we want the framing of ECE / reliability diagrams. Lightweight cite.

3. **Conformal prediction.** Vovk et al., Romano-Patterson. Only needed if we keep conformal as a deployment-polish claim. The converged plan demoted it; probably skip.

4. **XGBoost reference.** Chen & Guestrin 2016 (KDD). Standard.

5. **Survival analysis / hazard prediction lineage.** If §3 frames the labels as a hazard-prediction problem, brief cite for the framing (Cox proportional hazards or a modern survival-analysis-with-NN reference). One sentence; not load-bearing.

## Bibliography priorities

Tier 1 — must cite, must differentiate from in prose:

- HALT (2602.02888) — closest signal/predictor precedent
- Limits of Learned Importance (2601.14279) — the positioning lever
- ASR-KF-EGR (2512.11221) — closest controller proposal (conceptual)
- ERGO (2510.14077) — entropy-monitoring template
- DefensiveKV (2510.13334) — stability assumption fragility
- ForesightKV (2602.03203) — +147% finding

Tier 2 — cite for context, single sentence each:

- StreamingLLM, SnapKV, ExpectedAttention (the presses)
- "What Must We Give in Return?" (2407.01527)
- "Hold Onto That Thought" (2512.12008)
- "Pitfalls of KV Cache Compression" (2510.00231)
- "Quantization Hurts Reasoning?" (2504.04823)
- "Closing the Curious Case" (2310.01693)
- ThinKV (2510.01290)

Tier 3 — appendix or skip for the predictor paper:

- All hybrid methods (§3 of SOTA)
- All GQA/MQA / MLA architectural papers (§7)
- Most adaptive-policy papers (§8) except where directly relevant
- Most reasoning-architecture papers (§6) — defer to Phase 2 if features expand
- All quantization papers except the cliff one — different compression axis

## What the SOTA's §10 synthesis says vs. what this paper's intro should say

SOTA §10 ends at: "no closed-loop controller exists; this is the catastrophe-aware-control opportunity." That is the Phase 4 thesis.

This paper's intro ends at: "no prior work predicts compression-induced catastrophes from logits before they manifest; HERALD does, and the signal generalizes across presses." That is the predictor thesis.

Both are true. Different paper.

## Open question for the user

The current intro lead is *scientific* ("LLMs broadcast their impending failure through their own logits"). The SOTA's framing is *systems* ("compressors are open-loop and lack runtime monitoring"). The two are compatible but emphasize different review audiences. The current intro decision in `introduction.md` already picked scientific; this mapping respects that and keeps systems framing as the "so what" closer.
