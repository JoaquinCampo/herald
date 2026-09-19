# 11. Compression damage: concentrated source deletion, demand-dependent readout, and late decision formation

**Status: independently recomputed analysis of the returned 2026-09-18 GPU package.** No new language-model forward passes were made for this record. The intervention results are from the user's GPU run. The source package README and execution corrections were read before analysis. Layer, query-head and KV-group indices below are zero-based.

## Scientific result in one paragraph

The earlier account was too close to “evidence remains, routing drifts, a fixed prior wins.” Correctly targeted interventions establish that source values still influence compressed decisions. Much of the observed early tail-attention loss is already present under fixed-query deletion and is concentrated in one grouped-query-attention KV group, rather than requiring accumulated query drift. However, this early loss occurs in successful arms too. At the sampled wrong decisions under a correct reference prefix, the strongest difference in signed decision-margin accounting is expressed through late MLP residual updates: the reference's summed MLP updates support the correct token, whereas the compressed updates oppose it in 28/28 cases. The working explanation is a demand-dependent source-reading impairment followed by altered downstream decision construction. The full causal mediation chain remains a staged hypothesis, not an established result.

The paths `scripts/`, `artifacts/` and `tests/` below refer to the companion `herald-gpu-reanalysis.zip` delivered with this investigation. A standalone direct-from-vectors cross-check is additionally versioned as `herald-offline/scripts/gpu_followup_core.py`. Raw GPU arrays remain outside Git.

## Scope, assumptions, and evidence units

The task is exact seven-digit NIAH recall in Qwen2.5-7B-Instruct, with one checkpoint, eight prompts and four base/planted families. The older archive has 80 arms. This GPU collection excludes eight excision arms and contains **72 arms: 22 damaged, 50 preserved**. Among nonzero ordinary Knorm arms there are 22 damaged and 10 preserved; the other preserved controls are eight no-ops, 16 pinned and 16 streaming arms. These are repeated treatments of the same prompts, not 72 independent examples.

There are **406 paired comparisons at offsets 0, 4 and 6**, with two explicit native-prefix cells outside the saved trajectory. They represent **297 distinct (arm, token prefix, logical position) contexts**; 109 comparisons repeat a context through the two prefix-owner designations. The repeated 56 onset comparisons from the sentinel run are not an additional independent sample.

Each pair uses identical emitted tokens and logical positions. Generated K/V can nevertheless differ. “Reference-owned” means both forwards are conditioned on the reference's correct token prefix, not that the compressed state is identical to reference. “Compressed-owned” supplies the native compressed prefix. Oracle truth digits cease to be a legitimate next-token correctness label after an earlier wrong or closed first answer; these post-error rows are not used for the wrong-decision mechanism results below.

Final arm labels mean original free-running exact recall. They do not label every counterfactual forward or continuous prose quality. A correct number plus repetition can still score preserved. No continuous quality scale is inferred from margin, entropy or edit distance.

The independent exported-array audit verified **812 NPZ hashes, 406 usable comparisons and 24 exact no-op array equalities**. The common-scale reconstruction in this analysis has maximum error 7.11e-15; the separate original exported-array auditor gives 2.84e-14. All **1,246 package-manifest payload hashes and sizes** match. Maximum recorded manual-FP32/native-BF16 attention relative L2 discrepancy is 0.0021475. This is a vector-level gate, not a signed-projection error bound. Full unsaved-cache and native-replay assertions remain producer evidence.

The executed token-reader correction is retained: equal-logit top-k ordering is not greedy argmax ordering. The collector's four corrected inferred IDs occur after the first numeric answer. No input JSON was rewritten. The failed 47-comparison collection is not counted as completed evidence.

**Regeneration and exact exceptions:** `scripts/analyze_gpu_followup.py` produces `artifacts/final/summary.json`, `signals.json`, `coordinate.json`, `fixed-direction.json`, `cells.csv`, and `layer-accounting.csv`. All patterns below are indexed there. Two complete regenerations were byte-identical. A second standalone implementation computes the central margin results directly from residual-vector differences rather than trusting the stored projection summaries, and reproduces the main counts and ranges. The reader streams the 543.8 MB trace rather than constructing its entire Python object tree in memory.

## R1. Corrected interventions reopen value dependence

The corrected mapping uses actual packed-cache columns. Of seven compressed intervention conditions, **four change the selected token**; the legacy mapping changed none. Two reference conditions are also supplied, for nine conditions at six states in total. These are not nine independent prompts.

The cleanest factorial example is `02-planted/knorm:0.1`, at its first wrong digit. Define M as logit(correct `2`) minus logit(distractor `7`).

| Mean-V replacement at the same starting state | M | Selected token |
|---|---:|---|
| None | -4.750 | 7 |
| Needle | -7.125 | 7 |
| Distractor | +15.875 | 2 |
| Both | -1.000 | period |

Replacing distractor values changes the immediate decision to the correct digit. Replacing both sources does not: an output outside the binary digit contest wins. The factorial interaction is `-1 - (-7.125) - 15.875 + (-4.75) = -14.5` logit units. Additive extrapolation from the two individual interventions would predict +13.5, not -1.

This demonstrates a non-additive response to these particular interventions. It does not assign unique amounts of factual information to the two sources. Mean replacement is not source deletion, “both” changes values in both spans, and a one-token correction is not a demonstrated full-answer rescue.

Three other details prevent overgeneralization. In `00-base/knorm:0.1`, needle replacement leaves period first, but the correct-digit/period margin changes from -5.25 to -8.125: unchanged argmax is not inertness. In `00-planted/knorm:0.1`, at the correct penultimate digit, needle replacement changes `5` to period and moves the `5`/period margin from +0.625 to -14.875. In preserved `03-base/knorm:0.1`, needle replacement changes the correct `6` to `1`. Thus surviving compressed arms can still depend on source values.

The precise revisions are: withdraw “compressed values are ruled out,” withdraw any routing-only interpretation of pinned rescue, and retain only the observation that retained-position identity matters. Keys and values were retained jointly in the pinned/streaming experiments. The old excision intervention removed the full 22-token needle union across heads, not a small globally sparse set of anchors.

## R2. A GQA-weighted bottleneck explains much of the early anatomical deficit

At native first-digit onset, **26/32 nonzero Knorm arms** have less than 20% of reference L23 attention to the last two source digits. This includes **21 damaged and five preserved arms**. In all 26, the tail mass of **L23 KV group 2, query heads 14–20**, is zero already in the fixed-reference-query deletion bridge, RS.

Before deletion, that group supplies **83.08–93.00%** of the layer's summed reference tail attention. It accounts for **93.33–98.23%** of the deletion-stage loss. Across those arms, RR-to-RS accounts for **98.50–105.31%** of the eventual RR-to-CC mean tail-mass loss. Values above 100% mean later bridge stages partly offset the deletion loss.

This model has 28 query heads but four KV groups per layer: one group's source-column deletion affects seven query heads. Counting the fraction of surviving layer/KV-head entries without weighting their current role can therefore be misleading. A large fraction of nominally preserved needle entries is compatible with losing a group that carries most of this particular source-position attention.

This is an observed concentration of access, not proof that patching this group alone will repair recall. The five preserved severe-deficit arms are:

`02-base/knorm:0.25`, `02-base/knorm:0.5`, `03-base/knorm:0.1`, `03-base/knorm:0.25`, `03-planted/knorm:0.1`.

The damaged exception to the severe-deficit marker is `02-planted/knorm:0.1`. Six nonzero-Knorm arms in total are outside the severe-deficit subset: `02-base/knorm:0.05`, `02-base/knorm:0.1`, `02-planted/knorm:0.05`, `02-planted/knorm:0.1`, `03-base/knorm:0.05`, `03-planted/knorm:0.05`.

## R3. Anatomical source loss is not a fixed loss of decision support

A head-output contribution depends on its values and the direction in which later computation reads the result. To hold the semantic contrast constant, project the L23 fixed-query deletion output onto the **last truth digit minus period** readout direction, at reference-owned onset and reference-owned last-digit demand. This is the same digit contrast at both times, not the current first digit at one time and a different target at the other.

In the recorded FP32 bridge, **26/26 severe-deficit arms** have a positive onset projection and a negative final-digit projection. Onset values range from **+0.00238 to +0.43684** reference-normalized logit units; final-digit values range from **-1.89330 to -0.09884**. There are no recorded sign exceptions, and five of these arms preserve final recall.

This rules out interpreting the missing anatomical mass as a uniformly negative direct contribution to that future digit at every moment. It also supplies a negative result for a simple anticipatory unembedding probe: early deletion projected directly onto the later correct digit does not uniformly point toward the later error.

Two cautions matter. Query vectors and generated cache history both differ between onset and the later decision; the comparison does not isolate query demand alone. The smallest positive projection is not protected by a native, per-contrast numerical error bound. The statement is about the saved FP32 computation, not 26 certified native-BF16 sign tests. A query-by-cache cross experiment would separate query and history effects; none was run here.

## R4. The main decision-margin difference is expressed through late MLP updates

Consider only reference-owned correct token prefixes. The sampled compressed forwards make **28 wrong next-token decisions**, spanning **21 of the 22 damaged arms**. They include 16 closure decisions and 12 wrong-digit decisions. The unexposed arm is `00-planted/knorm:0.05`, whose native first error is at an unsampled offset. All **150 sampled reference-owned decisions of the 50 preserved arms** select the correct digit.

### Exact accounting, with normalization separated

Let `h` be the final unnormalized residual state, `s` its RMS denominator, and

`d = gamma * (W_correct - W_competitor)`.

The competitor is the strongest non-target candidate in the compressed output, held fixed when comparing reference and compressed states. The candidate set contains both actual output winners. This is an oracle analysis, not a deployable signal.

On a common reference scale,

`M_C - M_R = d·(h_C-h_R)/s_R + (1/s_C - 1/s_R) d·h_C + numerical residual`.

Decompose the state difference using the **stored** attention and MLP residual increments: `post_attention_add - block_input` and `block_output - post_attention_add`. They include the effect of BF16 residual addition rounding; they are not assumed equal to the modules' unrounded outputs. This makes the accounting a telescoping identity over actual saved states. It is not a causal attribution of the whole effect to whichever module contributes the largest final readout projection.

### The repeated pattern

In **28/28 wrong decisions**, summed reference MLP increments support the correct token; summed compressed MLP increments oppose it. In **28/28**, the magnitude of the MLP contribution change exceeds that of the attention-increment contribution change. In **28/28**, L25–27 MLP increments contribute negatively to the reference-to-compressed margin change. Those last three MLP increments account for **60.01–96.50%, median 78.63%**, of the total signed margin loss in this particular accounting.

L22–23 attention increments also lose support in **28/28**, with common-scale changes from -11.999 to -0.518, median -4.172. This suggests a reader-to-late-computation interface worth intervening on. It does not prove that the earlier attention changes causally mediate the later MLP changes.

The loss is not merely positive-vector attenuation. If the summed compressed MLP increment is decomposed as `m_C = alpha*m_R + m_perp`, with `m_perp` orthogonal to the reference sum, then **alpha is positive in 28/28** (0.478–0.846). The orthogonal component contributes -36.32 to -15.80 logit units to the correct-versus-competitor direction on the compressed normalization scale. The summed-vector cosine remains positive, 0.596–0.917, while its decision support changes sign. Thus a scalar shrinkage description is insufficient; the realized update changes direction in a consequential subspace.

### Why “the last MLP causes the flip” is also too strong

The correct-versus-competitor projection is already negative before the final MLP in **27/28** sampled errors. The final MLP increment itself remains positive in **8/28** errors, but does not restore a positive final margin. These eight cells are enumerated in `summary.json/errors/last_mlp_positive_exceptions`. In all **20/20 later sampled decisions of preserved ordinary Knorm arms**, the final MLP increment remains positive.

An illustrative matched comparison is `00-base/knorm:0.1` at the last digit. Reference updates from MLP25/26/27 contribute approximately +5.02/+2.47/+17.75 toward correct `2` over period. Compressed updates contribute about +0.31/-0.09/-1.60. The final margin changes from +28.125 to -5.25. The difference is not a small final-layer nudge to an otherwise completed readout; the reference builds much of its support late, and the compressed trajectory does not build the same support.

Preserved arms also lose support. Their 20 later ordinary-Knorm decisions lose between 0.96 and 23.81 units of total MLP support on the common reference scale, median 13.77. Large effect and failed task are therefore different events. Comparing only catastrophic failures with no-op or pinned controls would conceal this overlap.

### A concrete unverified mechanism inside the MLP

For a gated feed-forward block, schematically `f(x)=W_down [s(x) * u(x)]`, the exact two-factor product difference can be written

`product_C-product_R = (s_C-s_R)*(u_C+u_R)/2 + (u_C-u_R)*(s_C+s_R)/2`.

This separates changed gating from changed feature amplitude without arbitrarily choosing one factor first. Native activation/product rounding must be recorded separately. The identity is CPU-tested in the staged script; actual gate/up vectors were not exported and no claim about which factor dominates is made. The head/MLP interchange experiment below is the immediate next causal test. Gate tracing is a later refinement, not a result.

## R5. The clean competitor error combines selective deletion with query reranking

In `02-planted/knorm:0.1`, reference-owned offset 4, mean L22 source-digit masses are:

| Bridge | Truth digits | Distractor digits |
|---|---:|---:|
| RR: reference query, full reference K/V | 0.3645 | 0.0624 |
| RS: reference query, reference K/V restricted to kept columns | 0.2691 | 0.1281 |
| RC: reference query, actual compressed K/V | 0.2705 | 0.1294 |
| CC: compressed query, compressed K/V | 0.2294 | 0.1885 |

The source rebalancing starts with deletion; query change increases it. At L22, KV group 0 has lost at least one of the discriminative source positions. Among the 21 query heads retaining both positions, **21/21** reduce truth/distractor log odds with the changed query, but only **one**, head 12, crosses from truth-dominant to distractor-dominant. “Every head switches source” is false.

For surviving source positions at a fixed query, deleting unrelated positions cannot change their within-head odds: `log(a_i/a_j)=q·(k_i-k_j)/sqrt(d_head)`. All **967,974 retained-pair checks** preserve fixed-query odds in the recorded bridge. The query-predicted log-odds identity has maximum numerical discrepancy 1.34e-5. These are dependent arithmetic checks, not 967,974 independent examples.

In the same swap, L23 still assigns more aggregate digit mass to truth than distractor (0.1983 versus 0.0383), despite the wrong digit winning. Aggregate source mass is not a sufficient output rule. Weighted values, per-head output projection, residual composition and later computation matter. The ordered L22+23 fixed-readout contributions in this case are -7.264 from deletion, -0.021 from generated-cache content change, and -1.047 from query change. This is one order-dependent finite decomposition, not unique causal percentages for those pathways.

## R6. Token-history corruption is not necessary for the measured first errors

There are **16 native-prefix snapshots exactly at the first answer error**. On all 16, the full-reference forward conditioned on those same emitted tokens selects the correct digit, while the compressed forward selects a wrong token. Before the first error, both select the correct digit at all 33 sampled native damaged-arm snapshots. Incorrect emitted digits are therefore not necessary for those observed first errors. Generated K/V may already differ under the identical emitted prefix.

Forcing reference tokens also does not guarantee correct compressed computation: it exposes wrong sampled decisions in 21/22 originally damaged arms. The remaining arm fails at a missing sample time, not proof of token-history necessity there. Post-error snapshots cannot be used as “rescues” merely because the original truth digit reappears at the old offset: the semantic context may already have shifted or the number may have closed.

Normalization is a separate magnitude channel. For the 28 sampled errors, the explicitly separated RMS term ranges from -2.912 to +0.339 units, median -0.361. With a bias-free readout and positive RMS, scaling alone cannot reverse the exact sign of the correct-versus-competitor numerator. Rounding is explicitly retained in the reconstruction. The principal margin reversal is not explained by a positive scalar becoming smaller.

## R7. Signal inventory: early information, specificity and costs

Counts below concern 72 arms, not the earlier 80-arm inventory. Thresholds are exploratory descriptive definitions, not fitted or held-out validated decision rules. Every false positive, false negative and lead time is listed in `signals.json`.

| Candidate | Total detections among 22 failures | Specificity among 50 preserved | Specificity among 10 preserved nonzero Knorm | Strictly earlier decision | At least two decisions early |
|---|---:|---:|---:|---:|---:|
| Digit entropy >0.02 | 18/22 | 45/50 | 5/10 | 7/22 | 3/22 |
| Digit winner/runner-up logit gap < ln(10,000) | 21/22 | 44/50 | 4/10 | 14/22 | 5/22 |
| L23 onset tail-mass ratio <0.2 | 21/22 | 45/50 | 5/10 | 21/22 | 21/22 |
| L23 onset tail-mass ratio <0.1 | 15/22 | 47/50 | 7/10 | 15/22 | 15/22 |
| L23 onset relative query-Jacobian difference >0.25 | 21/22 | 43/50 | 3/10 | 21/22 | 21/22 |
| L23 onset relative query-Jacobian difference >0.5 | 18/22 | 45/50 | 5/10 | 18/22 | 18/22 |
| Negative summed MLP support, sampled until first error | 16/22 | 48/50 | 8/10 | 0/22 | 0/22 |

The first two inspect only the emitted-stream decision distributions over the first numeric attempt. For entropy, detections split **7 early, 10 at the error decision, one late**; for logit gap, **14 early, six at, one late**. Logits at the error-producing step are available before its token is emitted, but that is not a positive-token-lead forecast.

The internal onset markers arrive **2–6 decisions early** in the detected failures. They require reference access. Tail mass additionally requires an oracle source span; the Jacobian norm does not need a named span but still compares two query-response computations. These are not cheap, own-stream deployment metrics. The negative-MLP marker additionally uses an oracle correct token and a selected competitor, and is sampled only at 0/4/6. It catches all 16 observed first-error snapshots, yet supplies no observed advance warning. Sparse sampling leaves earlier changes between checkpoints unresolved.

The onset tail marker's single failure miss is `02-planted/knorm:0.1`. Its five false positives are the surviving severe-deficit arms listed in R2. The gap marker adds `02-base/knorm:0.1` as a sixth false positive. The negative-MLP marker's two preserved false positives are `02-base/knorm:0.5` and `03-base/knorm:0.25`. Remaining misses and threshold-specific exceptions are explicit in the signal ledger.

The query-Jacobian quantity is

`||J_comp(q_ref)-J_ref(q_ref)||_F / ||J_ref(q_ref)||_F`,

aggregated in quadrature over L23 query heads. It measures altered local response geometry, not a realized future error. Its false positives reinforce the distinction between lost capacity and failure to satisfy a particular future demand. Simply replacing entropy with a more internal but direction-agnostic magnitude does not solve specificity.

A useful conceptual ordering is therefore:

1. **Anatomical availability:** which source-bearing pathways remain?
2. **Functional response:** how has response to a relevant query changed?
3. **Decision consequence:** what signed support and reserve remain after downstream computation?
4. **Trajectory consequence:** does the changed decision change final task utility?

No single measured scalar is shown to identify all four. The early structural evidence is meaningful; it is not yet a validated forecast. Correct reference directions and future queries in diagnostic formulas are additional information, not free operational inputs.

## Replacing the simple three-family story

Tail collapse, competitor swap and cascade remain useful descriptions of observations, but should not be treated as proven disjoint causal mechanisms. This evidence supports a factorized working account.

Compression changes source availability nonuniformly across GQA groups. The effect of that change depends on current query, cache history and value geometry. Changes in the resulting residual state alter later decision construction, often expressed here through upper MLP residual additions. Whether the correct token survives depends on the signed final margin, not the size of attention loss alone. A wrong digit, a period, or a still-correct digit then creates different subsequent histories and different eventual behaviors.

The distinction between “retrieval evidence” and “prior” cannot be assigned mechanically to attention versus MLP. Reference MLP updates supply large *correct* support in these data. Loss of that support can be a transformed consequence of impaired retrieval rather than an unchanged prior overwhelming a clean readout. Conversely, large late-MLP changes may be downstream correlates rather than the necessary causal mediator. The current data localize a candidate interface, not a complete circuit.

The decisive-digit-position association remains a four-family, confounded observation. Source disagreement is necessarily invisible in shared digits under a pure source-copy interpretation, but that does not establish protection from a longer shared prefix. The current follow-up did not vary source similarity or disagreement position within a family. The earlier staged geometry experiment remains unexecuted.

## Next GPU experiment: explicit causal mediation, not another uncontrolled failure gallery

`scripts/staged_head_mlp_mediation.py` prints a CPU-only plan by default. It stages **30 pre-error-or-at-error cells**, all eight Knorm0.1 prompts at native offsets 0/4/5/6 with two post-error cells censored. There are 13 conditions per cell, including two self-patch identities, a same-sized different-group control, single/joint reader patches, individual/joint MLP patches, and reverse-direction patches.

The primary hypothesis is that changing the candidate source-reader outputs changes later MLP construction of the decision. The decisive condition restores reference outputs at L22 KV0 and L23 KV2 while **holding late MLP outputs at their original compressed values**. If normal head restoration corrects a decision but freezing those MLP outputs removes the correction, that supports mediation along this defined computational path. If normal head restoration changes the decision despite the MLP freeze, the explanation needs a direct or different downstream path. If it does not change the decision at all, the selected group bottleneck is insufficient or the effect has already propagated elsewhere. All outcomes are allowed; no GPU prediction is promoted to a conclusion.

Patching reference late-MLP outputs into compression asks sufficiency at the intervention site. The reverse patch into reference asks a complementary impairment question. A positive result alone does not make the patched vector a unique natural cause, and these hybrids may be off-manifold. Self-patches must reproduce exact full logits. Full native calibration, same-prefix/logical-position construction, source hashes, BF16/SDPA runtime checks, and capture-hook neutrality are gates. No new ratio or compressor is used.

This staged script has been compiled, its CPU plan executed, and its patch indexing and algebra checked with synthetic tests. **It has not been GPU-executed.** It reports immediate decisions, not full-answer rescue. The next package should contain all baselines, all conditions including failures/shams, the emitted ledger, runtime and source hashes, and failed-gate logs. Continuation-level utility after successful patches is an additional required measurement before claiming repaired recall.

After the mediation question, the most valuable expansion is new held-out prompt families with independently varied source position, number/distractor geometry and answer format. More ratios of these same four families do not establish generalization. Collecting the existing missing offsets 1/2/3/5 with the already executed demand-trace runner would resolve timing gaps without changing the method, but does not replace causal tests or held-out families.

## Connection to established work

Kobayashi et al. (2020), *Attention is Not Only a Weight*, motivate distinguishing attention weights from value-dependent outputs. Geva et al. (2022), *Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space*, motivate examining feed-forward residual updates in vocabulary directions. Wu et al. (2024), *Retrieval Head Mechanistically Explains Long-Context Factuality*, provide independent evidence for specialized retrieval heads in other settings. None of those studies validates the specific L22/L23 groups, the numerical thresholds, or the mediation hypothesis in this pilot.

Primary sources: https://aclanthology.org/2020.emnlp-main.574/ ; https://aclanthology.org/2022.emnlp-main.3/ ; https://arxiv.org/abs/2404.15574 .

## Validation and nonclaims

69 targeted tests passed, including 14 new tests and the supplied package tests. Two regenerations of all seven analysis outputs were byte-identical. The separate exported-array audit passed. The full repository suite was not run; Ruff and mypy are not installed. No raw evidence was edited. The new stage has no GPU results. Claims about general long-context quality, continuous severity, a deployable early-warning rule, all-layer necessity/sufficiency, or a unique causal percentage would exceed the evidence.
