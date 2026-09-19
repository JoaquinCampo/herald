# 14. Mediation is context-dependent; token repair is not cache repair

**Status: reanalysis of the supplied 2026-09-18 mediation packet. No new model forward passes.** The complete expanded analysis, all parsed evidence/exception ledgers, tests and staged GPU wrapper are delivered in `herald-mediation-reanalysis.zip`. This versioned core records the deductions and counterexamples needed for the next reader. `herald-offline/scripts/mediation_packet_core.py` independently reproduces the central arithmetic using only the readable packet and Python standard library.

## Evidence and scope

Input: `HERALD_MEDIATION_TEXT_PACKET.md`, SHA-256 `e57e53661e9e35196ca59cc669d2b9943959efddde2ac5e3d86e0d38831a7be7`. Eight prompts/four paired families; five sampled first errors in three families. Thirty cells contain 13 pre-error snapshots of damaged arms, five first errors and 12 preserved-arm snapshots. There are 390 patches, 60 baseline outputs and 75 complete continuations. Repeated conditions are not independent prompts.

All are existing Knorm0.1 arms. Donor/recipient emitted prefixes match; generated K/V may differ. Reference donors are privileged, task-conditioned observations. Output interchange is not source deletion or proof of a unique natural circuit.

The owner's prior analysis rerun regenerated five outputs byte-identically and two with floating-point serialization differences, maximum 7.105427357601002e-15 across 518 numbers. Do not repeat a blanket seven-byte-identical claim. This continuation independently checks the new text tables, not a new GPU replay or all unsaved caches.

## 1. Continuous mediation is not the binary 3/5 result

Use a fixed contrast: correct digit minus the original compressed winner (period for four truncations, 7 for the swap). B is compressed baseline, H both reference-reader patches, F the same patches with MLP25/26/27 outputs clamped to their original compressed vectors.

| Cell | B | H | F | H-B | F-B | H-F |
|---|---:|---:|---:|---:|---:|---:|
| 00-base/d6 | -5.250 | 5.625 | -0.125 | 10.875 | 5.125 | 5.750 |
| 00-planted/d6 | -4.875 | 6.125 | .500 | 11.000 | 5.375 | 5.625 |
| 01-base/d6 | -13.250 | .375 | -10.000 | 13.625 | 3.250 | 10.375 |
| 01-planted/d6 | -6.500 | 3.125 | -4.000 | 9.625 | 2.500 | 7.125 |
| 02-planted/d4 | -4.750 | 10.250 | .250 | 15.000 | 5.000 | 10.000 |

Releasing the late-MLP clamp adds correct-token support in **5/5**, not merely the three cells whose argmax changes back. The two small positive frozen margins do not establish an absent MLP path. Conversely F-B is positive in all five, so not all reader benefit requires those outputs to respond.

This controlled contrast describes mediation of the **oracle reader intervention**. It is not a natural indirect effect or percentage of original compression damage. F-B includes other MLPs, intervening attention, normalization and every unblocked path; it is not a pure direct head effect. Near-zero signs are pinned-runtime observations, not robustness guarantees.

All 325 patch decisions at the 25 non-error cells remain correct. In 8/9 later preserved cells, releasing the clamp nevertheless improves the fixed native-competitor margin by 1.5-12 logits (exception: 03-base/d6, -.25). At preserved 03-base/d4, margin 3 becomes 14.875 with H but only 4 with F. A large causal response is not specific to failure.

## 2. The same compressed outputs are not intrinsically bad

The exact captured compressed MLP25/26/27 outputs coexist with 0/5 correct first-error decisions in native compression, 2/5 with restored readers, and 3/5 in a reference recipient. The executed code clones the same donor outputs; exported projections agree. This does not independently verify the unsaved full vectors.

Their effect depends on the residual/recipient background. Comparing reference-reader gain in compression against compressed-reader loss in reference gives:

| Cell | Gain in C | Loss in R | Recipient interaction |
|---|---:|---:|---:|
| 00-base/d6 | 10.875 | 3.250 | 7.625 |
| 00-planted/d6 | 11.000 | 3.5625 | 7.4375 |
| 01-base/d6 | 13.625 | .1875 | 13.4375 |
| 01-planted/d6 | 9.625 | .625 | 9.000 |
| 02-planted/d4 | 15.000 | 17.000 | -2.000 |

The four reference truncation states are less susceptible to this interchange, not merely farther from the boundary. The swap is the exception: reference loses 17 logits but retains +4. Both response and reserve matter. These are finite two-background effects, not identified gates, redundancy or compensation mechanisms.

## 3. Joint rescue is not positive synergy

For A=L22G0, B=L23G2, fixed-margin factorial interaction M_AB-M_A-M_B+M_native is negative in all five error cells: **-1.125, -1.375, -1.75, -.625, -.25**.

At 01-base/d6, A alone gives -.5 and B alone -10.625, but AB gives +.375. Joint binary correction occurs despite a subadditive continuous response. Threshold crossing is not evidence of positive mechanistic synergy.

L23G2 alone improves the error margins by .5-2.625 logits but repairs 0/5. The alternative L23G1 patch repairs 4/5: it is an active comparator, not an inert sham. Its donor also comes from the correct reference and may carry relevant information. These results limit uniqueness, not show that arbitrary perturbations work.

Restoring MLP25 makes MLP26's fixed-contrast raw projection smaller in 5/5 errors: -13.962210, -15.319194, -15.374483, -17.100208, -3.446091. These are raw projections, not normalized logits. After H, MLP26 remains negative in 5/5 repaired cells; MLP27 remains negative in 01-base and 01-planted. Output repair need not restore every module's sign or reference-like output. Gate/up features were not measured.

## 4. A formal decoder constraint: the final MLP has no independent cache-writing path

The executed hooks replace attention outputs at o_proj's input and MLP outputs after their forward. In the pinned Qwen2 implementation, K/V is written before attention-output/MLP operations within each layer; after MLP27 only final norm/readout remains. Primary source: https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py (cache update 140-149; decoder order 210-229; final readout 346-360/417-428).

**Conditional deduction:** with identical initial cache/input/positions/persistent state, deterministic greedy decoding and hooks removed after the patched forward, a final-MLP27-only patch cannot change the returned K/V. If its selected token is also unchanged, the entire subsequent continuation must be identical by induction. This is not an additional measured GPU result. The staged wrapper asserts exact cache and trajectory equality to test its premises.

Earlier patches can change the newly appended K/V for the input token in higher layers, but not old slots. Thus a single patched decision can influence the future through (a) a changed selected output token and (b) changed newly written persistent state. The output token itself is written on the next forward.

At 00-planted/d5, MLP27 replacement changes current minimum margin .625 to 15.875 and probability .649751 to .999999762 while still selecting 5. The theorem predicts the later native failure remains. All 13 pre-error MLP27-only conditions have unchanged immediate tokens; their early futures were not measured. Under sampling, changed logits would change token probabilities, so the unconditional greedy null does not transfer; conditional on the same token and cache, future evolution remains identical.

At the four actual truncation errors, MLP27-only patches change the last digit and repair all four complete first answers. Combining measured continuations with the decoder constraint shows a token channel can repair these answers without repairing prior K/V. It does not show the recipient can recover the fact without the oracle donor or that future source-reading capacity is restored.

## 5. Complete answers, persistent state and utility must be separated

All 44 immediately correct outcomes give correct first numbers; all 31 wrong outcomes give wrong first numbers. There are 9 distinct complete token sequences globally, or 11 distinct (arm, sequence) pairs (2,2,2,3,2 by arm; the 01 twins share two response strings).

All five patch sites are the final independent truth/distractor distinction: four last digits, and the swap followed by shared suffix 04. **0/5 tests another independent source distinction after repair.** Full completion is measured, but ongoing retrieval-capacity repair is not.

18/31 successful compressed-recipient patches have lower winning probability than the wrong baseline. At 01-base/d6, wrong period probability .999995 becomes correct digit probability .592598 under H and the answer completes correctly. Confidence restoration is not necessary for these deterministic repairs; this is not a population-calibration claim.

The sole same-emitted-prefix group with divergent futures is 01-planted/d6. Compressed baseline produces wrong 486015. and EOS. Reverse MLPs in a reference recipient produce that same wrong first number/period, then a 104-token explanation containing distractor 4860455 and truth 4860155. Recipient cache/background differs, so this is not an isolated new-slot intervention.

First-number score is 44/75; original any-exact-number score is 45/75. The sole discrepancy is 01-planted/d6/comp_MLPs_into_ref. It is not demonstrated semantic self-correction: the text contains conflicting numbers and an unsupported derivation. Once the first wrong number is closed, append-only continuation cannot repair first-number correctness but can earn anywhere credit. Wrong-token onset and eventual utility loss are therefore different targets.

## 6. A new transient endogenous conflict signal, with negative checks

For current full-vocabulary winner w and runner-up v, define C_t=[gamma*(W_w-W_v)] dot (f25+f26+f27). The f values are raw MLP module outputs. The packet's target anchor cancels: P(target-v)-P(target-w)=P(w-v). No gold digit or reference donor is needed for this arithmetic. It is readout support, not causal attribution, and the evaluation window remains task-defined answer checkpoints.

The still-correct winner is opposed at 00-planted/d4 (lead2, -9.828436), 00-planted/d5 (lead1, -14.005844), 01-base/d5 (lead1, -14.732691), and 01-planted/d5 (lead1, -20.373560). In all five native first errors the three-layer sum supports the new wrong winner. Conflict can disappear when the winner changes; agreement is not correctness.

On this five-failure/three-survivor slice: last-three opposition detects3/5 strictly early, specificity3/3, one at least two steps early; misses00-base/02-planted. Final-MLP-only opposition detects4/5 with three early and specificity2/3; false-positive03-base, miss00-base. Its preserved03-base/d5 final contribution is -2.004536 but three-layer sum +12.439397.

Other sampled signals: entropy>.02 detects5/5, specificity1/3, early4/5 (two at least2early); digit gap<ln10000 detects5/5, specificity0/3, early4/5 (three at least2early); period strongest alternative to a digit catches4/5 at lead1, specificity3/3, miss02swap. Paired reference-reader gain>10 catches4/5, specificity0/3, early2/5, miss01-planted. These are exploratory observations, not fitted/held-out performance. Layer selection used prior evidence from these same families.

Stress test: among65 first-error patch conditions, last-three opposition appears in five correct and five wrong first answers. Correct: reverseMLPs in00-base/00-planted/02-planted and frozenMLPs in00-planted/02-planted. Wrong:01-base L22/alternative-group/MLP25,01-planted MLP26,02-planted MLP27. The sign is not a universal correctness label.

Broader checks: period runner-up anywhere in the old first numeric attempt detects9/22 failures, specificity40/50, seven early/two late. Restricting to before digit7 improves specificity49/50 (FP02-base/.5) but uses known answer length. Rounded probability zero was not treated as absence.

A related final-three stored-residual-increment feature was recomputed on199 eligible older NPZ cells, each hash verified. Full-vocabulary runner directions are absent in all72 onset exports. The explicitly smaller candidate-set variant detects2/22 by sampled error, specificity50/50, only1/22 early:00-planted/.1/d4 and03-planted/.25/d6. All15 raw-output/stored-increment overlap signs agree, but raw modules and rounded residual increments are not silently equated. The older0/4/6 grid misses the new offset5 conflict. These data justify dense timing, not extrapolation of3/5.

## 7. Focused next measurement

For post-forward cache K and selected token a, the relevant object is the declared utility of the continuation U(G(K,a)). A current change in logits, a future decision change and a utility change are not equivalent.

The companion `scripts/staged_token_cache_cross.py` implements, but has not GPU-run:

- **cross:** five first-error cells, six cache conditions, two explicit first tokens =60 continuations. Conditions: compressed baseline, both readers, joint lateMLPs, MLP27, reference baseline, reverseMLPs. Cross the original wrong token and target with each independent cloned cache. Known natural branches must reproduce the published trajectories.
- **early:** all25 native-correct cells under five conditions =125 continuations. BaselineC, L23G2, both readers, joint lateMLPs, MLP27. Require the same immediate token; test later effect and the final-MLP null.

Both capture gold-free conflict and logits at every ordinary continuation step with no extra model forward. Gates include exact original source/runtime, native replay, capture neutrality, published immediate logits, original-slot cache parity, upstream-layer cache parity, whole-cache MLP27 equality and complete known natural/self continuations. Use new approved project-home output directories. Return complete ledgers, runtime/source hashes and failures. See companion GPU_HANDOFF.md for paths/commands.

First acceptance slices:02-planted/d4 for token/cache separation,01-planted/d6 for utility divergence. A later independent disagreement/new target is a separate required capacity test. Gate/neuron hypotheses remain unmeasured; no gating conclusion follows from output patches.

## Reproduction and validation

`python herald-offline/scripts/mediation_packet_core.py --packet /path/HERALD_MEDIATION_TEXT_PACKET.md` independently checks core counts, fixed contrasts, native conflict hits and the metric discrepancy. Full companion scripts additionally preserve all exceptions, earlier-array checks and the staged runner.

Current checks:450 output records,390 projections,75 continuations,60 self-summary identities,90 frozen/donor layer-projection equalities, zero candidate-margin arithmetic error,75 exact prior-tokenizer decodes,30 native snapshot/top-five/logit parities.199 older NPZ payloads were independently hash-checked, not all812. Thirty-nine targeted CPU tests pass (33 new,6 preserved owner wrapper tests); no full-repository or nativeGPU rerun is claimed. New model scripts are compiled/CPU-tested, not GPU-certified.

The explanatory advance is context-dependent decision construction with two persistent channels and a separately declared utility. Reader-to-MLP responsiveness is real in these interventions, but comparable responsiveness occurs in survivors. Neither an early lesion, an opposing module nor a successful oracle patch alone identifies future quality loss.
