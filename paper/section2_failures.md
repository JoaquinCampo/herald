# Section 2: Catastrophic failures under KV-cache compression

## Job of this section

Define what we mean by *catastrophic*, give the detection rules, and quantify how often these failures occur across compressors and ratios. By the end of §2 the reader knows:

1. The two failure modes we formalize and why those two.
2. How we detect each one mechanically (rules, thresholds, onset definitions).
3. That catastrophic failures are common at heavy ratios and that compressors differ wildly in their fragility.
4. That uncompressed generation has a small but non-zero baseline rate, so our predictor must distinguish *compression-induced* catastrophes from generic generation failures.

This section earns Contribution 1 ("we characterize…") and produces the labels that §3 will train on.

## Structure (4 subsections)

### 2.1 What we mean by *catastrophic*

- **Distinguish two error regimes.** Graceful accuracy degradation (the model gets the math wrong) is normal, it happens even without compression and is well-handled by accuracy benchmarks. *Catastrophic* failures are discrete, qualitative breakdowns where the generation itself goes wrong, independent of whether the answer is correct.
- **Two formal modes:**
  - **Looping**, the model emits the same chunk of tokens over and over.
  - **Non-termination**, the model never produces a stop token within the budget.
- **Out of scope, deliberately:** wrong answers (a normal model failure even on uncompressed inference; tracked separately as accuracy), instruction amnesia (no robust automatic detector), coherence collapse (likewise), format failure (likewise). These are noted as future work in a single closing line.
- **Why these two and not more.** They are (a) detectable from the token stream alone, with no labeled judge needed, (b) demonstrably caused by compression (rates rise sharply with ratio; near zero on uncompressed runs), (c) operationally severe, both produce unusable outputs and both run the model to its token budget.

### 2.2 Detection rules and onset definitions

- **Non-termination.** A run is non-terminating if its stop reason is `max_tokens` or `timeout`, i.e. the generator hit the budget without producing EOS. Onset is defined as the final token position; non-termination is a property of the whole trace, not a moment. We acknowledge that an effective onset for hazard labeling needs to be earlier than the budget cutoff and explain how §3 handles this with the `nt_onset_frac` parameter (set to 0.75 in our pipeline; see `models/analysis/per_press.json`).
- **Looping.** A run is looping if any window of $W = 20$ contiguous tokens appears at least $r = 3$ times within the generated sequence. Onset is the start position of the *second* occurrence of the first window that hits the threshold, the first time the model repeats instead of producing new content. Why $W = 20$ and $r = 3$: short enough to catch tight phrase loops, long enough to avoid false positives on natural repetition (e.g. a list of items); $r = 3$ requires the model to commit to the loop, not just produce a coincidental pair.
- **Co-occurrence is the norm.** Looping almost always escalates into non-termination, since a looping generation will run until the token budget. We present them as separate detectors but treat them as a single hazard label in §3 (the union of either firing).
- **Wrong-answer carve-out.** We measure wrong-answer rate alongside, but it is *not* a hazard label: math errors occur on the uncompressed baseline at non-trivial rates and would inject label noise. Wrong-answer rate appears in §6 as an accuracy metric, not in the hazard target.

### 2.3 Prevalence across compressors

Numbers come from `models/analysis/per_press.json` at horizon $H{=}10$, evaluated on the test split (~2500 sequences per press; ~500 for the uncompressed baseline).

| Compressor | Catastrophic rate | $n$ catastrophic / $n$ sequences |
|---|---:|---|
| KNorm | **71.0\%** | 1774 / 2500 |
| SnapKV | **68.7\%** | 1690 / 2460 |
| Random | 35.9\% | 898 / 2500 |
| TOVA | 15.4\% | 385 / 2500 |
| StreamingLLM | 13.7\% | 342 / 2500 |
| ExpectedAttention | 4.8\% | 121 / 2500 |
| *None (uncompressed)* | *1.2\%* | *6 / 500* |

**Reading the table:**

- **Catastrophic behavior is compressor-specific.** A 15× spread between the most fragile (KNorm, SnapKV) and most robust (ExpectedAttention) compressors at the same ratios.
- **Even the random baseline is bad but not the worst.** Random eviction sits in the middle, suggesting that some "principled" compressors (KNorm, SnapKV) are actively *worse than random* in our regime, a finding worth flagging as more than a curiosity.
- **The uncompressed baseline is non-zero (1.2%).** Some catastrophic-looking behavior is intrinsic to the model on long-form chain-of-thought; this is the floor any predictor has to clear.

Possible figure here: a bar chart of catastrophic rate per compressor, ordered worst to best, with the uncompressed baseline drawn as a horizontal reference line. Useful, low-cost.

### 2.4 Prevalence across compression ratios

Numbers from `models/analysis/per_ratio.json`, $H{=}10$, pooling all six compressors at each ratio (~2992 sequences per ratio; ~500 at ratio 0).

| Ratio | Catastrophic rate |
|---:|---:|
| 0.000 (uncompressed) | 1.2\% |
| 0.250 | 10.1\% |
| 0.500 | 25.5\% |
| 0.625 | 35.9\% |
| 0.750 | 45.9\% |
| 0.875 | 56.7\% |

**Reading the table:**

- **Roughly linear in ratio.** Each 0.125 step in compression adds roughly 10 percentage points of catastrophic rate.
- **No "safe" heavy regime.** By ratio 0.875, more than half of all generations across all compressors fail catastrophically. This is what makes the runtime-intervention angle interesting: at heavy ratios the failure mode is the modal outcome, not a tail.
- **Cross-product implication.** The rates at high ratios are dominated by the fragile compressors (KNorm, SnapKV); the robust ones (ExpectedAttention) stay tolerable even at 0.875. This tension between memory savings and reliability is exactly what HERALD lets a system navigate at runtime.

Possible figure here: catastrophic rate vs. ratio, one line per compressor, with the uncompressed baseline as a horizontal floor. This is probably the most useful figure of §2 because it shows both the linear trend and the cross-compressor spread in one image.

### 2.5 Qualitative example (one paragraph + reference to Figure 1)

Walk through `gsm8k_79` (random press, ratio 0.75), the same sequence shown in Figure 1's left panel. The arc:

1. Model solves the problem correctly (200 kg total).
2. Model second-guesses itself ("However, the question asks for...").
3. Loop crystallizes: four near-identical repetitions of `\boxed{75 kg and 125 kg}`.
4. Generation runs to the 512-token budget without an EOS.

Use this to make concrete what looping + non-termination look like in practice and forward-reference Figure 1 instead of repeating it.

## Decisions

- **One table OR one figure for prevalence, not both.** A figure with two panels (per-press, per-ratio) is the strongest version. A table-only version is the cheapest version. Default: figure.
- **No standalone "what we don't measure" subsection.** Fold it into one closing line of §2.1, since reviewers don't need a paragraph telling them what is *not* in the paper.
- **Reuse Figure 1 for the qualitative arc.** Don't introduce a new figure for it.

## Open questions before drafting prose

1. Do we want the per-press numbers at H10 (what's in `per_press.json`) or at the H1 ground-truth label (what §3 mostly trains on)? They differ slightly because the horizon affects which sequences count as "catastrophic" within the labeled window. **Recommendation:** state the prevalence numbers at the *sequence level*, not at any horizon, i.e. "this run contains a catastrophic event at any token", and clarify that the horizon-dependent labels in §3 are derived from these.
2. Confirm the wrong-answer carve-out reads honestly. We claim wrong answers occur on the uncompressed baseline; we should probably state the uncompressed wrong-answer rate as a one-line piece of evidence.
3. Should §2 introduce $H$ (the hazard horizon) at all, or defer entirely to §3? Probably defer, §2 talks about what counts as a catastrophe; §3 talks about how far ahead we predict it.
