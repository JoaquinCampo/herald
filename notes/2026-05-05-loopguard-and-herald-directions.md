# LoopGuard read + HERALD direction brainstorm

**Date**: 2026-05-05.
**Status**: Scratchpad. Not canonical. Brainstorm for future iterations,
not commitments. Move concrete plans to `gold/` only after pressure
testing.
**Trigger**: Reading *LoopGuard: Breaking Self-Reinforcing Attention
Loops via Dynamic KV Cache Intervention* (Xu et al., arXiv 2604.10044,
Apr 2026), an adjacent concurrent paper.

## 1. LoopGuard summary

Same neighborhood as HERALD. Different scope and approach.

- **Failure mode**: persistent repetition loops only. Not non-term, not
  amnesia, not format break.
- **Signal**: surface-output statistics (TTR, compression ratio,
  top-1 confidence streak, suffix self-alignment) over a 256-token
  sliding window with K-of-3 debounced vote.
- **Action**: structural KV cache surgery — keep anchors, sparse middle,
  tail-cleaned recent under fixed budget B. Progressive aggressiveness
  level if it re-triggers.
- **Theory** (App. A): RoPE + KV reuse → block-wise relative-offset
  scoring → score invariance lemma under periodic tail repetition →
  tail attractor with low-diversity stability.
- **Benchmark**: LoopBench-DC (300 prompts, JSON entity extraction
  from MultiNews) and LoopBench-RI (200 prompts, recursive-protocol
  on NarrativeQA). Built specifically to induce loops.
- **Models**: Qwen3-1.7B and Llama3.2-1B. Smaller models; scale-
  dependent survival.
- **Result**: loop rate 100% → 1.3-2.7% across baselines, F1 on
  2WikiMultihopQA preserved (53.61 vs 50.49 Full Cache).
- **Trigger timing**: Figure 4b shows triggers concentrated at token
  800-1300, i.e. **post-onset**. They detect collapse and break it,
  not prevent it.

## 2. Where HERALD already differs (defensive position)

- **Scope**: 5 failure modes, not 1.
- **Signal class**: logit features (entropy, KL, top-k Jaccard,
  rolling/EWMA aggregates) instead of output text statistics.
- **Output**: damage *magnitude* prediction (`sum_js`, ROUGE-L drop)
  not just binary loop flag.
- **Cross-press transfer**: held-out compressors evaluated; LoopGuard
  doesn't show this.
- **Larger model + natural tasks**: Qwen2.5-7B on 4 LongBench tasks
  vs LoopGuard's small-model synthetic-loop benchmarks.

## 3. Concordance worth using

LoopGuard's evidence actually *corroborates* HERALD's lead-time
inversion finding (predictor AUROC 0.27-0.33 in [-200, 0] window for
looping/non-term).

- Their Fig 2a "abrupt collapse point" + low-diversity attractor =
  why HERALD's JS-divergence label drops at onset.
- Their Lemma 1 (score invariance under periodic repetition) gives a
  theoretical home for "looping is low-divergence repetition, so a
  divergence-based label correctly assigns it low scores."
- Reframes HERALD's inversion from "limitation" to "predicted by a
  published mechanism." Same finding, stronger framing.

## 4. Leverage list — import-and-extend ideas

These borrow from LoopGuard. Useful but not original on their own.

1. **Cite Lemma 1 to home our inversion finding.** Free. Do it.
2. **Add TTR + CR + p1-streak as cheap baselines** in
   `predictor_baselines.py`. ~1 day. Forces honest comparison;
   defends against "you didn't try the obvious output-surface
   detector."
3. **Train HERALD on a TTR-collapse or CR-collapse window label**
   instead of `future_sum_js`. Phase 2 already flagged "different
   label family needed" for pre-onset lead time on looping; LoopGuard
   tells us which family. Probably fixes the inversion for looping
   specifically. ~1 week including dataset rebuild.
4. **Run HERALD on LoopBench-DC and LoopBench-RI.** External eval on
   their turf. Either we beat their detector at lower cost, or we
   match at lower cost. Either is a figure. ~1 week.
5. **Add their scale axis** (Qwen3-1.7B, Llama3.2-1B). Cross-scale
   transfer claim they can't make. Orion time, otherwise straightforward.
6. **Phase 4 = HERALD trigger + LoopGuard cache surgery** rather than
   a from-scratch controller. Compare HERALD-trigger + LG-action vs
   LG-trigger + LG-action on LoopBench. Reframes Phase 4 as "better
   detector for an existing intervention."
7. **Borrow figure recipes**: collapse-point trajectory (Fig 2a) and
   trigger-position histogram (Fig 4b). We have the data.
8. **Borrow debounce/cooldown hyperparameters**: K-of-3, persistence,
   warmup tmin=64, cooldown=32. Use as priors instead of re-searching.

## 5. Original ideas — actually pushing past LoopGuard

These are not in either paper. Ranked by how much they would change
HERALD's identity, not by how easy they are.

### 5.1 Early-warning signals from dynamical systems theory

Single biggest framing upgrade. LoopGuard shows phase-transition
behavior (sharp collapse point + bistability). The bifurcation /
critical-slowing-down literature (Scheffer 2009 *Nature*; Dakos et al.)
gives **universal precursor signatures** for phase transitions:

- Variance increases as the basin of attraction shallows
- Lag-1 autocorrelation increases (perturbations decay slower)
- Skewness shifts (system spends more time near alternative attractor)
- Flickering (brief excursions toward new state)

Used in ecology, epileptology, finance. **Never applied to LLM
decoding dynamics.** HERALD's rolling/EWMA features are accidentally
low-fidelity versions of these — that's why they work. Proper
formulation gives:

- Theoretical justification for cheap online features (matches
  LoopGuard's attractor theory at the *prediction* layer)
- Better features: lag-1 AC of entropy, variance of top-1 prob over
  window, skewness of KL
- Cross-disciplinary framing: HERALD as critical-transition theory
  applied to neural-network compression dynamics

Risk: precursors may be weak in practice. Have to actually compute
and test on existing data. Tractable because everything is post-hoc
on logged signals.

**Why this is the strongest contender**: gives HERALD a theoretical
identity matching LoopGuard's, with experimental work that's
incremental on existing data.

### 5.2 Attention-concentration as a pre-onset feature

LoopGuard's Fig 5 shows barcode-like attention stripes during loops.
They use it diagnostically. The unstated research question:
**does attention narrow before output diversity collapses?** Almost
certainly yes — head locking is the cause; output collapse is the
effect.

Adding attention-entropy and attention-mass-on-recent-K as features
gives a signal class neither paper uses. Concrete prediction:
attention features lead output features by 50-200 tokens, beating
LoopGuard's TTR/CR detector and our current logit features at
pre-onset prediction.

Cost: small. Attention is in the same forward pass.

**Why this is a strong empirical bet**: legitimately fixes the
lead-time inversion (not via relabeling) and uses signal LoopGuard
cannot use because surface text statistics are downstream.

### 5.3 Conformal safety budgets

Reframes HERALD output. Instead of binary classifier on future
divergence, output a **calibrated upper bound on tokens-to-failure**
at the current compression ratio. Per-token output:
*"With 90% confidence, this run will remain healthy for at least N
more tokens at ratio r."*

Conformal prediction territory. Well-developed theory. Gives runtime
guarantees. LoopGuard provides nothing comparable.

Risk: requires exchangeability assumptions that may not hold across
runs. Conformal-time-series literature has solutions but adds depth.

**Why this is publication-worthy on its own**: it's a contribution
in HERALD's *output type*, not just in training. Practical value
is high.

### 5.4 HERALD as teacher signal for compression-policy learning

Most ambitious. If HERALD predicts damage, then HERALD's score is a
reward signal that can train a KV-cache eviction policy via RL. The
policy learns to compress aggressively where damage is forecast as
low and conservatively where high.

Pivots HERALD from "monitor that watches a fixed press" to
"teacher that produces a better press." Pure leapfrog of LoopGuard.

Cost: full RL setup, environment, baselines. Months not weeks. But
the contribution magnitude is correspondingly larger.

**Why this is the most ambitious paper option**: changes the
contribution from measurement to improvement.

### 5.5 Multi-class catastrophe-type prediction

LoopGuard predicts loop-or-not. HERALD has tagged data for looping,
non-termination, format-break, drift, amnesia. A multi-class predictor
that says **which** failure is about to happen lets a downstream
system pick the right intervention (surgery for loops, max_new_tokens
bump for non-term, regen for amnesia). LoopGuard's TTR/CR signal
class cannot disambiguate failure modes; logit features can.

Cost: modest engineering lift on the existing pipeline.

**Why this is a clean, contained contribution**: tractable, original,
and directly load-bearing for any closed-loop demo.

### 5.6 Counterfactual decompression as the intervention

LoopGuard's action damages the cache (irreversible pruning, info
loss). Alternative: when HERALD fires, **temporarily decompress** for
the next M tokens, then re-compress. Reversible, structure-preserving,
correctness-friendly. Trade-off: transient memory cost vs damage
avoided.

LoopGuard never considers this because they assume strict budget
always. HERALD + dynamic budget = different design philosophy.

**Why this is a clean alternative action**: not a paper on its own
but a strong Phase 4 design choice that distinguishes from LoopGuard.

## 6. My honest ranking

- **Single thesis pick**: 5.1 (early-warning signals). Theoretical
  identity + incremental experimental work + cross-disciplinary
  story.
- **Single biggest empirical bet**: 5.2 (attention features).
  Legitimately addresses lead-time inversion.
- **Most ambitious option**: 5.4 (HERALD as teacher). Different
  magnitude of contribution.
- **Don't try to do all of them**. Pick a thesis, let the rest be
  future-work bullets.

The biggest risk for HERALD is not LoopGuard. It's doing six things
poorly instead of one thing crisply.

## 7. Pending decision

Choose one of:

1. **Thesis = early-warning signals**. Reframe HERALD as
   critical-transition forecasting for compression-induced phase
   changes. Add EWS features. Fold attention features in if cheap.
2. **Thesis = attention-driven detection**. Reframe HERALD as
   first-mover on attention-feature-based pre-onset detection. EWS
   becomes a sub-claim about why it works.
3. **Thesis = closed-loop teacher**. Reframe HERALD as a learning
   signal for better compression policies. Bigger paper, longer
   timeline.
4. **Thesis = multi-mode failure prediction**. HERALD as a
   failure-type classifier with intervention-routing. Cleanest, most
   contained, lowest narrative ambition.

Decision goes in `gold/research-plan.md` once chosen, not here.

## 8. Standing actions regardless of thesis

Cheap, defensive, do-anyway:

- Cite LoopGuard. Frame our inversion under their Lemma 1.
- Add TTR / CR / p1-streak baselines.
- Run on LoopBench (one subset minimum) for external comparison.
- Borrow debounce/cooldown hyperparameters as priors for Phase 4.
