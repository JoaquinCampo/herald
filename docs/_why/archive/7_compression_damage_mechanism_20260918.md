# How KV-cache compression damages NIAH recall: a mechanistic account

> **Status: research record (2026-09-18).** Documents the 113-competitor-v1
> pilot on Qwen2.5-7B-Instruct with Knorm at removal fraction 0.1: the
> damage table, the probe protocol that established causation, and the
> resulting mechanism. Intended as appendix material. Raw run records live
> on Orion (`herald-v4/results/113-competitor-v1/`); probe scripts and
> derived summaries are archived alongside this record (see Regeneration).
> Scores follow `docs/goal.md`: damage is task-grounded (exact number
> recall) and compression-attributable (reference correct, compressed
> wrong).

## Setup

NIAH (needle-in-a-haystack) tests long-context retrieval: one fact hidden
in a long irrelevant text. Here the haystack is a Paul Graham essay sliced
to the official ~4k-token budget (17,778 chars / 3,826 tokens), and the
needle is a sentence of the form "One of the special magic numbers for
{entity} is: {answer}" with a 7-digit answer. Four entities give four
prompts; each has a base twin and a planted twin carrying one extra
distractor sentence with a similar-but-wrong number ("The ledger recorded
{distractor} among its entries"). That is 8 runs.

Knorm keeps low-norm keys and evicts 10% of positions per layer-head
(~394 of ~3,935). The reference decodes from the full cache; the
compressed arm decodes from the pressed cache. Both are greedy and
deterministic. Scoring is numeric-run based (`[0-9]{5,}`), phrasing
agnostic; see `src/herald/niah_damage.py`.

## Damage table

Reference 8/8 correct; noop arm (`knorm:0`) 8/8; compressed (`knorm:0.1`)
3/8. Five damaged runs in three modes:

| Row | Truth | Compressed output | Mode |
| --- | --- | --- | --- |
| 00-base/planted, 01-base/planted | 3705852, 4860155 | 6 digits then `.`+EOS | truncation |
| 02-planted | 6109204 | 6109704 (the distractor, exactly) | single-token swap |
| 00-planted (also) | 3705852 | correct digits but uncertain, then early stop | cascade |

Damage is always delivered at ~0.99 top-1 probability. Confidence does
not discriminate: wrong answers look exactly like right ones. A carrier
wobble (skipping the word "provided" at position 12) appears in damaged
and preserved arms alike and is uncorrelated with damage.

## How it was established

Each step below re-derives the previous one's claims with an independent
check; every GPU result below was verified by exact reproduction.

1. **Eviction audit.** Recomputed the Knorm kept-set per layer-head and
   mapped it against needle/distractor token spans. The needle survives
   in ~95% of heads (4-7 of 112 evicting vs 11.2 average). Eviction
   concentrates in essay body/tail. Conclusion: damage is *not* missing
   evidence. This kills the naive hypothesis first, before any other
   interpretation.
2. **Twin-step replay.** Rebuilt reference and arm trajectories
   token-by-token from the same engine primitives, capturing per-step
   top-5/entropy/margin. All 24 trajectories bit-identical to the saved
   records, so every captured distribution is the true one.
3. **Causal span ablations.** At each critical decision step, rebuilt the
   exact cache state by teacher-forcing the recorded prefix (6/6
   baselines reproduce the recorded argmax), then neutralized span values
   (per-layer mean replacement). Reference decisions flip without their
   evidence (needle removal moves ref to the distractor digit at 0.88;
   removing 2 digit positions moves it to a wrong digit with entropy
   0 to 0.88). Compressed decisions do not move (0.991 to 0.982;
   0.994 to 0.994).
4. **Layer-wise ablation (140 runs).** No single layer reverses any
   decision: evidence reading is token-localized but depth-distributed.
5. **Eviction-mask transplant.** Rebuilding the reference state and
   ablating exactly the positions compression evicted (per head) changes
   nothing (`2` stays 0.999). Values are ruled out; the cause is routing
   plus accumulated drift, not any localized content.
6. **Manual attention mass.** With sdpa hiding attention weights, the
   layer-wise query/key computation was re-derived from the layer's own
   weights (layernorm included) and validated against cached keys to
   cosine 1.00000 before trusting any mass number. Two methodology traps
   were caught here: comparing queries to keys (meaningless), and
   forgetting `input_layernorm` (match decayed with depth until fixed).

## Mechanism

Compression shifts the model from evidence-grounded copying to
prior-driven continuation at unchanged confidence:

- **Swap.** Total number-evidence attention is conserved at the peak
  digit-reading layer (L22: 0.428 vs 0.420) but *split* between
  competitors: needle 0.379 to 0.241, distractor 0.064 to 0.191 (3x).
  At the single discriminative digit the focus ratio mirrors almost
  exactly (ref 0.156/0.043 for truth; comp 0.041/0.158 for distractor)
  and the output follows it. Only that position can manifest damage:
  the other six digits agree in both sources, so any mixture reads them
  correctly. Hence the surgical single-token flip.
- **Truncation.** Focus on the decision-relevant last-two digit
  positions collapses (L22: 73% to 41% of digit mass; L23: 63% to 2%)
  while total digit attention stays comparable. The weakened readout
  cannot overrule the autoregressive length prior, so the model emits a
  confident early stop.
- **Cascade.** The intermediate regime: readout partially works
  (uncertain but correct digits, entropy 0.48-0.67) while the prior
  pulls toward stopping; ablations still modulate the distribution.
- **Canary.** Carrier-word confidence halves in all compressed arms
  (top-1 ~0.9 to ~0.5-0.6). Compression injects decision noise
  everywhere; strong localized readout overrules it, degraded readout
  surrenders to it.

Small per-layer routing shifts compound over 28 layers (hidden cosine
1.0 to ~0.89) into a flip that lives in a small subspace the head
amplifies. First visible divergence (position 12) does not predict
damage; the discriminative event sits at the digit tokens.

## Lineage and cross-checks (added 2026-09-18)

113 is the latest in a numbered mechanism series (105-113) logged in
Orion `herald-v4/experiments/` (docs 001-093) and `results/105-*`
through `results/113-*`. The directly relevant predecessors:

- **090** (mechanism map, b16 prompts): hypothesized this route, derived
  the local-error identity (missing mass alone does not determine error
  direction), recorded a truncation and a middle-digit skip, and left 4
  open explanations. 113 decides them (above) on fresh prompts.
- **091/092** (history-vs-eviction hybrid): replacing generated history
  with reference history does not repair b16 truncation (margins +26.56
  vs -23.44); original eviction plus within-forward consequences
  suffice. 113 confirms with identical-prefix twins and adds the
  within-forward routing measurements 092 called for.
- **093** (residual-to-logit accounting): the final flip materializes as
  residual projection onto the winner/runner-up direction; sign changes
  need projection changes, not rescaling.
- **024** (oracle needle-rescue, 12/12 incl. 8 failures): restoring
  needle content rescues — in tension with the ablation-inertness found
  here. Reconciliation hypothesis H3 below (restore changes routing;
  ablate-values does not).
- **015** (prefix disagreement may be nonspecific): confirmed by the
  carrier wobble in preserved 113 rows.
- **106** (required-vs-competitor geometry per head), **109**
  (source-exchange; native arm truncates 5847921 to 584792),
  **107/108/110/111/112** (token mediation, matched history, cache
  order incl. score-ordered kept sets, a failed format mediation, and
  per-head needle masses ref-vs-hybrid).

The frozen magnitude STOP (Knorm MSE skill +4.97% but worse on
positive-damage rows, negative MAE skill) is consistent with this
mechanism: confident, discrete damage is the wrong shape for graded
magnitude regression, so failure concentrates exactly on damaged rows.

## Landscape extension (added 2026-09-18, same 8 runs)

Dose-response (knorm 0.0/0.05/0.1/0.25/0.5, replay harness verified by
exact match on the 0.0 and 0.1 arms): damage rate 0/8, 4/8, 5/8, 6/8,
7/8 — monotonic, with heterogeneous thresholds (00/01 break at 0.05,
02-planted at 0.1, 03-planted at 0.25, 03-base at 0.5, 02-base never).
Modes progress truncation to distractor-swap to garbage with ratio
(03-planted: preserved, swap, garbage; 01-base emits no numeric run at
all at 0.5). Needle-eviction counts do not predict fragility (02 has
the highest needle votes and the toughest base row).

Victim-identity, three complementary interventions at matched budgets:

- **H3 needle-pinned Knorm** (same kept count, needle force-kept):
  16/16 correct at 0.1 and 0.25, including every case damaged
  unpinned. Decides H3 and reconciles 024: restore works via routing
  restoration, consistent with ablation-inertness (values neutralized,
  routing fixed, decision unchanged).
- **H2 needle-excise** (remove only the union of needle positions
  evicted anywhere, ~22 columns of ~3920, keep everything else): 8/8
  damaged — worse than full knorm. Planted rows fall back to the
  distractor; base rows collapse with no numeric run at all. Necessity
  of the needle columns, completing the causal square with H3/024
  sufficiency and Phase C/E inertness.
- **StreamingLLM-style eviction** (4 sinks + recent window, same
  budgets): 16/16 correct at 0.1 and 0.25. Same amount removed,
  different victims, opposite outcome — the first cross-compressor
  evidence that the mechanism is victim-identity (routing anchors),
  not amount.

Refined mechanism: the ~5% evicted needle positions are high-leverage
routing anchors. Their values are not solely decisive (ablations
inert), but their columns attract the mass whose redistribution tips
digit decisions; restoring them (H3/024) or never losing them
(streaming) keeps decisions grounded, while losing just them (H2)
destroys recall entirely.

## Behavioral atlas: 80 labeled arms (added 2026-09-18)

All dose (knorm 0.0/0.05/0.1/0.25/0.5), pinned, streaming, and excise
arms with per-step top-5/entropy/margin streams: 30 damaged, 50
preserved. Modes: preserved / truncation / swap / wrong-number /
collapse (no numeric run).

WHEN-map (knorm dose then pinned/streaming/excise): 00 and 01 break at
0.05; 02-planted at 0.1; 03-planted at 0.25; 03-base at 0.5; 02-base
never. Modes progress truncation to swap to wrong-number/collapse with
ratio. Pinned and streaming arms: all preserved. Excise arms: base rows
collapse, planted rows swap to the distractor.

Signal inventory (sens/spec over the 80 arms, thresholds in parens):

- carrier top-1 < 0.7: 8/30, 37/50. Weak both ways.
- early-window entropy > 0.05: 30/30, 0/50. Fires everywhere:
  nonspecific, confirmed at scale.
- digit entropy > 0.02: 19/30, 45/50. Best single signal; misses the
  ~40% of damage delivered confidently (swap, confident truncation,
  excise fallback).
- digit entropy > 0.10: 15/30, 47/50.
- min margin < 0.99: 30/30, 0/50. Nonspecific (every stream has a soft
  step somewhere).
- digit runner-up mass > 0.005: 18/30, 45/50.
- length < 25: 16/30, 40/50. Misses full-length swaps; preserved
  carrier-skip rows (len 24) false-alarm.

Earliness: damaged arms often have NO soft digit step at all (first
signal absent); when softness exists it sits at the digit positions
themselves (steps 17-22), not earlier. Carrier softening (step 12)
precedes but does not discriminate.

High-ratio collapse faces (0.25-0.5): (a) distractor looping to the
128-token budget (03-planted 0.25 repeats 6024064 eight times);
(b) confabulated justification — the model emits the wrong digits,
then fabricates prompt quotes supporting them ("as it mentions ... is:
3705", repeated 3x) and concludes with the wrong number; (c) rare-token
wander then stop. The wrong answer always comes first; the false
memory follows (autoregressive consistency pressure). The looping face
matches the corpus catastrophe family (non-termination).

## General laws (added 2026-09-18, Phase J: 44 mass records, 0 errors)

Attention focus at L22/L23 (validated method) for every damaged knorm
arm at digit onset (tA) and at its decision step (tB). All baselines
reproduce (44/44). Three families plus one variant:

- **J1 onset intactness.** At first-digit onset routing is near-identical
  (low ratios: cosine 0.93-1.00, equal needle mass) or drifted but
  undecided (high ratios: cosine ~0.73, needle mass still matched). The
  decision is always lost later, at the discriminative or tail
  positions. Mechanistic earliness limit: there is nothing to see at
  onset because nothing has gone wrong yet.
- **J2 tail-readout collapse** (truncation, wrong-number, collapse:
  every base-row arm plus 03-planted, 01-base novel digit,
  03-base transposition). Focus on the last-two digit positions
  collapses ref-to-comp (e.g. 0.29 to 0.05, 0.14 to 0.00) while total
  needle mass may stay equal or even rise. The weakened tail readout
  surrenders to the prior, which fills stop (truncation), invented
  digits (wrong-number, always mid-to-late positions), or wander/loop
  (collapse). First three digits never flip in any of the 30 damaged
  arms.
- **J3 competitor surge** (swap: all 5 planted mid-ratio swaps).
  Distractor discriminative-position mass surges (e.g. 0.043 to 0.158,
  0.007 to 0.150), with or without needle drain. The output follows
  the surge exactly at the disagreement position.
- **J4 cascade variant** (00-planted 0.1/0.25). Decision-point routing
  intact with even elevated needle focus (0.052 to 0.176); the model
  reads correctly then stops early anyway. The prior overrules a
  working readout.
- **J5 drift is global, damage is local.** Hidden cosine ref-vs-comp
  falls with ratio (tB: ~0.9 at 0.05, ~0.4-0.7 at 0.5) but damage
  always localizes to one span (last2 or disc_d). Drift magnitude does
  not predict the mode.

Structural correlates (CPU taxonomy + tokenizer geometry, same 80 arms):

- Qwen tokenizes digits singly, so each pair has exactly one
  discriminative token position: 00 at 3, 01/02 at 4, 03 at 6.
  Fragility follows it inversely: later disagreement breaks later
  (00/01 at 0.05, 02-planted at 0.1, 03-planted at 0.25, 03-base at
  0.5, 02-base never). More agreed prefix means more autoregressive
  support before the decision. Near/far-distractor plantings (H5)
  would test this directly.
- Competitor-copy errors land exactly on disagreement positions;
  novel errors land late or transpose neighbors. Carrier-skip is
  truncation-specific (9/13), not damage-general; collapse arms have
  the most confident carriers (0.91). No single upstream event flags
  damage.
- Repetition pressure at high ratio is mode-general: 02-base at 0.5
  repeats the true number 3x (scores preserved), 03-planted repeats
  the distractor 8x to budget (swap). Same dynamics, outcome decided
  by routing.

## Next hypotheses (from the cross-checks)

- **H3 (decisive, cheap):** needle-pinned Knorm on the 8 prompts. Damage
  gone: rescue works via routing restoration (reconciles 024). Damage
  stays: other evictions suffice (strengthens distributed routing).
- **H1' (corpus-only):** tiny entropy elevation (~0.03-0.05 vs 0.0) at
  answer-region tokens as a weakened-readout marker at scale.
- **H2 (corpus-only):** truncation dominance — shortened generations and
  EOS rates ref-vs-compressed on NIAH-like tasks.
- **H5 (moderate GPU):** near- vs far-distractor plantings; swap rate vs
  truncation rate as a function of competitor similarity.
- **H7 (engine extension + GPU):** evict only top-anchor sentences vs
  random spans; finds which evictions matter (090's open question).
- **H6 (schema check first):** (winner, runner-up, margin) fingerprints
  per mode as a basis for mode-specific forecaster heads.

## Limits

Qwen2.5-7B-Instruct, Knorm, ratio 0.1, NIAH, 8 runs. The 01 pair was not
probed (same truncation signature as 00, mechanism assumed shared).
Dose-response across ratios, other compressors, and other tasks are open
follow-ups, not claims made here.

## Regeneration

- The Orion pilot script (`herald-v4/scripts/run_pair_pilot.py`, not in
  this repo) with `--manifest data/113-competitor-v1/pair-manifest.json
  --actions 0.10` reproduces the run records (10s smoke check per Orion
  rules).
- Probe phases (evict, replay, span/layer ablations, eviction transplant,
  attention mass) rerun from the archived scripts against those records;
  each phase asserts exact reproduction before its measurements count.
- Bulk records stay off git per project rules; frozen derived summaries
  (damage table above, per-layer masses, ablation outcomes) are the
  citable artifacts.
