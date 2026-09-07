# Next diagnostic: needle-span rescue

## Decision

Do one small mechanistic diagnostic before any new predictor fit or confirmation
population. Reuse four fixed NIAH development cases, selected from generator
answer-position metadata after development exposure but before this diagnostic:
IDs `000`, `004`, `011`, and `001`, covering early, middle, and late positions
and both observed prompt-length regimes. This is a causal probe of the
compression mechanism, not a new unseen-population evaluation.

## Competing explanations

1. **Sparse-head signal:** global mean EA excess hides evidence concentrated in
   a small set of layers or heads.
2. **Query mismatch:** mean prefill queries do not represent the later retrieval
   query that produces the answer.
3. **Needle damage:** Knorm removes the answer-bearing context span, so the
   aggregate feature is weak because it measures the wrong object.
4. **Replay instability:** small fitted results reflect nondeterministic masks,
   generation, or implementation state rather than a stable mechanism.

## Minimal experiment

Use the existing Qwen checkpoint, tokenizer, prompt rendering, shared boundary,
seed, and official RULER scorer. Work at removal fraction `.10` only. Locate the
answer-bearing needle span by tokenizing the actual rendered prompt and record
the generator position as a provenance cross-check, rather than using it to
select cases.

For each case, create three equal-size cache branches from the same boundary:

- standard Knorm `.10`;
- **oracle span rescue**, force-retain every token in the known needle span and
  evict the same number of retained non-span, non-sink tokens that Knorm would
  otherwise retain;
- a matched control that adds the same per-head number of nearest
  needle-adjacent, baseline-excluded, non-needle, non-sink positions.

Both rescue and control evict the exact same lowest-priority retained
non-needle, non-sink positions and fill those sorted vacated slots, so they
have the same cardinality, victim set, and perturbation count. If the
missing-span count is zero for a head, both branches equal the standard mask
for that head.

Preserve sink handling, per-layer/head lengths, cache isolation, and all existing
physical and no-op gates. Persist the actual retained indices, span-retention
fractions by layer/head, continuation tokens, official score, and signed loss.
Repeat the standard branch once with the same seed and compare cache, mask, and
continuation hashes. Do not fit a model, tune a threshold, or use continuation
tokens to define a feature or branch.

## Interpretation gate

- Rescue recovering the answer while the matched control does not supports
  needle damage. It does not validate an oracle feature for deployment.
- Rescue helping while the global mean remains weak is consistent with sparse
  evidence, but this all-head oracle cannot identify a causal head or layer
  subset; report only aggregate rescue versus control behavior.
- Standard retention of the needle plus failed rescue, or rescue and control
  behaving alike, weakens the needle-damage explanation and points toward query
  mismatch or an insufficiently causal span definition.
- Any exact replay mismatch supports replay instability and blocks mechanistic
  interpretation until resolved; an exact replay weakens that explanation.

Stop after this four-case diagnostic. A rescue result can justify a separately
designed hypothesis, but it cannot reopen EA predictor fitting or confirmation
without a new prespecified design.
