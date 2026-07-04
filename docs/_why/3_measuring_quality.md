# How we measure quality, and why this way

**Status: provisional.** This decision is the current working plan. The
dense per-position labelling in particular is gated on a pilot
(described at the end) that has not yet run. If the pilot fails, the
dense target changes and this document is revised.

Damage is a quality delta between the paired runs, computed offline on
completed outputs: $\text{damage}(s) = q(\text{reference}) -
q(\text{hybrid at } s)$. Everything here concerns how $q$, the quality
of a finished output, is measured.

## Why a quality delta, not a difference

An earlier plan used the cosine distance between sentence embeddings of
the two outputs as the damage signal. We dropped it. Cosine measures
how *different* two outputs are, not how much *worse* one is. A direct
probe of the intended encoder showed the failure is not a matter of
encoder quality but structural: a one-digit change to a final answer
(real damage) lands far closer to the gold solution than a
meaning-preserving paraphrase (no damage), with the ordering inverted
on essentially every item, and the same for a one-character code bug
versus a correct re-implementation. Paraphrase-invariance and
error-sensitivity are opposite demands on one geometry, so no encoder
resolves both. A measure of difference cannot serve as a measure of
damage. The probe is retained as evidence.

Quality delta avoids this by construction: it compares how *good* each
output is, not how *similar* they are. The paired counterfactual and
greedy decoding make the delta attributable to compression.

## Why quality is measured offline, never mid-generation

The checks that define quality (running unit tests, checking a final
answer, reading the whole output) only make sense on a finished output.
This is not a limitation, because the measurement and the prediction
happen at different times. We measure damage offline, after each run
completes, to build the training labels. The predictor is the only
thing that runs online, and it never measures quality; it forecasts the
offline-measured quantity from the logits seen so far. So a task whose
quality can only be scored after generation (all of them) poses no
problem for the measure.

## Why quality is composite, not one instrument

No single instrument measures quality well across the board.

- Where correctness is mechanically verifiable, the exact check is the
  ideal measure of that correctness, not a weak proxy. A right GSM8K
  answer is right; code that passes its tests works. A model reading
  the output adds nothing to these verdicts.
- What the exact checks cannot see is graded and within-verdict
  quality: a correct answer reached through garbled reasoning, a wrong
  answer that degraded into gibberish, partial degradation on free-form
  outputs. Only something that reads and understands the text can score
  these.

So the composite uses each where it is reliable: exact checks for the
correctness they verify, and a judge for the graded residual they are
blind to. This is stronger than judging everything (it keeps the exact
parts exact) and stronger than checks alone (it sees the residual).

It also corrects a misframing we worked through. The exact checks are
not "cheap and therefore untrustworthy." They are exact ground truth
where they apply. Their only weakness is resolution on within-verdict
quality, and that weakness is precisely what the judge covers.

## Why the judge is the right instrument for the residual

Measuring graded quality means understanding the output, and the
ideal reader is a competent human applying a rubric (correctness,
completeness, faithfulness, coherence, instruction adherence),
comparing the two outputs pairwise rather than scoring each in
isolation, because comparison is what human judgement is reliable at.
The realizable, self-hosted form of that ideal is a strong open-weight
model performing the same comparison, calibrated against a
human-labelled anchor set so its agreement with human judgement is a
reported number rather than an assumption. This judge measures quality
by reading the output; it is not validating a difference metric, which
is the misuse we rejected with cosine.

## Why the measure and the training target are separated

The same quality delta serves two purposes with very different costs.

The reported **measure** (how much compression degrades output) needs
one comparison per run, the fully compressed output against its
reference. That is affordable across the full sweep and can use the
judge directly.

The predictor's **training target** needs a damage value at every
switch position of every run, on the order of a dozen per run, because
the entire point of a per-position target (rather than the run-level
target we already rejected) is that damage depends on *where*
compression activates. That density is a different cost class, and
because this is training data, its quality directly caps the final
model. Keeping the two separate lets the measure use the judge freely
while the target is designed under its own constraints.

## Why a pilot gates dense judging

Dense judging has a failure mode that has nothing to do with cost. The
target needs damage at adjacent switch positions to differ (compression
at one position versus sixteen tokens later), but those two outputs
differ by only a few tokens, exactly the regime where a graded judge is
noisiest. If the judge's own variation when asked the same question
twice is larger than the real difference across positions within a run,
dense judging produces a noisy copy of the single run-level number, the
thing we already discarded, at many times the cost.

So before committing the training set to dense judging, a pilot
densely judges a handful of full curves, measures the judge's
test-retest wobble on identical pairs, and checks that the real
across-position spread within a run exceeds that wobble. Only then is
dense judging worth doing. The pilot also yields the true per-label
cost, which sizes the rest of the experiment. Under greedy decoding a
large fraction of late-switch hybrids are byte-identical to their
reference and are exact zeros by string comparison, needing no judge
call, so the real judging workload is far below a dozen per run.
