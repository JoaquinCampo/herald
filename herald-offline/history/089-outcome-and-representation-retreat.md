# Retreat after B16 JS: outcome and representation audit

The previous goal turn made progress: study087 produced verified negative evidence.
No validated predictor exists. Failed gates and exposed populations stay closed.

## Current data evidence

CPU-only audit of all32 exposed071 discovery records, saved on Orion as
results/b16-outcome-morphology.json with source-record hashes:21 damaged,
11 unchanged. Among damaged cases19 have a single numeric output differing only
by deletions, one has a digit substitution, one repeats shortened numbers.
All32 reference/action pairs terminate at EOS. Case023 changes text while keeping
the correct number and zero loss. These facts reinforce both delayed damage and
harmless divergence. They do not identify an admissible predictor, nor justify
changing exact-match scoring to make a signal look better.

Instrumented087 observation costs a median2.516 times the remaining reference
forward time; median remaining reference duration .09498s and8tokens. This is
not optimized deployment latency: _probe_logits calls compress_knorm, including
its validation/fingerprint work. No speed benefit is established. Preserve the
raw measured cost and do not label it a minimal implementation cost.
See results/b16-js-relative-cost-audit.json.

## Correcting the scope of negative evidence

Independent Luna review found no executed supervised hidden-state representation
predictor in v4. Studies013/014 use hidden states to calculate attention;035/036
fit mask-retention features. The blanket hidden-state-family closure in053/064
is therefore unsupported by the inspected executed evidence. Historical notes
remain unchanged; this correction governs future interpretation. It does not
reopen any failed scalar, representation, threshold or evaluation population.

A targeted primary source supplies a distinct rationale:
[No Answer Needed, v3](https://arxiv.org/html/2509.10625v3) studies linear
correctness probes of pre-generation residual-stream activations. Its factual
transfer results weaken the assumption that only output probabilities can carry
advance correctness information. Its mathematical-transfer limitations and
uncompressed setting do not establish numeric retrieval or compression-risk
prediction. Our inference is that signed direction in paired activations is
an untested representation, unlike scalar JS that discards direction.

## Competing hypotheses and smallest next action

1. Paired intermediate activations encode loss-relevant direction before the
   immediate argmax changes. A later fresh predictor could add value beyond
   reference-only representations and scalar perturbation baselines.
2. Activations encode ordinary prompt difficulty only. A reference-only probe
   would match any apparent candidate gain; no action-specific value follows.
3. Signals reflect template/answer-progress artifacts or overfitting. Genuine
   prompt-group isolation and training-only transformations must defeat this.
4. No useful advance signal exists at B16. Richer representations also fail;
   no layer/horizon sweep should follow a fixed negative experiment.
5. Instrumentation changes behavior. Exact raw-logit/no-op/state replay is the
   cheapest rejection test before considering more data or any learning.

Select only a one-case technical capture at the existing B16 boundary and action,
on discovery000. Capture residual output after block14 (zero-based13 of28),
chosen as architecture midpoint before measurement, for reference/noop/Knorm.1
pending-token forwards. No layer search, future token input, new continuation,
fit, predictor claim or fresh outcome exposure. Reproduce087 raw logits exactly,
source fingerprints and071 native mask; require one finite3584-vector per arm,
reference/noop vectors exact, hook removal and uninstrumented replay exact.
Keep costs and raw arrays. Passing authorizes designing a separate prospective
study only. It does not justify fitting a3584-dimensional model on32 examples.
