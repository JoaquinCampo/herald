# Strategic retreat after value-head study 035/036

## Current read

The frozen result is negative for the predeclared head-fraction predictor. On
the locked 48 prompts, the outcome is exactly 41 losses of 1 and 7 losses of
0, with no negative losses. The all-head Ridge has MSE 0.124375 versus the
constant mean 0.124566, only 0.153% relative improvement, 24/48 prompt wins,
and AUC 0.509. Its bootstrap MSE-improvement interval crosses zero. Metadata
and mean-retention baselines also miss every development gate.

This is evidence against this prompt-only mask representation under native
Knorm .10 on this NIAH slice. It is not evidence that value-token damage is
noncausal: 033/034 restored all eight old failures by restoring value entries.
It is also not evidence for a universal impossibility result, because the
current sample is a narrow exact-schema task with 48/48 correct references,
zero adapter misses, one length, one horizon, and a nearly binary outcome.

## Competing explanations

1. **Action saturation and metric discreteness.** Native Knorm .10 may push
   this task into an almost deterministic failure regime. Expected: a single
   predeclared milder action creates more zero outcomes or transitions while
   the current head features remain weak. Learning: calibrate the action and
   label variation before fitting any predictor.

2. **Head-wise eviction is the wrong causal observable.** Value restoration
   is sufficient at the token-region level, but per-head absent fractions do
   not encode cross-layer routing, redundancy, or which surviving values are
   actually read. Expected: materially different outputs at the same coarse
   head-fraction geometry, and weak ordering even after action calibration.
   Learning: stop treating head fractions as a locator; investigate a small
   causal/state observable only after the label assay is informative.

3. **The failure is digit copying or termination after retrieval.** The
   exposed diagnostic shows many compressed outputs begin with a correct
   answer prefix and then omit, reorder, or truncate digits. Expected: the
   compressed boundary lowers conditional likelihood for later answer digits,
   even when the prompt-derived value is known. Learning: separate damaged
   value access from decoding dynamics before choosing a predictor observable.

4. **Template and population confounding.** Four-key NIAH with one literal
   queried value is a useful controlled slice, but it can make prompt geometry
   stand in for task quality and says nothing about semantic retrieval. Expected:
   a layout or wording block changes the damage pattern without changing the
   nominal adapter rule. Learning: any later predictor claim needs a blocked,
   semantically varied population; do not open confirmation from this study.

## Selected next experiment: functional digit-copy probe

Run one exploratory, no-fit probe on 16 discovery contexts, stratified 8
positive and 8 zero losses under the already observed native Knorm .10 arm.
Keep Qwen2.5-7B, B0, length, prompt text, and the existing action fixed. At
the shared boundary, compare native and .10 compressed continuation logits
under teacher forcing for the deterministic response prefix followed by the
seven prompt-derived answer digits. The adapter may derive these digits only
from the known schema in the prompt; no reference answer or outcome enters
the probe. Do not search ratios, models, features, horizons, heads, or query
variants.

Record per-digit log probabilities, native-minus-compressed drops, the first
digit at which the drop appears, EOS-versus-digit logit margins, probe timing,
and the already recorded unforced signed loss. This isolates a mechanism; it
does not claim that teacher-forced likelihood predicts the unforced score.
The frozen 035/036 result remains unchanged and confirmation stays closed.

The probe supports the digit-copy explanation if compressed loss cases show a
larger or later-digit-specific likelihood drop than zero-loss cases, with the
same direction in the unforced examples. If forced digit likelihood is intact
but unforced outputs still fail, the mechanism is likely autoregressive
termination or token-transition dynamics, so a likelihood probe alone is not
an admissible predictor. If neither separates, treat the output-prefix audit
as descriptive and retire the exact-schema value path pending a new population.
Only after this read should action calibration at one milder removal, or a
decision-time state observable, be considered.

No confirmation study is justified yet. No literature lookup is needed for
this decision; the existing causal partition and locked evaluation provide
the controlling evidence.
