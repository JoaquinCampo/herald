# Goal

## The problem

KV-cache compression makes long-context LLM inference cheaper, but it can
silently make the answer being generated worse than it would have been with
the full cache. Today that harm is measured after the fact, by averaging
benchmark scores over completed outputs. A serving system running compression
needs a different answer: while this generation is decoding under an active
compressor, is it going to come out damaged, and can we tell early enough to
do something about it?

## The thesis

While decoding under a known compressor at a known ratio, the risk that this
generation's final task quality ends lower than its uncompressed counterpart
is visible in causal, zero-cost statistics of the model's own next-token
distributions, early enough in the generation to be actionable.

This is a current-state claim: compression is already active and stays active.
Forecasting what would happen if compression were activated on a full cache is
a separate, prospective problem and is not an objective of this paper.

## What "damage" means in this paper

Damage is task-grounded and compression-attributable. For a prompt, the
uncompressed reference run and a compressed run are both scored on the task.
The run is damaged when the compressed run's final quality is strictly lower
than the reference run's. A wrong answer the model would have gotten wrong
anyway is not damage; different tokens, distributions, or wording are not
damage unless final task quality is lower.

The primary endpoint is risk, not magnitude: the probability that a run ends
damaged, forecast at each decoding step from information available at that
step. Divergence between compressed and uncompressed distributions is at most
a feature; it is never the claim.

## Contribution

1. A current-state, compression-attributable operational definition of
   quality damage, with a per-step risk target that is well-defined
   throughout a compressed generation.
2. A causal risk forecaster over zero-cost logit statistics, confirmed
   one-shot on quarantined prompts against a comparator that knows the
   compression action and position but not the stream.
3. An earliness analysis: how many tokens into a generation the forecast
   becomes reliable, overall and for catastrophic versus graceful damage.
4. A reproducible account of where the forecast holds and where it degrades,
   reported separately by task, compressor, and ratio.

## Protocol and status

The locked protocol is `docs/implementation/quality_risk_protocol.md`. The
earlier pre-switch magnitude study (`docs/methodology.md`) and the
current-state divergence forecaster (`results/recovered/current-state-damage-v1/`)
are retained as research records; neither defines this paper's claim.
