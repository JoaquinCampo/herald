# Goal

## The problem

KV-cache compression is the dominant way to make long-context LLM inference
cheaper, but it silently degrades output quality at heavy ratios. Today this
degradation is measured after the fact, in averaged task-accuracy benchmarks
that smooth over both the magnitude and the timing of damage. Whether
compression is already making the output worse than it would have been
uncompressed is a question deployed systems answer only once decoding has
finished.

## The thesis

Compression-induced output damage has an online signature in the model's own
next-token distribution, and that signature is strong enough to forecast the
damage from cheap per-token logit statistics alone, before the damage is
visible in the produced text.

## What "damage" means in this paper

Damage is the difference between the output a user actually receives under
compression and the output they would have received without it, measured on
the same prompt. It is paired, counterfactual, and operational: compression
has caused damage when the compressed output is worse than its paired
uncompressed counterpart in a way that matters for the task the output is
being used for.

## The contribution

1. An operational, paired-counterfactual definition of compression damage.
2. A streaming online predictor that forecasts this damage from cheap
   logit features alone.
3. An evaluation that establishes the predictor transfers across
   compressors, compression ratios, tasks, and model families.