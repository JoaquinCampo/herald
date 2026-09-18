# The current digit error survives replacement of generated cache history

One exposed071 discovery000 diagnostic completed on Orion, exec90397 exit0.
No predictor was fitted; no new population or complete hybrid answer was generated.
Frozen protocol091 and its implementation passed independent Luna review before
execution. All exact replay, source, mask, clone, no-op and initial hybrid gates
passed. Raw logits and reports: results/b16-history-one on Orion; small JSONs
also local. Owner independently verified raw hash, parity and float64 margins.

At the first wrong token, generated index21, the full branch selects0. The actual
compressed branch and the hybrid both select period. Their raw-logit margins
m=logit(0)-logit(period), subtracted in float64, are:

| State | Margin | Selected token |
| --- | ---: | --- |
| Full reference | 26.5625 | 0 |
| Actual compressed history | -22.25 | . |
| Original eviction plus reference-written later history | -23.4375 | . |
| Full no-op | 26.5625 | 0 |

The ordered margin contrasts are50.0 for full minus hybrid and -1.1875 for hybrid
minus actual, totaling48.8125 for full minus actual. They are exact differences
of the saved logits, not probability changes or layer-wise additive attributions.
Replacing later history therefore does not repair this token decision. Under
this intervention, original eviction and its within-forward consequences suffice
to produce the wrong punctuation even with reference-written later history.
The actual later history slightly favors0 relative to the hybrid on this margin,
so accumulated history difference is not required for this particular error.
This does not prove it irrelevant in other cases or identify the causal head.

Generated-history tensors really differ:54/56 per-layer K/V tensors differ by
step21. Their absence of difference in layer1 is consistent with identical token
inputs before first-layer attention; this is a consistency observation, not a
layer-local causal attribution. Original retained K/V entries match exactly.
The initial B16 hybrid matches the actual cache and logits exactly, confirming
that the later hybrid contrast arises only after generated-history writes.

## Reporting precision correction

The initial report subtracted bfloat16 logits before converting to a Python float,
rounding reference margin26.5625 to26.5 and hybrid -23.4375 to-23.5. The raw logits
are intact. Independent owner recomputation discovered this, and the authoritative
values above and owner-audit.json use float64 subtraction. Preserve the initial
report and producer-source.py unchanged; the corrected script explicitly casts
before subtraction. Its exact saved-logit CPU reproduction passed. No GPU replay
was needed because this was reporting arithmetic, not model/state behavior.
Argmax choices, controls and qualitative interpretation are unchanged.

## Next understanding step

This removes one proposed prerequisite for000's failure. It does not establish
that a particular source digit's K/V carries a literal copy of that digit, nor
that all omission cases work this way. The relevant unresolved path is now the
current-forward response to missing original cache entries, including query and
residual changes within that forward. Inspect how contextual source K/V and
query position can favor punctuation instead of the final digit before proposing
another observation. No new predictor study or head/layer search is selected.
