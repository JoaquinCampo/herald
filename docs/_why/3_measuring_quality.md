# How we measure quality, and why

> **Status: current initial protocol.** IFEval instruction-level loose
> accuracy is the implemented primary quality measure. Strict IFEval is a
> planned robustness analysis. No LLM-judge score is part of the current
> training label.

## Quality is measured on completed paired outputs

For each validated intervention at switch position $s$, quality is scored
offline after both branches complete:

$$
d_{c,r}(s)=q(y_{\mathrm{control}})-q(y_{\mathrm{compressed}}).
$$

The predictor never observes either completed output or either quality score.
It forecasts this offline-measured quantity using only causal information
available before compression.

Which paired intervention produces the two outputs is load-bearing. Live-state
forks are preferred when the compressor supports them; otherwise the protocol
uses a matched no-press and pressed re-prefill. The classification and parity
requirements are in `docs/_why/6_intervention_semantics.md`.

## Primary IFEval quality

For the initial paper,

$$
q(y)
=
\frac{\text{number of prompt instructions satisfied by }y}
     {\text{number of instructions in the prompt}}.
$$

The implementation uses IFEval instruction-level **loose** accuracy: an
instruction is satisfied when any official loose response transformation
passes its checker. The score is deterministic, mechanically verifiable, and
graded whenever a prompt contains multiple instructions.

Example: if the uncompressed branch satisfies three of three instructions and
the compressed branch satisfies two, then $d=1-2/3=1/3$.

## Why task quality, not output difference

Different tokens or distributions are not necessarily damage. A paraphrase
can be harmless, while one changed digit or violated formatting constraint can
destroy task success. Token mismatch, lexical overlap, KL or JS divergence,
embedding distance, entropy, and perplexity measure change, not whether the
user received a worse answer.

Those quantities may be predictor inputs or diagnostics. The label remains a
difference in task-relevant final quality.

## Why the label remains signed

- $d>0$: compression degraded measured quality.
- $d=0$: no measured quality effect.
- $d<0$: the compressed branch scored better.

Negative values are retained during training and primary evaluation. Positive
harm $d^+=\max(0,d)$ is reported separately. Averaging signed effects and
averaging positive harm answer different questions, so neither substitutes
for the other.

Any-damage and major-damage indicators are derived secondary outcomes. Their
thresholds must be preregistered before primary magnitude results are
inspected.

## Strict scoring and broader quality are separate work

Strict IFEval scoring is not currently implemented as a dataset label. Adding
the official strict variant is a planned robustness check and requires its own
versioned scorer output.

An LLM or human pairwise judge could measure residual qualities that IFEval
does not capture, such as coherence within equally scoring outputs. That would
be a new measurement instrument requiring a frozen rubric, test-retest
analysis, order-bias checks, and human calibration. It is not silently mixed
with the mechanical IFEval target.

The initial claim must therefore be phrased precisely as forecasting final
IFEval instruction-following degradation, not universal prose quality.

## Provenance requirements

Before historical labels are reused, record and verify:

- scorer implementation and version;
- prompt instruction IDs and checker arguments;
- exact output text and token IDs for both branches;
- deterministic decoding and generation budget;
- whether the stored ratio is removed or retained fraction;
- equality of stored `dq` and recomputed
  `q_control - q_compressed`;
- intervention and sham-parity status for the compressor.

If these facts cannot be established, the label is not paper evidence and must
be regenerated under the frozen protocol.
