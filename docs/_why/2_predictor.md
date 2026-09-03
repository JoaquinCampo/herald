# What the predictor predicts, and why

> **Status: current protocol.** This document supersedes the earlier
> compressor-agnostic transfer design. The current paper trains and evaluates
> one pre-switch magnitude forecaster per known compressor.

## The prediction target

At switch position $s$, exactly $s$ generated tokens are fixed and compression
has not yet activated. For known compressor $c$ and known removal ratio $r$,
the predictor emits

$$
\hat d_c(s,r),
$$

an estimate of the signed final task-quality effect of activating $c$ at $s$
and continuing under compression to completion. The measured target is the
paired live-fork or matched-reprefill contrast defined in
`docs/_why/6_intervention_semantics.md`.

This is not a post-switch reaction monitor. It uses no compressed tokens,
grace window, probe continuation, rollback outcome, or future information.

## Why the target varies by switch position

A run-level target would ask the predictor for the same value at every
position and could not localize when the generation is vulnerable. Switching
at $s$ and switching sixteen tokens later are distinct interventions with
distinct paired continuations. The measured switch curve supplies one
supervised magnitude at each sampled decision boundary.

The target is activation now through completion, not damage over a fixed
number of future tokens. The serving decision is whether activating the known
compressor now will harm the completed answer. A fixed horizon would introduce
another policy choice and answer a different question.

## Why one model per compressor

Different compressors remove different cache entries and produced
compressor-dependent label distributions in the historical experiments. The
current paper does not require one invariant mapping to transfer to an unseen
compressor. Instead, each compressor gets its own model and must support its
own claim on unseen prompts.

Compressor identity is therefore implicit in model selection rather than a
feature column. A pooled or macro result cannot rescue a compressor whose
model fails.

## Why ratios are pooled

Removal ratio is a numeric intervention setting known before activation.
Pooling a compressor's supported ratios gives its model more prompt-level
evidence and asks it to learn dose dependence. The ratio is an explicit input,
and every result is also reported separately by ratio so aggregate performance
cannot hide a failed setting.

This is interpolation over ratios represented during training, not a claim
about an unseen ratio. A separate model per ratio is an ablation only if the
pooled model's diagnostics justify it.

## Which inputs are admissible

The first model may use only information available at the decision boundary:

- the known removal ratio;
- absolute generated position $s$;
- causal reference-stream `feat__*` values observed through $s$.

It may not use final reference length, relative position computed from that
length, completed-output quality, labels, prompt identity, compressed probes,
hybrid-stream summaries, grace-window features, or any other post-switch
quantity. Feature timing must be checked against the exact boundary in the
generation loop; an off-by-one row is leakage.

Compressor-specific pre-switch measurements may be studied later only after a
fixed first model fails and the missing information is identified. They must
be available before activation and receive a separate cost and causality
analysis.

## Why signed scalar regression

The primary label retains magnitude and sign. Positive values are degradation,
zero is no measured effect, and negative values are lift. Clipping before
training would change the estimand and allow improvements to disappear from
the evidence.

One fixed squared-error XGBoost model is the first learned candidate. It must
beat grouped ratio-and-position baselines on held-out prompts before any broad
model search. Mean-squared skill is primary because the target is heavily
zero-inflated and median prediction can look strong under MAE without
forecasting graded effects.

Positive harm, any-damage risk, and major-damage risk are derived reporting
targets. Risk prediction is a preregistered secondary endpoint with thresholds
fixed before primary results are inspected; it is not a post-hoc replacement
for failed magnitude regression.

## Why the streaming constraint remains strict

The forecaster may use quantities already produced by the uncompressed
generation pass and a constant amount of causal aggregation per token. It may
not run an additional model forward pass. This preserves the claim that damage
is forecast before compression from the model state already available to the
serving system.
