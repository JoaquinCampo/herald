# HERALD v4 strategy, initial landscape and smallest test

## Scope and facts

The estimand is signed final loss, `D = score(reference) - score(action)`.
Positive means compression degraded the final answer; negative means it improved
it. A feature is admissible only if it is available at the compression decision
boundary. The first study should use one model, one physically revalidated
compression action, one fixed boundary and one fixed retention level. This
isolates prompt-to-prompt predictability before asking whether a predictor
generalizes across actions.

The literature establishes several different mechanisms, but it does not
establish a predictor of signed final task loss. H2O reports heavy hitters and
recency; StreamingLLM reports attention sinks; SnapKV uses an end-of-prompt
observation window; PyramidKV reports layer-wise information funneling; Ada-KV
derives a bound on attention-output change; Expected Attention estimates future
query attention without materializing the full attention matrix; and InfoKV
argues that predictive uncertainty captures long-range influence. These are
mechanistic claims or compression results, not evidence that any one signal
predicts HERALD's signed final loss on unseen prompts.

The current project state makes the old IFEval singleton population unavailable
for confirmation. V3's immediate and delayed logit features are negative,
retrieval is untested, and none of those results imply that all other
representations are uninformative. Do not claim that any proposed population is
absent from model pretraining. Fresh here means unseen by this project and
sealed before confirmation, not text that the model could never have observed.

## Competing hypotheses

### H1, future-query salience is the main predictor

**Measurement.** At the boundary, compute an action-specific summary of how
retained versus removed KVs are expected to receive future attention, such as
head-wise expected-attention mass or a closely matched salience score.

**Meaning and cost.** It measures anticipated future routing demand, which is
closer to the causal use of a KV than current-token logits. It requires a cache
scan and per-head reductions, so it is decision-time available but materially
more expensive than length or entropy metadata. Expected Attention specifically
targets this future-query problem without requiring the full attention matrix.

**Smallest distinguisher.** On the same prompts and action, compare cross-fitted
salience-only prediction with a prompt/action metadata baseline. A positive
result must remain after prompt-grouped splitting and after matching for context
length. Compare the salience score with its future-attention oracle only as an
audit, never as an admissible feature.

### H2, predictive uncertainty identifies information-bearing tokens

**Measurement.** Use boundary next-token entropy, top-k mass, and a compact
summary of entropy over the available prompt/decode positions. If logits already
exist, this is nearly free; computing additional positions or a short probe
rollout has a real forward-pass cost and is a separate feature family.

**Meaning and cost.** The hypothesis is that uncertain or information-rich
tokens affect distant future behavior even when their current attention is low.
InfoKV makes this forward-influence claim, while the proposed test asks whether
the cheap boundary signal predicts final signed loss before compression.

**Smallest distinguisher.** Compare entropy-only, salience-only and their
pre-registered combination at the same action and boundary. If entropy adds no
out-of-fold reduction beyond salience and metadata, this mechanism is not useful
for the first predictor, even if it correlates with ordinary model difficulty.

### H3, local attention-output perturbation predicts downstream loss

**Measurement.** For the available boundary query or prompt-query set, estimate
the norm and direction of the attention output change caused by the candidate
compression, including separate key and value contributions where possible.

**Meaning and cost.** This is an intervention-proximal quantity: it measures
immediate residual disturbance, not future quality. It is available before the
action but needs an extra attention-style reduction or local simulation, making
it more expensive than metadata and usually cheaper than generating a reference
continuation. Ada-KV's attention-output loss bound motivates this family.

**Smallest distinguisher.** Fit the perturbation summary alone, then test whether
it beats H1 and H2 on prompt-equal MSE and preserves the sign of improvements.
Use a deliberately different action severity in development only to check that
the score tracks dose; do not tune the sealed confirmation on that result.

### H4, position and prompt structure dominate semantic measurements

**Measurement.** Use only decision-time structure: context length, switch
position, distance to the end, token-position bands, segment boundaries,
attention-sink indicators, and retained/removed recency statistics.

**Meaning and cost.** It tests whether compression damage is largely a routing
geometry effect, such as sinks, recency or layer-wise funneling, rather than a
content-specific effect. These features are almost free and always available,
but can predict only systematic structure unless content is encoded indirectly.
StreamingLLM and PyramidKV provide the mechanistic motivation.

**Smallest distinguisher.** Treat this as the mandatory matched baseline. If
H1-H3 do not beat it out of fold, the project should stop adding expensive
signals and first revisit the action, boundary, or estimand. If structure wins,
test portability across lengths and task families before interpreting it as a
general semantic predictor.

## Recommended smallest acceptance slice

Use a sealed, project-new RULER population generated from held-out seeds, with
four task families represented: multi-needle retrieval, multi-hop tracing,
aggregation and a non-retrieval control. Stratify three context lengths and
keep all variants of one prompt in one split. A practical first slice is 240
instances, 120 training, 60 development and 60 sealed confirmation, with the
same reference/action boundary and exact task score for both continuations.
RULER is preferable to a judge-scored benchmark for this first test because its
task configuration and exact answers are controlled. HELMET or LongBench v2
should be a later external confirmation, not a tuning source.

Before opening confirmation, register: raw prompt-equal MSE as primary, a
matched metadata-only baseline, a 10% relative MSE improvement target, a 95%
prompt bootstrap interval for the MSE difference, sign accuracy, and interval
coverage. Preserve unscorable or failed continuations. Validate that paired
reference and action states are equivalent at the boundary and that the action
has a measurable physical cache effect.

The first acceptance result is useful only if one admissible family lowers
out-of-fold MSE against metadata, retains calibration on the sealed set, and
does so at a measured cost that is plausible relative to the compression
decision. If all families fail, redirect to understanding the intervention and
population, not to a larger feature search. If structure alone wins, test
length/task portability. If salience or perturbation wins, add the cheapest
component that improves the held-out result, then test a second action. If
entropy adds independent signal, reserve it for a later long-horizon study.

## Primary sources

- H2O, https://arxiv.org/abs/2306.14048
- StreamingLLM, https://arxiv.org/abs/2309.17453
- SnapKV, https://arxiv.org/abs/2404.14469
- PyramidKV, https://arxiv.org/abs/2406.02069
- Ada-KV, https://arxiv.org/abs/2407.11550
- RULER, https://arxiv.org/abs/2404.06654
- HELMET, https://arxiv.org/abs/2410.02694
- LongBench v2, https://arxiv.org/abs/2412.15204
- Expected Attention, https://arxiv.org/abs/2510.00636
- InfoKV, https://arxiv.org/abs/2606.26875
