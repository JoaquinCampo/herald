# Restart literature check and scope of negative evidence

2026-09-14. Targeted primary-source reading during UNDERSTAND, not a systematic
literature review or a claim that no relevant method exists.

## Sources checked live

- [The risk of KV cache compression](https://arxiv.org/html/2607.01520v1),
  Haverbeck et al., July 2026, section 3: its formal risk concerns approximation
  of one attention head across future queries. This supports studying cache/query
  geometry, but does not validate a predictor of signed final task-score loss.
- [CompilerKV](https://arxiv.org/html/2602.08686v1), Yang et al., February 2026,
  sections 3.2 and 4.4: uses attention entropy and local perplexity to inform
  compression thresholds. Its training reward is a compressed-minus-full NLL
  difference over a token window, with penalties. Its aggregate task evaluation
  does not establish that this reward predicts an individual request's signed
  task-score loss. It is relevant neighboring work missing from the prior short
  restart list, not evidence to repeat our entropy or attention feature search.
- [Compression-Aware Abstention](https://arxiv.org/html/2608.29934v1),
  Khodabandehlou and Krishnamachari, August 2026, sections 2 and 3: trains an
  adapter using evidence-survival labels constructed from supporting spans.
  This provides semantic supervision and changes model behavior. Those labels
  and abstention outputs are not the unmodified model's prospective signed
  compression loss; gold answer spans cannot become inference features.

Owner inference: these sources distinguish approximation, compression policy,
and task-grounded evidence. None of the inspected claims supplies a validated
drop-in HERALD predictor. They do not justify a new model/task/feature sweep.

## Correct interpretation of prior results

Study 046 failed the required improvement over all matched baselines. Its modest
improvement over the mean and failure against aggregate OLS mean that added
head-level detail has no demonstrated incremental value in that fixed study.
They do not imply absence of every predictive signal in the underlying cache.
Its bootstrap intervals condition on the fitted model and omit fitting uncertainty.

The prefix-disagreement bit failed its frozen usefulness gate in 020. This
limits the tested bit/model combination. A richer counterfactual representation
remains unproven; the old failure cannot establish that the entire information
source is empty. Reusing those outcomes to tune it would still be exploratory.

The future-query result in 048 is explicitly not an information upper bound.
Similarly, flat VT and MuSiQue losses are failures to produce an informative
test setting, rather than tests rejecting the existence of a predictor.

The September 7 operational stop was reasonable. Preserve every failed criterion
and exposure boundary, while distinguishing that stopping decision from an
empirical impossibility statement. A new collection needs an articulated mechanism
or new supporting evidence; the 066 audit first tests whether the strongest
remaining positive assumption deserves continued weight.
