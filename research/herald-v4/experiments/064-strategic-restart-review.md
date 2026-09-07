# Strategic restart review

## Decision

Do not run the proposed QMSum counterfactual-lookahead experiment. The targeted
review of 015, 017, and 020 shows that a 16-token paired rollout from cloned B0
state is already an admissible, paid counterfactual observation in this project.
Its fixed prefix-disagreement bit failed the matched task/action-mean baseline:
the residual model improved MSE by only 6.82 percent, with three of four fold
wins, below the frozen 10 percent gate. No predictor was validated.

Teacher-forced token probabilities would retain more of the same rollout than
the prior one-bit summary, but that is a related feature-family extension, not a
new decision-time mechanism. Changing both the feature representation and the
task to long-form summarization would make either result hard to attribute.
There is no current evidence that graded path divergence resolves the decisive
failure: prefix differences occur for unchanged and improved outcomes, while
some degradations preserve the first 16 tokens. The right decision is a bounded
impasse, not another collection.

This does not claim that expected signed task-quality loss is unpredictable. It
closes no untested mechanism and does not alter the validated engineering status
of paired B0 states. It says only that the reviewed evidence does not justify a
new GPU experiment now.

## Corrected prior evidence

- Study 015 explicitly defined the first 16 reference and action tokens as
  information computable before the live action by simulating both arms from
  cloned boundary state. It allowed early EOS and recognized up to 16 generated
  tokens per arm plus a shared reference branch as observation cost.
- Study 017 found action and reference prefixes differed for 20 of 30
  degradations, 8 of 25 unchanged outcomes, and 1 of 5 improvements. Adding the
  bit improved grouped OOF MAE by 5.65 percent, below its 10 percent gate, and
  within-task rank association was not positive in both tasks.
- Study 020 corrected the baseline. The prefix model had MSE 0.18616 against
  0.15035 for the train-fold task/action mean. A prespecified residual fit using
  only the prefix bit reached 0.14009, a 6.82 percent gain, and still failed.
  This closes that fixed observation and model, while leaving the broader
  counterfactual-rollout family unproven rather than absent.
- The failed future-query attention oracle further lowers the prior for another
  path-discrepancy statistic, although it does not logically cover full-logit
  counterfactuals. The 50 percent privileged cross-rate gain shows stable prompt
  susceptibility, but other action outcomes remain unavailable at decision
  time and supply no replacement signal.

## Competing ways forward

1. **Richer 16-token counterfactual features.** Mean reference-token log
   probability shift, argmax mismatch fraction, and first mismatch add graded
   action-conditioned information that the bit discarded. They cost one short
   reference rollout and one compressed evaluation per candidate. They remain
   on the same reference path, so they have no demonstrated way to distinguish
   harmful divergence from harmless or beneficial divergence. Do not select.
2. **The same probe on QMSum.** A long output makes the probe a smaller fraction
   of the response and ROUGE-L provides a graded final score. This changes the
   task substrate, not the information mechanism, while adding uncertainty
   about Qwen reference quality, action variation, metric sensitivity, and the
   number of independent meeting documents. Do not select.
3. **Static prompt semantics or self-evaluation.** These observations are
   admissible, but prompt embeddings lack an established action-specific link
   and self-evaluation primarily predicts answer correctness. Both would need a
   new causal account and substantially more prompt-grouped data. Do not select.
4. **Stop collection pending a distinct mechanism.** This preserves the fixed
   target and closed evidence without claiming impossibility. Select this path.

## What would justify a later experiment

A later proposal must state what new variable becomes available before the live
action and why its sign or magnitude should track the signed change in completed
task score. It must distinguish that variable from token-path divergence,
attention reconstruction, hidden state, mask retention, generic likelihood, and
other closed families. A task or model change can support that mechanism, but it
cannot serve as the mechanism by itself.

If a future hypothesis returns to paid rollouts, it must first explain how it
addresses both observed error modes: damage after identical short prefixes and
harmless or beneficial early divergence. A prospective fresh subcase is
justified only if this explanation fixes the representation before outcomes and
the full model must beat a train-fold task/action mean plus a baseline receiving
the same reference-rollout information. Signed final task-score loss remains the
target; KL, log-probability shift, prefix agreement, and correctness confidence
remain candidate observations only.

For QMSum or any multi-query corpus, all queries, prompt variants, and actions
from the same full meeting document must stay in one split. The independent
sample size is the number of meeting documents, not the number of queries. No
discovery or sealed-evaluation size should be fixed until that group structure
and prior exposure are verified without reading outcomes.

Any short rollout must stop at actual EOS. A reference arm ending before token
16 supplies only its realized prefix, and an action arm whose next-token argmax
is EOS must not be teacher-forced beyond that termination. Report processed
tokens and wall time per arm, cache-copy and compression cost, peak memory, and
the probe-token fraction of each actual completed response. The ratio 16 to a
maximum generation ceiling is not an overhead measurement.

No new task, action, feature, model, threshold, dataset, or execution is selected
by this review. The next scientific move requires a genuinely distinct
decision-time information mechanism or new external evidence supporting one.
