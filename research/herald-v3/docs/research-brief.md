# Starting question

## Long-term ambition

Build a controller that adjusts KV-cache compression during generation to
reduce memory consumption while keeping task-quality degradation within a
specified tolerance. This is a future objective, not the current deliverable.
Lowering compression after eviction may require reconstructing lost state;
recovery behavior and its cost must eventually be measured explicitly.

## First research objective

Determine whether information available at a generation decision point can
predict the quality consequences of candidate compression actions well
enough to inform a future choice.

The agreed target is expected final task-quality loss from a candidate
compression action, conditional on information available at the decision
point. Use the signed reference score minus the compressed score: positive
means degradation, negative means improvement. Retain both in the labels.
The quality metric, initial task, observations, and experiment remain to be
selected. Changes in token probabilities cannot be assumed to imply
task-quality loss. Controller design and unacceptable-loss thresholds are
outside the current scope.

## Questions to settle before implementation

1. What does damage mean, relative to which reference, and over what horizon?
2. Does the first experiment start from an uncompressed state, an already
   compressed state, or a precisely defined switch boundary?
3. Which candidate actions have valid, comparable continuation semantics?
4. What information is available at decision time, and what does collecting
   and evaluating it cost? Logits, hidden states, and cache information are
   candidates to assess, not an input whitelist.
5. Which existing data answer that question, and which missing comparisons
   require collection? What data remain genuinely unseen?

## Candidate minimal experiment, subject to those choices

Use a small set of prompts and decision points. For each point, independently
continue matched copies of the state under a reference action and a few
compression choices. Validate state isolation and intervention semantics
before interpreting output differences. Score completed answers; retain
causal observations at the decision boundary. If local future measurements
are collected, evaluate their connection to final task quality explicitly.

First establish that the actions produce meaningful, measurable outcome
differences. Then ask whether a predictor improves on a baseline with the
same action and clock information on unseen prompts. Select additional
baseline information to match the claimed contribution. Define useful
improvement and the evaluation procedure before opening confirmation data.

The deliverable is an interpretable feasibility result and its limitations,
including a negative result if warranted. A controller comes only after
evidence that the prediction can inform an action; its actual memory,
quality, and runtime effects require a separate closed-loop experiment.
