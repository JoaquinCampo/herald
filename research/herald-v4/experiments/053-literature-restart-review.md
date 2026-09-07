# Literature restart review after action identifiability

## Decision

The bounded review found no primary work that predicts, for a new request and
candidate KV-compression action, the signed change in final task score using
only information available before that action. Current work supports three
nearby claims: cache or attention approximation can be bounded, compressors can
be ranked by aggregate task performance, and correctness or abstention can be
estimated after extra generation or model adaptation. None is the HERALD
estimand.

Study052 changes one assumption only. The same prompt's losses are stable across
Knorm rates in a small exposed NIAH population, which supports stable prompt differences in those observed outcomes. It does not reveal an admissible prompt feature. The
closed NIAH probe, mask, hidden-state, and functional-feature families remain
closed.

## Primary sources and exact relevance

1. [Haverbeck et al., *The risk of KV cache compression*](https://arxiv.org/abs/2607.01520)
   characterize minimax single-head attention approximation risk under query
   distributions and report targeted LongBench results. Their risk is not
   signed final task-quality loss and is not a per-request predictor of it.
2. [Luo et al., *How Query Visibility Changes KV-Cache Compression Rankings*](https://arxiv.org/abs/2607.11942)
   hold model, budget, instances, and decoding fixed while comparing query-aware
   and query-agnostic compression. They show that the decision-time information
   boundary can reverse method rankings. Their final benchmark scores audit
   methods in aggregate rather than predict a new prompt's action-specific loss.
3. [Devoto et al., *Expected Attention*](https://arxiv.org/abs/2510.00636)
   estimate future attention from an activation distribution and use it to rank
   KV entries. This is a mature training-free compressor available in KVPress,
   but attention preservation is an intervention rule, not a calibrated final
   quality-risk estimate.
4. [Khodabandehlou and Krishnamachari, *Compression-Aware Abstention*](https://arxiv.org/abs/2608.29934)
   train a LoRA adapter from evidence-survival labels and evaluate actual
   compressed-cache decoding on MuSiQue. It predicts an answer-versus-abstain
   behavior after changing the model; it does not estimate signed counterfactual
   task loss before choosing an action.
5. [Kadavath et al., *Language Models (Mostly) Know What They Know*](https://arxiv.org/abs/2207.05221)
   predict whether a model-generated answer is correct using self-evaluation.
   This supports output-side confidence as a mature neighboring method, but its
   candidate answer is future information at HERALD's pre-action boundary and it
   does not isolate compression-induced loss.

Next-token KL or JS, attention reconstruction error, cache compressibility, and
perplexity may be useful observations. They become evidence for HERALD only if a
prospective paired study shows that they improve prediction of signed completed
task-score loss over an information-matched baseline. None can substitute for
that outcome.

## One different practical direction

Test a joint prompt/action response model on graded multi-hop QA, not another
NIAH proxy. Use MuSiQue two-hop questions with token-F1 final quality and the
published Expected Attention implementation at three fixed budgets. This changes
the benchmark mechanics, outcome granularity, and compressor family while using
mature components.

Set the decision point after the complete context and question are available and
before answer decoding. Admissible inputs are the literal prompt, task identity,
native token length, decision position, compressor identity, and candidate
budget. Exclude compressed continuations, action logits, future queries,
reference answers, gold evidence spans, survival labels, and any outcome from
another action on the same prompt. Record the cost of feature extraction and
candidate scoring.

Use a fixed stateless word-and-character HashingVectorizer on the prompt, then a
single regularized linear model with candidate budget and prompt-by-budget
interactions. Fit one model across all actions. This is a prospective test of
whether ordinary prompt content carries transferable susceptibility, not a new
cache-derived proxy search. Freeze hashing dimensions, normalization, penalty,
and feature order before outcomes. Do not tune learners or promote subgroups.

The primary matched baseline is the discovery task-by-action mean, because it
has the same task, action, and decision-clock information. Also report a
structural baseline with task, action, token length, and decision position. Keep
all prompt variants and all actions in one prompt-grouped split; retain imperfect
references, zero losses, and signed improvements.

A small acceptance slice is 48 discovery and 48 locked evaluation questions,
each run under one uncompressed reference and all three predeclared actions. Fit
only if discovery contains usable graded variation, without filtering cases.
Freeze the model before opening evaluation outcomes. Preserve the existing
prediction standard: at least 10% lower MSE and at least 32 of 48 strict
prompt-level squared-error wins against each matched baseline. Because every
prompt has three actions, also require within-prompt action concordance at least
0.65 and at least 0.10 above the structural baseline. A pass warrants untouched
confirmation only; a failure closes this text-based joint model without feature,
budget, learner, or task-subgroup search.

This direction is justified as a prospective design, not selected for execution.
Its strongest limitation is external validity: even a pass would establish
prediction only for the frozen model, compressor, decision boundary, and QA
population until separately confirmed.

## Owner decision before execution

The benchmark and compressor are candidates for a small feasibility pilot. The
text-hashing predictor is not selected:48 prompt groups do not yet justify a
high-dimensional interaction model, and a fair comparison would also need a
text-only additive baseline to isolate action-specific interactions. First check
real QA competence, score variation and independent-state EA intervention
semantics. No48+48 prediction collection or model fit is authorized by this note;
the owner's existing autonomous authority remains unchanged.
