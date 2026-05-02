# HERALD Ultimate Goal

The unconstrained vision. This document captures what the work is
ultimately *for*, independent of what the first paper can fit. The
staged plan in `research-plan.md` is a path toward this, not a
replacement for it.

When in doubt during execution, the question to ask is: "does this
move us closer to the ultimate goal, or does it optimize for a local
metric that the ultimate goal would not care about?"

## The Goal in One Sentence

Make KV-cache compression a controllable cost-quality knob, by
predicting compression-attributable damage from cheap online signals
during generation, so that any LLM serving system can dynamically
trade compute for quality with calibrated, well-understood risk.

## Why This Matters

KV-cache compression is currently a static decision: pick a press,
pick a ratio, accept the resulting average quality loss. There is no
way to know, at the point of generation, whether the current
compression configuration is about to break this particular request,
and no way to act on that knowledge. The result is that production
systems either over-compress (and degrade silently on hard prompts)
or under-compress (and pay for headroom they rarely use).

A successful HERALD turns compression from a static knob into a
closed-loop controller. The serving system observes the model's own
logit signals, predicts whether the current compression is degrading
*this* generation, and intervenes before the damage manifests in
user-visible output. The cost-quality frontier moves outward.

## The Three Pillars (Maximal Form)

### 1. Measurement Methodology

A rigorous, reproducible framework for evaluating compression damage,
adopted by the field as the standard way to compare compression
methods.

- **Multi-resolution damage hierarchy**: token-level approximation
  error (matched-prefix replay), trajectory divergence (NLL ratios,
  first-divergence-points), sequence-level semantic distance
  (BERTScore, embedding cosine), task-level outcome change (paired
  accuracy with all four cells), human judgment on calibrated
  rubrics.
- **Cross-resolution alignment**: the alignment matrix shows when
  cheap proxies are faithful to expensive ground truth and where the
  proxies break. This is what lets practitioners pick the right
  metric for their use case.
- **A released benchmark**: matched-prefix replay data, all metrics,
  all tags, across the full grid. The artifact a future paper can
  use as ground truth without redoing the data collection.

### 2. Predictability of Compression Damage

A predictor that, given only signals available during compressed
generation, forecasts compression-attributable damage at multiple
horizons, with calibrated risk and quantified lead time, generalizing
across models, tasks, and compression methods.

- **Calibrated multi-horizon risk**: not "will it fail" but "with
  what probability and severity in the next 5 / 10 / 25 / 50
  tokens." Reliability diagrams, ECE, conformal intervals where
  applicable.
- **Lead time analysis**: the distribution of "predicted onset minus
  actual onset" across runs. Lead time is what makes the predictor
  useful, not just accurate.
- **Generalization across the full matrix**: held-out model family,
  held-out task, held-out compression method, held-out ratio. The
  predictor that only works on the training distribution is not the
  goal.
- **Mechanistic story**: why the prediction works. Which features
  carry the signal. How the signal relates to the underlying KV
  eviction dynamics. The point is not just "XGBoost beats entropy
  threshold" but "here is what compression damage looks like at the
  logit level, and here is why it is detectable."
- **Counterfactual target discipline**: the predictor is trained to
  forecast compression-attributable damage relative to the paired
  uncompressed baseline, not to detect generic abnormal text. Looping
  and non-termination are diagnostic failure modes; the primary target
  is outcome, sequence, or trajectory damage caused by compression.

### 3. Closed-Loop Control

Predictor-guided dynamic compression that demonstrably improves the
cost-quality Pareto frontier across compression methods and
deployment regimes.

- **Multiple compression methods**: mask-based (StreamingLLM),
  eviction-based (SnapKV, ExpectedAttention), score-based, with the
  controller adapted to each press's intervention vocabulary.
- **Controllability before control**: before training a full
  controller, explicitly measure whether a generation can recover
  after an intervention and which interventions are meaningful for
  each press. Reactive control, anticipatory control, and oracle
  recompute fallback are different regimes and must not be conflated.
- **Per-segment and per-token gating**: deployable per-segment as the
  practical controller, per-token as the oracle upper bound on what
  responsiveness buys.
- **Pareto curves vs every meaningful baseline**: fixed compression,
  no compression, random gating at matched compute (the load-bearing
  control), oracle gating with ground-truth labels (upper bound).
- **Production realism**: latency budget for the predictor itself,
  memory overhead, integration cost. The controller has to be cheap
  enough to deploy.

## Dataset Scope (Unconstrained Vision)

If compute and time were not constraints:

- **6 models**: 3 families (Llama, Qwen, Mistral) at 2 sizes each
  (~7-8B, ~13-14B), plus one base (non-instruction-tuned) variant of
  one family for instruction-tuning ablation.
- **6+ tasks** across orthogonal generation regimes: math reasoning
  (GSM8K, MATH), code (HumanEval, MBPP), instruction following
  (IFEval), knowledge (MMLU subset), long-context QA (LongBench,
  NarrativeQA), open-ended (MT-Bench or similar judge-scored).
- **Full press matrix** plus a random-eviction baseline.
- **Dense ratio sweep** concentrated near the high-compression cliff.
- **Multi-seed**: 5 seeds per cell where stochasticity matters
  (sampling-based decoding, prompt perturbations for greedy).
- **Paired uncompressed run** for every prompt, amortized.
- **Matched-prefix replay** at every position for full-trace runs on
  a 10% sub-sample, every 4-8 tokens for the rest, with dense
  triggered windows for analysis.
- **Human-annotated severity slice**: 1-2k examples with paid
  annotators on a calibrated rubric (looping, drift, instruction
  amnesia, format break, semantic error). LLM-judge calibrated
  against the human slice for scaling.
- **Calibration sets** for noise floors at every level (numerical
  precision, semantic equivalence, sampling stochasticity).

The full dataset is closer to a benchmark project than a single
paper. That is the point: it is the artifact the field uses for
years.

## Predictor Scope (Unconstrained Vision)

- **Streaming sequence model** (small Transformer or Mamba, ~50-200k
  params) over the per-token feature stream, conditioned on
  (press_id, ratio).
- **Multi-task multi-target**: simultaneously predicts matched-prefix
  JS at multiple horizons, sequence BERTScore at completion, task
  outcome, judge severity. Multi-task regularization plus a richer
  output that can support different downstream uses.
- **Tier 0, 1, and 2 features**: zero-cost logit signals, cheap
  derived signals, and model internals (attention entropy, hidden
  state dynamics) for the ablation that defends the "lightweight"
  claim.
- **Conformal wrapper** with task-conditional calibration where it
  can be made to hold.
- **Sub-millisecond per-token inference cost** so it is deployable
  on the same GPU as generation.

## Closed-Loop Scope (Unconstrained Vision)

- **All major compression methods supported**, with method-specific
  intervention policies (toggle, ratio reduction, recompute fallback,
  KV restoration where the press allows).
- **Press-specific action vocabularies**: the controller's action
  space is not universal. Mask-based methods can often be toggled or
  relaxed mid-generation; continuous eviction methods may allow
  "stop further eviction"; prompt-time eviction methods may require
  reprefill or recompute fallback. The action space must be measured,
  not assumed.
- **Per-token, per-segment, and per-request gating** evaluated and
  compared.
- **Production deployment study**: latency, throughput, cost on real
  serving infrastructure, not just offline simulation.
- **Multi-objective control**: trade off compute, quality, latency,
  and tail risk simultaneously. Not just "recover quality" but
  "recover quality subject to a latency SLA."

## What "Done" Looks Like

The work is fully realized when:

1. The matched-prefix replay framework is the standard way the field
   evaluates compression damage. New compression methods report on
   it.
2. The predictor (or its successors built on the released dataset)
   is integrated into at least one production serving system, and
   the cost-quality improvement is measured in the wild.
3. The community has a shared understanding of *which* compression
   damage is predictable, *which* is not, and *why*. The negative
   results matter as much as the positive ones.

## What Makes It A True Contribution

The contribution is not the broad observation that logits or entropy
contain quality information. That is already known. HERALD earns its
place only if it validates the narrower claim documented in
`contribution-validation.md`: future compression-attributable damage
can be forecast from cheap online signals well enough to generalize
across meaningful axes and improve intervention decisions.

The ultimate version therefore requires:

- Strong baselines beaten, especially entropy/EWMA, position-only, and
  single-feature thresholds.
- Cross-compressor transfer measured as a first-class result.
- Cross-ratio, cross-task, and cross-model limits reported honestly.
- Failure definitions grounded in prior literature where possible.
- Evidence that online features contain information beyond trivial
  position/ratio proxies.
- At least one closed-loop controller demo that beats random gating at
  matched compute.

If those do not hold, the work can still be valuable as a measurement
methodology or negative-result paper, but it should not be framed as a
full controller contribution.

## Why Staging Does Not Compromise the Goal

The phased plan in `research-plan.md` builds toward this vision in
the order that maximizes information gain per unit compute:

- Phase 0-1 establish the measurement methodology. If only this
  succeeds, the field still gets the standard benchmark.
- Phase 2-3 establish predictability and generalization. If only
  these succeed on top of Phase 1, the field gets a working
  predictor and a clear scope of where it works.
- Phase 4 establishes the control story. This is the application
  layer; everything above it is the foundation.

The staging also includes explicit controllability probes before the
full controller dataset is collected. These probes bound whether
reactive intervention can recover quality, or whether the predictor
must be anticipatory with enough lead time to prevent derailment.

The vision above is what the work is *for*. The phased plan is *how
to get there without burning out on a benchmark project before the
methodology is proven*. The deferred items (human annotation, Tier 2
features, larger model sweep, production deployment) are not cuts;
they are sequenced.

## The Discipline This Demands

When evaluating any decision during execution, ask:

- Does this move us toward closed-loop, calibrated, generalizable
  control of compression damage, or does it optimize for a local
  metric that the ultimate goal would not value?
- Does this clarify the controller's state, action, cost, and outcome
  variables, or does it merely add more measurement without improving
  future control decisions?
- Does this build infrastructure that compounds across phases, or
  does it solve only the immediate phase?
- Are we shipping a real understanding of compression damage, or are
  we shipping a number on a leaderboard?

The local metric, the phase-specific solution, and the leaderboard
number are all useful, but they are not the goal. The goal is to
make compression damage measurable, predictable, and controllable in
a way that survives contact with reality.
