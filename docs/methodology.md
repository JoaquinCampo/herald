# Methodology

> **Status: archived research record (2026-09-02).** This pre-switch
> magnitude protocol produced a negative frozen result (section 10) and no
> longer defines the paper's claim. The current protocol is
> `docs/implementation/quality_risk_protocol.md`; the thesis is in
> `docs/goal.md`.

This document specifies the paper on pre-switch magnitude forecasting for
known compressors. Historical controller, rollback, and cross-compressor
experiments remain research records; they do not define this protocol.

## 1. Research question

For a known compressor $c$ and known removal ratio $r$, can statistics
available from the uncompressed generation state forecast the final
task-quality effect of activating compression at the current switch position
$s$?

One model is trained per compressor. A compressor's supported ratios are
pooled and $r$ is an input. The current paper makes no claim about an unseen
compressor or an unseen ratio.

## 2. Damage estimand

A switch position $s$ means that exactly $s$ generated tokens are fixed and
the intervention occurs immediately before computing the next token. Let
$S_s$ denote the complete decoder state at that boundary.

For a compressor with a valid live-state operation, the preferred target is

$$
d^{\mathrm{fork}}_{c,r}(s)
=
q\!\left(\operatorname{continue}(S_s)\right)
-
q\!\left(\operatorname{continue}(\mathcal C_{c,r}(S_s))\right).
$$

The branches begin from independent, identical state forks. Only one fork is
compressed. Both use the same prompt and emitted prefix, remaining token
budget, EOS rules, decoding configuration, attention backend, and numerical
precision.

Some compressors require auxiliary state or are defined by a compressed
re-prefill rather than a pure cache transformation. Those compressors use the
matched target

$$
d^{\mathrm{refill}}_{c,r}(s)
=
q\!\left(\operatorname{continue}(R(S_s))\right)
-
q\!\left(\operatorname{continue}(\mathcal C_{c,r}(R(S_s)))\right),
$$

with the same no-press re-prefill path on the control arm. Compressor
classification, state contents, clone isolation, and the parity gates are
specified in `docs/_why/6_intervention_semantics.md`.

## 3. Intervention validation

Before labels are recovered or generated, every intended compressor and ratio
must pass the applicable checks:

1. **Fork-clone parity and isolation:** two uncompressed state forks continue
   identically; compressing one cannot mutate the other.
2. **Sham re-prefill parity:** compare a live full-cache continuation with a
   no-press re-prefill from the same token prefix.
3. **Intervention parity:** where a live operator exists, compare its
   compressed continuation with pressed re-prefill.

Each check records exact continuation match rate, first divergence position,
and final quality delta. Historical `q_reference` values are shortcuts for a
no-press re-prefill only if sham parity passes. Historical hybrids may be
described as live-cache activation only if intervention parity passes.

## 4. Quality and labels

Quality is measured offline on completed outputs. The initial paper uses
IFEval instruction-level loose accuracy:

$$
q(y)
=
\frac{\text{instructions satisfied by }y}
     {\text{instructions in the prompt}}.
$$

The primary label is signed $d$: positive is degradation, zero is no measured
effect, and negative is lift. We retain negative values during training.
Positive harm $d^+=\max(0,d)$, any-damage risk, and major-damage risk are
derived reporting quantities, not replacements for the primary magnitude
target.

Strict instruction-level IFEval accuracy is retained as the frozen secondary
robustness score for every reference, matched control, and treatment. Token
mismatch, KL or JS divergence, embedding distance, and perplexity change are
diagnostics or candidate features; none is a quality label. Detailed rationale
is in `docs/_why/3_measuring_quality.md`.

## 5. Dataset

The final sweep contains one row per
`(model, task, prompt, compressor, ratio, s)` with a completed-output quality
delta and causal reference-stream features. StreamingLLM and Knorm branches
fork and compress the live uncompressed cache with pending-token semantics.
ExpectedAttention uses a matched sham re-prefill control and pressed re-prefill
treatment from the exact prompt and reference prefix.

The frozen dataset has 200 references and 59,076 switch cells across all three
compressors and four ratios, with no skipped rows. Its manifest verifies scorer
identity, ratio semantics, causal feature timing, exact cell keys, raw-artifact
hashes, and complete prompt coverage. Historical quality labels were rejected;
all labels in this dataset were regenerated under the validated protocol.

## 6. Predictor

For each compressor $c$, fit one scalar regressor

$$
\hat d_c(s,r)=f_c(x_{\le s},r,s),
$$

where $x_{\le s}$ contains only information available before activation.

The first-pass input whitelist is:

- the known removal ratio;
- absolute generated position $s$;
- causal pre-switch `feat__*` statistics available at the decision boundary.

Forbidden inputs include final reference length, relative position computed
from that length, reference or hybrid quality, labels or label-derived fields,
prompt identity, probes, hybrid-stream summaries, grace-window features, and
all other post-switch information. Exact feature timing at $s$ must be
validated to exclude an off-by-one leak.

The first learned model is one fixed XGBoost squared-error regressor per
compressor. A linear Huber regressor is a learned baseline. Broad model and
hyperparameter searches are out of scope until a fixed model demonstrates
skill over non-learned baselines.

## 7. Splits, weights, and baselines

Splits are prompt-disjoint. Every ratio and switch position belonging to a
prompt remains in one split. Early stopping and model selection use training
prompts only.

Each `(prompt, ratio)` trajectory receives equal total training mass. If
trajectory $i$ has $n_i$ switch rows, each row receives weight

$$
w_{ij}=\frac{1}{n_i}.
$$

Evaluation macro-averages ratios so different trajectory lengths or missing
rows cannot alter ratio weighting.

For each compressor, establish these training-only baselines before fitting a
learned model:

1. global mean and median;
2. mean and median by ratio;
3. mean and median by ratio and 16-token absolute-position bucket;
4. linear Huber regression on the same whitelisted inputs.

## 8. Evaluation and decision rules

Report results separately for every compressor and ratio. A pooled or macro
average cannot rescue a failed compressor.

Primary magnitude evidence:

- held-out-prompt MSE and RMSE skill over the strongest grouped-mean baseline;
- MAE skill over the strongest grouped-median baseline;
- prompt-cluster bootstrap confidence intervals.

Diagnostics:

- calibration by predicted-damage bins;
- positive, zero, and negative label prevalence;
- error on positive and non-zero damage rows;
- mean prediction versus mean target;
- rank correlation.

A compressor supports the magnitude claim only if its MSE-skill 95% prompt
bootstrap interval is strictly above zero, error on positive-damage rows does
not worsen, calibration is graded rather than collapsed near zero, and
per-ratio results show that one setting does not create the aggregate gain.
Otherwise the current features do not establish magnitude forecasting for
that compressor.

Any-damage and major-damage risk are preregistered secondary endpoints. Their
thresholds and metrics must be frozen before primary results are inspected;
they are not post-hoc fallbacks for failed magnitude regression.

## 9. Initial scope

The first study is Llama-3.1-8B-Instruct on IFEval, using the real compressors
whose intervention semantics and historical provenance can be validated.
Expansion to other tasks, models, ratios, or new pre-switch sensors requires a
frozen IFEval result and a separately stated replication question.

## 10. Frozen result

The prompt-disjoint split contains 12,972 training rows, 2,088 validation rows,
and 4,632 held-out rows per compressor. The learned regressors are compared
with the validation-selected ratio-and-position mean or median baseline.
Prompt-cluster intervals use 1,000 bootstrap resamples.

| Compressor | Positive damage | Major damage | MSE skill (95% CI) | MAE skill | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| ExpectedAttention | 7.94% | 5.99% | +0.89% [-3.50%, +5.50%] | -44.39% | Stop |
| Knorm | 27.21% | 23.35% | +4.97% [+0.006%, +9.40%] | -23.92% | Stop |
| StreamingLLM | 24.65% | 21.16% | -2.10% [-7.96%, +3.13%] | -37.53% | Stop |

ExpectedAttention improves MSE at ratios 0.25, 0.5, and 0.875, but not 0.75;
its bootstrap interval includes zero. Knorm has a positive overall MSE-skill
interval and improves ratios 0.5, 0.75, and 0.875, but is worse at ratio 0.25.
StreamingLLM is worse at ratios 0.25 and 0.5 and its overall interval includes
zero. All three models worsen MSE on positive-damage and major-damage rows and
have negative MAE skill. Therefore none satisfies the frozen magnitude claim.

This is a negative result, not a license to select a favorable compressor or
metric post hoc. The present causal pre-switch logit features do not establish
reliable signed final-quality forecasting under the preregistered gate. The
regenerated dataset, fitted native model artifacts, and complete evidence
remain useful as a reproducible benchmark for a separately preregistered
sensor or modeling hypothesis.

Frozen evidence is in
`results/recovered/ifeval-intervention-v1/paper_evidence.json`; native XGBoost
models and fit state are in the adjacent `magnitude_evidence_v2_models/`
bundle.
