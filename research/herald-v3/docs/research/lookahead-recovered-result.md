# Eight-step lookahead, recovered analysis

The recovered calculation does not show a predictive gain from delayed JS.
This is a descriptive exploratory result. The original unamended execution
contract has status `operational_integrity_failure`: its source-lock rule was
violated by validator repairs. The numerical evaluator's `negative` status is
not a claim that the original execution contract passed.

The r4 prediction seal was fixed before any test outcome. After the first
outcome was generated, a source-only execution amendment repaired resource
metadata, score-schema parsing, and partial-resume checks. No model was refit,
no probe was recollected, and no outcome was resampled during that recovery.
The first raw result was preserved byte-for-byte. The r5b execution seal pins
the r4 parent and inherits its models and all 1,656 predictions exactly.

Earlier engineering repetitions occurred before test outcomes and reproduced
all prediction inputs and fitted predictions exactly. These repairs must be
reported; this study is not an uninterrupted confirmation run.

## Numerical result

There were 76 reserved candidates, 69 eligible prompts, seven immutable early
terminations, and 138 action outcomes. Thirty-eight prompts had a nonzero loose
loss, exceeding the locked information floor of 20. All 483 saved responses
passed recomputation with the pinned official IFEval scorer before evaluation.

| Predictor | Loose MSE | Strict MSE |
| --- | ---: | ---: |
| Per-action training mean | 0.189700 | 0.184987 |
| B0, action/size and probe length | 0.186989 | 0.184914 |
| B1, adds reference entropy/margin | 0.187704 | 0.192867 |
| B2, adds scalar action-probe features | 0.186899 | 0.188347 |
| B3, adds immediate JS | 0.185922 | 0.187233 |
| B4, adds mean delayed JS | 0.189879 | 0.194937 |

Errors weight each prompt equally and average its two candidate actions.
B4's loose MSE is 2.13 percent higher than B3's. The paired 95 percent interval
for B3 MSE minus B4 MSE is [-0.015702, 0.006899], so these data do not establish
a gain or a reliable deterioration. B4 fails the unchanged requirement of at
least five percent improvement and a positive paired interval against every
comparator. The strict-score calculation has the same unfavorable direction.
The intervals use 2,000 seed-0 paired prompt bootstrap resamples.

Loose outcomes comprise 48 degradations, 80 unchanged scores, and ten
improvements. At Knorm 0.25 these counts are 20/44/5; at Knorm 0.5, 28/36/5.
Signed improvements were retained, as specified.

## Measured cost and scope

Across 138 candidate probes, median instrumented H8 time was 0.272 seconds
(95th percentile 0.276 seconds), with median peak allocated-memory increment
26.16 MB. Complete probe collection, including controls and validation, took
1.194 seconds per eligible prompt at the median. These are Orion measurements
for this assay, not deployment overhead or net memory savings.

This result concerns Qwen2.5-7B-Instruct, the fixed 32-token boundary, two Knorm
actions, IFEval compliance, the specified features, and Ridge with alpha 1.
It does not establish that final quality loss is generally unpredictable.
The numerical comparison supplies no basis for deploying this predictor or
adding a controller. Do not tune the horizon, summary, actions, or estimator
on these now-exposed test labels.

## Evidence

- Numerical report: `results/lookahead-evaluation-r5.json`, SHA-256
  `b6f4e3efd6bbc923676f3b75b843aec5a9861f4b76eef38feb22be08387a942e`.
- Completed outcome index: `results/lookahead-outcomes-r5/outcome-index.json`,
  SHA-256 `eeac8f61143b5c7e0bffc872d36a8655fe58ecf7131612bf8c4aecf4be324bc0`.
- Authoritative pre-outcome fit: `results/lookahead-prediction-seal-r4.json`.
- Linked execution seal: `results/lookahead-execution-seal-r5b.json`.
- Recovery audit: `results/lookahead-recovery-independent-audit-r5.json`.
- Costs and action counts: `results/lookahead-cost-and-coverage-r5.json`.
- Final independent audit: `results/lookahead-independent-audit-r5.json`,
  SHA-256 `935842b95071a87c1e91c5b8c7313514d9b528d5dccaeb6d5d9a5c452ca4db5b`.
  Independent reconstruction matched the metrics and all 2,000 bootstrap
  resamples exactly and verified all 276 raw artifact files.

## Owner decision

Close the tested cheap-logit and delayed-JS branch for this scope. The broader
HERALD research objective remains open. Any next experiment requires a
materially different prespecified mechanism and a fresh independent population;
these exposed labels cannot provide another untouched test. The next ownership
step is a bounded review of alternative mechanisms and available populations,
before allocating further GPU work.
