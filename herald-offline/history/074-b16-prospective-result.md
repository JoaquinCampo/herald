# B16 prospective replication failed

The frozen study071 failed on fresh evaluation data. It does not justify
confirmation or a validated-predictor claim. All 80 generated instances are now
exposed development evidence and must not support another fit or feature search.

Discovery had 32 eligible, complete cases: 21 positive losses, 11 zero losses,
and reference mean 1.0. It passed the frozen feasibility gate. The four models
were fit once, with the candidate using all six structural columns plus z.
Frozen artifact SHA:
6f3ace205f096c5a34c5dd692220412ca544b61587fac8d7bbba3527ae5a6c92.
The collector loaded this artifact and persisted every evaluation prediction
before generating its paired continuations or scoring their outputs.

Evaluation had 48 eligible, complete cases: 35 positive losses, 13 zero losses,
and reference mean 1.0. No failure, reference mistake or improvement was excluded.
This population simply contained no observed improvements. All state, mask,
reference/no-op, uninterrupted-path, feature and official-scoring checks passed.

| Model | Evaluation MSE | Candidate gain against it | Candidate strict wins |
| --- | ---: | ---: | ---: |
| Discovery mean | 0.2027994792 | -35.7754% | 28/48 |
| Structural Ridge | 0.2752692737 | -0.0300% | 29/48 |
| Structural OLS | 0.3195383378 | 13.8282% | 24/48 |
| Candidate | 0.2753518648 | | |

The candidate missed the required 10% gain against mean and structural Ridge,
and the 32/48 strict-win criterion against all three comparators. Raw positive-direction
z AUC was 0.654945, below the required 0.80. The simultaneous Bonferroni 95%
gain intervals were [-132.98%, 18.35%] against mean, [-0.1466%, 0.1195%] against
structural Ridge, and [3.988%, 25.709%] against OLS. These are paired-prompt
bootstrap intervals conditional on the fixed fit, not training-set uncertainty.
Beating the weaker OLS model in MSE does not rescue the failed study.

The added z coefficient was already near zero in discovery. On evaluation the
candidate behaved almost identically to structural Ridge, while both lost to
the mean. The old 12-prompt AUC of .8125 did not replicate at its frozen threshold.
This rejects the selected measurement/model protocol; it does not prove that
all query information, compression risk, or signed quality loss is unpredictable.

Independent reconstruction matched all 240 official arm scores across discovery
and evaluation, with zero salience reconstruction error and maximum z discrepancy
3.87e-9. Reconstruction starts from saved probabilities and value norms, not
independently recomputed QK products. Owner arithmetic using Python's standard
library independently matched MSE, strict wins and pairwise AUC, and verified
the frozen artifact hash remained unchanged.

The evaluation report initially failed while taking a median over a nested
timing dictionary. The original source is preserved locally under
results/restart-20260915/failures/evaluator-before-cost-report-fix.py. Restricting
the timing summary to numeric fields fixed the exact failed command. No fit,
prediction, label, feature or criterion changed; the rerun correctly reports failure.

Median synchronized observation time was 0.22723 seconds, plus 0.0000101 seconds
for four predictions. The recorded cache-feature transfer was 458,063,872 bytes,
with another 3,578,624 structural-summary bytes and separately recorded Q transfer.
Diagnostic probes and repeated state checks are separately identified. This is
measurement cost, not an assertion of useful latency savings or deployment readiness.

Authoritative Orion artifacts: results/b16-replication-v1-{models,fit}.json,
results/b16-replication-v1-{discovery,evaluation}-audit.json, and
results/b16-replication-v1-evaluation-report.json. Raw paired cases and tensors
remain in the discovery/evaluation result directories. Small local copies and
owner-metric-check.json are under results/restart-20260915/b16.

Return to UNDERSTAND. Do not tune these 80 cases or open confirmation. Reassess
which target/data/modeling assumption or genuinely untested observable deserves
a prospective test. A targeted history check already established that prompt-only
answer-likelihood probes were tested in038/039; they are not a new mechanism.
