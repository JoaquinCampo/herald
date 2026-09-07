# First EA predictive-value result

The locked mean-query, value-weighted, all-head EA excess feature did not meet
007's exploratory proceed rule. On 20 fresh development prompts and 60 Knorm
actions, grouped out-of-fold MAE changed from 0.349474 to 0.343477, a 1.72% gain
against the required 10%. All four folds improved slightly, but within-task
Spearman correlations were negative for the EA model: NIAH -0.1480, CWE -0.1743.
This is a scoped negative result for this feature aggregation and fixed model,
not evidence that attention cannot predict compression damage.

## Evidence and controls

The new population contains 12 NIAH and 8 CWE prompts at removal fractions
0.05, 0.10 and 0.20. Four balanced prompt folds were fixed before outcomes.
Both models use training-fold StandardScaler and Ridge(alpha=1), with unbounded
raw predictions. The EA model adds only the locked excess to seven baseline
features. Task names, answers and continuation information do not enter fitting.

All 20 prompts and 80 arms completed with no failed controls or unscorable rows.
Shared reference/no-op tokens and termination matched exactly. Cache/source
isolation, physical compression and 112-head finite feature checks passed.
There are 60 compressed effects: 5 improvements, 25 unchanged and 30 degradations.
The NIAH reference scored 1 on all 12; CWE reference mean was 0.825.
Even .05 removal lost retrieval on 7/12 NIAH cases, while CWE effects were mostly 0.

Raw results and scores: results/ea-dev-v1-first/ and results/ea-dev-v1-rest/.
Collector source 302fc3f590818844a384eed7cf2ff95643be23019b1dde610a29cf5a0fd4861f
is frozen in results/ea-dev-v1-launch/frozen-collector.py. Later active-file
changes add redundant audit fields and were not used in this run.

Analysis: results/ea-dev-v1-analysis/summary.json and predictions.jsonl/.csv.
Frozen analyzer 3646e6b51e215fbaa76528ad5b7ef62ad0ab667209e439864e7d37770745f698
is in results/ea-dev-v1-analysis-launch/. Before the first real fit, review removed
unprespecified clipping and fixed a merged-output field overwrite. Neither issue
was selected or changed using fitted real outcomes.

Owner verification: results/ea-dev-v1-owner-verification.json.
Independent audit PASSED in results/ea-dev-v1-audit.json: all 80 scores and
60 OOF prediction rows independently recomputed, with the same NO-GO conclusion.
No confirmation population was opened. Do not promote the feature, change its
threshold, or search many fitted variants on these outcomes.

## Reflection

Milder removal restored some retrieval variation, so complete severity saturation
cannot explain this result. The aggregate EA feature adds little beyond the
mechanism/geometry controls in this slice. It remains unresolved whether global
head averaging hides sparse useful evidence, mean prefill queries miss the later
retrieval query, or the small population makes the fitted comparison unstable.
Next return to those competing explanations with a bounded mechanistic diagnostic
before another predictor fit or a larger population.
