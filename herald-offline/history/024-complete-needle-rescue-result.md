# Complete exposed NIAH rescue result

Across all12 exposed NIAH prompts at .10 Knorm, reference and oracle rescue
scored12/12; standard compression and matched non-needle control scored4/12.
Oracle restoration rescued all8 standard failures and harmed none of the4
correct cases. The control rescued none and harmed none. Paired oracle-minus-
control mean score is8/12. This broadens the original selected-four mechanism
result within the exposed population; it is not a confirmation study.

The additional eight cases completed without failures using unchanged source
ace1754a. Every persisted control passed, including prior reference/standard
replay. The checks dictionary contains27entries including its aggregate flag;
prior_checks also records enabled status and both exact-token matches.
Independent audit PASSED:84 official branch score recomputations,324 recorded
checks,6720 custom mask head checks and1344 oracle/control swap comparisons.
Artifact: results/needle-rescue-all12-audit.json. Raw outputs remain in
results/needle-rescue-all12-additional with the original four in
needle-rescue-v1-first/rest. Owner summary:
results/needle-rescue-all12-owner-summary.json.

Correct standard/control cases:001,006,007,009. All other cases000 through011
are rescued failures. Same per-head budget, victims and slot replacement rule
is preserved. This supports sentence-content retention as a causal mechanism
in these eight failures, but does not identify causal heads or answer-free
importance estimates. No predictor is validated. The next design question is
whether decision-time query-conditioned evidence can identify the vulnerable
content without oracle spans or answer labels.
