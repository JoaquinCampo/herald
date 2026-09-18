# Query-located span result

The fixed CPU understanding audit failed all three gates. The locator covered
all oracle sentence tokens in ten of twelve prompts; cases000 and005 selected
the introductory instruction instead. Raw positive-direction AUC of mean
located-span eviction was0.3125, compared with0.4375 for cyclic wrong questions.
Similarity-only AUC was0.71875 and normalized-position AUC0.3125. No predictor
was fitted. Close this fixed prompt-only locator/statistic without parser repair,
orientation reversal or alternate reduction search.

All twelve measurements completed. Root independently checked native masks
against historical rescue masks, exact tokenization/character-to-token mapping,
cyclic question identity, all24per-head reductions and24official reference/action
scores. Scalar discrepancy was zero. See results/lexical-span-owner-audit.json.

Frozen script SHA2564833fc091a9c7c6bc7f11056e38f8b9e29b33182ede20392466dab1124d76bdc.
Feature output SHA256b6e63a5dca8010d86739c138d9ca6f4d7a719813c581ea88e736326298f9280a.
Restricted input SHA256496f88c28810edca7dd6ca5c11ce0de1f3f8bdcb5f0b395a9e47147bbfefd365.
Measurement results/lexical-span-v1/measurement.json. Total measured wall time
0.768s, CPU0.747s, peak process RSS729,972,736bytes includes Python/runtime and
loading all persisted masks, not isolated incremental deployment memory.

An explicitly exploratory follow-up separated the locator and reduction errors:
using the perfect oracle sentence positions with the same mean eviction fraction
gave raw positive-direction AUC0.34375. See results/oracle-span-mass-diagnostic.json.
This is inadmissible as a feature and was not fitted or orientation-reversed.
It shows that correcting this locator alone would still leave the simple mean
removal statistic uninformative on these cases. Full-sentence rescue causality
survives, but the identity of affected heads/tokens or their interactions matters
more than this count. No claim follows for every possible content-aware feature.

No GPU work or confirmation access was performed. Return to understanding the
causal granularity or a prespecified independent discovery population before any
new predictive fit.
