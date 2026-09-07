# Direct-query salience result

The fixed025proxy failed its causal slice: it rescued0/3 failed cases000/004/011,
matched control rescued0/3, and correct case001 stayed correct. All4 runs
completed and all18 recorded controls per case passed, including actual full-
cache instrumented/plain logits and prior reference/standard token replay.
Independent audit PASSED:72controls,24officialscores, raw probability/Vnorm
aggregation and mask/selection reconstruction, allpriorreplays. Artifact:
results/query-salience-v1-audit.json. RawK/V were not persisted, so this doesnot
claim independent query-key dot-product recomputation.

Selected KV heads:14,12,13,12. Scalar gaps z:2.178726,1.834671,1.878895,1.977556.
CPU salience and selection cost0.76-0.89seconds/case, in addition to the full-
cache query probe and transfer. This implementation cost is not production cost.
Raw results: query-salience-v1-first/rest, owner summary:
results/query-salience-v1-owner-summary.json. Frozen source:
results/query-salience-v1-launch/frozen-diagnostic.py,
fca6a8f37b6c2342d09867687c6631547bcb339008dc41169c8c118e49129a8a.

Posthoc inspection of000 shows selected positions mostly at the prompt end
(13/14 in its last16 cached positions), none in oracle needle positions300:320.
This inspection uses answer annotations only AFTER the fixed feature/test, for
mechanism interpretation. Across all4cases, final16position nominations were13/14,12/12,13/13,11/12.
None overlapped needle spans in the3failures; one head overlapped in correct001. It suggests the
actual last-prompt query is dominated by recent prompt structure. It does not
show that later queries cannot identify needed content or that all attention
statistics fail. Close this fixed proxy without fitting a predictor or sweeping
heads/positions/thresholds. Choose any new observation or decision time only
through an explicit strategy revision, retaining changed estimand semantics.
