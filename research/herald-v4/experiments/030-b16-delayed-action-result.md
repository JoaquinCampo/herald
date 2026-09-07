# Fixed B16 delayed action result

The fixed study missed its development gates. All 12 exposed NIAH prompts
remained eligible; eight degraded and four were unaffected, exactly the same
label vector as B0. The uncompressed B16 reference reproduced each original
uninterrupted reference. These are new action outcomes at a changed boundary,
not relabeled B0 measurements.

LOO MSE was 0.242747208 against the training-mean baseline 0.264462810,
a reduction of 8.2112%. The model improved seven of twelve prompts. Raw positive
AUC was 0.8125. This passes the AUC gate but misses the frozen 10% error reduction
and eight-prompt gates, so the fixed B16 observation is closed.

The exact one-sided AUC permutation count was 27/495, p=0.05455. The descriptive
fixed-OOF-pair bootstrap gain interval was [-0.26953, 0.33785]. It excludes model
refitting uncertainty and cannot establish a validation claim. No confirmation
population was opened.

Raw measurements: results/b16-v1-first and results/b16-v1-rest. Locked analysis:
results/b16-model/summary.json and rows.json. The collector recorded all controls
passing for all cases. Independent audit passed all twelve cases, including 36 official scores and
closed-form LOO reconstruction. Maximum scalar z discrepancy was 2.74e-9,
salience discrepancy 1.79e-7 and score discrepancy zero. See
results/b16-owner-audit.json and scripts/audit_b16.py. Raw K/V were not saved,
so this reconstructs features from persisted probabilities and value norms,
not an independent QK dot product or another GPU run. CPU reproduction and source freeze are in 029.

Return to understanding the observation. B0 and B16 broad removed-mass averages
have not demonstrated useful final-loss prediction. Oracle needed-sentence rescue
remains causal evidence that those locations matter, but it does not supply a
decision-time method to locate them or a general predictor.

Median recorded costs: ordinary 16-token advancement 0.2986s, extra probe
0.01168s, CPU feature construction including transfer 0.1420s, scalar reduction
0.01602s. Feature transfer median 458,293,248 bytes. These exclude cache cloning
and other orchestration, so they are partial observation costs, not deployment
latency or savings. The repeated advancement is verification work, not required
predictor work.

Literature checked during the retreat: CompressKV distinguishes streaming from
semantic retrieval heads and argues that grouping all heads can obscure useful
middle-context evidence (https://arxiv.org/html/2508.02401v1). That motivates
aggregation loss as a competing explanation; it does not establish that a new
head selection will predict signed final loss. The query-visibility audit
(https://arxiv.org/abs/2607.11942) also motivates keeping query-aware versus
query-agnostic scope explicit. HERALD's current boundary includes the question.
