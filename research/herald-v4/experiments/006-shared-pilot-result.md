# Shared-state pilot: valid variation, no predictor yet

The new protocol completed all12 development prompts on Orion, exit0.
Reference and no-op continue independent copies of the same split-prefix
boundary, before the last prompt token. All48 arm checks passed. An independent
audit recomputed every score from raw text and verified all36 compressed
effects, including three improvements. V1 remains a separately failed assay.

For eight retrieval prompts, reference quality was1 throughout. Removal.25
caused six full losses and two unchanged outcomes. Removal.5 and.75 caused
full loss on all eight, leaving little prediction variation at those severities.

For four aggregation prompts, reference mean quality was.8. Compressed mean
quality was.75, .775 and.25 at the three severities. The12 compressed outcomes
include improvement, unchanged quality and degradation. These tiny development
observations are not population estimates or evidence of predictive gain.

Full-prefill still differs from the split path on CWE002/003 and is retained
as a separate partition audit. Results describe the explicitly constructed
decision state, not ordinary whole-prompt-prefill intervention semantics.

Run time was about107seconds including setup and checks; summed continuation
and arm times about75seconds. This is experiment cost, not deployment overhead.
Next: one transparent pure-scorer EA measurement check before new labels/fits.

Evidence: results/pilot-v2-shared/{run.json,scores.jsonl,owner-verification.json,
independent-audit.json}; source snapshot results/pilot-v2-launch/frozen-runner.py.
Source27045a21..., manifest92838c1c..., with full digests in owner approval.
