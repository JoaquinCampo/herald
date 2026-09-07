# Optional prior evidence

Read this index when a concrete question needs historical context. Do not
load the old experiment narrative as the default starting methodology.

Sources inspected during bootstrap on 2026-09-05:

- `/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v2`, main at
  `b19d18b`: `docs/methodology.md` records the pre-switch experiment;
  `docs/_why/6_intervention_semantics.md` describes state and intervention
  concerns; `docs/goal.md` records the later current-state question.
- `/Users/joaquincamponario/orca/workspaces/herald-v2/cero`,
  `codex/quality-risk-v1` at `f47a47b`, with additional uncommitted work:
  `docs/implementation/quality_risk_audit_2026_09_04.md` and
  `docs/implementation/quality_risk_feasibility_2026_09_05.md` provide newer
  evaluation and feasibility records. Working-tree material may change.

These are provenance pointers, not imported conclusions. Before reusing an
artifact, inspect its exact target, data lineage, split exposure, evaluation
validity, and source revision. Reuse tested utilities only after checking
their semantics against the newly chosen experiment. Do not treat old
negative results as impossibility proofs or old positives as validated
evidence without examining their audits.

No historical results, model weights, datasets, or predictor implementation
were copied into this bootstrap.

## Paper review for the agreed magnitude target

The main checkout's `paper/attic/03_problem.tex`, `04_intervention.tex`,
`05_dataset.tex`, and `06_method.tex` directly address signed final-quality
loss at a switch point. Reusable concepts are matched independent state
forks, explicit intervention timing, task-grounded scores, and causal
feature availability. `docs/_why/3_measuring_quality.md` explains IFEval's
instruction-level partial credit, which is a candidate quality measure,
not a v3 task selection. It measures instruction compliance rather than
all aspects of answer quality.

The newer `paper/sections/03_corpus.tex` describes always-compressed
trajectories. Those trajectories cannot automatically supply outcomes of
activating compression later from a shared full-cache state. Validate data
compatibility before reuse. Several manuscript sections contain TODOs and
historical status claims; consult actual scorer and experiment artifacts
before treating them as implemented or verified.

Do not inherit the attic draft's fixed regressor, input whitelist, or
compressor choices. Its corresponding negative result in
`docs/methodology.md` is evidence to inspect when forming a new hypothesis,
not a universal limit on the agreed prediction target.
