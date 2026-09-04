# Protocol: current-state compression quality-risk forecasting (v2) — PROPOSAL

Status: DRAFT proposal for lead approval. Not locked. No v2
`protocol_lock.json` may be written and no confirmation row may be read
until the lead approves this file. Nothing here weakens v1; v1's verdict
(development-gate failure on gate 4, confirmation unopened) stands.

## 0. Relation to v1

v1 (see `docs/implementation/quality_risk_protocol.md` and
`results/recovered/quality-risk-v1/DEV_GATE_VERDICT.md`) cleared gates 2,
3, and 5 but failed gate 4: a shared global mapping cannot fit the
task-conditional sensor-to-damage directions found on development (99 of
112 features flip correlation signs across tasks;
`results/recovered/quality-risk-v1/sign_flip_analysis.json`). v2 is
technically distinct: it routes by task regime. Two variants below; only
the primary carries the paper's headline claim.

## 1. Estimand (unchanged)

At a compressed-state row `(run_id, t)` decoding under press `p` at ratio
`r`, forecast `risk(run, t) = P(damage(run) = 1 | causal information
available at t)` with `damage(run) = 1[baseline_quality_score(prompt) -
compressed_quality_score(run) > 0]`. Scope, checkpoints `{8, 16, 32, 64,
128}`, eligibility, and the section 2.1 null-score rulings are unchanged
from v1 (prompt-level gaps excluded and recorded; run-level nulls imputed
to the task minimum; reference nulls imputed to the task minimum after
failure-indicator verification; any other pattern blocks).

## 2. Variants

- PRIMARY (headline): stream-only routing. A causal regime classifier maps
  the frozen raw material at row `t` (same raw material as v1 section 5,
  rows `<= t` of the same run only) to soft regime probabilities over the
  four development tasks. The risk model takes the v1-final feature roster
  plus the regime probabilities as features. Task identity is never an
  input at inference; the paper's claim stays "from the logits alone".
- SECONDARY (upper bound): oracle task identity as an additional risk-model
  input. Reported as the achievable ceiling and as the measured cost of
  inferring vs knowing the regime. Not the headline.

## 3. Regime classifier (primary only)

- Target: task (stratification metadata, allowed as a dev-only target).
- Model: XGBoost multiclass classifier (`multi:softprob`), one model
  shared across positions, `t` as a feature; hyperparameters identical in
  spirit to the risk model (v1 values, nested early-stopping round
  selection on development folds).
- Quarantine: fit on development prompts only; only out-of-fold regime
  probabilities may enter risk-model rows. In-sample probabilities are
  forbidden (they would leak task identity into training rows).
- Causality: per-feature automated audit exactly as v1 section 5 (recompute
  from rows `<= t`); the classifier's outputs at row `t` derive from rows
  `<= t` alone.
- Unseen tasks: soft probabilities have no unseen level; no fallback is
  needed for the primary variant.

## 4. Comparators (matched to candidate information)

- `prevalence`, `action_only`: unchanged from v1.
- Matched clock: the candidate's own model class on action categoricals
  plus `log1p(token_pos+1)` (and raw `token_pos`) plus exactly the
  regime/task information the candidate sees: regime OOF probabilities for
  the primary, oracle task identity for the secondary. No sensors.
- Hard comparator: per checkpoint, the lowest development out-of-fold loss
  among the three.
- A candidate must beat both its matched clock and the hard comparator.
  The increment over the matched clock is the value of the online logit
  signal given the regime.

## 5. Models and calibration

- Risk model: XGBoost binary classifier as in v1 (shared across positions,
  `t` as a feature, nested scheme, isotonic-or-Platt calibration on
  development OOF only, frozen before confirmation).
- The v1-final sensor/divergence roster carries over; the only additions
  are regime probabilities (primary) or oracle task identity (secondary).
- TCN challenger: dropped (v1 evidence: tied overall, worse on gate 4;
  CPU cost disproportionate). If the lead wants it back, it trains on the
  same roster under the v1 scheme before selection.

## 6. Metrics, weighting, inference (unchanged)

Prompt-equal, then action-equal, then token-equal weighted log loss; macro
over checkpoints; AUROC per checkpoint; Brier skill vs hard comparator;
earliness overall and by catastrophe flag; task-stratified prompt-cluster
bootstrap, 10,000 resamples, seed 314159, shared draws, studentized max-T
simultaneous bounds, 95% confidence; candidate-minus-comparator convention.

## 7. Gates (v1 numbers, matched comparators)

1. Every integrity, label, causality, leakage (including regime-classifier
   OOF/causality), and quarantine audit passes.
2. At every checkpoint `t >= 16`, the simultaneous one-sided 95% bound
   shows the candidate beats its matched clock and the hard comparator on
   log loss and on AUROC.
3. Usefulness floor: at `t = 32`, the simultaneous one-sided 95% lower
   bound on candidate AUROC is at least 0.70.
4. No harm: every task, press, and ratio subgroup has a non-negative point
   estimate of AUROC improvement over the matched clock at `t = 32`.
5. Calibration: at every checkpoint, the simultaneous 95% interval for the
   calibration slope contains 1, and ECE (10 quantile bins) has an upper
   bound at most 0.05.
6. Leave-one-task-out stress (development only): with each task held out,
   gates 2 and 3 still hold on the held-out task at `t = 32`. SECONDARY:
   the held-out task level is unseen, so the fallback path (shared v1-final
   model) engages; the test is no-crash plus non-harmful vs the matched
   comparator. PRIMARY: no fallback exists or is needed.

Confirmation pass rule: repeat gates 2 through 5 on the untouched 256
confirmation prompts without any change. Gate 6 is development-only.
Retirement: any valid confirmation gate failure retires v2; no rescue.

## 8. Quarantine and confirmation reuse

v2 reuses v1's 508/256 partition and five folds if and only if the audit
re-verifies that no quality-risk confirmation label, feature, or
prediction has been read by any v1 or v2 script (grep proof over both
trees). The v1 confirmation predictions file stays sealed until the v2
prefit lock. Confirmation rules are v1's: open once, fit once, predict
once, score once.

## 9. Secondary endpoints (frozen now, opened only after the primary decision)

Same as v1 (major damage with `theta_major` from the development label
distribution, catastrophe-risk diagnostic, descriptive magnitude), plus
the inference-vs-oracle gap: secondary minus primary characterization as
the measured cost of inferring the regime.

## 10. Proof artifacts

- `results/recovered/quality-risk-v2/protocol_lock.json` (written only
  after lead approval of this file, before the first v2 label is read).
- `results/recovered/quality-risk-v2/confirmation_prefit_lock.json`
  (includes regime classifier architecture, rounds, normalization, code
  hash; risk-model feature orders including regime inputs; calibration
  choice and parameters; matched-comparator definition).
- `results/recovered/quality-risk-v2/confirmation_result/report.json`
  with top-level `pass: true`.
- `results/recovered/quality-risk-v2/final_audit.json` with
  `independent_confirmation_evidence_verified: true` and
  `confirmation_attempts: 1`.
- `uv run poe check` green; every new script has a CPU-runnable test.

Scripts follow the v1 naming with a v2 infix for the regime classifier
(`materialize_quality_risk_regime_dev.py`,
`train_quality_risk_regime.py`, `audit_quality_risk_regime.py`).
