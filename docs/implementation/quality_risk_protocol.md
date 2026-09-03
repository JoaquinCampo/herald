# Protocol: current-state compression quality-risk forecasting (v1)

Date: 2026-09-02. Status: DRAFT pending lead approval; becomes the locked
source of truth once approved. Supersedes the divergence target of
`results/recovered/current-state-damage-v1/protocol_lock.json` as the paper's
claim while reusing its dataset, quarantine, features, and inference machinery.

Numbers in this file are the standard. Do not paraphrase them elsewhere;
point here.

## 1. Estimand

At a compressed-state row `(run_id, t)` of a generation decoding under press
`p` at ratio `r`, forecast

    risk(run, t) = P( damage(run) = 1 | causal information available at t )

where `damage(run) = 1[baseline_quality_score(prompt) - compressed_quality_score(run) > 0]`,
both scores taken from the sequence-level fields of the release. Scores are
the task's own metric as recorded in the corpus; the label is computed from
the two score columns, never from the sign of `quality_delta` (verify its
sign convention in the audit and record it).

Supported scope: Qwen2.5-7B-Instruct, the six observed compressors, the seven
observed ratios, the four observed tasks, fixed active compression for the
whole run. Unsupported: prospective activation from an uncompressed state,
action changes within a run, unseen compressors or ratios, other model
families.

None-state semantics: if no compression is active, risk is identically zero
by definition and no model is invoked. `none` runs are label references only;
they never enter training rows.

## 2. Dataset

- Repo `Jocana/herald-logits`, revision `2a88d65a0205d831ff87ea01e52e506ec5c5987d`,
  file hashes as recorded in the v1 protocol lock.
- Integrity gate identical to v1: one task per prompt, one `none` sequence
  and all 42 compressed sequences per prompt, unique `run_id`, contiguous
  `token_pos`. Any failure blocks; no outcome-selected deletion.
- Additional label audit before any fitting: finite scores for every run;
  label prevalence per task, press, and ratio recorded; the fraction of
  prompts whose reference run is itself wrong recorded (these prompts stay in;
  their compressed runs can only be damaged if scored strictly lower).

## 3. Quarantine

Reuse the v1 development/confirmation partition and the v1 five-fold
development assignment verbatim (hash salts in the v1 lock): 508 development
and 256 confirmation prompt groups. Justification: v1 never read any quality
or catastrophe field, so these labels are untouched on both sides. The audit
must prove this by grepping every v1 artifact and script for the label
columns before the first label is read.

Confirmation rules are v1's: open once, fit once, predict once, score once.
No exclusions, alternate candidate, recalibration, or threshold change after
opening.

## 4. Training rows, checkpoints, and eligibility

- A training row is a compressed token row `(run, t)` with the run's label.
  All eligible tokens of a run share its label.
- Evaluation checkpoints are absolute positions `t in {8, 16, 32, 64, 128}`.
  A run contributes to checkpoint `t` iff it has a token row at `t`
  (generation not yet finished). Runs that end before `t` are excluded from
  that checkpoint; never zero-fill or carry forward.
- Relative progress, final length, and anything derived from them are
  forbidden (they leak the outcome).

## 5. Inputs

Allowed: exactly the v1 whitelist (action categoricals, `log1p(token_pos+1)`,
the twelve instantaneous sensors, the causal-history transforms), plus,
optionally, the frozen v1 divergence forecaster's four horizon outputs as
features. If used, they are computed by the frozen v1 model only; refitting it
is out of scope.

Forbidden: every field in the v1 forbidden list, all quality and catastrophe
fields, `relative_progress`, `output_length_so_far` beyond its identity with
`t`, task identity as a predictor (stratification only), and any
future-looking quantity.

## 6. Comparators

- `prevalence`: training-fold prompt-equal damage rate.
- `action_only`: training-fold prompt-equal damage rate per press-by-ratio cell.
- `action_clock`: the candidate's own model class restricted to action
  categoricals and `log1p(token_pos+1)`, no sensors.
- Hard comparator: per checkpoint, the lowest development out-of-fold loss
  among the three.

A candidate must beat both `action_clock` and the hard comparator. The
increment over `action_clock` is the value of the online logit signal.

## 7. Models

First candidate: XGBoost binary classifier, one model shared across
positions, `t` as a feature; hyperparameters frozen on development folds via
the v1 nested early-stopping scheme. Challenger: the v1 causal TCN with a
sigmoid head. Selection rule as in v1: if both qualify, prefer XGBoost unless
the TCN's one-sided 95% prompt-clustered macro-loss upper bound is below
XGBoost's point estimate.

Calibration: isotonic or Platt fit on development out-of-fold scores only,
chosen and frozen before confirmation.

## 8. Metrics, weighting, and inference

- Loss: prompt-equal, then action-equal, then token-equal weighted log loss.
  Macro loss is the mean over checkpoints.
- Discrimination: AUROC per checkpoint. Skill: Brier skill relative to the
  hard comparator.
- Earliness: AUROC and Brier skill as a function of checkpoint `t`, overall
  and separately for runs with any catastrophe flag versus none.
- Independent unit: prompt. Task-stratified prompt-cluster bootstrap,
  10,000 resamples, seed 314159, shared draws, studentized max-T
  simultaneous bounds, 95% confidence. Difference convention: candidate
  minus comparator; for loss and Brier negative favors the candidate, for
  AUROC positive favors the candidate.

## 9. Gates

Development gates (all required before confirmation may open):

1. Every integrity, label, causality, leakage, and quarantine audit passes.
2. At every checkpoint `t >= 16`, the simultaneous one-sided 95% bound shows
   the candidate beats `action_clock` and the hard comparator on log loss and
   on AUROC.
3. Usefulness floor: at `t = 32`, the simultaneous one-sided 95% lower bound
   on candidate AUROC is at least 0.70.
4. No harm: every task, press, and ratio subgroup has a non-negative point
   estimate of AUROC improvement over `action_clock` at `t = 32`.
5. Calibration: at every checkpoint, the simultaneous 95% interval for the
   calibration slope contains 1, and expected calibration error (10
   quantile bins) has an upper bound at most 0.05.
6. Leave-one-task-out stress: with each task held out, gates 2 and 3 still
   hold on the held-out task at `t = 32`.

Confirmation pass rule: repeat gates 2 through 5 on the untouched 256
confirmation prompts without any change. Gate 6 is development-only.

Success claim on pass: a calibrated current-state compression quality-risk
forecaster on the pinned scope, with the earliness curve as the operational
characterization. A pass does not establish magnitude forecasting,
prospective activation, or transfer.

Retirement: any valid confirmation gate failure retires v1 of this protocol.
No alternate candidate, threshold, subgroup relaxation, or secondary endpoint
may rescue an opened confirmation. Record the failure and open v2 only with a
technically distinct hypothesis.

## 10. Secondary endpoints (frozen now, opened only after the primary decision)

- Major damage: `baseline - compressed >= theta_major`, with `theta_major`
  frozen from the development label distribution before confirmation opens
  (record the value and rationale).
- Catastrophe risk: `has_looping or has_non_termination` as label, same
  pipeline, reported as diagnostic only.
- Magnitude: `E[baseline - compressed | damage]`, reported descriptively; no
  claim.

## 11. Proof artifacts

- `results/recovered/quality-risk-v1/protocol_lock.json`: this file's
  parameters, frozen before the first label is read.
- `results/recovered/quality-risk-v1/confirmation_prefit_lock.json`.
- `results/recovered/quality-risk-v1/confirmation_result/report.json` with
  top-level `pass: true` and every gate's bounds.
- `results/recovered/quality-risk-v1/final_audit.json` with
  `independent_confirmation_evidence_verified: true` and
  `confirmation_attempts: 1`, produced by an independent recomputation.
- `uv run poe check` green; every new script has a CPU-runnable test.

Scripts follow the v1 naming: `scripts/materialize_quality_risk_dev.py`,
`scripts/train_quality_risk_tabular.py`, `scripts/train_quality_risk_tcn.py`,
`scripts/evaluate_quality_risk_dev.py`, `scripts/open_quality_risk_confirmation.py`,
`scripts/run_quality_risk_confirmation.py`, `scripts/audit_quality_risk_final.py`.

## 12. Integrity rules

- Never read a confirmation label, feature, or prediction before the prefit
  lock is written.
- Never weaken a gate because a candidate misses it.
- Never use relative progress, final length, quality, or catastrophe fields
  as inputs.
- A failed hypothesis is evidence; record it and change the next test.
- Results and models never enter Git.
