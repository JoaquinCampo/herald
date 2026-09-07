# Delayed lookahead prediction proposal

Status: owner-approved implementation contract, 2026-09-05. Collection remains
blocked until the acceptance, provenance, and seal-mechanism checks below pass.
This is one post-pilot exploratory train/test study. It does not authorize a
controller, a deployment claim, a model or horizon sweep, or any change to the
locked population after collection starts.

## Evidence and scope

Pilot v1 found no support for the fixed one-step full-vocabulary JS addition:
B3 loose MSE was 0.194360 versus B0 at 0.182044, with the same strict-score
direction. That rejects the tested Ridge specification, not the larger
decision-time question. The passed engineering assay proves that a capped
reference-prefix probe can be acquired correctly, not that it predicts final
instruction-compliance loss.

Engineering evidence is `results/lookahead-engineering-01/measurement.json`,
SHA-256 `7013822552f194c41ac95ff347930f484f6f901b31aafab8a7a8dc7e27d2fe44`.
On exposed `ifeval_3335` with Knorm 0.5, all four top-level gates passed. The
probe covered output indices 32 through 39, reproduced the accepted full-model
step 0, preserved source state, and left four-token outcome continuations
unchanged. One prompt/action took 0.3007945 s total, including 0.1791296 s of
forwards, and increased allocated CUDA memory by 27,822,080 bytes. These are
one-prompt engineering measurements, not population cost or prediction
results.

A positive result is a post-pilot exploratory holdout result because this
feature was proposed after the negative pilot. It is not an independent
replication.

## Frozen H8 feature contract

At the decision boundary, output indices 0 through 30 are in KV and output
index 31 is the pending input. Step 0 processes index 31 and predicts output
index 32, exactly as B3 did. From independent no-action and action clones,
continue only on no-action greedy tokens. At step `j`, every action clone
receives the same reference input token and produces a distribution predicting
output `32 + j`. The cap is eight distributions, `j = 0..7`, predicting output
indices 32 through 39.

Let `L` be the realized number of distributions. Stop after a reference
distribution whose greedy argmax is EOS. `L` is therefore in `1..8`; an EOS at
step 0 gives `L=1`.

```text
immediate_js = JS_0
mean_delayed_js = mean(JS_1, ..., JS_(L-1))  if L > 1
                  0                           if L = 1
has_delayed = 1[L > 1]
```

The zero at `L=1` is a defined feature value, never an imputation. No
post-EOS token is forced. An action EOS never terminates the teacher-forced
probe because subsequent inputs remain the reference tokens. A short H8 probe
is valid when it follows this rule.

For every eligible prompt, the two action rows must have identical reference
forced-token IDs, `L`, reference EOS position, reference-state fingerprint and
reference-probe hash. Each action must have a complete H8 record. A difference,
missing action row, missing probe, non-finite feature, or source-parity failure
is an `operational_integrity_failure`, not a row exclusion. The synthetic
reference prefix is a paid intervention-time rollout, not information observed
at the original decision timestamp.

## Immutable populations and provenance

Training is exactly the 120 eligible pilot-v1 prompts and both Knorm actions,
for exactly 240 matched rows. No subset, row drop, replacement, or fallback
population is permitted. Recollect only H8 probes for these rows. Existing
signed outcomes may be reused only after each row matches the locked prompt
bytes, tokenizer and model snapshot, chat template, generation settings,
decision boundary, reference-prefix token IDs, action, source hash, and run
provenance. Any mismatch is an `operational_integrity_failure` and stops the
study before fitting.

The test population is the ordered 76-ID manifest specified by
`data/lookahead-v1/roster-derivation.json`: the seed-ranked unused tail 20 from
the original 160-prompt roster followed by the 56 unselected singleton
holdout IDs. The lock must preserve this order, every source hash, the exposure
review, and the derivation hash. No test selection, replacement,
near-duplicate disposition, or eligibility rule may use outcomes.

A reference ending before 32 committed outputs is an immutable, ineligible
ledger entry with ID, source hash, position, and reason. It is never a
prediction row and is never replaced. For every test prompt reaching the
boundary, both action rows are mandatory and must satisfy the full-pair H8
contract. A failure in either action is operational, not an opportunity to
score the other action or average a partial prompt.

## Pre-collection lock

Before any probe or test outcome work, write and hash one lock containing:

- ordered training- and test-manifest IDs, raw prompt hashes, exposure records,
  roster derivation, and every pilot-v1 outcome/run hash used for training;
- model and tokenizer revisions, chat template, EOS IDs, seed, BF16, SDPA,
  generation budget, boundary, both actions, scorer source/version, complete
  environment, source-tree hashes, probe code hash, and engineering evidence;
- the ordered feature columns and their dtypes: action, prompt token count,
  decision index, pre-action cache size, B1/B2 scalar fields, `immediate_js`,
  `L` as integer, `has_delayed` as integer, and `mean_delayed_js` as float;
- exact estimator parameters and versions: `StandardScaler`,
  `Ridge(alpha=1)`, prediction clipping to `[-1, 1]`, and their library and
  runtime versions; and
- the prediction-seal schema, the outcome-runner source/dependency hashes, and
  an assertion that the outcome runner refuses to start without a valid seal.

Any change after this lock, including a column order or dtype change, is an
`operational_integrity_failure`.

## Matched estimators

Fit each learned estimator once using exactly the 240 training rows. The five
pilot folds remain provenance only. They are not reused to tune, select, or
fit any test model. Fit `StandardScaler` on training rows only, then
`Ridge(alpha=1)` with no tuning, calibration, feature search, missing-value
search, or refit after test access. Clip predictions to `[-1, 1]`.

| Name | Inputs |
| --- | --- |
| `action_mean` | Plain per-action training mean, computed from all 120 training prompts. It intentionally has less information than the learned comparators. |
| `B0_L` | Original B0 inputs plus `L` and `has_delayed`. |
| `B1_L` | `B0_L` plus original uncompressed entropy and top-two margin. |
| `B2_L` | `B1_L` plus original scalar action-probe features. |
| `B3_L` | `B2_L` plus `immediate_js`. |
| `B4_L` | `B3_L` plus `mean_delayed_js`. |

`B4_L` is the only new learned predictive degree of freedom. `L` and
`has_delayed` appear in every learned comparator, so its gain cannot be
attributed to whether the reference stopped early. The no-op has a known zero
label and remains outside prediction metrics.

The primary target is signed loose IFEval loss,
`d = q_reference - q_action`. Fit the listed estimators to loose training
labels and score loose test labels. Strict scoring is a locked sensitivity
mirror: fit the same listed estimators to strict training labels and score
strict test labels, with unchanged rows, features, estimator parameters, and
decision rule.

## Prediction seal, one opening, and decision

Before any test outcome is generated, read, or scored, write and hash a
prediction seal containing the complete fitted loose and strict estimator
parameters, scaler parameters, lock and code hashes, every eligible test
`(prompt_id, action, model)` prediction, the full early-EOS ledger, and the
complete H8 probe records. The outcome collector must verify this seal hash and
its own locked dependency hashes before it can run. A missing, invalid, or
post-outcome seal is an `operational_integrity_failure`.

Primary error is prompt-equal loose MSE: average the two action errors within
each eligible prompt, then average prompts. Report per-action and combined
positive, zero, and negative test-label counts. The information floor is 20
distinct eligible test prompts with nonzero loose loss for either action.

For every comparator `C` in `action_mean`, `B0_L`, `B1_L`, `B2_L`, and
`B3_L`, calculate:

```text
relative_skill(C) = (MSE_C - MSE_B4_L) / MSE_C
```

If `MSE_C = 0`, set `relative_skill(C) = null` and `B4_L_beats_C = false`.
Use 2,000 seed-0 paired prompt-cluster bootstrap resamples, preserving both
action rows and repeated prompt multiplicities. The paired interval is for
`MSE_C - MSE_B4_L`.

The final status is exactly one of:

- `operational_integrity_failure`: any lock, provenance, full-pair, probe,
  seal, runner, or outcome-integrity violation. Make no scientific claim.
- `inconclusive_information_floor`: integrity passes but fewer than 20 eligible
  test prompts have nonzero loose loss for either action. Do not run a sweep.
- `negative`: integrity and floor pass, but B4 fails the required result against
  one or more comparators.
- `positive_exploratory`: integrity and floor pass, and against every comparator
  B4 has at least 5 percent relative skill and a 95 percent paired interval
  entirely above zero.

A strict-mirror reversal does not alter the loose status, but limits any
positive result to loose IFEval compliance. No status establishes a controller,
deployment memory saving, or general answer-quality claim. After any
non-operational result, do not tune the horizon, summary, actions, folds, or
model on these 76 labels.

## Checks before collection

1. Materialize and audit `data/lookahead-v1/roster-derivation.json` against the
   existing seed order and exposure ledger.
2. Prove exact 120-prompt, 240-row training provenance, including every raw
   run/outcome hash and full-pair boundary parity.
3. Freeze and test the lock, H8 record schema, prediction-seal schema, and
   outcome-collector seal dependency before test probes exist.
4. Demonstrate that an outcome collection invocation fails closed without the
   exact sealed prediction manifest.
5. Record population-level probe and outcome costs. The single engineering
   measurement does not estimate deployment overhead.
