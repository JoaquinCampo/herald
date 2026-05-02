# Cross-Press Transfer Analysis

CPU-only train-on-one-press / test-on-every-other-press matrix that
directly probes the HERALD black-box "compressor-agnostic" claim. If
the diagonal is high but the off-diagonal collapses, HERALD is a
per-compressor calibration framework, not a universal predictor, and
the paper framing must adjust.

## Why It Matters For HERALD

`gold/contribution-validation.md` ("Required Transfer Experiments")
makes cross-press transfer a headline result, not an appendix-only
analysis. The black-box claim is one of the six load-bearing words
in the HERALD contribution claim; without an explicit cross-press
matrix the predictor contribution silently degrades to "per-press
calibration." This analysis is the structural test: does a predictor
trained on press A retain its discrimination on press B at the
horizons HERALD wants to act on?

## What This Analysis Does Claim

- Per (horizon, feature_set, train_press, test_press) AUROC and
  AUPRC, with positive/negative counts and run counts on both sides
  of every cell.
- Diagonal mean vs off-diagonal mean per (horizon, feature_set), and
  the transfer gap `diagonal_mean - offdiagonal_mean`. A small gap
  is consistent with compressor-agnostic transfer; a large gap says
  the predictor is press-specific.
- Off-diagonal cells get a percentile bootstrap CI by resampling
  test `run_id`s. Diagonal CIs are the GroupKFold-fold spread
  (min/max across folds), not a bootstrap.
- Insufficient cells are flagged with a structured reason
  (`train_pos<5`, `test_neg<5`, ...) and excluded from the means.

## What This Analysis Does NOT Claim

- It is not the predictor. The cell scores come from a regularized
  logistic regression baseline, not the eventual XGBoost/HERALD
  model. The matrix is a *floor* for what cross-press transfer can
  look like with cheap features and a linear model.
- It is not Phase 1. Phase 0 has only **two** compressed presses
  (`snapkv`, `streaming_llm`), so the matrix is 2x2 (two diagonal
  cells, two off-diagonal cells). The transfer-gap point estimate
  is directional only at this scale and has no real CI; the summary
  marks this with `phase0_caveat`.
- It does not put a CI on the `transfer_gap` itself. With four
  cells in the entire matrix, a paired CI is not meaningful; that
  lands with Phase 1 once the press grid widens.

## Method

1. **Per-token table** — same as `information_ceiling.
   build_token_dataset`: compressed runs only, derived onset
   (looping first, non-termination proxy second), `relative_progress
   / output_length_so_far` position features, integer-encoded
   `task_code`, `compression_ratio` as ratio metadata, online
   features as-is, tokens at/past onset dropped.
2. **Cross-press metadata excludes `press_code`** — under
   train-on-A / test-on-B `press_code` is constant on each side
   *and* takes a literally unseen value at test, which would
   silently degrade the model and confound the matrix. The cross-
   press metadata baseline is `(token_pos, relative_progress,
   output_length_so_far, compression_ratio, task_code)`.
3. **Diagonal cells** (train_press == test_press) — `GroupKFold(
   groups=run_id)` over the runs of that press; the cell value is
   the mean fold AUROC. Run-level disjointness is enforced inside
   the press.
4. **Off-diagonal cells** (train_press != test_press) —
   `StandardScaler.fit_transform(train_rows)` then `LogisticRegression
   (class_weight='balanced')`, scored on `scaler.transform(test_rows)`
   and `clf.predict_proba(...)`. Run-level disjointness is automatic
   (a run lives in exactly one press).
5. **Insufficient-cell rule** — a cell is `insufficient` when *any*
   of `train_pos`, `train_neg`, `test_pos`, `test_neg` is below the
   threshold (default 5). Conflating train- and test-side imbalance
   would silently include broken cells in the means.
6. **Bootstrap** — off-diagonal cells: resample test `run_id`s with
   replacement, recompute AUROC, take 2.5/97.5 percentiles.
   Diagonal cells: report (min, max) across GroupKFold folds.
7. **Standardization** — fit on train rows only. Spelled out in the
   spec because it is easy to break by accident.

## Critical Decisions Locked In

- `press_code` is dropped from every cross-press feature set.
- Diagonal scoring is GroupKFold(run_id) within the press, not a
  random split.
- Off-diagonal cells train on the *full* train-press table. This
  is consistent with how HERALD would actually be deployed against a
  press it was not trained on; volume mismatches are reported via
  the per-cell counts so they can be inspected.
- Cells are marked insufficient *before* the model is fit, so
  degenerate cells never contribute a number to the matrix or to
  the diagonal/off-diagonal means.

## Files

- `src/herald/analysis/cross_press_transfer.py` — dataset reuse,
  per-cell evaluation (diagonal + off-diagonal), matrix builder,
  bootstrap CI, plotting, top-level entry point.
- `scripts/build_cross_press_transfer.py` — CPU CLI mirroring
  `build_information_ceiling.py`.
- `tests/analysis/test_cross_press_transfer.py` — synthetic two-
  press fixture, feature-set hygiene tests, insufficient handling,
  matrix-shape test, smoke run + CLI run, single-press blocker.

## Reproducing The Phase 0 Smoke Output

```sh
.venv/bin/python scripts/build_cross_press_transfer.py \
    --input results/phase0 \
    --output results/analysis/cross_press_transfer \
    --horizons 5,10,25,50 \
    --n-bootstrap 200
```

## Phase 0 Smoke Read

Phase 0 inputs: 114 compressed runs, 19,625 tokens after the at/past-
onset filter, two presses (`snapkv` 19 runs / `streaming_llm` 54
runs), 64 cells total, 0 insufficient.

Diagonal-mean / off-diagonal-mean / transfer gap (AUROC), per
(horizon, feature_set):

| H  | feature_set       | diag  | off   | gap   |
|----|-------------------|-------|-------|-------|
| 5  | online            | 0.609 | 0.359 | +0.250 |
| 5  | position_metadata | 0.579 | 0.467 | +0.112 |
| 5  | all               | 0.835 | 0.592 | +0.243 |
| 5  | entropy_only      | 0.594 | 0.394 | +0.200 |
| 10 | online            | 0.561 | 0.382 | +0.178 |
| 10 | position_metadata | 0.692 | 0.706 | -0.013 |
| 10 | all               | 0.832 | 0.642 | +0.190 |
| 10 | entropy_only      | 0.564 | 0.421 | +0.143 |
| 25 | online            | 0.514 | 0.444 | +0.070 |
| 25 | position_metadata | 0.797 | 0.779 | +0.018 |
| 25 | all               | 0.823 | 0.765 | +0.058 |
| 25 | entropy_only      | 0.507 | 0.459 | +0.048 |
| 50 | online            | 0.517 | 0.463 | +0.054 |
| 50 | position_metadata | 0.830 | 0.817 | +0.013 |
| 50 | all               | 0.832 | 0.801 | +0.032 |
| 50 | entropy_only      | 0.494 | 0.463 | +0.031 |

Pattern at this scale: the **online feature set transfer collapses
at short horizons** (H=5: 0.609 -> 0.359; entropy-only goes *below*
chance off-diagonal), suggesting the calibration of online-feature
distributions is press-specific. The **position_metadata transfer
is intact across horizons** (gap < 0.02 from H=10 onward), as
expected for universal proxies (position, ratio, task). The combined
`all` set inherits both: high diagonal (0.83-0.84 across horizons)
and a meaningful off-diagonal gap at short horizons (+0.24 at H=5,
+0.19 at H=10) that shrinks once position dominates the signal at
H=25/H=50.

**Phase 0 verdict on the predictor:** directionally consistent with
"online features need per-press calibration at short horizons"; *too
underpowered to claim* anything about cross-press transfer of the
HERALD predictor itself. Two presses with very different run counts
(19 vs 54) and a 2x2 matrix do not support a publishable transfer
claim. Phase 1's wider press grid is the real test; this scaffold
exists so the result can land immediately when the data does.

## Outputs

Under `<output>/`:

- `cross_press_transfer.parquet` — long format: rows of
  `(horizon, feature_set, train_press, test_press, auroc, auprc,
   ci_lo, ci_hi, n_train_rows, n_train_pos, n_train_neg,
   n_train_runs, n_test_rows, n_test_pos, n_test_neg, n_test_runs,
   n_features, insufficient, insufficient_reason)`.
- `cross_press_transfer_summary.json` — input paths, config,
  presses found, label counts by press/horizon, insufficient cells,
  diagonal/off-diagonal means + transfer gap per (horizon, feature
  set), Phase 0 caveat.
- `cross_press_transfer_h{H}.png` — heatmap (rows = train press,
  cols = test press), default at the smallest horizon and the `all`
  feature set; the script also writes one heatmap per
  (horizon, feature_set) pair.

## Connection To The Predictor

The matrix here uses logistic regression. The Phase 2 predictor is
expected to be XGBoost over a richer feature set; if it does *not*
beat this LR's diagonal, the predictor contribution collapses. If it
beats the diagonal but not the off-diagonal, the contribution narrows
to "per-press HERALD" and the paper's transfer wording must be
softened. If it beats both, that is the headline cross-press result
the contribution claim asks for, and this analysis is the baseline
against which it is reported.
