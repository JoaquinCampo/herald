# Preregistration: irreversible pre-compression selector

Date: 2026-07-11. Status: frozen before implementation and live execution.

## Hypothesis

A compressor-specific selector using only uncompressed reference features can choose one safe ExpectedAttentionStats ratio-0.25 switch before compression. Compressing the GPU cache in place once, with no grace window and no rollback image, will retain the host-rollback branch's quality and positive real KV savings while removing its repeated transfer and grace-attempt slowdown.

## Frozen data protocol

- Source table: `results/expected_stats_predictor_full_s0_v2/switch_dataset.parquet`, 4,923 provenance-valid rows from 200 IFEval prompts, bound to sweep config SHA-256 `5b9a0938c1669d8b99ff0aef25573496029fddafc35286d3b084c663b1efcfa6` and ExpectedAttentionStats artifact SHA-256 `ca7450b7c388fb8612a32ed954ae72fbd7df810bcba7f7e353d6a13c10a21eaf`.
- Frozen evaluation population: the existing 46-prompt alarm-bundle target split. It is excluded from all fitting and threshold selection.
- Five-prompt rejection triage: `ifeval-1069`, `ifeval-1075`, `ifeval-1087`, `ifeval-1107`, `ifeval-1128`, evaluation-only.
- Development population: the existing 154 non-target prompts. Sort prompt IDs, assign every fifth identity to calibration, and use the remainder for training. This deterministic split is frozen before reading outcomes.
- Features: columns prefixed `feat__` plus the constant ratio. Probe, press, hybrid, output, damage, quality, and prompt-identity fields are forbidden inputs.
- Labels: a row is safe only when `dq <= 0.01` and `major_damage` is false. This is compressor-specific supervised selection, not reuse of the cross-compressor alarm-imitation gate.

## Model and threshold selection

Fit the existing three-seed XGBoost binary classifier family with the repository's frozen alarm hyperparameters; do not search model hyperparameters. Force one training thread so local and Orion reductions are deterministic. Aggregate seed probabilities by their mean. On calibration prompts, evaluate a deterministic grid of unique scores. For each threshold, each prompt commits at the earliest stride-qualified row predicted safe, or never commits.

Select the threshold with greatest analytical retained-token opportunity subject to both prompt-cluster bootstrap upper 95% bounds being at most 1% for paired damage incidence and major-damage incidence. Analytical opportunity is selection-only and never counts as deployment memory evidence. If no nontrivial threshold satisfies calibration, reject the selector without live triage.

## Test-first implementation boundary

First add a failing shipped-behavior test. It must drive `run_episode` in irreversible mode and prove that the selector sees only pre-compression rows, a rejection leaves the reference cache untouched, a commit compresses the same GPU cache object once, no grace tokens or rollback bytes are retained, and committed output matches the corresponding recorded hybrid semantics.

Implement only the deterministic split, feature-only bundle, threshold calibration, immutable provenance, irreversible runtime path, CLI wiring, accounting, and tests. Do not alter compressor ratio, sustained interval 32, stride 16, model, task, or deployment thresholds.

## Validation and stopping

1. Validate source completeness and hashes before fitting; reject duplicate/stale prompt rows.
2. Run local pytest, Ruff format/check, MyPy, and real CPU entry-point tests.
3. Verify and flat-sync permitted content to Orion; preflight GPU/process state and smoke-check every launch for at least ten seconds.
4. If calibration yields a nontrivial frozen selector, run the five frozen prompts once with paired live baselines and 2,000 prompt-cluster bootstrap resamples.
5. Do not expand after any non-sample-size gate failure. If all four gates pass or only sample-size uncertainty remains, run the remaining frozen target prompts to at least 30 unique pairs.
6. Only paired live runtime evidence decides deployment. Training replay, analytical retained tokens, allocator peaks, and the host-rollback result are not substitutes.
7. A rejection closes only this exact feature-only irreversible selector. Record it, rerank the backlog, and proceed to quantized/mixed-precision KV or another technically distinct hypothesis.
