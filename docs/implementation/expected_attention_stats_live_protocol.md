# ExpectedAttentionStatsPress live-evaluation protocol

This is the prescribed path for a new ExpectedAttentionStatsPress candidate.
It produces a fresh frozen alarm bundle. Do not reuse the existing
`expected_attention` bundle, its replay targets, or its selector.

## Invariants

- Statistics use only prompt IDs outside the deterministic held-out split.
- The statistics artifact contains query-moment shapes, calibration IDs,
  excluded test IDs, settings, and a content digest.
- The alarm bundle records that digest. The live controller refuses a missing
  or mismatched digest.
- Hybrid streams are extracted from the same switch parquet and carry a
  row-identity key. Bundle export rejects a row-order mismatch.
- Run the live controller only on the frozen test prompt IDs from its new
  `fidelity_targets.json`.

## Commands

Run the GPU stages on Orion after its required preflight. Replace
`$STATS`, `$SWEEP`, `$PREDICTOR`, and `$BUNDLE` with new empty output paths.

```bash
HF_HUB_OFFLINE=1 uv run python scripts/collect_expected_attention_stats.py \
  --out-dir "$STATS" \
  --prompts-per-task 200 \
  --split-seed 0 \
  --max-calibration-prompts 100 \
  --max-prompt-tokens 1024

HF_HUB_OFFLINE=1 uv run python scripts/run_sweep.py \
  --models llama --tasks ifeval \
  --compressors expected_attention_stats --ratios 0.25 \
  --prompts 200 --switch-stride 16 --hybrid-batch 1 \
  --expected-attention-stats "$STATS" \
  --results-dir "$SWEEP"

uv run python scripts/build_switch_dataset.py "$SWEEP" \
  --tasks ifeval --models llama \
  --out "$PREDICTOR/switch_dataset.parquet"

uv run python scripts/extract_hybrid_streams.py "$SWEEP" \
  --parquet "$PREDICTOR/switch_dataset.parquet" \
  --out "$PREDICTOR/hybrid_streams_ifeval.npz" --task ifeval

uv run python scripts/export_alarm_bundle.py \
  --parquet "$PREDICTOR/switch_dataset.parquet" \
  --streams-npz "$PREDICTOR/hybrid_streams_ifeval.npz" \
  --compressors expected_attention_stats \
  --expected-attention-stats "$STATS" --out-dir "$BUNDLE"

HF_HUB_OFFLINE=1 uv run python scripts/run_live_controller.py \
  --bundle-dir "$BUNDLE" --out-dir results/live_expected_attention_stats \
  --references-dir "$SWEEP/llama/ifeval/references" \
  --compressors expected_attention_stats --ratios 0.25 \
  --expected-attention-stats "$STATS"

uv run python scripts/evaluate_live_fidelity.py \
  --live-dir results/live_expected_attention_stats \
  --targets "$BUNDLE/fidelity_targets.json"

uv run python scripts/evaluate_deployment.py \
  --live-dir results/live_expected_attention_stats \
  --bootstrap-resamples 2000
```

The five-prompt prompt-disjoint triage may use `--limit-prompts 5` on the
live command. It is not contract evidence. Only a strict survivor may run
the full frozen test set and then the N>=30 paired deployment contract.
