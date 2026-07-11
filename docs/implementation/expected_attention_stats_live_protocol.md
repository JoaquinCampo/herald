# ExpectedAttentionStatsPress live-evaluation protocol

This is the prescribed path for a new ExpectedAttentionStatsPress candidate.
It produces a fresh frozen alarm bundle. Do not reuse the existing
`expected_attention` bundle, its replay targets, or its selector.

## Invariants

- Statistics use only prompt IDs outside the deterministic held-out split.
- The statistics artifact contains query-moment shapes, calibration IDs,
  excluded test IDs, a tokenized-input fingerprint, settings, and a content
  digest.
- The alarm bundle records that digest. The live controller refuses a missing
  or mismatched digest.
- Hybrid streams are extracted from the same switch parquet and carry a
  row-identity key and source-parquet SHA-256. Bundle export rejects a
  row-order or source mismatch.
- The switch parquet embeds the SHA-256 of the completed sweep's root
  `config.json`. Bundle export verifies that config and its expected-stats
  digest before training an alarm.
- Bundle export requires a new output directory, stages every booster and
  target, and atomically publishes the complete bundle. Its source provenance
  is duplicated in every bundle and target, then revalidated before live use.
- Run the live controller only on the frozen test prompt IDs from its new
  `fidelity_targets.json`.
- Each live directory has an immutable identity manifest, including exact
  candidate ratios, required prompt coverage, and bundle and gate directory
  digests. Use a fresh empty directory for each candidate, prompt limit, and
  sustain mode. Resume only with the exact same configuration and explicit
  `--resume`.
- The memory gate uses end-to-end retained KV-cache bytes, including the
  concurrent held-reference and forked grace cache, never allocator memory.

## Commands

Run the GPU stages on Orion after its required preflight. Orion uses a
flat checkout, so run from `/clustergpu/home/jcampo/herald-v2` and omit the
local `scripts/` prefix. Replace `$STATS`, `$SWEEP`, `$PREDICTOR`,
`$BUNDLE`, and `$LIVE` with new empty output paths.

```bash
export PATH="/clustergpu/home/jcampo/.local/bin:$PATH"

HF_HUB_OFFLINE=1 uv run --no-sync python collect_expected_attention_stats.py \
  --out-dir "$STATS" \
  --prompts-per-task 200 \
  --split-seed 0 \
  --max-calibration-prompts 100 \
  --max-prompt-tokens 1024

HF_HUB_OFFLINE=1 uv run --no-sync python run_sweep.py \
  --models llama --tasks ifeval \
  --compressors expected_attention_stats --ratios 0.25 \
  --prompts 200 --switch-stride 16 --hybrid-batch 1 \
  --expected-attention-stats "$STATS" \
  --results-dir "$SWEEP"

uv run --no-sync python build_switch_dataset.py "$SWEEP" \
  --tasks ifeval --models llama \
  --out "$PREDICTOR/switch_dataset.parquet"

uv run --no-sync python extract_hybrid_streams.py "$SWEEP" \
  --parquet "$PREDICTOR/switch_dataset.parquet" \
  --out "$PREDICTOR/hybrid_streams_ifeval.npz" --task ifeval

uv run --no-sync python export_alarm_bundle.py \
  --parquet "$PREDICTOR/switch_dataset.parquet" \
  --streams-npz "$PREDICTOR/hybrid_streams_ifeval.npz" \
  --sweep-config "$SWEEP/config.json" \
  --compressors expected_attention_stats \
  --expected-attention-stats "$STATS" --out-dir "$BUNDLE"

HF_HUB_OFFLINE=1 uv run --no-sync python run_live_controller.py \
  --bundle-dir "$BUNDLE" --out-dir "$LIVE" \
  --references-dir "$SWEEP/llama/ifeval/references" \
  --compressors expected_attention_stats --ratios 0.25 \
  --expected-attention-stats "$STATS"

uv run --no-sync python evaluate_live_fidelity.py \
  --live-dir "$LIVE" \
  --targets "$BUNDLE/fidelity_targets.json" --skip-static

uv run --no-sync python evaluate_deployment.py \
  --live-dir "$LIVE" \
  --target-compressor expected_attention_stats --target-ratio 0.25 \
  --bootstrap-resamples 2000
```

The five-prompt prompt-disjoint triage may use `--limit-prompts 5` on the
live command with a distinct `$LIVE` directory. It is not contract evidence.
Run sustained candidates in their own directories, for example with
`--sustain-interval 32`, then evaluate with
`--target-sustain-interval 32`. Only a strict survivor may run a new full
frozen-test directory and then the N>=30 paired deployment contract. The
contract evaluator rejects partial manifest coverage, mixed modes, duplicate
prompt evidence, or a target not named in the frozen run manifest.
