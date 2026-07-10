# Deployment experiment log

Append-only decisions and reproducible negative results for the active
deployment mission. Raw result directories remain untracked by policy.

## 2026-07-10: legacy live-controller audit

Command:

```bash
uv run python scripts/evaluate_deployment.py \
  --live-dir results/live_controller_v3 \
  --report-only --bootstrap-resamples 500
```

All 12 compressor and ratio configurations failed the deployment contract.
The artifact has no isolated KV-cache measurement, no ratio clears the 5%
slowdown upper confidence bound, and quality failures remain at several
compressor and ratio cells. This result is diagnostic only because the old
artifact cannot prove real KV savings.

## 2026-07-10: sparse-calibration N=32 exhaustion

Commands:

```bash
uv run python scripts/experiment_sparse_sampling.py \
  --mode sparse --n-values 32 --seeds 0,1,2
uv run python scripts/experiment_sparse_sampling.py \
  --mode dense-cal --n-values 32 --seeds 0,1,2
uv run python scripts/experiment_sparse_sampling.py \
  --mode mixed25 --n-values 32 --seeds 0,1,2
```

Neither uniform nor stratified sampling passed in any mode. The merged
`results/predictor/sparse_sampling/summary.json` still reports
`smallest_passing: null`. Mixed25 increased savings in some cells, but its
mean costs reached 0.027 to 0.039 for ExpectedAttention and 0.021 to 0.044
for StreamingLLM. Dense calibration did not stabilize the design. Increasing
N or mixing dense rows is therefore not the next deployment lever.

## 2026-07-10: direct cache-fork Orion smoke

Configuration: one held-out IFEval prompt, StreamingLLM ratio 0.25, frozen
gate and alarm, Llama-3.1-8B-Instruct on Orion.

- Plain baseline: 8.155 seconds, quality 1.0, peak KV 91,095,040 bytes.
- Cache-fork controller: 8.150 seconds, quality 1.0, peak KV 93,847,552
  bytes, commit at token 16.
- Both attempts reported zero recomputed prefill tokens.

The cache fork removes the latency bottleneck on this prompt, but one-time
prompt compression does not guarantee peak memory savings because generated
tokens grow the cache again. A held-out campaign and sustained decode-cache
candidate were launched from this result.
