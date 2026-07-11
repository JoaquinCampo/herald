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

## 2026-07-10: sustained StreamingLLM interval-32 held-out pilot

On Orion, Llama-3.1-8B-Instruct ran five held-out IFEval prompts
(`ifeval-1069`, `ifeval-1075`, `ifeval-1087`, `ifeval-1107`, and
`ifeval-1128`) with the frozen StreamingLLM alarm, cache-fork attempts,
and decode-cache pruning every 32 tokens. The runner recorded five paired
plain baselines and 20 candidate episodes, four ratios per prompt.

Command:

```bash
export PATH="/clustergpu/home/jcampo/.local/bin:$PATH"
HF_HUB_OFFLINE=1 uv run --no-sync python \
  scripts/run_live_controller.py \
  --compressors streaming_llm \
  --ratios 0.10,0.125,0.20,0.25 \
  --sustain-interval 32 \
  --limit-prompts 5 \
  --out-dir results/live_controller_cachefork_sllm_r32 \
  --bundle-dir results/predictor/alarm_bundle \
  --references-dir results/sweep/llama/ifeval/references \
  --device cuda --dtype bfloat16 --stride 16
```

The strict five-prompt triage used every active threshold but temporarily
set `--min-pairs 5`; the active N=30 report was also written. Neither is a
deployment claim. Both reports use 2,000 prompt-cluster bootstrap resamples.

| Ratio | Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV mean, lower 95% |
| --- | ---: | ---: | ---: | ---: |
| 0.10 | 20.0% | 0.0% | 15.1% | 4.1%, -17.4% |
| 0.125 | 40.0% | 60.0% | 12.3% | 7.9%, -15.3% |
| 0.20 | 40.0% | 60.0% | 6.4% | 21.2%, -7.0% |
| 0.25 | 80.0% | 80.0% | 11.4% | 23.2%, -4.3% |

No ratio cleared the quality, speed, and positive lower-bound KV-savings
gates. Ratios 0.125 through 0.25 also breached the major-damage gate.
Therefore no cell was expanded to the full held-out split. All 20 cache-fork
attempts reported zero recomputed prefill tokens, and the GPU was clean after
completion. Raw artifacts are in `results/live_controller_cachefork_sllm_r32/`:
`baseline.jsonl`, `episodes.jsonl`, `pilot_triage_report.json`, and
`deployment_report_min30.json`.

## 2026-07-10: sustained StreamingLLM interval-1 mechanism pilot

The interval-32 pilot could lose its memory advantage between pruning
boundaries. To isolate that mechanism without changing the frozen alarm,
prompts, ratios, or controller path, the same five-prompt held-out grid ran
with `--sustain-interval 1`. This is a mechanism check, not a deployment
claim or an interval-selection procedure.

Command:

```bash
export PATH="/clustergpu/home/jcampo/.local/bin:$PATH"
HF_HUB_OFFLINE=1 uv run --no-sync python \
  scripts/run_live_controller.py \
  --compressors streaming_llm \
  --ratios 0.10,0.125,0.20,0.25 \
  --sustain-interval 1 \
  --limit-prompts 5 \
  --out-dir results/live_controller_cachefork_sllm_r1 \
  --bundle-dir results/predictor/alarm_bundle \
  --references-dir results/sweep/llama/ifeval/references \
  --device cuda --dtype bfloat16 --stride 16
```

The strict five-prompt triage again used every active threshold with
`--min-pairs 5`, and a separate active N=30 report was written. Both use
2,000 prompt-cluster bootstrap resamples.

| Ratio | Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV mean, lower 95% |
| --- | ---: | ---: | ---: | ---: |
| 0.10 | 20.0% | 0.0% | 16.1% | 7.4%, -17.4% |
| 0.125 | 20.0% | 0.0% | 93.6% | 8.6%, -16.1% |
| 0.20 | 40.0% | 60.0% | 8.9% | 23.0%, -6.7% |
| 0.25 | 60.0% | 60.0% | 9.3% | 29.8%, -3.0% |

Per-token pruning improved mean peak-KV savings at the higher ratios, but
no lower confidence bound became positive. It also did not repair the
quality failures caused by the initial compression decision, and it
breached the end-to-end speed limit at every ratio. No cell was expanded.
All 20 cache-fork attempts again reported zero recomputed prefill tokens.
The interval-1 and interval-32 grids therefore exhaust sustained-pruning
frequency as a standalone remedy. Raw artifacts are in
`results/live_controller_cachefork_sllm_r1/`: `baseline.jsonl`,
`episodes.jsonl`, `pilot_triage_report.json`, and
`deployment_report_min30.json`.

## 2026-07-10: frozen StreamingLLM gate did not alter sustained control

The next existing policy lever was the frozen StreamingLLM scorer gate at
the in-distribution ratio 0.25. It was evaluated with interval-32 sustained
pruning, the unchanged frozen alarm, and the same five held-out prompts. The
gate model and its threshold were loaded from
`results/predictor/gate_bundle/streaming_llm/`, not refit or adjusted.

Command:

```bash
export PATH="/clustergpu/home/jcampo/.local/bin:$PATH"
HF_HUB_OFFLINE=1 uv run --no-sync python \
  scripts/run_live_controller.py \
  --compressors streaming_llm --ratios 0.25 \
  --sustain-interval 32 \
  --gate-dir results/predictor/gate_bundle \
  --limit-prompts 5 \
  --out-dir results/live_controller_cachefork_sllm_r32_gated \
  --bundle-dir results/predictor/alarm_bundle \
  --references-dir results/sweep/llama/ifeval/references \
  --device cuda --dtype bfloat16 --stride 16
```

The gate skipped zero of the live attempts, so this policy produced the same
commit positions and failure pattern as ungated interval-32 ratio 0.25. Its
five-prompt triage report has quality upper 80.0%, major-damage upper 80.0%,
slowdown upper 11.6%, and peak-KV savings mean 23.2% with lower bound -4.3%.
It fails all non-sample-size deployment gates. No full split was run. All
attempts reported zero recomputed prefill tokens; Orion returned to the
keepalive-only 653 MiB state. Raw artifacts are in
`results/live_controller_cachefork_sllm_r32_gated/`.

## 2026-07-10: minimum-switch-position policy is not a new live candidate

A read-only review considered a fixed minimum switch position as a way to
veto early StreamingLLM commits. The existing cell position-gate experiment
already tests the stricter form of that idea: it selects the earliest
per-(task, ratio) safe position using donor and training compressors, then
keeps the first alarm-qualified position at or after that threshold. Its
canonical held-out results are recorded in
`results/predictor/experiments/position_gate/summary.json` and its
cross-compressor limitations in `docs/implementation/exhaustion_report.md`.
At the strict `epsilon=0.01` setting, the safe flat-block policy achieved
only 1.5% to 2.1% replay savings across the three held-out compressors.

This review does not create a new policy selection or deployment result.
The five current live triage prompts (`ifeval-1069`, `ifeval-1075`,
`ifeval-1087`, `ifeval-1107`, and `ifeval-1128`) belong to the frozen
canonical test population. Their live outcomes must remain evaluation-only:
no position threshold, tolerance, or cell scope may be selected or retuned
using them. Moreover, replay savings and `1 - s / ref_len` do not measure
end-to-end latency or isolated peak-KV savings.

No minimum-position live run was launched. A future test would require a
new preregistered selector trained only on non-triage, donor-compressor
prompt groups, with the alarm, threshold, attempt grid, and deployment
contract unchanged. It would still need to clear the five-prompt triage
before any N>=30 paired evaluation. This branch is otherwise exhausted as
a standalone policy lever.

## 2026-07-10: StreamingLLM cache-fork branch exhaustion

The direct cache-fork StreamingLLM branch is closed in
`deployment_exhaustion_report.md`. A code-level preflight rejected
storage-reusing fork implementations: exact rollback requires the full
reference cache throughout the grace decision, while the standard cache API
requires StreamingLLM's discontiguous retained tokens to be materialized as
a contiguous candidate tensor. In-place mutation instead needs an
order-equivalent rollback shadow buffer. Neither changes the exact
reference-plus-candidate peak-KV accounting.

A segmented-cache Llama attention implementation would be a distinct,
preregistered model-runtime project, not a legal configuration variant. The
report states the resulting scope limit explicitly and links the
hash-verified raw artifacts through `deployment_evidence_manifest.json`.
