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

## 2026-07-11: ExpectedAttentionStats cache-fork five-prompt triage

A complete 200-prompt Llama-3.1-8B-Instruct/IFEval ratio-0.25 sweep produced
4,923 provenance-valid switch rows and hybrid streams under immutable sweep
configuration SHA-256
`5b9a0938c1669d8b99ff0aef25573496029fddafc35286d3b084c663b1efcfa6`.
The frozen ExpectedAttentionStats calibration artifact digest was
`ca7450b7c388fb8612a32ed954ae72fbd7df810bcba7f7e353d6a13c10a21eaf`.
The exported bundle validates against its 46-prompt target split and records
source hashes for the switch parquet and stream archive.

Three fresh controller candidates used the same five frozen held-out prompts
(`ifeval-1069`, `ifeval-1075`, `ifeval-1087`, `ifeval-1107`, and
`ifeval-1128`), frozen bundle, compressor ratio, and cache-fork measurement.
The triage evaluator retained every active threshold, used five prompt
clusters and 2,000 bootstrap resamples, and wrote report-only results. It did
not select, retune, or promote any candidate. Interval 1 was the bounding
regrowth check after interval 32 missed both speed and KV gates.

| Candidate | Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV lower 95% | Failed gates |
| --- | ---: | ---: | ---: | ---: | --- |
| One-shot | 20.0% | 0.0% | 22.0% | -45.3% | quality, slowdown, KV savings |
| Sustained every 32 tokens | 0.0% | 0.0% | 27.1% | -41.0% | slowdown, KV savings |
| Sustained every token | 0.0% | 0.0% | 49.4% | -42.6% | slowdown, KV savings |

The artifacts are in `results/expected_stats_live_triage_one_shot_s0_v2/`,
`results/expected_stats_live_triage_sustained32_s0_v2/`, and
`results/expected_stats_live_triage_sustained1_s0_v2/`. Each has five paired
baselines and episodes, a complete immutable live manifest, and a
target-specific deployment report. No candidate may advance to N>=30,
broader model/task validation, or a deployment claim. This rejects the
evaluated ratio-0.25 controller modes, not ExpectedAttentionStats as a general
method.

## 2026-07-11: exact host-rollback ExpectedAttentionStats triage

The preregistered in-place host-rollback mechanism reused the candidate GPU
cache and retained an exact uncompressed rollback image on CPU. The shipped
runtime separately recorded host bytes and included layer-replacement
transients in isolated GPU-KV peak accounting. Two launch-preflight failures
(import path, then a non-empty evidence directory caused by log placement)
produced no episodes; both were corrected before the immutable run.

The five frozen prompts then ran once with ratio 0.25, sustained interval 32,
the unchanged ExpectedAttentionStats artifact and alarm, and 2,000
prompt-cluster bootstrap resamples. All live outputs preserved paired quality.

| Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV mean, lower 95% | Failed gate |
| ---: | ---: | ---: | ---: | --- |
| 0.0% | 0.0% | 29.3% | 24.4%, 11.5% | end-to-end slowdown |

This is the first evaluated mechanism in the program with a positive lower
confidence bound on real isolated GPU-KV savings and no paired quality damage,
but its 12.1% mean slowdown and 29.3% upper bound fail the frozen speed gate.
The non-sample-size failure forbids N>=30 expansion. It closes only synchronous
exact host rollback for this candidate. The result localizes the next problem:
avoid repeated device/host rollback transfers and grace attempts, rather than
retuning the compressor or weakening the contract.

## 2026-07-11: irreversible pre-compression selector triage

A new ExpectedAttentionStats-specific feature-only selector used the existing
4,923-row sweep with the frozen 46 target prompts excluded. Of 154 development
prompts, the sorted every-fifth rule assigned 31 to calibration and 123 to
training. The Orion-canonical three-seed bundle committed on 12 calibration
prompts with zero observed damage and 13.6% mean analytical opportunity. A
second Orion fit was byte-identical. Cross-architecture local fitting produced
a different threshold despite matching data and XGBoost versions, so the
bundle is explicitly bound to the Orion runtime; no frozen target outcome was
examined during that investigation.

The immutable five-prompt run made one irreversible in-place compression on
two prompts and no compression on three. All paired outputs preserved quality.

| Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV mean, lower 95% | Failed gates |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.0% | 0.0% | 12.7% | 9.6%, 0.0% | end-to-end slowdown, positive KV savings |

The repeated feature-only scoring path still added 10.8% mean end-to-end
slowdown, including on no-commit prompts, and conservative coverage left the
memory lower bound at zero. Both are non-sample-size failures, so no N>=30
expansion or threshold change is allowed. This closes the exact irreversible
feature-only selector branch. It strengthens two conclusions: decision
latency must be effectively free, and abstention-heavy token eviction cannot
prove population-level savings under the frozen contract.

## 2026-07-11: dependency-free int8 KV-cache triage

The preregistered model-native cache preserved every token, quantized completed
KV blocks to signed int8 with explicit per-vector float32 scales, and retained
the newest 128 tokens in bfloat16. Retained-byte accounting included quantized
values, scales, residual tensors, and conservative layer-replacement transient
peaks. No controller, alarm, rollback, or token eviction was involved.

The immutable five-prompt paired run had zero quality or major damage and a
positive real retained-KV savings bound, but failed end-to-end speed.

| Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV mean, lower 95% | Failed gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.0% | 0.0% | 16.8% | 21.5%, 13.9% | end-to-end slowdown |

Mean end-to-end slowdown was 8.2%, while per-token slowdown averaged 2.8% with
a 3.6% upper bound. The discrepancy is direct evidence that fixed quantize and
dequantize work on short generations breaks the deployment contract even when
steady-state token cost appears acceptable. The speed failure is non-sample-
size, so no N>=30 expansion or residual/scale retuning is allowed. This closes
only the exact dependency-free int8/per-vector-scale/residual-128 runtime.

## 2026-07-11: always-on low-ratio StreamingLLM triage

The preregistered no-controller path applied StreamingLLM ratio 0.05 directly
at prefill and maintained the same logical ratio every 32 decode tokens. It
used no cache fork, rollback, alarm, gate, or recomputation. All five paired
outputs preserved quality, but both runtime and retained-memory gates failed.

| Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV mean, lower 95% | Failed gates |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.0% | 0.0% | 98.7% | 3.7%, -15.6% | end-to-end slowdown, positive KV savings |

Per-token slowdown was consistently about 19%, showing sustained positional
pruning itself—not controller or rollback work—was expensive in this runtime.
At only 5% removal, pre-prune regrowth and prompt variation also overwhelmed
the small retained-byte benefit. Both failures are non-sample-size; no N>=30
expansion or ratio/interval retuning is permitted. This closes only the exact
always-on ratio-0.05 interval-32 implementation.

## 2026-07-11: always-on SnapKV ratio-0.25 triage

The first launch exposed KVPress's strict `query_length > window_size`
assertion on an exactly 64-token prompt. No parameter was changed. A uniform
wrapper was test-first added to skip compression whenever context length is not
larger than the default scoring window, matching the method's domain. The
incomplete artifacts were discarded and one fresh immutable run completed.

SnapKV produced large measured savings and apparently favorable end-to-end
wall time, but one prompt suffered major quality damage and terminated early.
The timing advantage is therefore not evidence of efficient equivalent work;
per-token slowdown was 33.9%.

| Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV mean, lower 95% | Failed gates |
| ---: | ---: | ---: | ---: | ---: | --- |
| 40.0% | 60.0% | -6.8% | 47.0%, 11.3% | quality, major damage |

The non-sample-size quality failures forbid expansion or ratio adjustment.
This closes only always-on SnapKV ratio 0.25 with its default window and the
uniform short-context guard. It also demonstrates why early termination must
not be interpreted as a speed win.

## 2026-07-11: native two-segment flash-attention preflight

After the additional runtime and quality stalls, the required strategic retreat
returned to the model attention path. Transformers 4.57.6 requires each cache
layer's `update` method to return contiguous key/value tensors to Llama's SDPA
interface. Orion's PyTorch 2.10 native flash-attention operator can instead
accept Llama's 32-query-head/8-KV-head geometry and returns per-query
log-sum-exp metadata, allowing mathematically exact attention over separate
sink and recent segments without repeating KV heads or materializing a
contiguous candidate.

A preregistered synthetic CUDA preflight compared that two-call representation
against one contiguous native flash-attention call. It used bfloat16 Llama
geometry, alternating measurement order, 50 warmups, and 200 synchronized
repetitions at each retained length.

| Retained tokens | Contiguous | Two segments | Slowdown | Max / mean absolute error |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0161 ms | 0.0747 ms | 362.5% | 0.00184 / 0.000188 |
| 1,024 | 0.0175 ms | 0.0750 ms | 329.3% | 0.00079 / 0.000082 |
| 4,096 | 0.0266 ms | 0.0778 ms | 192.4% | 0.00045 / 0.000041 |

Numerical acceptance passed, and separate segment storage avoids one full
contiguous candidate allocation. Runtime failed the frozen preflight rejection
criterion at every length: the second kernel dispatch and exact log-sum-exp
combination dominate one-token decode. No model-path implementation or frozen
prompt run is justified. This closes only the tested native two-call
flash-attention representation; a fused paged/segmented kernel remains a
distinct dependency- or kernel-development branch.

## 2026-07-11: always-on ExpectedAttentionStats sustained-32 triage

The frozen ExpectedAttentionStats artifact and ratio 0.25 ran directly at
prefill and every 32 decode tokens with no controller, alarm, gate, grace
window, cache fork, host copy, or rollback. All five paired outputs preserved
quality and the real retained-KV lower confidence bound was positive.

| Quality upper 95% | Major-damage upper 95% | Slowdown mean, upper 95% | Peak-KV mean, lower 95% | Failed gate |
| ---: | ---: | ---: | ---: | --- |
| 0.0% | 0.0% | 25.1%, 50.3% | 24.9%, 13.6% | end-to-end slowdown |

Per-token slowdown was 21.9% with a 22.8% upper bound. Removing all decision
and rollback machinery therefore did not remove the binding cost: repeated
ExpectedAttentionStats scoring, selection, and cache replacement is itself too
slow in the installed runtime. The non-sample-size speed failure forbids N>=30
expansion. This closes only the exact always-on ratio-0.25 sustained-32 path;
it does not retune the frozen ratio or interval.

## 2026-07-11: always-on Knorm ratio-0.25 one-shot triage

The weight-free Knorm press scored key vectors once at prefill and used no
controller, rollback, sustained decode compression, learned artifact, or
attention-score calculation. One of five prompts incurred 0.5 quality damage;
all four deployment gates failed.

| Quality mean, upper 95% | Major-damage mean, upper 95% | Slowdown mean, upper 95% | Peak-KV mean, lower 95% |
| ---: | ---: | ---: | ---: |
| 10.0%, 30.0% | 20.0%, 60.0% | 46.9%, 113.4% | -4.0%, -26.7% |

Per-token slowdown was 13.8%. One-shot prefill eviction was overwhelmed by
decode regrowth on this population, while full prefill scoring and cache
replacement still imposed material runtime cost. These are non-sample-size
failures, so no N>=30 expansion or ratio change is allowed. This closes only
the exact always-on Knorm ratio-0.25 one-shot path.

## 2026-07-11: reconstruction and adaptive-storage preflight

A post-stall code-level retreat inspected the installed KVPress mechanisms and
Transformers cache contract before another implementation. `KVzipPress`
performs context reconstruction through multiple additional model forwards and
its own documentation warns of 2–3x prefill overhead; its context manager also
compresses only after the enclosed initial forward, so it cannot be dropped
into the existing `generate()` path. `FastKVzipPress` requires model-specific
Hub gate weights that are not available in the offline runtime. AdaKV's
head-adaptive path retains dense fake keys and therefore cannot satisfy the
real-memory gate. Physical variable-length per-head storage and exact token
merging both require a custom attention representation or kernel because
Llama's cache API returns one dense tensor per layer.

These observations reject installed reconstruction/adaptive mechanisms as the
next practical deployment candidate; they do not claim that fused merge or
reconstruction kernels are impossible. The highest-value in-scope lever is now
the existing quality-safe int8 branch: remove its full-tensor float32 scale
arithmetic while preserving the same residual length and paired protocol.
