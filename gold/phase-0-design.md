# Phase 0 Design: Engineering Dry-Run

**Status**: design locked, implementation not yet started.

**Scope**: 20 GSM8K prompts, 1 task, 2 presses (StreamingLLM + SnapKV),
3 ratios (0.5, 0.875, 0.9375), greedy decoding, dense matched-prefix
replay. 140 runs total (20 baseline + 20 × 2 × 3 compressed).

**Goal**: validate the end-to-end measurement substrate that Phases
1-4 will scale on top of. No science claims. The deliverable is
infrastructure that compounds: replay code, parquet schema, metric
pipeline, repair path, sampling-rate study harness.

This document resolves seven design questions. Each decision is
justified against the ultimate goal in `ultimate-goal.md` (does this
build infrastructure that compounds, or only solve Phase 0?).

## Q1: Matched-Prefix Replay Mechanism

**Decision**: single teacher-forced forward pass per compressed run.

For a compressed run that emitted tokens `g_1..g_N` after prompt `P`:
call `model(input_ids = P + g_1..g_N)` with no press context, no
`generate`. Slice `logits[:, prompt_len-1 : -1, :]` to get the
uncompressed model's distribution at every generated position
conditioned on the matched prefix. Top-K selection is done
position-by-position on GPU; the full `(N, V)` tensor is never
materialized on CPU.

**Index alignment**: for generated token `g_t` (1-indexed), the
prediction distribution lives at sequence position `prompt_len + t -
1` in the forward pass output. That position predicts `g_t`.

**Sanity checks (gate 3 and 4 below)**:
- *t=0 identity*: replay's first-position distribution must match the
  paired baseline run's first-position distribution within fp16 noise
  (both condition on prompt only).
- *Press=none full-trace identity*: for baseline runs, replay along
  the realized sequence must yield ~0 per-token JS against the
  baseline's own generation-time scores.

**Rationale vs alternatives**: per-position separate forwards (B) is
N× more compute with no benefit; batched windowed replay (C) adds
complexity without saving compute. Teacher forcing is exactly
"conditioning on the matched prefix at every position" by definition.

**Compounds because**: Phases 1-3 reuse this exact code at scale; the
only thing that changes is the cell count.

## Q2: Storage Schema

**Decision**: Parquet, hive-partitioned by `(press, ratio)`. Three
artifact families joined on `(run_id, token_pos)`. Union top-K=128
storage **plus** GPU-time exact full-vocab scalars (no truncation
bias on headline metrics).

### Layout

Raw per-run artifacts are written during the sweep and concatenated
into the final hive-partitioned dataset by `metrics finalize`. The
two layers are separated so the live sweep never touches the final
dataset and finalize is purely a compaction step.

```
results/phase0/
  raw/
    runs/<run_id>.parquet                   # 1 row; per-run metadata + outcome
    tokens/<run_id>.parquet                 # 1 row / gen_pos; features
    replay/<run_id>.parquet                 # 1 row / gen_pos; both dists
  final/
    runs.parquet                            # all runs concatenated
    tokens/press=X/ratio=R/tokens.parquet   # hive-partitioned
    replay/press=X/ratio=R/replay.parquet   # hive-partitioned
  metrics/
    token_metrics.parquet                   # JS/KL/rank per (run, pos)
    trajectory_metrics.parquet              # NLL ratio, FDP, sum_KL per run
    sequence_metrics.parquet                # BERTScore, cosine, edit, ROUGE-L
    outcome.parquet                         # paired cells per (prompt, press, ratio)
    tags.parquet                            # diagnostic tags per run
    alignment.parquet                       # pairwise corr/AUROC matrix
  manifests/
    phase0-random-manifest.json             # canonical 20 prompts
    phase0-stress-manifest.json             # optional known-failure prompts
```

### `runs.parquet` columns

`run_id` (str), `prompt_id` (str), `press` (str), `compression_ratio`
(float), `model` (str), `model_revision` (str|null), `tokenizer_revision`
(str|null), `dtype` (str), `device_class` (str), `seed` (int),
`max_new_tokens` (int), `decoding_config` (struct: do_sample, temperature,
top_p, top_k, ...), `prompt_text` (str), `prompt_hash` (str, sha256 of
prompt_text), `task` (str), `baseline_run_id` (str, FK to runs.run_id;
**self-link for baseline rows**, never null — see below),
`generated_text` (str), `generated_token_ids` (list[int]),
`num_tokens_generated` (int), `stop_reason` (str), `predicted_answer`
(str|null), `ground_truth` (str), `correct` (bool|null), `catastrophes`
(list[str]), `replay_status` (enum: ok, failed, retried), `replay_error`
(str|null), `created_at` (timestamp), `herald_git_sha` (str).

The 14 generation-determinism columns (`model`, `model_revision`,
`tokenizer_revision`, `dtype`, `device_class`, `seed`, `max_new_tokens`,
`decoding_config`, `prompt_text`, `prompt_hash`, `task`, `press`,
`compression_ratio`, `herald_git_sha`) constitute the **determinism
manifest**. Any field that affects the realized token stream under
greedy decoding is persisted. Repair re-runs with these exact values;
if any field is missing, the repair is rejected. The realized
`generated_token_ids` is also persisted on `runs.parquet` (not only on
`tokens.parquet`) so the repair self-check can compare the
re-generated sequence against the original record without a parquet
join.

**`baseline_run_id` semantics**: for compressed runs (`press != "none"`),
`baseline_run_id` points at the FK of the paired `press="none"` run for
the same prompt. For baseline runs themselves, `baseline_run_id =
run_id` (self-link). This makes gate 2 ("every `runs.parquet` row
resolves its `baseline_run_id`") uniform across compressed and
baseline rows; consumers that want only "true paired baselines" filter
on `baseline_run_id != run_id`.

### `tokens.parquet` columns

`run_id` (str), `token_pos` (int, 0-indexed in generation), `token_id`
(int), `token_str` (str), plus the existing `TokenSignals` fields
(`entropy`, `top1_prob`, `top5_prob`, `top5_logprobs`, `h_alts`,
`avg_logp`, `delta_h`, `delta_h_valid`, `kl_div`, `top10_jaccard`,
`eff_vocab_size`, `tail_mass`, `logit_range`, `lookback_ratio`).

Note: `kl_div` here is the existing per-token *consecutive-step* KL
already computed in `signals.py` (KL between the distribution at step
`t` and step `t-1` of the same compressed run); it is **not** the
cross-distribution KL between compressed and uncompressed, which lives
in `replay.parquet` as `kl_unc_comp_full` / `kl_comp_unc_full`. The
column name matches the existing `TokenSignals.kl_div` field for
consistency with the codebase.

Predictor-feature-only by design. No per-position cross-distribution
metrics live here.

### `replay.parquet` columns

`run_id` (str), `token_pos` (int), `realized_token_id` (int),
`union_top_k_token_ids` (list[int], length up to 2K = 256),
`logprobs_compressed` (list[float], same length), `logprobs_uncompressed`
(list[float], same length), `tail_mass_compressed` (float),
`tail_mass_uncompressed` (float), `realized_logprob_compressed` (float),
`realized_logprob_uncompressed` (float), and the **GPU-time exact
scalars** computed before dropping to top-K: `js_full` (float),
`kl_unc_comp_full` (float), `kl_comp_unc_full` (float), `top1_match`
(bool), `top1_rank_comp_under_unc` (int), `top1_rank_unc_under_comp`
(int).

The two `top1_rank_*` integers are **exact** because they are computed
on GPU against the full vocabulary before truncation; they are not
recovered from top-K storage. The top-K storage is for analysis
flexibility (recompute KL with different K, audit truncation bias,
inspect the full head of either distribution).

Headline metrics never depend on top-K truncation.

### `replay.parquet` is also produced for baseline runs

Baseline runs (`press="none"`) emit replay artifacts too, with the
"compressed" and "uncompressed" distributions both being the same
uncompressed model. This is what powers gate 4 (press=none identity)
without a separate duplicate sanity cell.

## Q3: Metric Computation Pipeline

**Decision**: new subpackage `herald/metrics/`. GPU-bound replay +
exact scalars happen *inline* during the sweep; everything else is
offline parquet processing, batched and re-runnable.

### Module layout

```
herald/metrics/
  __init__.py
  io.py            # parquet schemas, readers, writers, FK joins
  replay.py        # GPU: matched-prefix replay forward + on-the-fly scalars
  token.py         # post-hoc top-K-based recomputation + truncation-bias audit
  trajectory.py    # NLL ratio, first-divergence-point, sum_KL per run
  sequence.py      # BERTScore, embedding cosine, edit, ROUGE-L per run
  outcome.py       # paired cells: gross_harm/help, net_delta
  tags.py          # diagnostic tags (looping, drift, format break, non-term)
  alignment.py     # pairwise Spearman + AUROC across all metric families
  cli.py           # `herald metrics finalize|build|repair`
```

`analysis/` is unchanged: it stays for post-hoc visualization and
slicing of trained-predictor outputs. `metrics/` is the
*measurement substrate*; `analysis/` consumes it.

### Streaming vs batch boundary

- **GPU-bound, streamed per run inside `replay.py`**: while the full-
  vocab logprob tensors for both distributions are alive on GPU at
  position `t`, compute every scalar that needs the full distribution
  (KL both directions, JS, top-1 match, realized-token NLLs,
  full-vocab tail mass before truncation). Then drop to union top-K
  and emit one row to a parquet writer. Nothing involving the full
  vocab survives the position loop.
- **CPU-bound, batched offline**: trajectory/sequence/outcome/tags/
  alignment all read parquets. Trivially parallelizable across runs.
  Re-runnable when new metrics are added without redoing the GPU
  pass — the central reason for the GPU/CPU split.
- **`token.py`** is the *audit* path: recompute KL/JS from stored
  union top-K + tail bucket, compare against `js_full`/`kl_*_full`,
  log per-position truncation bias as a column on
  `token_metrics.parquet`.

### `replay.py` API shape

`RunResult` alone does not carry everything replay needs. Introduce a
`GenerationArtifact` wrapper (Pydantic model in `config.py` or a
dataclass in `metrics/replay.py`) that is the sole input to
`replay_run`:

```python
@dataclass
class GenerationArtifact:
    run_id: str
    input_ids: torch.Tensor          # shape (1, prompt_len)
    input_len: int
    generated_token_ids: list[int]   # the realized greedy sequence
    compressed_scores: list[torch.Tensor]  # one (vocab,) per gen pos, on GPU
    run_meta: RunResult              # for downstream parquet writers

def replay_run(
    model, tokenizer,
    artifact: GenerationArtifact,
    output_dir: Path,
    top_k: int = 128,
) -> ReplayMetrics: ...
```

Called immediately after `run_single` returns the artifact, before
`compressed_scores` leaves GPU. Same model instance, no second forward
for the compressed run, no reload. `run_single` is updated to return a
`GenerationArtifact` instead of constructing the `RunResult` inline;
the `RunResult` is finalized only after replay completes (so
`replay_status` and the determinism manifest are written atomically
with the rest of `runs.parquet`).

## Q4: Replay Sampling Rate Study

**Decision**: Phase 0 produces dense replay; the rate study runs
*offline by subsetting* dense replay to every-4 / every-8 schedules.

### Operational definitions

For each subsetting (dense / every-4 / every-8):

1. *Predictive utility against intrinsic ground truth*: derive
   `future_max_JS_H` for `H ∈ {5, 10, 25, 50}` from the subset,
   linearly interpolate to dense positions, compute Spearman vs the
   dense-derived feature.
2. *Trajectory ranking preservation*: per-run `trajectory_sum_JS`
   computed under each rate; Spearman across the 120 compressed runs
   between (dense, every-4) and (dense, every-8).
3. *Onset displacement*: visual distribution of "first-position-where-
   future_max_JS_H > T" displacement between dense and subsetted.
4. *Optional extrinsic AUROC*: only computed if `baseline_correct ∧
   compressed_wrong` positives ≥ 5 across the 120 runs. With N=120
   and sparse failures, AUROC may be undefined; this is a directional
   read, never a Phase 0 gate.

### Phase 0 deliverable

A table + plots reporting items 1-3 (and 4 if positives suffice). The
**binding** rate decision is deferred to an early Phase 1 sub-experiment
at 200 prompts/task. Phase 0's job is to ship the harness and a
directional read, not to settle the question.

**Compounds because**: dense replay in Phase 0 means any future
sampling-rate question is a parquet query, not a recompute.

## Q5: Concrete Phase 0 Cells

- *Prompts*: 20 GSM8K test-split items, sampled with
  `random.Random(42).sample(...)`. Persisted to
  `gold/phase0-random-manifest.json` (canonical, strictly random) and,
  optionally, `gold/phase0-stress-manifest.json` (handful of known-
  failure prompts harvested from existing `results/`, for the metric-
  light-up sanity read). Random and stress slices reported separately
  and never mixed in headline reporting. Both manifests pin
  `prompt_id` and `prompt_hash` for auditability.
- *Presses*: **StreamingLLM** (mask-based, Phase 4 mask-based choice)
  and **SnapKV** (eviction-based, Phase 4 eviction-based choice). The
  pairing makes Phase 0 directly de-risk Phase 4 mechanics, not just
  Phase 1 measurement.
- *Ratios*: **0.5, 0.875, 0.9375**. One mild, one mid, one cliff.
  Strict subset of Phase 1's `{0, 0.5, 0.75, 0.875, 0.9375, 0.96875}`.
- *Baselines*: 20 baseline runs (`press="none"`, `ratio=0`), one per
  prompt, FK-linked via `baseline_run_id`. Replay artifacts are
  produced for baselines too (powers gate 4).
- **Total**: 140 runs.

## Q6: Orchestration & Resumability

**Decision**: extend `experiment.py`, do not replace. Add four CLIs
and one Poe task.

### CLIs

- `herald phase0 run` — orchestrates the 20 baselines first (so each
  compressed cell can resolve `baseline_run_id` immediately on
  completion), then iterates the 6 (press, ratio) cells. Replay is
  inline. Reuses the existing per-prompt JSONL checkpoint pattern for
  live resumability. Skip-existing means "all three per-run parquets
  exist *and* `runs.parquet` row has `replay_status='ok'`."
- `herald metrics finalize` — concatenates per-run parquets into the
  hive-partitioned dataset under `results/phase0/{runs,tokens,replay}/`.
  Idempotent.
- `herald metrics build` — runs offline (CPU) trajectory / sequence /
  outcome / tags / alignment from finalized parquets. Re-runnable.
  This is where new metric definitions land later.
- `herald metrics repair --run-id X` — re-runs compressed generation
  + replay for one failed run. Reads the determinism manifest from
  `runs.parquet`; rejects repair if any required field is missing.
  Determinism rests on the manifest, not on implicit defaults.

### Poe task

```toml
phase0 = ["phase0-run", "metrics-finalize", "metrics-build"]
phase0-run        = "uv run herald phase0 run"
metrics-finalize  = "uv run herald metrics finalize"
metrics-build     = "uv run herald metrics build"
```

`metrics build` is **manual by default** so metric iteration does not
require re-orchestrating. The Poe task gives a one-command smoke path
when wanted.

### Determinism manifest (repair invariants)

Every field in `runs.parquet` listed below must be present for a run
to be repairable. If any is missing, `metrics repair` errors out:

`model`, `model_revision`, `tokenizer_revision`, `dtype`,
`device_class`, `task`, `prompt_id`, `prompt_text`, `prompt_hash`,
`press`, `compression_ratio`, `max_new_tokens`, `decoding_config`,
`seed`, `herald_git_sha`.

`prompt_text` is persisted on `runs.parquet` directly (not only in the
manifest) so repair never depends on an external file. `prompt_hash`
is verified against `prompt_text` at repair time; mismatch is a hard
error.

Greedy is deterministic given these. If the realized
`generated_token_ids` from a repair diverge from the original record
(stored on `runs.parquet`), the repair fails loudly; this is the
self-check that the manifest is sufficient.

### Replay inline integration

```python
def run_single_with_replay(...) -> tuple[RunResult, ReplayMetrics]:
    # 1. compressed generation under press(model) context
    # 2. exit press context, keep compressed scores on GPU
    # 3. uncompressed forward on full sequence (P + g_1..g_N)
    # 4. position-by-position: compute exact scalars, drop to top-K
    # 5. emit per-run parquet rows for runs/tokens/replay
    # 6. set replay_status; on exception, set 'failed' + replay_error
```

## Q7: Phase 0 Success Gates

Phase 0 is **engineering-green** iff gates 1-7 pass. Gates 8-9 are
scientific reads on a small sample; they are reported but do not
block scaling.

1. **Completion**: ≥ 95% of 140 runs produce `replay_status='ok'`.
   Non-ok runs have a recorded failure mode and a documented
   attempted repair.
2. **Schema integrity**: every `(run_id, token_pos)` in
   `tokens.parquet` has a matching row in `replay.parquet` (Phase 0
   is dense). Every `runs.parquet` row resolves its
   `baseline_run_id`. All parquet schemas roundtrip cleanly through
   polars + pyarrow. Determinism manifest is fully populated for
   every run.
3. **Replay correctness — t=0 identity**: for every run, JS at the
   first generated position between the replay distribution and the
   paired baseline's first-position distribution is below the
   empirical fp16/CUDA noise floor. Calibrated on Orion (RTX 5090,
   Qwen2.5-7B-Instruct, fp16, 2026-05-02): cell_js_max=0.092,
   cell_js_median=0.079 across 20 no-press runs. Gate threshold:
   `JS_NOISE_FLOOR = 1e-1` (one order above the observed maximum;
   one order below the saturated-compression regime at ~ln 2).
4. **Replay correctness — press=none full-trace identity**: replay
   artifacts for the 20 baseline runs (same model, no press)
   produce per-token JS below `JS_NOISE_FLOOR` (1e-1) along the
   full sequence, with the headline cell-level summary recorded in
   `gold/phase-0-results.md`. Implemented via baseline replay rows,
   not a separate sanity cell.
5. **Index alignment audit**: spot-check 5 random runs; confirm that
   under `press="none"` greedy, the argmax of replay logits at index
   `prompt_len + t - 1` reproduces the realized token. For compressed
   runs, the realized token's logprob under uncompressed is non-
   degenerate (finite, not exactly equal to the compressed value).
6. **Truncation-bias bounded** (target, not hard blocker): report
   50/95/99-pct absolute difference between top-K-derived KL and
   `kl_*_full` per `(press, ratio)`. Engineering bar: no systematic
   catastrophic bias and exact full-vocab scalars present for every
   headline metric. Since headline scalars are exact, truncation bias
   only affects audit/flexibility.
7. **Metric pipelines complete without bug-induced NaN**: trajectory
   NLL ratio, first-divergence-point, sum_KL, BERTScore, embedding
   cosine, gross_harm/help/net_delta computed for every applicable
   cell.
8. **Sampling-rate study runs end-to-end** and produces the
   directional artifacts in Q4. Recommendation deferred to Phase 1.
   *Non-blocking.*
9. **Alignment matrix renders**: pairwise Spearman + AUROC across all
   metric families on 120 compressed runs, with bootstrap CIs. No
   bug-induced NaNs. Signs inspected and recorded; no significance
   gate (N=120 with sparse outcome failures cannot support one).
   *Non-blocking.*

If gates 1-7 fail, do not scale to Phase 1.

## Implementation Priorities

In order:

1. **Schema + determinism manifest** (`metrics/io.py`,
   `runs.parquet` schema, `RunResult` extensions). If artifacts cannot
   be repaired/recomputed exactly, Phase 1 cannot trust them.
2. **Replay forward + on-the-fly scalars** (`metrics/replay.py`).
   Includes both sanity checks gated by tests on a tiny model (e.g.,
   `Qwen/Qwen2.5-0.5B-Instruct`) before Orion runs.
3. **Per-run parquet writers + finalize** (`metrics/io.py`,
   `metrics/cli.py finalize`).
4. **Inline orchestration** (`run_single_with_replay`, `phase0 run`
   CLI, baseline-first sweep, repair CLI).
5. **Offline metrics** (`token.py`, `trajectory.py`, `sequence.py`,
   `outcome.py`, `tags.py`).
6. **Alignment + sampling-rate study** (`alignment.py`, study
   notebook/script).
7. **Poe task wiring** + smoke test on the 0.5B model end-to-end on
   Mac before scaling to Orion.

## What Phase 0 Explicitly Does Not Do

- No claim about predictability. Phase 2.
- No generalization across models or tasks. Phase 3.
- No closed-loop control. Phase 4.
- No binding sampling-rate decision. Early Phase 1.
- No human/judge severity annotation. v2.
- No Tier 2 features. v2.

## Alignment With the Ultimate Goal

Each major decision was selected to compound across phases, not just
to satisfy Phase 0:

- **Single-pass replay** is the same code Phases 1-3 use at scale.
- **Union top-K + exact scalars** means future analyses do not
  require recompute; the dataset is the artifact.
- **Hive-partitioned parquet** scales to ~24k Phase 3 runs without
  schema redesign.
- **Determinism manifest** makes the released benchmark
  reproducible by third parties — a precondition for the methodology
  becoming a community standard.
- **Inline replay with baseline replay rows** removes the need for
  separate sanity cells later; the same pattern serves Phases 1-3.
- **Manual `metrics build`** keeps metric iteration cheap, which is
  the whole point of separating GPU-bound extraction from CPU-bound
  metrics.

The scope is small. The substrate is not.

## Implementation-Plan Notes

These items are explicitly *not* design decisions; they are
preconditions to surface inside the implementation plan so they get
explicit work items and aren't paved over.

### Dependency decisions

`pyproject.toml` does not currently include the libraries the design
depends on. The implementation plan must split these into core vs
optional and decide install groups:

- **Core** (required for `phase0 run` + `metrics finalize`): `pyarrow`
  and/or `polars` for parquet IO.
- **Metric extras** (required only for `metrics build`):
  - `bert-score` for BERTScore (heavy; pulls a transformer model).
  - `sentence-transformers` (or `transformers` reuse) for embedding
    cosine.
  - `rouge-score` and `python-Levenshtein` (or `editdistance`) for
    ROUGE-L and edit distance.
- **Stats**: `scipy` for Spearman/bootstrap (likely already pulled in
  transitively but should be made explicit).

Recommendation for the plan: introduce an optional `metrics` extra in
`pyproject.toml` so a base install can run the sweep on Orion without
pulling BERTScore's model weights, and a Mac-side install for paper
work pulls `[metrics]`.

### `GenerationArtifact` introduction

The replay API needs `input_ids`, `input_len`, `generated_token_ids`,
and `compressed_scores` — none of which are on `RunResult` today. The
plan must introduce `GenerationArtifact` (see Q3 API shape) and
refactor `experiment.run_single` to return it, with the `RunResult`
constructed *after* replay so `replay_status` and the determinism
manifest land atomically. This is a real refactor of `experiment.py`,
not just a new module.

### Repair input persistence

`prompt_hash` alone is insufficient for `metrics repair`; the command
needs the exact `prompt_text` to feed into the tokenizer. The schema
update above persists `prompt_text` directly on `runs.parquet`. The
plan must verify that `prompt_text` round-trips byte-exact through
parquet (no Unicode normalization surprises) and that `prompt_hash` is
recomputed and verified at repair time. Manifest files
(`phase0-random-manifest.json`) are convenience indices, not the
source of truth for repair.

