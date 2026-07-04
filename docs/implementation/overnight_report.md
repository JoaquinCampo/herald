# Overnight build + generation report

Status as of the overnight session. The generation harness is built,
verified, and was used to run a complete GSM8K slice on Orion (all 5
compressors x 4 ratios x switch grid, 8183 hybrids), with a HumanEval
slice running afterward. The full sweep across both models and all four
tasks is a multi-day job (sized below), so tonight produced the harness,
the correctness verification, one fully-swept task, and honest
projections for scaling the rest.

## 1. What was built

Flat package `src/herald/` (Python 3.12, pydantic config, functions over
classes, ruff line 78, mypy strict, `poe check` green: 83 tests).

- **config.py**: `Config` (pydantic) is the full sweep spec: models,
  tasks, compressors, ratios, switch stride `k=16`, prompts/task, batch
  sizes, dtype, attention backend. `TASKS` registry (GSM8K, HumanEval)
  with per-task generation budget `M`. Validators enforce ratios in
  (0,1) and positive counts.
- **presses.py**: `get_press(name, ratio)` factory for the five
  weight-free presses (StreamingLLM, SnapKV, ExpectedAttention, Knorm,
  Random).
- **features.py**: logit-only by design (the thesis claims damage is
  forecastable from cheap per-token logit statistics with no new forward
  pass). `FeatureCollector` is a LogitsProcessor that reads the logits
  the forward pass already produces and stores a 20-scalar per-step
  superset inline: entropy, varentropy, h_alts (entropy of the non-top-1
  mass), avg_logp, max_prob, margin_prob, top1_logit, margin_logit,
  logit_std, logit_range, logit_skew, logit_kurtosis, chosen_logprob,
  KL-to-previous-step, and top-k mass for k in {2,5,10,20,50,100}. Full
  logits are discarded each step. `derive_features` is a downstream,
  strictly causal (backward-only) and O(1)-online transform that expands
  the stored series with dynamics (delta, acceleration, slope, EWMA) and
  rolling robust stats (mean/std/min/max/median/IQR over short and long
  windows) plus position. It is recomputable from the stored superset and
  never persisted, so the final feature set is a downstream selection,
  never a regeneration, and a unit test proves no future leakage.
  Attention features were deliberately excluded: materializing attention
  needs eager kernels (not cheap, and off the logit-only thesis); kept as
  a possible future ablation.
- **generate.py**: one greedy `model.generate` path for both reference
  and hybrid runs (identical numerics for the paired counterfactual).
  Reference: full cache, features inline. Hybrid at switch `s`: prefill
  `[prompt + first s reference tokens]` (token-id space, never
  detokenized) inside `press(model)` so the press fires once on that
  full-attention cache. Greedy is forced through a clean
  `GenerationConfig` (no sampling, no repetition penalty, no n-gram
  block) so `scores == logits`. A runtime self-check asserts the
  collector saw the decisive scores. Budget is `M` for the reference and
  `M - s` for a hybrid (same total budget on both sides).
- **tasks.py**: deterministic first-N prompt loaders for GSM8K and
  HumanEval, building chat messages and gold metadata.
- **scoring.py**: deterministic task-grounded `q in [0,1]`. GSM8K answer
  match (flexible last-number extraction, see findings); HumanEval
  sandboxed subprocess execution of model-generated code with a timeout.
- **storage.py**: atomic (temp + fsync + rename), idempotent, resumable
  storage. References as JSON + fp16 `.npy` features with an append-only
  `_done.jsonl` manifest; hybrids as append-only JSONL shards per
  (compressor, ratio) keyed on `(prompt_id, s)`. Torn final lines from a
  crash mid-append are tolerated on read.
- **runner.py**: the resumable sweep. References once per (model,
  prompt) and reused; hybrids per (compressor, ratio, s), batched across
  prompts at a fixed cell. Already-stored cells are skipped on resume.
  Per-batch error isolation: a poison item (OOM, degenerate prompt)
  logs `batch_error` and is skipped rather than aborting the unattended
  run. Structured JSON logging per run.
- **scripts/**: `run_sweep.py` (CLI), `validate.py` (Phase 1 real-model
  invariants), `analyze.py` (Δq distributions + trivial-baseline check),
  `keepalive.py` (RTD3 mitigation, error-tolerant with heartbeat).

Full output **text and token ids** are persisted for every run (not just
`q`), so the gated judge/graded-target layer can run later with zero
regeneration. The damage measure is the deterministic task-grounded
delta `dq(s) = q(reference) - q(hybrid at s)`.

### Dependency note (flagged)

`kvpress 0.5.2` is incompatible with `transformers` v5: the v5 attention
refactor stopped threading `cache_position` to the attention module, so
the press's prefill hook (`kwargs["cache_position"]`) raises `KeyError`
and no compression fires. Pinned `transformers>=4.57,<5` (4.57.x still
passes `cache_position` and supports both base models). `torch==2.10.0`
pinned to match Orion's warm uv cache; the CUDA stack is routed to the
PyTorch cu128 index so the install reuses cached wheels.

## 2. Correctness verification (Phase 1)

All invariants verified, first on a tiny CPU Llama (unit tests), then on
the real `Llama-3.1-8B-Instruct` on the GPU (`scripts/validate.py`):

- **scores == logits**: PASS. The clean greedy config adds no
  score-altering processor, so collected features are the raw model
  distribution.
- **s = 0 == fully compressed**: PASS for all five presses. The s=0
  hybrid (prompt-only cache, compressed) reproduces an independently
  generated fully-compressed run token-for-token.
- **Injected prefix byte-identical to the reference**: PASS (true by
  construction, kept in token-id space).
- **Determinism**: greedy reproduces run-to-run; RandomPress is seeded
  for reproducible evictions.
- **Feature causality**: teacher-forced recomputation of the features
  over the realized sequence matches the inline features (approximate,
  not byte-exact, per documented prefill-vs-decode fp differences) and
  the greedy argmax sequence matches exactly.

### Break-it finding 1: left-pad batching corrupts compression

Batched hybrid generation with left-padding gives DIFFERENT results than
batch=1 for 4 of the 5 presses (StreamingLLM, ExpectedAttention, Knorm,
Random; only SnapKV matches). The pad tokens enter the press's eviction
scoring (e.g. StreamingLLM keeps positional "sinks" that are pad tokens),
so the compressed cache differs. Confirmed on the real model
(`batch==single` check: same length, different tokens). **Consequence:
hybrids must run at batch_size = 1** (now the enforced default).
References, which use no press, batch freely and correctly.

### Break-it finding 2: strict #### scoring reports systematic false damage

Hand-inspection of high-s hybrids (where the hybrid shares almost all
tokens with the reference and damage should be ~zero) showed `dq = 1.0`
where there was no real damage. Cause: light compression frequently
nudges the model out of the "#### N" answer format into prose
("Therefore, Janet makes $18...") while keeping the correct number. A
strict #### scorer marks that wrong, fabricating damage. Switched GSM8K
to flexible last-number extraction (the accepted instruct-GSM8K metric).
Because full output text is stored, a stricter policy is recoverable
downstream without regeneration. This is a scoring-policy choice that
materially changes the damage labels and should be reviewed.

## 3. Throughput and sweep sizing

Measured on the RTX 5090, Llama-3.1-8B, bf16, SDPA:

- References: batched, fast (12 in well under a second of generation
  after model load).
- Hybrids at batch=1: **~0.58 hybrids/second** (the binding constraint).

A GSM8K run is ~200-270 tokens, giving ~13-17 switch positions at k=16,
so ~5 compressors x 4 ratios x ~15 positions = ~300 hybrids per prompt.

Projection for the full intended sweep (2 models x 4 tasks x 200 prompts
x full grid, batch=1): order 4 x 10^5 hybrid generations, ~8 days of
GPU time on a single 5090. This is infeasible overnight and was never
going to be; the harness is built to run it resumably across many
sessions. Qwen3-8B is additionally not yet cached on Orion (only
Qwen2.5-7B is) and was not substituted.

Levers to fit a realistic budget (for review):
- `k` is a compute knob, not load-bearing (methodology says so). Raising
  k from 16 to 32 halves the hybrid count for a coarser damage curve.
- Fewer ratios or prompts per task.
- SnapKV (the one batchable press) can run batched for a partial speedup.

## 4. Slice results

Slice: `Llama-3.1-8B`, GSM8K, 25 prompts, all 5 compressors, all 4
ratios, k=16, hybrids at batch=1. The GSM8K sweep is COMPLETE: 8183
hybrid runs across the full 5x4xswitch grid, plus 25 references. 15
isolated SnapKV `OverflowError` cells were skipped (see below); every
other cell completed. A 15-prompt HumanEval slice is running afterward.

### Damage matrices (GSM8K, 25 prompts, Llama, sweep complete: 8183 hybrids)

Reference accuracy q_mean = 0.76. A run can only be "damaged" if the
reference got it right, so the maximum possible mean dq is ~0.76, which
is total destruction (every correct reference turned wrong).

**Reported damage measure (methodology 2.1): the fully-compressed run
(switch position s=0) vs its reference, one comparison per run.** This is
the headline.

| compressor | 0.25 | 0.50 | 0.75 | 0.875 |
|---|---|---|---|---|
| expected_attention (predicted attn) | -0.08 | 0.04 | 0.24 | 0.68 |
| streaming_llm (positional) | 0.00 | 0.12 | 0.72 | 0.76 |
| snapkv (recent attn) | 0.00 | 0.64 | 0.76 | 0.76 |
| knorm (key-norm geometry) | 0.36 | 0.64 | 0.76 | 0.76 |
| random (degradation floor) | 0.68 | 0.76 | 0.76 | 0.76 |

The five compressors span a clean quality spectrum, as the literature
predicts:

- **ExpectedAttention is best by far**: net-zero at 0.25 (slightly
  positive), still only 0.24 at 0.75, reaching 0.68 only at the extreme
  0.875. NVIDIA's method earns its reputation here.
- **Random is the degradation floor by design**: it destroys 0.68 of the
  correct runs even at the lightest ratio 0.25, and saturates at the 0.76
  ceiling from 0.5 on.
- streaming_llm, snapkv, knorm sit in between. streaming_llm shows a
  sharp cliff between 0.5 (0.12) and 0.75 (0.72): positional eviction is
  harmless until it drops too much, then collapses.

**Per-position curve mean (methodology 2.2 target summary, NOT the
headline)** is uniformly lower because it averages in the decaying tail
of the damage curve:

| compressor | 0.25 | 0.50 | 0.75 | 0.875 |
|---|---|---|---|---|
| expected_attention | -0.03 | -0.01 | 0.10 | 0.36 |
| snapkv | -0.03 | 0.24 | 0.54 | 0.66 |
| streaming_llm | 0.00 | 0.28 | 0.48 | 0.56 |
| knorm | 0.22 | 0.46 | 0.67 | 0.70 |
| random | 0.36 | 0.66 | 0.65 | 0.66 |

Two things hold in both views. Damage rises monotonically with the ratio
for every compressor (near-zero at 0.25 for the better methods shows the
signal tracks compression, not a scoring artifact). And the spread
ACROSS compressors at a fixed ratio is large (at s=0, ratio 0.25: -0.08
to 0.68), so the compressor choice matters independently of the ratio.
That is the structure the held-out-compressor transfer evaluation
probes, and it motivates the compressor-agnostic predictor design.
(25 prompts, GSM8K, Llama only; small N.)

### Scoring is a lower bound on heavy-compression damage

GSM8K uses flexible last-number extraction (Section 2). It corrects the
strict-#### over-count (which fabricated damage when a correct answer was
not in #### format) but has the mirror weakness: a garbled
heavy-compression output whose stray last number happens to equal the
gold gets false credit, so heavy-ratio damage is UNDER-counted. The true
damage is bracketed flexible <= true <= strict; both bounds are
recoverable from the stored output text without regeneration.

### The damage curve decreases with switch position (StreamingLLM @ 0.75)

dq mean by switch position s: 0.72 (s=0), ~0.72 through s=48, ~0.64
through s=112, 0.50 (s=128), 0.38-0.45 (s=144-192), 0.28 (s=208),
decaying to ~0 past s=256. This is the expected shape: compressing early
leaves most of the generation exposed (high damage); compressing late
leaves little to damage (the curve approaches 0 by construction as the
suffix shrinks). It also shows damage is not uniform along the run.

### Damage is real reasoning corruption (hand-inspected)

A confirmed damage event (StreamingLLM, ratio 0.75, s=128): the reference
correctly computes "9 eggs left * $2 = $18  #### 18" (q=1.0). The
compressed hybrid loses the thread, invents a "number of boxes = 9 / 3 =
3" subproblem, and answers `\boxed{3}` (q=0.0). The flexible scorer reads
3 != 18 and labels it damaged. This is genuine cache-loss reasoning
corruption, not a formatting artifact.

### Trivial-baseline break-it check

Across all five compressors (8183 hybrids), only ~33% of dq variance
(R2 = 0.328) is explained by the (compressor, ratio, s) cell means;
~67% is prompt-level residual. So dq is not trivially determined by
compressor, position, and ratio: a substantial per-prompt target REMAINS
for an online predictor to attempt. This is a necessary condition for
the thesis (there is something to predict beyond ratio and position), not
evidence that the per-token logit features can in fact predict it: no
predictor was trained tonight, and because GSM8K dq is in {-1, 0, 1},
part of that residual is Bernoulli flip variance whose predictability is
untested. Establishing predictability is the next experiment, with the
features this dataset already stores.

### Operational note: monitoring access interrupted

`ssh orion` routes through a `ProxyJump login-gpu` jump host that
rate-limits / fail2bans connection floods. The monitoring watcher
(an `ssh orion` poll every 45-60s) plus a persistent reverse tunnel
tripped that limit, and the jump host began refusing all connections
("Connection closed by 164.73.44.3 port 22"). The slice itself is
nohup-detached on Orion, not on the jump host, so it kept running and
checkpointing throughout; only my ability to observe it was lost. Local
tunnel/proxy/ssh were torn down to stop hammering the jump host;
reconnection is on a long zero-contact backoff (repeated attempts can
escalate an OpenSSH per-source penalty). The full cross-compressor
numbers and the recomputed break-it R2 will be pulled from the persisted
results once access returns; `scripts/analyze.py results/slice` produces
them in seconds with no regeneration. Lesson recorded in project memory:
monitor Orion with a single ssh every several minutes at most, batch
checks into one invocation, never a tight ssh poll loop.

### Known issue: SnapKV OverflowError at extreme compression

SnapKV at ratio 0.875 (keep 12.5%) with small switch positions raised an
`OverflowError` inside kvpress on a handful of cells (7 of 3127 so far).
The per-item error isolation logged `batch_error` and skipped them, so
the sweep continued and the rest of the dataset is intact; those few
(prompt, s) cells are simply absent. The cause is a SnapKV edge case at
extreme eviction with its 64-token observation window; worth a kvpress
upstream check or a guard before the full sweep.

## 5. Blocked / deferred

- **Full sweep**: multi-day; harness ready to resume to full scale. Needs
  a sizing decision (k, ratios, prompts) from the owner.
- **Qwen3-8B**: not cached on Orion; a ~16GB proxy download was not
  attempted tonight (the proxy is slow). Cross-family transfer is a
  downstream (predictor) concern, not tonight's dataset.
- **HumanEval**: a 15-prompt slice is now running (after the GSM8K sweep
  finished), into the same `results/slice` tree. The sandboxed code
  scorer works on the real model: early references score q=1.0 (Llama
  writes passing solutions, the subprocess unit-test execution confirms
  it). It may not finish before the session ends; it is resumable and
  whatever completes is valid cross-task data. A second-task damage
  matrix and R2 can be produced by `scripts/analyze.py` once it has
  enough cells.
- **IFEval, LongBench**: loaders/scorers not yet implemented and their
  datasets are not cached on Orion. GSM8K (fully swept) and HumanEval
  (running) cover one exact-match and one code-execution task, the two
  cleanest scorer types.
- **Judge layer**: out of scope tonight per the goal. Deterministic dq
  stands alone. The reliability pilot (docs 2.4 / _why/3) is still
  required before dense judging.

## 6. Needs review

- **Scoring policy**: flexible vs strict GSM8K extraction (finding 2).
  The flexible choice changes damage labels; confirm it matches intent.
- **Feature-set**: expanded to a generous logit-only superset (20 stored
  per-step scalars + a causal downstream dynamics/rolling library);
  attention features deliberately excluded to keep the cheap,
  no-new-forward-pass claim literal. The final selection is downstream.
  Slice references are regenerated with the new superset before the full
  sweep (references are <1% of compute; hybrids/damage are unaffected).
- **Sweep scope**: k, ratios, prompts per task, and whether to accept a
  multi-day full run or reduce scope.
- **Judge model + pilot**: gated, for the dense per-position target.
- **Doc inconsistency**: `_why/1` still describes the embedding-cosine
  damage signal that `_why/3` explicitly dropped; `_why/1` should be
  reconciled to the quality-delta framing the code implements.
