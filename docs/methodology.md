# Methodology

The plan for measuring compressor-induced damage and training a streaming
predictor for it. Companion to `docs/goal.md`. Rationale for every design
choice below is recorded in the matching entry under `docs/_why/`.

## 1. Damage definition

Damage is defined per (prompt, compressor, ratio) triple, comparing
generations from the same base model on the same prompt under
deterministic decoding.

- **Reference run.** Full KV cache, no compression.
- **Hybrid runs.** Generated tokens up to a *switch position* use the full
  KV cache; from the switch position onwards, the chosen compressor is
  active at the chosen ratio. The switch position is set at multiples of
  $k$ tokens: $\{0, k, 2k, 3k, \ldots\}$. The hybrid run with switch
  position $0$ is the fully compressed run; the hybrid run with switch
  position equal to the run length is the reference run itself.

We fix $k = 16$ tokens, an absolute count, independent of prompt length.

A run is *damaged* if its output is operationally worse than the
reference output. The set of hybrid runs at varying switch positions
gives, per prompt, a damage curve as a function of when compression
becomes active.

## 2. Measuring damage

ref: `docs/_why/3_measuring_quality.md`

> **Status: provisional.** The scheme below is the current working
> plan. The dense per-position target (Section 2.2) is gated on a
> judge-reliability pilot and may change once that pilot runs.

Damage is a quality delta between the paired runs, computed offline on
the completed outputs:

$$\text{damage}(s) = q(\text{reference}) - q(\text{hybrid at } s)$$

where $q$ is the task-relevant quality of a finished output. Because
both runs are greedy and share the prefix up to $s$, the delta is
attributable to compression. Quality is never measured mid-generation:
every check and judge call runs on completed text.

The same quantity plays two distinct roles, with different cost
profiles.

### 2.1 The damage measure (reported)

How much compression degrades output, for the headline results. One
comparison per run (the fully compressed run against its reference),
reported per (compressor, ratio, task).

### 2.2 The predictor's training target

Damage at each switch position of each run: the per-position curve the
streaming predictor is trained to forecast. This is training data, so
its quality directly bounds the final model.

### Measuring quality $q$

Quality is composite, each component used only where it is reliable:

- **Exact task checks**, for the correctness they verify exactly:
  GSM8K final-answer match, HumanEval unit tests, IFEval constraint
  satisfaction, LongBench per-subtask metric. Where these apply they
  are ground truth, not a proxy.
- **A self-hosted judge**, for the graded quality the checks cannot
  see: subtle degradation, correct answers reached through broken
  reasoning, failures that collapse into degenerate text. The judge
  reads the full outputs and returns a graded "how much worse" score.

### Pilot gate on dense judging

Using the judge for the dense per-position target (2.2) is adopted
only if a pilot confirms its per-position signal is reliable: the
judge's test-retest wobble on the same pair must be smaller than the
real damage differences across switch positions within a run.
Otherwise dense judging collapses to a noisy copy of the run-level
number and is not worth its cost. Under greedy decoding many
late-switch hybrids are byte-identical to their reference and are
exact zeros that need no judge call.

## 3. The predictor

ref: `docs/_why/2_predictor.md`

At every generated token $t$, the predictor outputs a scalar
$\hat{D}(t)$: an estimate of the damage that would result if the
compressor activated at $t$ and ran to the end of the generation. The
target is the per-position damage of Section 2.2, read at position $t$.

- **Inputs.** Per-token statistics derived from the model's own
  next-token distribution, observed causally up to $t$, plus the
  compression ratio. Compressor identity is not an input.
- **Output.** Scalar regression against the per-position damage
  measure (Section 2.2). Binary damage calls, when needed, are
  obtained by thresholding the scalar downstream.
- **Streaming constraint.** No additional forward pass, full or
  partial, beyond the one the model already performs to generate. No
  access to future tokens. Per-token compute and memory are $O(1)$;
  causal rolling statistics over past tokens are permitted.
- **Supervision.** Labels come from the measured damage curve, which
  exists at switch positions on the stride. How sparse labels become
  per-token supervision is a training decision, deferred to the
  training section.

One predictor is trained across all compressors and ratios in the
sweep. Cross-compressor transfer is evaluated by holding out entire
compressors at training time.

### Future extension: cross-layer features

Current features are read from the final next-token distribution only
(`features.py`). A future direction is adding per-layer / cross-layer
statistics: cheap hidden-state geometry (per-layer norms, cosine
between consecutive layers' residual streams) as an early signature of
compression damage. SPOT (CVPR 2026) is published evidence that
cross-layer aggregation makes a lightweight predictor more reliable;
see `docs/related_work.md`. This stays within the streaming constraint:
hidden states come from the same single forward pass, so per-token cost
is O(depth), still O(1) in sequence length. The cheap version uses
hidden-state geometry, not attention-map moments, which would force
eager attention and O(sequence^2) memory. Crucially this is a
reference-only feature re-extraction and does not invalidate the damage
labels, so it needs no resweep.

## 4. Scope

**Status: in progress.** Only the base models are fixed; the remaining
sweep dimensions are still open.

- **Base models.** `meta-llama/Llama-3.1-8B-Instruct` and
  `Qwen/Qwen3-8B`: two distinct families (different tokenizer and
  pretraining), which is what makes the cross-family transfer claim
  non-trivial. Both are supported by the compression library and run
  within local hardware. `Qwen/Qwen3-8B` is run in non-thinking mode,
  to keep both models on the same direct-answer regime.
- **Tasks.** GSM8K, HumanEval, IFEval, LongBench (as used in
  Section 2).
- **Compressors.** Five, chosen to span distinct selection
  principles so that holding one out is a real transfer test:
  StreamingLLM (positional), SnapKV (observed recent attention),
  ExpectedAttention (predicted attention), Knorm (key-norm geometry),
  Random (degradation floor). All are weight-free, so they behave
  identically on both base models, and none require eager attention.
- **Ratios, prompts per task.** To be decided.
