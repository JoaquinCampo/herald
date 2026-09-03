# How we measure compression damage, and why this way

> **Status: historical rationale, partly superseded.** The paired,
> task-grounded counterfactual and switch-position family remain current. The
> embedding/cosine measure was rejected in favor of the quality delta in
> `docs/_why/3_measuring_quality.md`. The earlier assumption that a stored
> continuous reference and a pressed re-prefill form an exact pair is now
> gated by the live-fork, matched-reprefill, and parity hierarchy in
> `docs/_why/6_intervention_semantics.md`. The embedding subsections remain
> only as a record of a rejected approach.

Compression damage is defined as a paired counterfactual: the gap between
the output a user actually receives under compression and the output they
would have received without it on the same prompt.

## The measurement

For every (prompt, compressor, ratio) triple we generate a family of
runs from the same base model under the same deterministic decoding
policy.

- The **reference run** uses the full KV cache from the first generated
  token onwards.
- A **hybrid run** at switch position $s$ uses the full KV cache for the
  first $s$ generated tokens, then activates the chosen compressor at
  the chosen ratio from position $s$ onwards.

Switch positions take values in $\{0, k, 2k, \ldots\}$ up to the run
length, with $k = 16$. The hybrid run with switch position $0$ is the
fully compressed run; the hybrid run with switch position equal to the
run length is the reference run itself.

For each (hybrid, reference) pair we ask the same question: is the
hybrid's output operationally worse than the reference's?

The result, per (prompt, compressor, ratio), is a damage curve indexed
by switch position: the value at position $s$ measures the damage of
activating compression at $s$ and running it to the end. At $s = 0$ the
curve gives the standard whole-run compression damage; at $s$ near the
run length the curve goes to zero by construction.

### Task-grounded outcome delta

The task itself has a deterministic correctness check (final-answer match
for GSM8K, unit-test pass for HumanEval, constraint check for IFEval,
per-subtask metric for LongBench). We apply this check to both runs and
record the cell of the 2x2:

| reference | compressed | label |
|---|---|---|
| correct | correct | no damage |
| correct | wrong | **damage event** |
| wrong | correct | lift (rare) |
| wrong | wrong | model+task failure, not compression-attributable |

Only the second row counts as damage. The fourth row is explicitly carved
out: the model would have failed regardless, so compression did not cause
the failure.

### Semantic similarity between the paired outputs

For each switch position $s$, the suffix of the hybrid run from $s$
onwards and the suffix of the reference run from $s$ onwards are
embedded independently with a single sentence encoder, and the
per-position damage signal is the cosine distance between the two
suffix embeddings.

- **Encoder.** `Qwen/Qwen3-Embedding-0.6B`, used across all four tasks.
- **Granularity.** Suffix-only. The token range $[s, \text{end}]$ is
  embedded on each side; the shared prefix is never included in the
  embedding.
- **Minimum suffix length.** Cosine is not computed when either suffix
  is shorter than 16 tokens. Switch positions with suffixes below the
  floor are omitted from the damage curve. Runs whose total length is
  below the floor are flagged as degenerate.
- **Storage.** Per-(run, switch position) suffix embeddings are stored
  as fp16 vectors in parquet. Embeddings are the persisted artifact;
  the cosine metric is computed downstream and can be iterated on
  without recomputing embeddings.
- **Per-run damage label.** At $s = 0$ the suffix is the entire
  generated output, so the cosine at $s = 0$ is the cosine distance
  between the full compressed output and the full reference output.
  This is the per-run damage label.
- **Damage curve.** The full set of cosines at $s \in \{0, k, 2k,
  \ldots\}$ above the floor is the per-prompt damage curve.

## Why this way

### Why paired, counterfactual, and operational

Damage has to be observable in principle, or any claim about it is a
claim about an unmeasured quantity. Pairing each compressed run with its
uncompressed counterpart on the same prompt makes damage a comparison
between two real generated artifacts, both of which we have on disk.
Counterfactual framing isolates compression from every other source of
model failure: the reference run already contains the model's intrinsic
error rate on that prompt, so any disagreement between the two runs is
attributable to compression. Operational framing (grounded in the task)
keeps the measurement honest about what "worse" means: we never claim
the model is damaged in some abstract distributional sense without
showing that the output the user receives is worse than the output they
would have received.

### Why deterministic decoding on both sides

Sampling variance would introduce another source of difference between paired
continuations. Greedy decoding removes that variance, but it does not by
itself establish attribution: both branches must also begin from matched
decoder states and use the same numerical path except for compression. The
parity protocol verifies those conditions rather than inferring them from
determinism.

### Why a family of hybrid runs rather than a single compressed run

A single (reference, compressed) pair collapses the damage of an entire
generation into one number. That number cannot say where along the run
compression caused harm, only that some amount of harm occurred by the
end. It also cannot distinguish damage that accumulated gradually from
damage that arrived all at once.

Generating a family of paired interventions, indexed by switch position,
replaces the single damage scalar with a damage curve along the run. The first
$s$ emitted token IDs are shared by construction. Whether the underlying
decoder states are also matched depends on the live-fork or re-prefill
semantics and must pass the parity protocol. Only then may the resulting
quality difference be attributed to activating compression at $s$.

The curve carries information the scalar cannot:

- It attributes damage to positions. The marginal damage of activating
  compression at position $s$ rather than at $s + k$ is the difference
  between successive points on the curve.
- It supplies a dense per-position label for the predictor. The
  predictor can be trained to forecast damage at each switch position,
  given features observed up to that position, without recourse to
  expensive synthetic continuations.
- It removes a class of attacks on the whole-run measurement: a critic
  who asks "but where did the damage happen?" has an answer in the data
  rather than in the discussion.

### Why a fixed switch granularity $k = 16$, independent of prompt length

The granularity $k$ controls how dense the damage curve is along each
run. A smaller $k$ gives more switch positions, a finer curve, and a
denser per-position label.

We fix $k$ at an absolute token count rather than letting it scale with
prompt or run length so that switch positions occupy a consistent scale
across prompts. A relative $k$ (such as "every 10% of run length") would
make positions on the damage curve mean different numbers of tokens in
different runs, which would in turn make the per-position predictor
target inconsistent in what it predicts.

$k = 16$ is dense enough to give around a dozen switch positions on a
typical 200-token run while keeping the family of hybrids manageable.
We expose this value as a configuration parameter so sensitivity at
$k = 8$ or $k = 32$ can be reported on a subset if a reviewer asks.

Note: $k$ is a compute knob, not a load-bearing definition. The damage
curve is defined on whatever stride we run. If hybrid generation turns
out cheaper than expected, we should reduce $k$ and report on a denser
curve; nothing in the definition changes.

### Why compression-onset-at-position-$s$ is the operationalisation we measure

When a hybrid run reaches its switch position, we activate compression
from that point onwards. The cache state up to that moment was built
under full attention; from the switch onwards the compressor acts on
that cache and on every cache state thereafter.

This corresponds to a deployment pattern where compression is turned on
mid-generation, for example when the cache reaches a budget threshold.
It is not identical to "compression that has been active continuously
from token 0," because the cache contents at the moment of activation
depend on the past. For switch position $s = 0$ the two coincide; for
$s > 0$ they differ. We measure the onset-at-$s$ operationalisation
because it is the cleanest paired counterfactual: the only intervention
on the trace is the activation event at $s$, and everything before $s$
is shared with the reference.

### Why an embedding-based semantic measure in addition to task-grounded

The task-grounded outcome delta has known gaps. It is binary, so it cannot
express degrees of damage. It is silent on partial damage where the final
answer is correct but the reasoning trace is garbled. The "wrong / wrong"
cell is carved out as not compression-attributable, but compression can
still turn a *plausibly wrong* answer into a *broken* one within that
cell. And on tasks where the correctness check is fuzzy (free-form
summarisation, open-ended generation), the binary label is itself a
proxy. A continuous semantic measure fills these gaps. It is defined on
every (reference, compressed) pair regardless of task and is not
confused by paraphrase the way surface-form lexical metrics are.

### Why a single encoder across all tasks

The cosine distance between paired embeddings lives in a metric space
defined by the encoder. Cosine values from different encoders are not
directly comparable: a cosine of 0.7 under one encoder is not the same
amount of damage as cosine 0.7 under another. If we used a different
encoder per task (for example a code-specialised encoder on HumanEval
and a general encoder elsewhere), the per-run damage target would live
in different metric spaces depending on the task, and a unified
predictor trained on the union of all tasks would have to learn an
implicit, task-conditional rescaling of its target. That hidden
confound contaminates any cross-task transfer claim.

Using one encoder across all tasks puts the headline target in one
metric space. Cross-task evaluation then measures whether the predictor
generalises across tasks, not whether different encoders happen to be
calibrated comparably.

### Why `Qwen3-Embedding-0.6B` specifically

The encoder must cover every task's output in a single pass without
chunking, must be open-weight and self-hostable, and must produce a
strong general-purpose embedding. `Qwen3-Embedding-0.6B` satisfies all
three: 32K context window covers every output in our sweep, the weights
are released under a permissive licence, and the model is at the top of
recent retrieval and similarity benchmarks for its size class.

We considered `BGE-large`, `E5-large`, and `Stella-large`. They are
strong embedding models but their context windows (typically 512 tokens)
would force us to chunk long outputs, which adds an aggregation step
we would have to defend. The 32K context of `Qwen3-Embedding-0.6B`
removes that complication entirely.


### Why the cosine is computed on the suffix from the switch position onwards, not on the whole output

At switch position $s$, the first $s$ tokens of the hybrid run and the
reference run are identical by construction: both were produced from
the same uncompressed cache under the same deterministic decoding. The
shared prefix therefore carries zero damage signal by definition, not
by approximation.

A cosine taken over the full outputs at $s > 0$ averages this
known-zero prefix into the non-zero suffix. The resulting number is
structurally biased toward zero in proportion to how much of the run
predates the switch. That is not noise to be tolerated; it is a known
contamination of the signal we are trying to measure.

Embedding only the suffix from $s$ onwards removes that contamination.
At each switch position, the cosine reflects damage in the affected
region only.

The per-run damage label is unchanged. At $s = 0$ the suffix is the
whole output, so suffix-cosine and full-output cosine coincide. A
reader who only ever looks at the per-run label sees a cosine between
two complete generations, exactly as before. Readers who follow the
damage curve see an honest per-position signal rather than a diluted
one.

Two caveats follow from suffix-only embedding:

- **Variable suffix lengths.** A 200-token suffix and a 30-token
  suffix produce embeddings with different length-conditioned
  statistics, so cosines at different $s$ are not strictly comparable
  in absolute scale. Comparisons within a switch position are clean;
  comparisons across positions should be read as a curve shape, not
  as point-to-point arithmetic.
- **Short-suffix noise.** Below a threshold the embedding of a very
  short token sequence is too noisy to carry a reliable cosine. We
  set a floor of 16 tokens, matching the switch granularity $k$;
  positions whose suffixes are below the floor are dropped from the
  damage curve.

### Why we store embeddings rather than precomputing a metric

Embedding both runs of every (prompt, compressor, ratio) triple is the
expensive step. Once the fp16 vectors are on disk, computing cosine,
applying a threshold, calibrating against a human-labelled subset, or
ensembling with a second encoder are all cheap downstream operations.
We avoid baking any specific metric choice into the artifact: the
artifact is the embeddings, the metric is whatever we report at write-
up time, and changing the metric does not require recomputing anything
on the GPU.
