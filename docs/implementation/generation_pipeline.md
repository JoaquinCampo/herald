# Generation pipeline

Implementation plan for producing the hybrid-run dataset as fast as the
design allows. This is an implementation document: it carries the
specifics that the methodology and `_why` files deliberately omit.

## What it produces

Per (model, prompt, compressor, ratio): one reference run and a family
of hybrid runs at switch positions $s \in \{0, 16, 32, \ldots\}$. The
reference run also yields per-token logit features. Each run yields the
generated text used for damage scoring downstream.

## Where the cost is

The combinatorial sweep dominates, not the per-token kernel. Being
structure-aware about what is generated once and reused is worth far
more than low-level tuning, so the plan is organised around reuse.

## Fixed settings

- **Models.** `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-8B`
  (non-thinking). bf16, no weight quantization (keep this version to a
  single compression axis).
- **Compressors.** StreamingLLM, SnapKV, ExpectedAttention, Knorm,
  Random.
- **Decoding.** Greedy.
- **Attention backend.** One consistent fast backend (SDPA or
  FlashAttention-2) for the reference and every compressor. No
  eager-only presses. A single backend everywhere keeps the base
  model's outputs from varying by kernel in a way that would correlate
  with the compressor.
- **Ratios.** Sweep axis, still to be decided.

## Compression mechanism (one-time at the switch)

kvpress fires its compression once, at the end of prefill (the forward
where `q_len == k_len`); during decode the hooks are registered but
skip. This matches the design's one-time-at-$s$ semantics directly.

A hybrid at switch position $s$ is produced by prefilling
`[prompt + first s reference tokens]` inside `press(model)`. The press
fires once on that full-attention `[prompt + s]` cache, evicting to the
ratio, and decoding then continues greedily to the end on the fixed
compressed cache.

- The first $s$ tokens are the reference's own tokens, injected rather
  than regenerated, so the pre-switch prefix is shared with the
  reference by construction.
- $s = 0$ compresses the prompt-only cache (standard prompt
  compression, the fully compressed run). $s$ at the run length is the
  reference.

## One uniform mechanism across all five presses

All five presses use the single re-prefill path above. We do not mix
snapshot-based cache reuse for some presses with re-prefill for others:
the two paths differ numerically (full-prefill versus
incremental-decode floating point), and that difference would correlate
with the compressor, biasing the exact paired comparison the
cross-compressor transfer claim depends on. With one mechanism, any
prefill-versus-decode floating-point differences are constant across
compressors and cannot confound transfer.

## Reuse (the unconditional win)

- **Reference generated once per (model, prompt).** The reference is
  full-cache and compressor- and ratio-independent, so it is generated
  a single time and reused across the entire compressor x ratio x
  switch grid.
- **Features come from the reference for free.** The predictor's
  per-token features are read off the uncompressed stream, which is
  exactly the reference run. Hybrids are not feature-extracted; they
  contribute only their final text for scoring.

## Feature extraction

Reference-only. Use a decode loop (or `generate` with logits returned)
that computes the per-token logit statistics inline and discards the
full-vocabulary logits each step. Not storing full logits keeps memory
low, which is what allows larger batches.

## Batching

- References batch across prompts (identical config).
- Hybrids batch across prompts at a fixed (compressor, ratio, $s$).
  Left-pad; all presses support batch size >= 1.

## Gated optimization for long prompts (LongBench)

The uniform path re-prefills `[prompt + s]` for every hybrid. This is
cheap when the prompt is short (GSM8K, HumanEval, IFEval) and expensive
when the prompt is long (LongBench), where prompt prefill dominates.

The optimization is to compress at $s$ from a snapshot of the reference
cache instead of re-prefilling, applied uniformly across all five
presses (the score-based presses recompute only the small
observation-window queries they need). It is promoted to headline use
only after it is validated to agree with the uniform path and to be
consistent across presses. Correct-and-consistent first; optimize where
proven identical.

## Storage

Per run: generated text, task score, and (reference only) the per-token
features. fp16 where applicable. No full logits persisted.

## Validation before the full sweep

- **Smoke check** every launch: process alive, past argument parsing,
  no backend or proxy hang.
- **Correctness:** $s = 0$ reproduces the fully compressed run; $s$ at
  run length reproduces the reference; the injected prefix matches the
  reference tokens exactly.
- **Determinism:** greedy generation reproduces run to run.
