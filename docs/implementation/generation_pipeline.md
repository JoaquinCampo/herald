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
- **Ratios.** Stored as removal fractions. The current configuration lists
  `0.25`, `0.5`, `0.75`, and `0.875`; recovered artifacts must be checked
  against their own frozen sweep configuration before reuse.

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

## Historical uniform re-prefill mechanism

The offline sweep used the same pressed-reprefill path for all five presses.
That uniformity prevented the generation mechanism from varying by compressor,
but it did not prove that the stored continuously decoded reference was a
numerically matched control. Full prefill and incremental decoding can differ
in floating point even under greedy decoding.

The current paper therefore treats these hybrids as historical
pressed-reprefill interventions. Reusing `q_reference` as their control
requires sham re-prefill parity. Describing them as equivalent to live-cache
activation additionally requires intervention parity for that compressor.
The canonical classification and validation ladder are in
`docs/_why/6_intervention_semantics.md`.

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

## Live-controller cache-fork optimization

The offline dataset sweep keeps the uniform mechanism above. The live
controller has a separately validated fast path for StreamingLLM and
Knorm, whose scores depend only on positions or cached keys. At an
attempt it shallow-forks the held reference cache, gathers the exact KV
pairs that kvpress would retain, and decodes from that fork. It does not
recompute `[prompt + s]`.

kvpress compresses after the prefill attention forward, so the first
post-switch token is still the uncompressed reference token. The fast
path reuses that token and its logit features, then continues from the
compressed fork. Tests on a real tiny Llama require exact retained
keys, values, output tokens, text, and alarm feature rows against the
uniform kvpress path for both supported presses. Unsupported presses,
including ExpectedAttention, retain the uniform re-prefill fallback.

Each live attempt records `recomputed_prefill_tokens`. It is zero for
the validated cache-fork path and `prompt_length + s` for the fallback,
so deployment evidence can verify that the intended mechanism actually
ran.

The live runner also exposes an experimental `--sustain-interval N`
mode for StreamingLLM and Knorm. Every N decoded tokens it prunes cache
growth back to the configured fraction of the logical sequence length.
This addresses the fact that one-time prompt compression loses its
memory advantage as uncompressed generated tokens accumulate. It is a
candidate, not the default: promotion requires held-out quality and
wall-time evidence under the deployment contract.

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
