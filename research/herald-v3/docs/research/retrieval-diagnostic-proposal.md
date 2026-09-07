# Retrieval causal diagnostic proposal

Status: candidate protocol for owner selection. No GPU run, implementation,
dependency change, outcome access, predictor fit, or new data collection is
approved by this document.

## Decision this protocol would support

The diagnostic asks one narrow question: when ordinary Knorm evicts user-message
KV entries in independently detected retrieval heads, does restoring a small,
fixed subset recover final IFEval compliance more than a position-matched
non-user restoration at the same cache budget?

A pass would establish a controlled rescue for the manipulated entries. It
would not show that the naturally observed eviction exposure predicts signed
loss. That later claim requires a separately frozen decision-time feature and
an independent outcome population.

## Existing structural evidence

The completed CPU audit covers the 69 materialized r5 prompts at the token-32
boundary. It verifies exact prompt rendering, token IDs, offsets, chat template,
and user-content spans for every record under tokenizer revision
`a09a35458c702b33eeacc393d103063234e8bc28`.

Qwen2.5-7B-Instruct has 28 layers, 28 query heads, and four KV heads per layer,
so the audit contains 7,728 `(prompt, layer, KV head)` cells for each action.
At Knorm 0.25, restoring every evicted user entry has enough retained non-user
donors in all 7,728 cells. At Knorm 0.5, full restoration is infeasible in 1,120
cells, although every cell permits at least one bounded equal-budget swap. The
0.25 action has 7,510 cells with at least one such swap. These findings support
one common bounded intervention across both ratios, not full restoration.

The audit does not establish that a recency-matched non-user control exists in
the retrieval cells eventually selected. That must be checked without labels
after head discovery and before a roster is materialized.

## Phase A: synthetic retrieval-head discovery

### Source and fixed inputs

Adapt only the detection metric from Wu et al.'s paper and authors' code at
commit [`3ac171a6f71ce7ef1cda57d4215c390fb6ab51f2`](https://github.com/nightdessert/Retrieval_Head/tree/3ac171a6f71ce7ef1cda57d4215c390fb6ab51f2).
Their method generates a known answer from a needle-bearing context and scores
a query head when its maximum-attended prompt token lies in the needle span and
has the same token ID as the generated token. The authors average this copy
fraction over successful retrieval cases and use 0.1 as the retrieval-head
threshold. Their released code filters successful cases at ROUGE-1 recall
greater than 50 percent.

Use the three released needle, question, and answer triples and their associated
`haystack_for_detect/part1`, `part2`, and `part3` corpora from that commit. At
implementation time, record every source-file hash and concatenate each part's
UTF-8 files in sorted relative-path order. Do not generate or select needles
from IFEval.

Run the exact pinned Qwen2.5-7B-Instruct checkpoint in BF16, batch one, greedy
decoding, with Python, NumPy, and PyTorch seed 0. Render each case with the
pinned chat template. Decode at most 50 tokens, stopping at EOS or the first
decoded line break after a non-whitespace token. Locate the inserted answer as
an exact token subsequence; a missing or ambiguous span is an integrity failure.

### Small fixed grid

Use 24 synthetic cases, with no additions after results are visible:

- Discovery panel: three triples times context lengths 1,024 and 4,096 times
  depths 20 and 80 percent, for 12 cases.
- Validation panel: the same three triples and lengths at depths 35 and 65
  percent, for 12 disjoint positions.

For each panel, include only cases whose generated answer has ROUGE-1 recall
strictly above 50 percent, matching the authors' filter. Require at least eight
successful cases per panel and at least one success for each context length.
Failure of that floor stops the proposal; do not lower it or add cases.

For every successful case and query head, calculate the authors' token-copy
fraction. Average within each panel. A query head is stable when both panel
means are at least 0.1. Before using the metric, run four fixed sentinel cases,
the lexicographically first and last case in each panel, once with the
attention-producing eager decode and once with the frozen SDPA path. Generated
token IDs must agree through the detection stop point. Any mismatch stops the
assay because instrumentation changed the observed behavior.

### GQA mapping and bounded selection

The implementation must verify the installed Transformers 4.57.6 Qwen2
`repeat_kv` ordering. Under the expected contiguous seven-query-head grouping,
query head `q` maps to KV head `floor(q / 7)` within the same layer. Map stable
query heads to `(layer, KV head)` cells and deduplicate shared KV heads before
any intervention.

For a KV cell, define its stability score as the largest, over its seven query
heads, of `min(discovery_mean, validation_mean)`. Keep cells with stability
score at least 0.1, sort by descending score with layer and KV-head index as tie
breakers, and retain at most the top eight. Zero retained cells is a scientific
stop. Do not replace threshold failures with a larger top-k set.

This reduced grid is an adaptation, not a reproduction of the paper's roughly
600-instance study. The authors report that a small sample can identify some
strong heads, which makes it suitable only for this falsification gate.

### Discovery cost cap

The maximum is 24 attention-bearing synthetic continuations plus four SDPA
sentinel continuations, with contexts no longer than 4,096 and outputs no
longer than 50 tokens. Stop if this phase exceeds 30 synchronized GPU minutes.
Record wall time, forward time, peak allocated memory, successful-case counts,
all query-head scores, GQA mappings, and selected-cell hashes. No current
measurement supports a tighter runtime or memory estimate.

The released detector uses `rouge-score`, which is absent from the current
project lock. The owner must either approve that dependency before
implementation or reject this exact success filter. Substituting a homemade or
post-result success metric would create a different protocol.

## Phase B: label-blind swap and control feasibility

Run this phase on all 69 structural records without reading quality scores or
loss signs. Let `N` be the full cache length at the decision boundary. For each
selected KV cell and action, partition original positions into:

- `U`, user-content positions evicted by ordinary Knorm;
- `C`, non-user positions evicted by ordinary Knorm; and
- `D`, non-user positions retained by ordinary Knorm.

Construct at most two `(u, c)` pairs per cell. A valid pair has `u` in `U`, `c`
in `C`, and `abs(u - c) <= ceil(0.10 * N)`. Choose the maximum-cardinality
one-to-one matching up to two pairs, then minimize total absolute distance;
break remaining ties by the lexicographic sequence of `(u, c)` pairs. This
caliper makes source recency explicit and refuses distant matches caused by the
different user and non-user spans.

For each accepted pair, choose one unused donor `d` from `D` closest to the
rounded midpoint of `u` and `c`, with lower position breaking ties. The same
donor is removed in both manipulated branches. A prompt-action cell is
structurally eligible only if it yields at least eight swaps across at least
four selected KV cells. The per-cell cap of two and the eight-cell head cap
limit the intervention to at most 16 entries per prompt-action continuation.

Freeze all pair and donor positions before reading signs. Report eligibility by
ratio, source and donor distance distributions, swap counts, selected cells,
and reasons for every exclusion. If any later outcome requires relaxing the
caliper, minimum dose, or tie rules, stop rather than amend the roster.

## Phase C: exposed diagnostic roster

Only after the structural lock may a separate roster materializer read the
existing r5 loose-score signs. Use 12 prompt-action cells, six at each ratio:
two existing degradations, two unchanged cases, and two existing improvements.
Within each ratio and sign stratum, rank structurally eligible cells by
SHA-256 of the UTF-8 bytes
`retrieval-diagnostic-v1\0<prompt_id>\0<action>` and take the first two. Do not
rank by loss magnitude, exposure, instruction type, swap count beyond the
fixed eligibility floor, or prior response content.

Require 12 distinct prompt IDs. Resolve a cross-ratio duplicate by keeping the
cell with the lower hash and taking the next eligible cell in the other
stratum. Iterate to a fixed point using hash order. If any of the six strata
cannot supply two distinct eligible prompts, stop without changing the dose or
sign balance.

The roster, existing source hashes, head-discovery seal, structural lock, and
all branch specifications must be sealed before generating a diagnostic
continuation. This is explicitly selected exposed development evidence.

## Phase D: three matched continuations per cell

Regenerate the accepted token-32 boundary from the pinned source and make three
independent cache clones:

1. **Knorm:** apply ordinary Knorm at the rostered removal ratio.
2. **Targeted:** start from the identical Knorm kept set, restore each original
   user entry at `u`, and remove its frozen donor at `d` in the same KV cell.
3. **Matched control:** start from the identical Knorm kept set, restore the
   paired non-user entry at `c`, and remove the same donor at `d` in the same KV
   cell.

Sort the final entries by original position. Preserve each entry's original K
and V tensors without recomputation. All other cells remain byte-identical to
ordinary Knorm. Targeted and control branches therefore use the same layers,
KV heads, swap counts, donor removals, cache lengths, and bytes; their restored
source positions differ by no more than the fixed 10 percent cache-length
caliper.

Continue all branches greedily to EOS or the existing 1,024-token total output
cap and score them with the pinned official IFEval loose scorer. Keep strict
scores as a descriptive mirror. The first roster cell is the engineering gate:
ordinary Knorm must reproduce the saved r5 token IDs exactly, source clones
must remain unchanged, targeted and control cache sizes must match Knorm, and
every branch must preserve the frozen intervention record. If it passes, the
same checks apply to the other 11 cells. Any failure is operational, not a row
exclusion.

The cap is 36 full continuations, including the first cell's acceptance run.
Record synchronized latency and peak allocated memory by branch. Existing r5
timings do not measure this intervention and must not be reused as its cost.

## Frozen causal decision rule

For cell `i`, let `q0` be its saved uncompressed reference score, `qK` the
reproduced ordinary-Knorm score, `qT` the targeted score, and `qC` the matched
control score. For the four selected degradation cells, define:

```text
target_recovery_i = (qT - qK) / (q0 - qK)
control_adjusted_i = (qT - qC) / (q0 - qK)
```

The denominator is positive by the frozen stratum. Do not clip either value.
The diagnostic is a **go for further mechanism research** only if all of these
hold:

1. At least three of the four degradation cells have
   `target_recovery_i >= 0.50` and `control_adjusted_i >= 0.25`.
2. Those qualifying cells include at least one Knorm 0.25 case and one Knorm
   0.5 case.
3. The mean `control_adjusted_i` over all four degradation cells is at least
   0.25.
4. Across the eight unchanged and improvement cells, targeted restoration is
   lower than both ordinary Knorm and matched control by at least one
   loose-scored instruction in no more than two cells.

Report every cell, instruction-pass vector, and effect sign. The unchanged and
improvement strata need not show the degradation pattern; criterion 4 only
limits collateral harm. Do not add a bootstrap interval or treat this
purposefully enriched sample as a population estimate. Failure of any
scientific criterion is a stop for this mechanism in the current scope, with
no search over thresholds, heads, calipers, positions, or roster replacements.

## Natural decision-time feature remains untested

Before any later predictor study, freeze the naturally available scalar for
ordinary action `a`:

```text
exposure(a) = weighted mean across selected KV cells of
              evicted_user_count(cell, a) / user_token_count
```

Use each cell's frozen stability score as its weight. This calculation uses the
ordinary Knorm kept indices and prompt span only, so it needs no rescue branch
or extra model forward at decision time. Confirm label-blind that it is finite
and has nonzero within-ratio variation across the 69 structural records.

Even if the rescue criteria pass, the enriched 12-cell experiment cannot show
that `exposure(a)` predicts signed final loss or improves on action and clock
baselines. That question requires a new prespecified comparison on an
independent population, with improvements retained as negative labels. If the
feature is constant within either ratio, or if the rescue gate fails, do not
collect that population.

## Remaining owner decisions

The owner retains final selection and must explicitly decide whether to:

- accept the reduced 24-case discovery adaptation, top-eight KV-cell cap,
  two-swap cell cap, 10 percent recency caliper, and eight-swap eligibility
  floor;
- authorize the exact ROUGE-1 implementation or its missing dependency;
- accept the 12-cell exposed roster and numerical rescue and collateral-harm
  rules;
- approve implementation and a bounded GPU run after the label-blind control
  audit passes; and
- define the next independent task population only if this manipulated rescue
  and the natural-feature variation gates both pass.

## Primary sources

- Wu et al., [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574), especially the retrieval-score definition, 0.1 threshold, successful-case filtering, and causal head masking.
- Wu et al., [official Retrieval Head code at the pinned commit](https://github.com/nightdessert/Retrieval_Head/tree/3ac171a6f71ce7ef1cda57d4215c390fb6ab51f2), including the three released detection triples and per-token attention calculation.
- Qwen Team, [Qwen2.5-7B-Instruct model card and configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), for the 28 query-head and four KV-head GQA architecture.
