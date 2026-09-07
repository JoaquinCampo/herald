# Alternative decision-time mechanisms after delayed JS

## Verdict

One materially different mechanism merits a small causal falsification before
HERALD spends another fresh label: **loss of access to the user instruction in
retrieval-capable attention heads**. The candidate observation is the
retrieval-score-weighted fraction of user-message KV positions that a proposed
Knorm action would evict. It uses cache identity and head function, rather than
another summary of the resulting token distribution.

This is only a mechanism hypothesis. Existing papers establish specialized
retrieval heads and the value of selective cache retention in other settings.
They do not establish per-prompt prediction of signed IFEval loss for
Qwen2.5-7B-Instruct, Knorm, or the token-32 boundary. A causal rescue test on
already exposed cases should precede predictor fitting and any new outcome
collection.

## 1. Retrieval-head instruction exposure

### Candidate observation

Identify retrieval heads on the frozen model with a label-free synthetic
retrieval assay. For grouped-query attention, map each identified query head to
its associated KV head. For candidate action `a`, compute one prespecified
score:

```text
exposure(a) = sum over retrieval heads h of
              retrieval_score(h) * fraction of user-message positions
              evicted by a in h
```

The user-message span is observable from the chat template and does not require
IFEval instruction annotations. The Knorm implementation already records kept
indices by layer and KV head, so the eventual observation should require no
extra model forward after the retrieval-head set is fixed. It is action
specific and available before choosing an action.

### What the literature establishes

Wu et al. identify a sparse set of retrieval heads across several model
families, including Qwen1.5 base and chat variants. Their retrieval score is
defined by copy behavior on synthetic needle tasks. Masking the detected heads,
but not random non-retrieval heads, causally damages retrieval; they also report
damage on extractive QA and chain-of-thought tasks. This supports a real causal
role for some heads in consulting earlier context.

H2O shows that removing attention heavy-hitter tokens degrades task performance,
while SnapKV reports that head-specific prompt positions selected from a prompt
observation window can preserve long-context performance under compression.
These results support selective retention, but neither paper predicts a
candidate action's signed final quality effect. Generic attention mass or
heavy-hitter retention is therefore supporting evidence, not a sufficient
HERALD candidate by itself.

### What remains inference

- IFEval compliance at output token 32 may depend on repeated retrieval of the
  user instruction through the same kind of heads. The cited causal evidence
  concerns retrieval, extractive QA, and reasoning, not this exact behavior.
- A synthetic assay may identify the relevant heads in Qwen2.5-7B-Instruct.
  Head identity cannot be inherited from another Qwen generation or model size.
- Evicting more user-message positions in those heads may cause larger final
  compliance loss. Kept-position identity alone does not show that a position
  would have been used later.
- The score is naturally a degradation-risk mechanism. It qualifies for the
  signed target only if the causal diagnostic also distinguishes the observed
  improvements, rather than merely separating positive loss from everything
  else.

### Strongest inexpensive falsification

Use a small, frozen diagnostic roster drawn transparently from the already
exposed r5 outcomes, with equal seeded counts of degradation, unchanged, and
improvement cases where possible. This is mechanism triage, not a new estimate
of predictive performance.

For each selected prompt and action, run three matched continuations at the
same retained-entry budget:

1. Apply ordinary Knorm.
2. Restore the user-message positions that Knorm evicted in independently
   detected retrieval KV heads, then evict the same number of other positions
   in those KV heads.
3. Apply a budget-matched control restoration using non-retrieval heads or
   seeded random prompt positions.

Continue each branch to the existing EOS or token cap and score it with the
pinned IFEval scorer. Before seeing these diagnostic outcomes, freeze the
retrieval assay, head threshold, roster, replacement rule, control, and a
minimum rescue criterion. The mechanism fails if targeted protection does not
recover final compliance on degradation cases more than the matched control,
or if its intervention effect does not vary in the expected signed direction
across the three outcome strata. A failed rescue ends this branch before any
regression or fresh labels.

Passing this gate would justify a new prespecified predictor study on an
independent population, using `exposure(a)` as the sole new information beyond
the established action and clock baseline. The diagnostic outcomes and all 76
r5 labels would remain development evidence.

## 2. Reserve candidate: signed task-state displacement

Function-vector and task-vector work reports compact internal representations
of an in-context task, with causal effects from interventions at middle layers.
That suggests a second, materially different candidate: project the
action-induced residual-stream change onto one task direction fixed from
synthetic instruction-following contrasts. The signed projection, rather than
hidden-state distance, could in principle represent movement toward or away
from the task state.

The transfer is weak. Those studies focus on functions induced by
demonstrations, while HERALD uses natural-language instructions and an
instruction-tuned Qwen model after 32 generated tokens. A residual difference
alone would be another correlational divergence measure. It becomes a credible
mechanism only if replacing the compressed branch's value along the fixed
direction with the uncompressed value causally restores final task quality
more than same-norm random-direction and unrelated-layer controls.

This is a reserve candidate because the direction, layer, and extraction
procedure introduce more researcher degrees of freedom than retrieval
exposure. They must be fixed using synthetic data, with no IFEval outcome use.
Makelov et al. further show that subspace activation patching can change output
through a dormant parallel pathway and create a false localization story.
Consequently, even a successful patch is evidence of controllability, not
proof that the patched direction is the naturally used mediator.

## Impasse condition

Declare an impasse for a cheap online predictor in the current
model, action, task, and boundary if retrieval-targeted protection fails the
matched causal rescue, and the fixed task-state patch also fails its controls.
Do not respond by mining layers, heads, token spans, horizons, or summaries on
the exposed labels. The next scientifically distinct move would require
changing the task population, compression intervention, or estimand, with a
new rationale and independent evidence.

Also stop on deployment grounds if a mechanism is visible only by retaining a
full reference cache or running near-complete duplicate continuations at every
decision. Such a result could explain damage, but it would not satisfy the
research objective of useful decision-time information at plausible cost.

## Sources

- Wu et al., [Retrieval Head Mechanistically Explains Long-Context Factuality](https://arxiv.org/abs/2404.15574), 2024.
- Zhang et al., [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048), 2023.
- Li et al., [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469), 2024.
- Todd et al., [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213), ICLR 2024.
- Hendel et al., [In-Context Learning Creates Task Vectors](https://aclanthology.org/2023.findings-emnlp.624/), Findings of EMNLP 2023.
- Makelov et al., [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://arxiv.org/abs/2311.17030), 2023.

## Review-process limitation

The project advisor instructions were inspected, but this worker had no
separate advisor model-call interface and was explicitly restricted from
further delegation. The recommendation above therefore rests on the cited
primary papers and the local r5 evidence, without an independent advisor pass.
