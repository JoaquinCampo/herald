# MuSiQue data feasibility for HERALD v4

## Decision

MuSiQue is feasible for a small, fresh HERALD v4 QA feasibility pilot. Use the
official MuSiQue-Ans two-hop dev rows, preserve the complete annotated context,
score final answer token F1 with the published scorer, and keep the pilot
diagnostic. It can establish model competence, intervention semantics, and
whether signed loss varies. Sixteen rows cannot validate a predictor or support
a generalization claim.

No MuSiQue data is currently present in the v4 workspace. This review made no
dataset download, generation run, model run, or code change.

## Official source and license

The authoritative release is the [Stony Brook MuSiQue repository](https://github.com/stonybrooknlp/musique).
Its [download script](https://raw.githubusercontent.com/stonybrooknlp/musique/main/scripts/download_data.sh)
retrieves the release archive from Google Drive. The repository documents
MuSiQue-Ans and MuSiQue-Full train, dev, and test files, together with the
single-hop ID leakage audit file. The archive is distributed under
[CC BY 4.0](https://raw.githubusercontent.com/stonybrooknlp/musique/main/LICENSE).
Acquire the archive only when the pilot is authorized, record its SHA-256, and
retain the source URL, release contents, and local acquisition timestamp in the
pilot manifest. Do not commit the archive or derived rows.

The repository's [data information file](https://raw.githubusercontent.com/stonybrooknlp/musique/main/data/.all_data_information.json)
lists the raw and official model-format files and their Drive identifiers. The
official conversion code is
[raw_data_to_official_format.py](https://raw.githubusercontent.com/stonybrooknlp/musique/main/scripts/raw_data_to_official_format.py).
Use the repository's conversion rather than making a second schema, because the
converter defines the released IDs, evidence indices, answer aliases, and the
label-hiding operation.

## Task, splits, and two-hop construction

MuSiQue-Ans contains 24,814 answerable questions:

| split | 2-hop | 3-hop | 4-hop | total |
|---|---:|---:|---:|---:|
| train | 14,376 | 4,387 | 1,175 | 19,938 |
| dev | 1,252 | 760 | 405 | 2,417 |
| test | 1,271 | 763 | 425 | 2,459 |

MuSiQue-Full adds a paired unanswerable example for each answerable question,
for 49,628 rows. The answerable and unanswerable members share the question and
have closely related contexts, so the pair is one leakage and evaluation atom.
For this feasibility study, MuSiQue-Ans dev is preferable because every row has
a directly scorable answer and the small pilot does not need an unanswerability
classifier.

The paper, [MuSiQue: Multihop Questions via Single-hop Question Composition](https://arxiv.org/html/2108.00573),
constructs a two-hop chain by composing two verified single-hop questions. The
first answer is a bridge entity mentioned in the second question, the final
answer is not already named in the first question, and the supporting
paragraphs are distinct. Disconnection filtering removes chains whose graph
structure does not require the intended connected reasoning. The 20-paragraph
context contains the supporting paragraphs and hard distractors retrieved from
the filtered single-hop gold paragraph pool. This makes the full context useful
for cache-compression evaluation, but it also means that dropping or reordering
paragraphs would change the task.

The paper's split procedure treats two multihop questions as overlapping when
they share a single-hop question, any single-hop answer, or an associated
paragraph. It greedily minimizes this overlap between train, dev, and test and
matches source and hop distributions. This is useful protection, but it does
not remove the need to group variants created inside a HERALD study.

## Official row and evidence format

The converted official row contains:

- `id`, with canonical forms such as `2hop__...`,
- `paragraphs`, each with `idx`, `title`, `paragraph_text`, and
  `is_supporting`,
- `question`,
- `question_decomposition`, with each component's single-hop `id`, question,
  answer, and `paragraph_support_idx`,
- `answer` and `answer_aliases`,
- `answerable`.

For a two-hop pilot, retain all 20 paragraphs, their original order and indices,
the decomposition, the final answer and aliases, and support annotations in a
sealed manifest. The model input must expose the question and paragraph text,
but must not expose `answer`, `answer_aliases`, `answerable`, decomposition
answers, or `is_supporting`. The support labels remain available for audit and
secondary reporting.

The converter maps raw composition shapes to stable official IDs. In particular,
two-hop rows use the `2hop__` family, while the raw triple and quadruple graph
variants map to distinct `3hop` and `4hop` families. Do not reconstruct IDs from
row order. Use the official ID as the stable key and retain the raw ID in
provenance when both are available.

## Scoring and HERALD estimand

Use the repository's
[evaluate_v1.0.py](https://raw.githubusercontent.com/stonybrooknlp/musique/main/evaluate_v1.0.py)
for the primary answer score. It consumes prediction JSONL in gold-row order,
checks matching IDs, and reads `predicted_answer`,
`predicted_support_idxs`, and `predicted_answerable`. Answer scoring lowercases,
removes punctuation and articles, normalizes whitespace, and computes token
precision, recall, and F1. The answer F1 is the maximum over the gold answer and
its aliases. Support scoring is set based paragraph precision, recall, and F1.

For HERALD, define the task score as answer token F1 on MuSiQue-Ans, scaled to
the interval 0 to 1. For each prompt, run an uncompressed reference and the
candidate compressed action from equivalent independent state. The signed final
loss is

```
reference_answer_f1 - action_answer_f1
```

Preserve negative values, zero values, partial failures, and unscorable rows.
Report support F1 as a diagnostic only unless the owner explicitly changes the
task score. The full scorer's answerability and paired sufficiency metrics are
useful for a later MuSiQue-Full study, but they are unnecessary for this first
answerable-only slice.

## Prompt, boundary, and horizon

MuSiQue defines the question and context, but it does not prescribe a chat
template, system prompt, decoding configuration, or `max_new_tokens`. These
must be frozen by HERALD before opening pilot outcomes. Use the current v4 Qwen
chat-template convention: one user message containing the complete formatted
context followed by the question and an answer-only instruction, then
`add_generation_prompt=True`. Do not add an assistant prefix unless the frozen
source row explicitly supplies one. An answer-only output matters because
explanatory text can reduce token F1 even when the reasoning is correct.

Use a deterministic formatting template that labels every paragraph with its
official `idx`, preserves title and text, and ends with the exact question. Do
not truncate the 20-paragraph context before the model's native context limit;
record the rendered prompt token count and fail closed if the limit is exceeded.

The current v4 runner treats the decision boundary as the complete rendered
prompt. It pre-fills the prompt through the boundary, feeds the final pending
prompt token through the same continuation path, and then generates only
post-boundary answer tokens. The reference and action must use the same prompt,
seed, decoding parameters, and answer horizon, with independent state created
before applying the action.

Choose the numeric generation horizon after inspecting the real answer-length
distribution in the acquired labeled split, and freeze it before generation.
The inspection may define a safety margin, but must not use pilot outcomes to
change the horizon. Record the chosen horizon and the reason in the manifest.

## Leakage and exposure controls

The official split reduces cross-split leakage, but the HERALD study needs
stronger grouping at its own boundary:

1. Keep all MuSiQue-Full answerable and unanswerable counterparts in one group.
2. Group rows sharing a normalized composed-question hash, any decomposition
   single-hop ID, any constituent answer, or any supporting paragraph title and
   text hash.
3. If a predictor is later fit, split these groups before fitting any text
   transform, calibration, or model. A dev-only 16-row feasibility pilot has no
   predictor fit and therefore cannot be used as discovery training evidence.
4. Keep answers, aliases, support flags, and decomposition answers out of all
   decision-time features. They are labels or post-decision evidence.
5. Hash the candidate rows and compare them with every protected v4 manifest
   before execution. A fresh folder is not evidence of a fresh population.

The complete context is part of the prompt, so paragraph text and titles are
admissible input. Gold support annotations and hidden decomposition answers are
not. Do not use the official test split for tuning, prompt changes, horizon
changes, action choice, or pilot selection.

## Cheapest fresh acceptance slice

After authorization to acquire the release, use the following bounded slice:

1. Convert the official MuSiQue-Ans dev file, select `answerable == true` and
   two-hop IDs, sort by the official ID, and take the first 16 rows without
   filtering on answer length, source, difficulty, or expected model behavior.
   Record the complete selection rule and row IDs.
2. Validate the 16 rows before any generation: exactly 20 paragraphs, unique
   paragraph indices, at least two supporting paragraphs, a nonempty question,
   a nonempty gold answer or alias set, and decomposition support indices that
   resolve into the retained context. Compare content and group hashes with
   protected v4 data.
3. Run one uncompressed reference and one predeclared fixed Expected Attention
   action budget per row. Do not sweep budgets or fit a predictor. Use matched
   independent states and the frozen prompt, seed, decoding, and horizon.
4. Score every row with the official answer metric and compute signed loss.
   Keep the raw generations, scorer output, state and boundary audits, and all
   failures in the run record.

Before the run, freeze these feasibility checks: at least 12 of 16 reference
answers have answer F1 at least 0.80, the reference has no silent or
unscorable rows, the action produces at least three distinct answer-F1 levels,
and at least four rows have positive signed loss. These are competence and
variation checks, not predictor success gates. If reference competence fails,
the MuSiQue slice is not interpretable for compression risk. If the action has
no measurable variation, close this action and task slice without parameter,
feature, or predictor search. If both pass, the next study may define a
prompt-grouped discovery and locked evaluation design with the owner.

## Recommendation

Proceed with MuSiQue-Ans two-hop dev as a fresh 16-row feasibility pilot after
the owner freezes the prompt, numeric horizon, and one Expected Attention
budget. Do not download the full archive, open test outcomes, or fit a predictor
yet. MuSiQue provides the graded final answer score and hard multi-paragraph
contexts that the current synthetic and proxy studies lack, while the small
pilot keeps acquisition and interpretation cheap. A successful pilot would
justify designing a larger grouped study, not claim that a predictor works.
