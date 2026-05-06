# Phase 1 run_damage.parquet

This is the canonical per-run compression-damage table for Phase 1.
One row per compressed run, joined to its uncompressed baseline by
`baseline_run_id`. Every analysis after the fixed-ratio sweep should
read from this table rather than reaching for `runs.parquet` and
re-deriving labels.

## Build pipeline

```sh
# After scripts/phase1_severity.py and herald metrics build have produced
# severity_phase1.parquet / trajectory_metrics.parquet / tags.parquet:
uv run python scripts/build_run_damage.py --root results/phase1

# Then the corrected alignment + success criterion:
uv run python scripts/build_phase1_alignment_final.py --root results/phase1
uv run python scripts/phase1_success_criterion_final.py --root results/phase1
```

The builder does NOT rerun generation. It re-scores IFEval and
LongBench/qasper deterministically from `runs.parquet.generated_text`,
the IFEval `instruction_id_list` / `kwargs` and the qasper gold-answer
list. GSM8K and HumanEval keep their sweep-time labels (they had
working evaluators during the sweep). To recover the post-hoc
metadata, the builder reloads `google/IFEval` and the qasper subset
from `THUDM/LongBench/data.zip` with the same `seed=42` and
`num_prompts=200` the sweep used.

`gold/phase-1-results.md`'s open question lists this table as the
methodological substitute the research plan calls for when task
correctness is saturated; this is the artifact that turns the open
reading into something testable.

## Schema

Metadata:

| column | type | notes |
| --- | --- | --- |
| `run_id` | str | compressed-run id |
| `prompt_id` | str | shared between baseline and compressed |
| `task` | str | `gsm8k` / `humaneval` / `ifeval` / `longbench_single` |
| `press` | str | compressor name (never `none`) |
| `compression_ratio` | float | |
| `baseline_run_id` | str | links to the paired uncompressed row |

Intrinsic trajectory metrics (from `trajectory_metrics.parquet`):

| column | notes |
| --- | --- |
| `sum_kl` | matched-prefix KL accumulated over the trajectory |
| `sum_js` | matched-prefix JS accumulated over the trajectory |
| `nll_ratio` | log P_uncompressed − log P_compressed over the realized sequence |
| `first_divergence_point` | first token-position where compressed top-1 ≠ uncompressed top-1 |

Sequence drift (from `severity_phase1.parquet` if present):

| column | notes |
| --- | --- |
| `rouge_l_drop` | `1 − ROUGE-L(baseline, compressed)` |
| `char_edit_ratio` | normalized character-level edit distance |
| `length_diff_ratio` | normalized length difference |
| `embedding_cosine_drop` | placeholder; null until `[metrics]` extra is wired in |

Diagnostic tags (from `tags.parquet`):

`has_looping`, `has_non_termination`, `has_format_break`, `has_drift`.

Post-hoc task scoring detail columns (filled per-row where the
scorer applies):

| column | notes |
| --- | --- |
| `qasper_em` | max normalized EM over gold answers |
| `qasper_f1` | max SQuAD-style F1 over gold answers |
| `qasper_threshold` | F1 threshold used for binary correctness |
| `qasper_n_golds` | number of gold answers compared against |
| `ifeval_num_constraints` | total instructions in the row's instruction list |
| `ifeval_num_supported` | how many our partial scorer can verify |
| `ifeval_num_unsupported` | how many fall in instruction families we don't check |
| `ifeval_num_satisfied` | of the supported, how many passed |
| `ifeval_score` | `num_satisfied / num_supported`; null if no supported |
| `ifeval_threshold` | binary threshold on `ifeval_score` |
| `ifeval_unsupported_types` | sorted list of instruction-id strings we skipped |

Unified outcome columns:

| column | notes |
| --- | --- |
| `baseline_quality_score` | float in [0,1]; binary tasks use 0.0/1.0; LongBench uses qasper_f1; IFEval uses ifeval_score |
| `compressed_quality_score` | same shape for the compressed side |
| `quality_delta` | `baseline_quality_score − compressed_quality_score`; null if either side is null |
| `baseline_correct_final` | post-hoc binary correctness; null when undefined |
| `compressed_correct_final` | post-hoc binary correctness; null when undefined |
| `gross_harm_final` | `baseline_correct_final AND NOT compressed_correct_final`; null when either side is null |
| `gross_help_final` | `NOT baseline_correct_final AND compressed_correct_final`; null when either side is null |
| `quality_label_source` | one of `gsm8k_exact`, `humaneval_pass`, `qasper_f1`, `ifeval_constraints`, `unavailable` |

## Quality label sources

A row's `quality_label_source` records which post-hoc grader yielded
the binary `*_correct_final`:

- `gsm8k_exact` — `parse_gsm8k_answer` matched gold; same as the
  sweep-time `correct` field.
- `humaneval_pass` — sweep-time HumanEval check (presence + `ast.parse`).
  This is intentionally weak (no sandboxed exec); the column carries
  the sweep-time label without re-grading.
- `qasper_f1` — token-level F1 between generation and the row's gold
  answer list, max over multiple golds, `qasper_correct =
  (qasper_f1 >= qasper_threshold)`.
- `ifeval_constraints` — partial deterministic scorer over the
  instruction families listed in `task_scoring.py:SUPPORTED_INSTRUCTIONS`;
  `ifeval_correct = (ifeval_score >= ifeval_threshold)` only when
  every constraint in the row is supported.
- `unavailable` — no usable post-hoc label could be produced
  (Qasper missing gold answers, IFEval missing or all-unsupported
  instruction list, baseline `correct` was None).

## Thresholds

| param | default | rationale |
| --- | --- | --- |
| `--qasper-threshold` | 0.5 | Standard SQuAD/Qasper-paper threshold for "correct enough" on token-level F1; CLI flag exposed for sensitivity studies |
| `--ifeval-threshold` | 1.0 | Match the official IFEval binary "all instructions satisfied" definition; CLI flag exposed for partial-credit studies |

## IFEval support coverage

The deterministic partial scorer covers these instruction families
(see `src/herald/metrics/task_scoring.py:SUPPORTED_INSTRUCTIONS`):

- `punctuation:no_comma`
- `change_case:english_lowercase`, `change_case:english_capital`,
  `change_case:capital_word_frequency`
- `length_constraints:number_words`, `length_constraints:number_sentences`,
  `length_constraints:number_paragraphs`
- `keywords:existence`, `keywords:forbidden_words`,
  `keywords:frequency`, `keywords:letter_frequency`
- `detectable_content:number_placeholders`,
  `detectable_content:postscript`
- `detectable_format:number_bullet_lists`,
  `detectable_format:number_highlighted_sections`,
  `detectable_format:title`, `detectable_format:json_format`,
  `detectable_format:multiple_sections`
- `startend:end_checker`, `startend:quotation`
- `combination:repeat_prompt`

Anything outside this list is reported as unsupported. If even one
instruction in a row is unsupported, that row's `ifeval_correct_final`
is null and `quality_label_source = unavailable`. The summary printed
by `scripts/build_run_damage.py` lists unsupported instruction-id
counts so the gap is visible to the analyst.

The durable upgrade path is installing Google's official
`instruction_following_eval` package and routing
`SUPPORTED_INSTRUCTIONS` through it. The current code exposes a
`task_scoring.has_official_ifeval()` probe but does **not** dispatch
to the official library — `ifeval_score` always uses the partial
scorer in this module. Wiring the dispatch is a follow-up; do not
assume the official lib is in use just because it is installed.

## How `gross_harm_final` interacts with undefined labels

`gross_harm_final` is True iff both sides are well-defined AND
`baseline_correct_final is True AND compressed_correct_final is False`.
When either side is None, `gross_harm_final` is null (not False). The
corrected alignment scripts (`scripts/build_phase1_alignment_final.py`,
`scripts/phase1_success_criterion_final.py`) drop null rows from the
binary AUROC computation but still use them for `quality_delta` when
both quality scores are defined.

The pre-registered Phase 1 success bar (AUROC ≥ 0.65 with bootstrap
CI lower bound > 0.5; ≥ 3 of 4 tasks) does NOT change. Only the
extrinsic label substitutes per the research plan's anticipated
"correctness saturated → use sequence/trajectory severity" rule. The
old strict-grader artifact at
`results/phase1/metrics/phase1_success_criterion.json` is preserved
unchanged.
