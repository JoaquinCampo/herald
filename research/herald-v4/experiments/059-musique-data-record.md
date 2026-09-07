# MuSiQue fixed pilot data record

Prepared 2026-09-07 for the frozen `058` QA feasibility pilot. This is data and scoring preparation only. No model generation, GPU run, predictor fit, or outcome-based row filtering was performed.

## Source and transformation

The source is the official MuSiQue repository and its linked MuSiQue-Ans development file. The downloaded raw JSONL contains 2,417 rows and has SHA256 `76eb07a2cd7b60b3336374a28097e90867f979791023f5626e7397dc2d083dd5`. The official converter was preserved locally and produced a 2,417-row official-format JSONL with SHA256 `310e4dab69ffc2db367ac8956634d8eda2ef839a16a0e26ab73f1e366a619efe`.

Sources and provenance are recorded in `data/musique-pilot-v1/metadata.json`:

- Repository: https://github.com/stonybrooknlp/musique
- Raw source: Drive file `1TRXU68wveSehVbQrRRtWsUsFkKUF43QS`
- License: official repository `LICENSE`, CC BY 4.0
- Converter: `data/musique-pilot-v1/provenance/official_converter/raw_data_to_official_format.py`, SHA256 `8c2dbb07efbf2c71a739a29406d5e5cae1c53a6609df5d411140cc82925c66aa`
- Scorer: `data/musique-pilot-v1/provenance/official_scorer/evaluate_v1.0.py`, SHA256 `f5fe66ae61dbea5172cba9d428d9924a5811f3457edc945fc1d81369c30e74b7`

The converter's raw `double__` IDs are paired with official `2hop__` IDs. The complete original 20-paragraph contexts and official labels are retained in `selected_raw.jsonl` and `selected_official.jsonl`; no test rows were inspected.

## Frozen selection and manifest

Selection is deterministic: answerable official MuSiQue-Ans dev rows whose IDs start with `2hop__`, sorted lexicographically by official ID, first 16. The selected IDs are:

```text
2hop__10017_18974
2hop__10114_599630
2hop__10122_18974
2hop__103889_86452
2hop__104095_40501
2hop__104095_40502
2hop__10515_21567
2hop__10620_49084
2hop__10620_79092
2hop__106864_80460
2hop__107185_64006
2hop__107238_64918
2hop__107548_124896
2hop__107601_110222
2hop__107690_124896
2hop__107905_110222
```

The runner-facing contract is `data/musique-pilot-v1/manifest.jsonl`, with one record per row: `id`, unique `group_id`, rendered user `prompt`, `answers` (gold answer followed by aliases), `task`, `max_new_tokens: 64`, and `seed: 0`. The manifest prompt contains all paragraphs in original order, followed by `Question:` and `Answer:`. Gold JSON fields, decomposition labels, support flags, and answerability labels are excluded from the prompt. Natural answer text can occur in source passages, and those occurrences are recorded by the preparation audit rather than removed.

The manifest JSONL SHA256 is `09b491f58372adbb5949eb9cf8847e4b78b0efdde0822c6299aec54f773f9bb6`; the pretty JSON mirror SHA256 is `3e8ac1fa264d94efbffaab8159cb09775c3cec390935f933c2fe50d1350302d6`. The selected official and raw JSONL hashes are recorded in metadata.

## Validation

`scripts/prepare_musique_pilot.py` validates both full source counts, raw and converted ID families, exact 20 paragraph indices `0..19`, at least two supporting paragraphs, two resolved decomposition supports, nonempty questions and answers, aliases, raw to converted ID pairing, unique group keys, and complete context preservation. It also checks that the selected prompt plus 64 generated tokens fits the 32,768-token context limit.

The pinned tokenizer is the read-only v3 Qwen tokenizer at `data/retrieval-tokenizer-a09a354`. Selected prompt lengths are 2,255 to 4,324 tokens. Gold answer lengths are 2 to 19 tokens, with maximum 19 and 45 tokens of horizon margin. The tokenizer files and hashes are recorded in metadata. The prompt uses the Qwen chat template with `add_generation_prompt=True` and no assistant answer prefix.

Protected v4 manifests have zero exact-prompt, context, and normalized-question overlaps with all 16 selected rows. There are 16 unique MuSiQue component group IDs. The preparation rerun passed Ruff formatting and linting, Python compilation, and all in-script scorer checks.

## Official scorer interface

For each prediction JSONL line, the official scorer requires `id`, `predicted_answer`, `predicted_support_idxs`, and `predicted_answerable`, in exactly the same order as the gold JSONL. For this answerable-only slice, the relevant outputs are mean `answer_f1`, `answer_em`, and `support_f1`; sufficiency metrics are only enabled when unanswerable gold rows are present. The runner should preserve the raw decoded answer and write one prediction line per manifest row, with no additional answer extraction or manual correction.

Answer F1 is the official maximum over the gold answer and aliases after lowercasing, punctuation removal, article removal, whitespace normalization, and token overlap F1. Support F1 is set-based paragraph-index F1 against official supporting indices. The preparation directly exercised the official normalizer, alias maximum, and partial-F1 behavior, all passing.

This record establishes a reproducible, leakage-audited fixed data/scoring slice. It does not establish Qwen competence, action loss variation, EA or Knorm behavior, or any predictive signal. Those claims require the separately frozen runner and real-model evidence.
