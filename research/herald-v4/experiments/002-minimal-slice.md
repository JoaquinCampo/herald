# Minimal RULER development slice

## Owner-adopted development protocol, 2026-09-06

The proposed 240 cases are too large for the first question. First establish
that RULER prompts, paired continuations, physical Knorm eviction, and the
exact scorer produce non-degenerate signed labels. Use 12 project-new
development prompts, with no confirmation set or numerical success gate.
This is an engineering and variation pilot, not predictor evidence.

## Population and generator

Generate 8 `niah_single_2` prompts and 4 `cwe` prompts through NVIDIA/RULER's
`scripts/data/prepare.py`, using `scripts/synthetic.yaml` unchanged, fixed
seeds, and one `max_seq_length=4096` setting, which includes the generation
cap in RULER's sizing. `niah_single_2` is essay-haystack retrieval
with one word key and one numeric value; `cwe` is the explicit non-retrieval
common-word aggregation control. Do not vary lengths or task configuration.
Lock generator hashes, seeds, input, references, `others`, and token counts.
Seed, task/config name, needle depth, answer value, template, and RULER
metadata are never predictors. Keep retrieval and control rows separate. Do
not pool them into one model, because task/template identity is a shortcut;
if a later fit is justified, split by seed and fit within task family.

Use RULER's exact `postprocess_pred` followed by `string_match_all`: each
reference must occur as a case-insensitive substring, then reference hits are
averaged per prompt. Set `q = score / 100`, and retain raw/postprocessed
prediction, references, pass vector, null/failed status, and signed
`D = q(reference) - q(action)`. Do not substitute a judge or semantic score;
the substring behavior is an assay limitation to report.

## Fixed model, boundary, and action

Use the pinned Qwen2.5-7B-Instruct snapshot on Orion, batch one, BF16,
Transformers 4.57.6, torch 2.10.0, SDPA, greedy decoding, and each RULER
`tokens_to_generate` cap, 128 for `niah` and 120 for `cwe`. Reuse the v3
paired-cache runner only after recording its source hashes.

The owner chooses a pre-answer boundary: prefill all but the last prompt
 token, leave the last prompt token pending, and start with no generated IDs.
Both arms start from independently cloned equivalent prefix caches. Evict
before processing the pending prompt token, retaining original logical RoPE
positions. This avoids handing short tasks an uncompressed answer prefix.
Require split-prefix no-action output to equal ordinary full-prefill greedy
output and preserve both sequences on failure. Compare source-cache hashes
before and after every arm and validate independent tensor storage.

For each prompt run reference/no-op and Knorm removal fractions 0.25, 0.50,
and 0.75. The three fixed severities probe variation without selecting a dose
from outcomes. Knorm scores keys by negative L2 norm and retains
int(L * (1-r)) entries per head using the existing pinned v3 primitive.
Record cache lengths/bytes, fingerprints, retained-index digest and timings.
Save all raw output tokens/text, termination and errors without exclusions.

## Measurement scope

No feature extraction, head discovery or fitting in this first pilot. First
establish a valid task-quality assay and meaningful outcome variation. Expected
Attention remains a candidate for the next experiment, not a prerequisite for
this one. Task metadata, answer locations and references are scoring/provenance
only, never model features. Any later feature family needs precise decision-time
semantics, its matched information baseline and measured extraction cost.

## Pilot readout

Report eligibility/no-op parity, score distributions and signed losses by task
and ratio, physical cache effect and paired-run cost. The only decision is whether the assay is valid and its outcomes
variable enough to justify a larger development population. Do not declare a
predictor gain or choose a strict threshold from this pilot.

## Primary sources

- Generator: https://github.com/NVIDIA/RULER/blob/main/scripts/data/prepare.py
- Config: https://github.com/NVIDIA/RULER/blob/main/scripts/synthetic.yaml
- NIAH: https://github.com/NVIDIA/RULER/blob/main/scripts/data/synthetic/niah.py
- Control: https://github.com/NVIDIA/RULER/blob/main/scripts/data/synthetic/common_words_extraction.py
- Scorer: https://github.com/NVIDIA/RULER/blob/main/scripts/eval/synthetic/constants.py
- Evaluation: https://github.com/NVIDIA/RULER/blob/main/scripts/eval/evaluate.py
- Expected Attention paper: https://arxiv.org/abs/2510.00636
- Expected Attention code: https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/expected_attention_press.py
- Model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
