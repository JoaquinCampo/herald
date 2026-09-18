# herald-offline pack (staged 2026-09-18, no Orion/SSH needed)

Self-contained material for mechanistic research on KV-cache compression
damage (Qwen2.5-7B-Instruct, NIAH pilot 113-competitor-v1). Everything
here was pulled from Orion; no further access required for the tracks
below.

> One-time exception, owner-approved 2026-09-18: this directory commits
> raw records to git so an agent without Orion access can work from
> them. Standing rule ("bulk records stay off git") otherwise applies;
> do not extend this precedent without a fresh override.

## Contents

- `mech-records/`: all Phase G/H/H2/I/J records pulled from Orion
  `/tmp/mech2/`: `dose.json` (5 ratios x 8 rows, per-step top-5/entropy
  streams), `pinned.json`, `streaming.json`, `excise.json`,
  `mass_all.jsonl` (44 attention-focus records at L22/L23, ref vs comp),
  `targets.json`, phase logs, plus the exact probe scripts
  (`mech2.py`, `replay.py`, `probe.py`, `attent.py`, `mass_all.py`).
- `pilot-data/`: `pair-manifest.json` + `manifest.json` (prompts,
  entities, truth/distractor numbers, row ids).
- `pilot-results/`: the 8 original run records + `run.json`.
- `history/`: 87 experiment docs (`001`-`093` series + others) from
  `herald-v4/experiments/`: lineage 105-113, ablations, rescue results.
- `tokenizer/`: exact Qwen2.5-7B-Instruct tokenizer + model config
  (usable offline with the `tokenizers` pip package or raw JSON;
  no transformers install required for encode/decode work).
- `scripts/`: analysis scripts (`atlas.py`, `taxonomy.py`, `mass_agg.py`)
  and a local copy of `mass_all.jsonl`.

## Ground truth established so far (see repo docs/_why/7_*)

Three damage families (tail-readout collapse, competitor surge,
cascade), onset-intactness (earliness limit), decisive-digit-position
law (later = tougher), ~40% of damage signal-silent at full
confidence, repetition pressure mode-general, carrier-skip is
truncation-specific.

## Workable offline tracks (no GPU)

1. Record mining: cross-mode statistics, edit-structure taxonomy,
   length/termination distributions, carrier behavior, combined-signal
   characterization (NOT predictor building; characterization only).
2. History synthesis: mine `history/` for unresolved tensions and
   untested predictions; extend the lineage cross-checks.
3. Prompt/token geometry with the local tokenizer: entity rarity,
   needle-span structure, decisive-position analysis, H5
   near/far-distractor experiment design (write the protocol + script
   for later GPU execution).
4. Theory: residual-to-logit accounting extensions, routing-anchor
   models, confabulation/looping mechanism sketches.
5. Code: extend `src/herald/niah_damage.py` scoring, local
   pytest/ruff/mypy work in the repo (repo root is one level up).
6. Experiment staging: write Orion-ready scripts (following the
   patterns in `mech-records/*.py`) so GPU time is turnkey later.

## Hard limits offline

No new forward passes: no new ratios, compressors, pairs, ablations,
replays, or attention masses. New claims requiring GPU evidence must
be marked as predictions with a staged script, not conclusions.
