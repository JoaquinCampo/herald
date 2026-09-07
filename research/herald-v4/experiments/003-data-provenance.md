# RULER pilot v1 provenance

Status: READY for the pair runner's 12 development rows. This is an exposed
development pilot, not confirmation evidence and not a predictor fit.

Source and generation:

- Official NVIDIA/RULER checkout: `vendor/ruler/`, detached at
  `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.
- Official generator: `scripts/data/prepare.py`, with unchanged
  `scripts/synthetic.yaml`, `niah.py`, and `common_words_extraction.py`.
- Tasks: 8 `niah_single_2` rows at seed `2026090601`, cap 128; 4 `cwe` rows
  at seed `2026090602`, cap 120. Both use official `max_seq_length=4096`.
- Public corpus required by official NIAH generation: three Paul Graham URLs
  listed in `data/ruler-pilot-v1/PaulGrahamEssays_URLs.txt`, combined by the
  pinned RULER downloader into `PaulGrahamEssays.json`.
- No model weights were downloaded. The local Qwen tokenizer is read-only at
  `/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3/data/retrieval-tokenizer-a09a354`.

Artifacts and hashes:

- `manifest.json`: `05db8d34df8022d99c200164990cfb28d41cbb20dfd5e8238aa42641eec40f54`.
- NIAH raw JSONL: `e63ba97df62059395905e02b1a3fbfea6c846a3eae3d4169226156e9abaf61db`.
- CWE raw JSONL: `fa16b25be3f286661e233610b4d405f2bbb90fb4fd75a2a790d38fb11675ee01`.
- Public corpus JSON: `807e9e9a259efb99cd1cc13018c254e5fb74ea714c827b9c0d5c1706a2e49571`.
- Corpus URL list: `a1f156bd460fe6c0c4fc063deb3ed79cff1a17a26cc5b061fac3f05e14f68db3`.
- Pinned corpus downloader: `d89622edfe08d4011718bb2634da742d69a56728139fa5e236e00cca703cc84c`.
- Qwen `tokenizer.json`: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

The manifest preserves each official `input` string and `outputs` references.
It records task, seed, generation cap, official sizing, raw Qwen prompt tokens,
and rendered Qwen chat-template tokens. The pair runner must apply
`tokenizer.apply_chat_template([{"role":"user","content": prompt}],
tokenize=True, add_generation_prompt=True)`; this rendering is deliberately
separate from the official generator's 4096 sizing.

Scoring is pinned to RULER's `evaluate.py::postprocess_pred` and
`synthetic/constants.py::string_match_all`. `score_prediction` stores raw and
postprocessed text, the 0-100 exact substring score, score fraction and the
per-reference pass vector. Their source hashes are `b9a5fcbded7209663d97670496c74dbcd6358ddbe3d9887a0c5f0c4f974bfe1c` and
`6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f`.
Empty, missing or failed predictions remain marked.

Reproducibility checks passed on 2026-09-06: rerunning both official generator
commands with the same tokenizer, corpus, seeds and config produced byte-identical
NIAH and CWE JSONL hashes. The scorer self-check passed exact hit, miss and
multi-reference examples. Reproduce with:

```text
.venv-v4/bin/python scripts/prepare_ruler_pilot.py
.venv-v4/bin/python scripts/score_ruler_pilot.py --self-check
```
