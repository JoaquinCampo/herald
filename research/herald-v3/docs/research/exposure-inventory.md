# IFEval exposure inventory

The reproducible inventory builder is
[`scripts/build_exposure_ledger.py`](../../scripts/build_exposure_ledger.py).
It reads the pinned official IFEval roster and the named v2 source files,
records source bytes and SHA-256 hashes, maps every source ID to the official
key, and writes `data/exposure-ledger.json`. The output is covered by the
existing `data/` ignore rule and is intentionally generated locally.

Run it from the v3 checkout with:

```sh
uv run python scripts/build_exposure_ledger.py \
  --output data/exposure-ledger.json
```

The builder fails if a required source is missing, malformed, or changes the
documented inventory counts. It does not read model outputs or other outcome
artifacts, choose a roster, or decide that any remaining prompt is fresh.
For the old reference JSON files it uses only the exact `prompt_id` field;
stored generated text, scores, token IDs, and feature values are ignored.

## Counts from the current sources

The generated ledger records these verified counts:

| Set | Count |
|---|---:|
| Official IFEval rows | 541 |
| Old sweep reference IDs | 200 |
| Current state development IDs | 133 |
| Current state confirmation IDs | 67 |
| Current state union | 200 |
| Old/current overlap | 79 |
| Exposed union | 321 |
| Official IDs remaining outside that union | 220 |
| Hidden state development IDs | 80 |
| Hidden state moderate IDs | 80 |

The builder asserts that the hidden development and moderate sets are the
same 80 IDs and are a subset of the current state union. It also asserts
`200 + 200 - 79 = 321` and `541 - 321 = 220`. These are exposure facts about
the named artifacts, not a freshness guarantee, because benchmark exposure,
other prompt variants, and selection history remain separate questions.

Each official prompt entry contains its canonical `ifeval_<key>` ID, raw and
normalized prompt hashes, instruction IDs, an `exposed` flag, and
`source_records` preserving each exact source ID. The `source_files` section
contains the absolute path, byte count, SHA-256, and relevant record count or
ID list for the official file, all 200 old reference JSON files, both current
state files, and both hidden state files.

## Near-duplicate review candidates

The builder compares every one of the 220 remaining official prompts with
every one of the 321 exposed prompts, and every unordered pair within the
remaining set, using Python's
`difflib.SequenceMatcher(..., autojunk=False).ratio()` after
`unicodedata.normalize("NFKC", text)`, collapsing whitespace, and stripping
the result. The review threshold is `0.70`. `real_quick_ratio()` is used only
as an upper-bound shortcut before calculating the exact ratio, so no match at
or above the threshold is skipped. These candidates are reported for owner
review and do not filter or exclude any prompt.

The current run reports 12 candidates, including one pair within the
remaining set so later split construction can keep a possible variant group
together:

| Pair kind | Left ID | Right ID | Ratio |
|---|---:|---:|---:|
| remaining vs exposed | 2337 | 2739 | 0.7724 |
| remaining vs exposed | 2337 | 1139 | 0.7699 |
| remaining vs exposed | 3750 | 3752 | 0.7661 |
| remaining vs exposed | 288 | 1139 | 0.7658 |
| remaining vs exposed | 288 | 2739 | 0.7500 |
| remaining vs exposed | 2337 | 1129 | 0.7284 |
| remaining vs exposed | 288 | 1129 | 0.7222 |
| remaining vs exposed | 3224 | 1480 | 0.7176 |
| remaining vs remaining | 288 | 2337 | 0.7119 |
| remaining vs exposed | 2337 | 1281 | 0.7091 |
| remaining vs exposed | 288 | 1281 | 0.7061 |
| remaining vs exposed | 288 | 1012 | 0.7033 |

Several high-scoring pairs share generic instructions such as “First repeat
the request word for word,” while their task content differs. The scores are
therefore screening evidence only. The owner must decide whether any pair is
semantically equivalent and whether the remaining set is suitable for a later
experiment.

The current source hashes recorded by the ledger include the official roster
`67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`, current
state development `70a72e48...`, current state confirmation `46268b01...`,
hidden development `b778be39...`, and hidden moderate `7b4f3ea7...`. The
complete per-file hashes and all exact IDs are in the generated ledger.
