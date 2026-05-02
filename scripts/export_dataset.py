"""Export HERALD run JSONs to parquet shards for HuggingFace release.

Reads run JSONs under <results-dir>/<press>/<model>_<ratio>_<n>p.json and
writes:

  <output-dir>/tokens/<press>.parquet      one row per (prompt, token)
  <output-dir>/sequences/<press>.parquet   one row per prompt
  <output-dir>/summaries.parquet           one row per run
  <output-dir>/README.md                   dataset card scaffold

Designed to run wherever the raw results live (orion or local).
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RATIO_RE = re.compile(r"_([0-9]+\.[0-9]+)_")

SIGNAL_KEYS = [
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "delta_h",
    "delta_h_valid",
    "kl_div",
    "top10_jaccard",
    "eff_vocab_size",
    "tail_mass",
    "logit_range",
]


def extract_ratio(name: str) -> float:
    m = RATIO_RE.search(name)
    return float(m.group(1)) if m else 0.0


def export_press(
    press_dir: Path,
    out_tokens: Path,
    out_sequences: Path,
) -> tuple[int, int, list[dict]]:
    token_rows: list[dict] = []
    sequence_rows: list[dict] = []
    summary_rows: list[dict] = []
    press = press_dir.name

    for path in sorted(press_dir.glob("*_*p.json")):
        ratio = extract_ratio(path.name)
        with path.open() as f:
            run = json.load(f)
        s = run["summary"]
        summary_rows.append(
            {
                "press": press,
                "compression_ratio": ratio,
                "total": s["total"],
                "accuracy": s.get("accuracy"),
                "catastrophic_failure_rate": s.get(
                    "catastrophic_failure_rate"
                ),
                "avg_tokens": s.get("avg_tokens"),
                "catastrophe_counts": json.dumps(
                    s.get("catastrophe_counts", {})
                ),
            }
        )

        for ex in run["results"]:
            sequence_rows.append(
                {
                    "press": press,
                    "compression_ratio": ratio,
                    "prompt_id": ex["prompt_id"],
                    "prompt_text": ex["prompt_text"],
                    "generated_text": ex["generated_text"],
                    "ground_truth": ex["ground_truth"],
                    "predicted_answer": ex["predicted_answer"],
                    "correct": ex["correct"],
                    "stop_reason": ex["stop_reason"],
                    "catastrophes": ex["catastrophes"],
                    "catastrophe_onsets": json.dumps(
                        ex.get("catastrophe_onsets", {})
                    ),
                    "num_tokens_generated": ex["num_tokens_generated"],
                    "max_new_tokens": ex["max_new_tokens"],
                    "seed": ex["seed"],
                }
            )
            for t, sig in enumerate(ex["signals"]):
                row = {
                    "press": press,
                    "compression_ratio": ratio,
                    "prompt_id": ex["prompt_id"],
                    "token_idx": t,
                }
                for k in SIGNAL_KEYS:
                    row[k] = sig.get(k)
                lps = sig.get("top5_logprobs", [])
                for i in range(5):
                    row[f"top5_logprob_{i}"] = (
                        lps[i] if i < len(lps) else None
                    )
                token_rows.append(row)

    if token_rows:
        out_tokens.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(token_rows).to_parquet(
            out_tokens, compression="zstd", index=False
        )
    if sequence_rows:
        out_sequences.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sequence_rows).to_parquet(
            out_sequences, compression="zstd", index=False
        )

    return len(token_rows), len(sequence_rows), summary_rows


DATASET_CARD = """\
---
language:
- en
license: mit
pretty_name: HERALD Logit Signals
size_categories:
- 1M<n<10M
task_categories:
- other
tags:
- llm
- kv-cache-compression
- logits
- gsm8k
- qwen
configs:
- config_name: tokens
  data_files: "tokens/*.parquet"
- config_name: sequences
  data_files: "sequences/*.parquet"
- config_name: summaries
  data_files: "summaries.parquet"
---

# HERALD Logit Signals

Per-token logit-derived signals from Qwen2.5-7B-Instruct generating
GSM8K solutions under six KV-cache compression methods, plus an
uncompressed baseline. The dataset accompanies the paper *HERALD:
Hazard Estimation via Real-time Analysis of Logit Distributions*.

## Configurations

* **`tokens`** — one row per `(press, compression_ratio, prompt_id,
  token_idx)`. Columns: 12 zero-cost logit features (entropy,
  top1_prob, top5_prob, h_alts, avg_logp, delta_h, delta_h_valid,
  kl_div, top10_jaccard, eff_vocab_size, tail_mass, logit_range)
  plus the five top-k log-probs (`top5_logprob_0` ... `_4`).
* **`sequences`** — one row per `(press, compression_ratio,
  prompt_id)`. Includes the prompt, the full decoded text, the
  predicted answer, the ground truth, the stop reason, the
  catastrophe labels, and the catastrophe onset positions.
* **`summaries`** — one row per run with aggregate accuracy, the
  catastrophic failure rate, and per-mode counts.

## Coverage

* Model: `Qwen/Qwen2.5-7B-Instruct`.
* Task: GSM8K test split, 500 prompts per run.
* Compressors: `streaming_llm`, `snapkv`, `expected_attention`,
  `knorm`, `tova`, `random`, plus `none` (uncompressed baseline).
* Compression ratios: 0.25, 0.50, 0.625, 0.75, 0.875 for each
  compressor; `none` runs at ratio 0.0.

## Loading

```python
from datasets import load_dataset

tokens = load_dataset("joaquinCampo/herald-logits", "tokens")
sequences = load_dataset(
    "joaquinCampo/herald-logits", "sequences"
)
```

## Reproduction

The full sweep is reproducible from the paper's GitHub repository
(`uv run herald sweep --num-prompts 500 --model
"Qwen/Qwen2.5-7B-Instruct"`).

## License

MIT. GSM8K prompts are redistributed under their original MIT
license.

## Citation

```bibtex
@misc{campo2026herald,
  title  = {HERALD: Hazard Estimation via Real-time Analysis of
            Logit Distributions},
  author = {Campo, Joaquin and Dufrechou, Ernesto and
            Moncecchi, Guillermo},
  year   = {2026},
}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_release"),
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokens").mkdir(exist_ok=True)
    (out / "sequences").mkdir(exist_ok=True)

    press_dirs = sorted(d for d in args.results_dir.iterdir() if d.is_dir())
    if not press_dirs:
        raise SystemExit(f"No press directories under {args.results_dir}/.")

    total_tokens = 0
    total_seqs = 0
    all_summaries: list[dict] = []

    for press_dir in press_dirs:
        n_tok, n_seq, summaries = export_press(
            press_dir,
            out / "tokens" / f"{press_dir.name}.parquet",
            out / "sequences" / f"{press_dir.name}.parquet",
        )
        total_tokens += n_tok
        total_seqs += n_seq
        all_summaries.extend(summaries)
        print(
            f"  {press_dir.name:>20}:  "
            f"{n_seq:>5,} sequences,  {n_tok:>9,} tokens"
        )

    if all_summaries:
        pd.DataFrame(all_summaries).to_parquet(
            out / "summaries.parquet",
            compression="zstd",
            index=False,
        )

    (out / "README.md").write_text(DATASET_CARD)

    print()
    print(f"  total: {total_seqs:,} sequences, {total_tokens:,} tokens")
    print(f"  output: {out}/")
    print()
    print("Next: upload to HuggingFace with")
    print(
        f"  huggingface-cli upload joaquinCampo/herald-logits "
        f"{out}/ . --repo-type dataset"
    )


if __name__ == "__main__":
    main()
