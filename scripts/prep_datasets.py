"""Pre-download IFEval + LongBench datasets and NLTK corpora.

Run once on Orion through the proxy (online) so later sweep runs work
offline. IFEval needs punkt/punkt_tab for some instruction checks;
LongBench loads its English subtasks.
"""

import os

# Force online for this one-time prep regardless of ambient env.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from datasets import load_dataset  # noqa: E402

from herald.longbench import LONGBENCH_EN_TASKS  # noqa: E402

print("downloading google/IFEval", flush=True)
load_dataset("google/IFEval", split="train")

for task in LONGBENCH_EN_TASKS:
    print(f"downloading THUDM/LongBench {task}", flush=True)
    load_dataset("THUDM/LongBench", task, split="test")

import nltk  # noqa: E402

for corpus in ("punkt", "punkt_tab"):
    print(f"nltk download {corpus}", flush=True)
    nltk.download(corpus)

print("PREP_DATASETS_DONE", flush=True)
