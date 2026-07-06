"""Diagnose live-vs-recorded reference divergence (one prompt).

Teacher-forces the RECORDED reference tokens through today's model at
batch 1 and reports every position where the greedy argmax disagrees
with the recorded next token, with the logit margin at each mismatch.

If the recorded tokens are argmax-consistent at batch 1, today's
batch-1 numerics reproduce the recording and the divergence would lie
elsewhere. If they are NOT consistent (mismatches at near-tie margins),
the recorded trajectory came from different kernel numerics (the sweep
generated references left-padded in batches of 16), and batch-1 live
references legitimately sample a different greedy path.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")

from herald.generate import build_input_ids, load_model  # noqa: E402
from herald.storage import safe_id  # noqa: E402
from herald.config import TASKS  # noqa: E402
from herald.tasks import load_prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-id", default="ifeval-1069")
    ap.add_argument(
        "--references-dir",
        default="results/sweep/llama/ifeval/references",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    rec = json.loads(
        (
            Path(args.references_dir) / f"{safe_id(args.prompt_id)}.json"
        ).read_text()
    )
    gen_ids = list(rec["gen_ids"])
    records = {
        r.prompt_id: r
        for r in load_prompts("ifeval", 200, TASKS["ifeval"])
    }
    record = records[args.prompt_id]

    lm = load_model("llama", dtype=args.dtype, device=args.device)
    prompt = build_input_ids(lm, record).to(args.device)
    full = torch.cat(
        [prompt, torch.tensor(gen_ids, device=args.device)]
    ).unsqueeze(0)
    with torch.no_grad():
        logits = lm.model(full).logits[0].float()

    p_len = prompt.shape[0]
    mismatches = []
    for t in range(len(gen_ids)):
        row = logits[p_len + t - 1]
        top2 = torch.topk(row, 2)
        pred = int(top2.indices[0])
        margin = float(top2.values[0] - top2.values[1])
        if pred != gen_ids[t]:
            rec_logit = float(row[gen_ids[t]])
            gap = float(top2.values[0]) - rec_logit
            mismatches.append((t, pred, gen_ids[t], margin, gap))
    print(
        f"{args.prompt_id}: {len(gen_ids)} recorded tokens, "
        f"{len(mismatches)} argmax mismatches under teacher forcing"
    )
    for t, pred, rec_tok, margin, gap in mismatches[:10]:
        print(
            f"  t={t:4d} argmax={pred} recorded={rec_tok} "
            f"top1-top2 margin={margin:.5f} "
            f"top1-recorded gap={gap:.5f}"
        )


if __name__ == "__main__":
    main()
