"""Phase 1 real-model correctness validation on Orion.

Loads a real base model and a handful of real task prompts, then checks
the invariants the dataset depends on. Prints PASS/FAIL per check. Run
offline (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1).
"""

import argparse

import torch

from herald.config import TASKS
from herald.generate import (
    build_input_ids,
    generate_hybrids,
    generate_reference,
    load_model,
    switch_positions,
)
from herald.presses import get_press
from herald.scoring import score
from herald.tasks import load_prompts

PRESSES = [
    "streaming_llm",
    "snapkv",
    "expected_attention",
    "knorm",
    "random",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    lm = load_model(args.model, dtype="bfloat16", device=args.device)
    pdtype = next(lm.model.parameters()).dtype
    print(f"[load] {args.model} param dtype={pdtype} eos={lm.eos_ids}")
    assert pdtype == torch.bfloat16, "model not bf16"

    records = load_prompts(args.task, args.n, TASKS[args.task])
    print(f"[data] {len(records)} {args.task} prompts")
    plens = [build_input_ids(lm, r).shape[0] for r in records]
    print(f"[data] prompt token lengths: {plens}")

    # References + features + q.
    refs = generate_reference(lm, records, args.max_new)
    for ref in refs:
        q = score(args.task, ref.text, _gold(records, ref.prompt_id))
        print(
            f"[ref] {ref.prompt_id} len={len(ref.gen_ids)} "
            f"feat={ref.features.shape} q={q}"
        )

    # scores == logits on the real model under the greedy config.
    ids = build_input_ids(lm, records[0]).unsqueeze(0).to(args.device)
    with torch.no_grad():
        out = lm.model.generate(
            ids,
            generation_config=lm.gen_config,
            max_new_tokens=8,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
    eq = all(
        torch.equal(s, l)
        for s, l in zip(out.scores, out.logits, strict=True)
    )
    print(f"[scores==logits] {'PASS' if eq else 'FAIL'}")

    ref = max(refs, key=lambda r: len(r.gen_ids))
    other = min(refs, key=lambda r: len(r.gen_ids))

    # s=0 hybrid == independent fully-compressed run, per press.
    for name in PRESSES:
        try:
            press = get_press(name, args.ratio)
            h0 = generate_hybrids(
                lm, [(ref, 0)], name, args.ratio, press, args.max_new,
                seed=0,
            )[0]
            seq = torch.tensor(
                ref.prompt_input_ids, dtype=torch.long
            ).unsqueeze(0).to(args.device)
            press2 = get_press(name, args.ratio)
            torch.manual_seed(0)
            with torch.no_grad(), press2(lm.model):
                o = lm.model.generate(
                    seq,
                    generation_config=lm.gen_config,
                    max_new_tokens=args.max_new,
                    return_dict_in_generate=True,
                )
            indep = o.sequences[0, len(ref.prompt_input_ids):].tolist()
            trimmed: list[int] = []
            for t in indep:
                trimmed.append(t)
                if t in lm.eos_ids:
                    break
            ok = h0.new_ids == trimmed
            print(f"[s0==full {name}] {'PASS' if ok else 'FAIL'} "
                  f"(hyb {len(h0.new_ids)} vs indep {len(trimmed)})")
        except Exception as exc:  # noqa: BLE001
            print(f"[s0==full {name}] ERROR {type(exc).__name__}: {exc}")

    # Hybrid batched (left-pad) vs batch=1, per press. If these differ,
    # left-pad corrupts compression and hybrids must run at batch 1.
    s = next(
        (x for x in switch_positions(len(ref.gen_ids), 16)
         if 0 < x < len(ref.gen_ids) and x < len(other.gen_ids)),
        None,
    )
    if s is None:
        print("[batch-check] skipped (runs too short)")
    else:
        for name in PRESSES:
            try:
                p1 = get_press(name, args.ratio)
                single = generate_hybrids(
                    lm, [(ref, s)], name, args.ratio, p1, args.max_new,
                    seed=0,
                )[0]
                p2 = get_press(name, args.ratio)
                batched = generate_hybrids(
                    lm,
                    [(other, s), (ref, s)],
                    name, args.ratio, p2, args.max_new, seed=0,
                )
                bref = next(
                    h for h in batched if h.prompt_id == ref.prompt_id
                )
                ok = single.new_ids == bref.new_ids
                print(f"[batch==single {name}] "
                      f"{'PASS' if ok else 'FAIL'} "
                      f"(b1 {len(single.new_ids)} vs bN "
                      f"{len(bref.new_ids)})")
            except Exception as exc:  # noqa: BLE001
                print(f"[batch==single {name}] ERROR "
                      f"{type(exc).__name__}: {exc}")

    # Prefix identity at s.
    if s is not None:
        press = get_press("knorm", args.ratio)
        h = generate_hybrids(
            lm, [(ref, s)], "knorm", args.ratio, press, args.max_new
        )[0]
        full = ref.gen_ids[:s] + h.new_ids
        ok = full[:s] == ref.gen_ids[:s]
        print(f"[prefix-identity s={s}] {'PASS' if ok else 'FAIL'}")

    print("VALIDATE DONE")


def _gold(records: list, prompt_id: str) -> dict:
    return next(r.gold for r in records if r.prompt_id == prompt_id)


if __name__ == "__main__":
    main()
