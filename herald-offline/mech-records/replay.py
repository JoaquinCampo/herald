"""113-competitor-v1 mechanistic replay (Orion /tmp scratch only).

Phase A: Knorm eviction audit -- per-position head-evict counts at
removal_fraction=0.1 versus needle/distractor token spans.
Phase B: twin-step replay (reference full-prefill + boundary arms) with
per-step logit capture, verified by exact token match to saved records.

Mirrors scripts/run_pair_pilot.py exactly (seed 0, bfloat16, sdpa, same
engine primitives) and never writes into the project tree.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

V4 = Path("/clustergpu/home/jcampo/herald-v4")
ENGINE_ROOT = "/clustergpu/home/jcampo/herald-v3/src"
MODEL = (
    "/clustergpu/home/jcampo/.cache/huggingface/hub/models--Qwen--"
    "Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
OUT = Path("/tmp/mech")

sys.path.insert(0, str(V4 / "scripts"))
sys.path.insert(0, ENGINE_ROOT)

import run_pair_pilot as R  # noqa: E402

engine = importlib.import_module("herald_v3.engineering.engine")


def load_rows():
    payload = json.loads(
        (V4 / "data/113-competitor-v1/pair-manifest.json").read_text()
    )
    manifest = json.loads(
        (V4 / "data/113-competitor-v1/manifest.json").read_text()
    )
    groups = {row["id"]: row for row in manifest["rows"]}
    return payload["prompts"], groups


def find_span(hay, variants, tok, label):
    hits = []
    for variant in variants:
        n = len(variant)
        for start in range(len(hay) - n + 1):
            if hay[start : start + n] == variant:
                hits.append((start, start + n))
    unique = sorted(set(hits))
    decoded = None
    if len(unique) == 1:
        decoded = tok.decode(hay[unique[0][0] : unique[0][1]])
    return unique, decoded


def phase_a(model, tokenizer, torch, rows, groups):
    device = next(model.parameters()).device
    eos = R.eos_ids(model, tokenizer)
    del eos  # spans only; no generation in phase A
    report = []
    for row in rows:
        prompt_ids = R.tokenize_chat_prompt(tokenizer, row["prompt"], None)
        boundary, _ = R.build_last_prompt_boundary(engine, model, prompt_ids)
        cache_ids = [int(v) for v in boundary.prompt_ids[0].tolist()][:-1]
        cache = engine.clone_cache(boundary.cache)
        ev = engine.compress_knorm(cache, 0.1)
        kept = ev.kept_indices
        n_layers = len(kept)
        heads = kept[0]
        n_heads = len(heads) if isinstance(heads, (list, tuple)) else 1
        full = len(cache_ids)
        votes = [0] * full
        per_head_kept = []
        for layer in kept:
            hs = layer if isinstance(layer, (list, tuple)) else [layer]
            for head in hs:
                kept_set = set(int(v) for v in head)
                per_head_kept.append(len(kept_set))
                for pos in range(full):
                    if pos not in kept_set:
                        votes[pos] += 1
        pair_id = row["group_id"]
        group = groups[pair_id]
        entity, answer = group["entity"], group["answers"][0]
        distractor = group["distractor"]
        needle = (
            f"One of the special magic numbers for {entity} is: {answer}."
        )
        distractor_sentence = (
            f" The ledger recorded {distractor} among its entries."
        )
        variants = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in (needle, " " + needle)
        ]
        d_variants = [
            tokenizer.encode(s, add_special_tokens=False)
            for s in (distractor_sentence, distractor_sentence.lstrip())
        ]
        n_hits, n_dec = find_span(cache_ids, variants, tokenizer, "needle")
        d_hits, d_dec = find_span(
            cache_ids, d_variants, tokenizer, "distractor"
        )
        votes_here = votes

        def span_votes(span, votes=votes_here):
            if span is None:
                return None
            start, end = span
            window = votes[start:end]
            return {
                "min": min(window),
                "mean": sum(window) / len(window),
                "max": max(window),
            }

        top = sorted(range(full), key=lambda p: -votes[p])[:15]
        thirds = [0, 0, 0]
        for pos, vote in enumerate(votes):
            thirds[min(2, 3 * pos // full)] += vote
        entry = {
            "row": row["id"],
            "cache_len": full,
            "layers": n_layers,
            "heads_per_layer": n_heads,
            "kept_per_head": per_head_kept[:4],
            "needle_hits": n_hits,
            "needle_decoded": (n_dec or "")[:90],
            "needle_votes": span_votes(
                n_hits[0] if len(n_hits) == 1 else None
            ),
            "distractor_hits": d_hits
            if row["id"].endswith("planted")
            else "n/a-base",
            "distractor_decoded": ((d_dec or "")[:90])
            if row["id"].endswith("planted")
            else None,
            "distractor_votes": span_votes(
                d_hits[0]
                if (row["id"].endswith("planted") and len(d_hits) == 1)
                else None
            ),
            "global_mean_votes": sum(votes) / full,
            "evict_thirds": thirds,
            "top_evicted": [
                {
                    "pos": p,
                    "votes": votes[p],
                    "ctx": tokenizer.decode(cache_ids[max(0, p - 4) : p + 3]),
                }
                for p in top
            ],
        }
        report.append(entry)
        print(
            f"[{row['id']}] needle={n_hits} votes={entry['needle_votes']} "
            f"dist_votes={entry['distractor_votes']} "
            f"mean={entry['global_mean_votes']:.1f} thirds={thirds}",
            flush=True,
        )
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (OUT / "evict.json").write_text(json.dumps(report, indent=1))
    print("wrote evict.json", flush=True)


def capture(torch, logits):
    with torch.no_grad():
        lf = logits[0].detach().to(torch.float32)
        prob = torch.softmax(lf, dim=-1)
        top_p, top_i = torch.topk(prob, 5)
        top_l, _ = torch.topk(lf, 5)
        ent = -(prob * torch.log(prob.clamp_min(1e-30))).sum().item()
    return {
        "top": [
            [int(i), round(float(p), 6)]
            for i, p in zip(top_i.tolist(), top_p.tolist(), strict=True)
        ],
        "top_logits": [round(float(v), 4) for v in top_l.tolist()],
        "ent": round(ent, 4),
        "mprob": round(float(top_p[0] - top_p[1]), 6),
        "mlogit": round(float(top_l[0] - top_l[1]), 4),
    }


def replay_reference(torch, engine, model, prompt_ids, cap, eos):
    device = next(model.parameters()).device
    prompt = prompt_ids.to(device)
    mask = torch.ones_like(prompt)
    engine._sync_device(device)
    with torch.no_grad():
        output = model(
            input_ids=prompt,
            attention_mask=mask,
            use_cache=True,
            return_dict=True,
        )
    engine._sync_device(device)
    cache = output.past_key_values
    logits = output.logits[:, -1, :]
    generated, steps, term = [], [capture(torch, logits)], "token_budget"
    while len(generated) < cap:
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos:
            term = "eos"
            break
        if len(generated) == cap:
            break
        token_tensor = torch.tensor(
            [[token]], device=device, dtype=prompt.dtype
        )
        logical = int(prompt.shape[1]) + len(generated) - 1
        output, _ = engine._pending_forward(
            model, token_tensor, cache, logical
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        steps.append(capture(torch, logits))
    return generated, term, steps


def replay_arm_from_cache(torch, engine, model, boundary, cache, cap, eos):
    """Step a pre-compressed boundary-cache clone; capture per-step stats."""
    device = next(model.parameters()).device
    pending = torch.tensor(
        [[boundary.pending_token_id]],
        device=device,
        dtype=boundary.prompt_ids.dtype,
    )
    output, _ = engine._pending_forward(
        model, pending, cache, boundary.logical_position
    )
    cache = output.past_key_values
    logits = output.logits[:, -1, :]
    generated = list(boundary.generated_ids)
    steps, term = [capture(torch, logits)], "token_budget"
    while len(generated) < cap:
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token in eos:
            term = "eos"
            break
        if len(generated) == cap:
            break
        token_tensor = torch.tensor(
            [[token]], device=device, dtype=boundary.prompt_ids.dtype
        )
        logical = int(boundary.prompt_ids.shape[1]) + len(generated) - 1
        output, _ = engine._pending_forward(
            model, token_tensor, cache, logical
        )
        cache = output.past_key_values
        logits = output.logits[:, -1, :]
        steps.append(capture(torch, logits))
    return generated, term, steps


def replay_arm(torch, engine, model, boundary, removal, cap, eos):
    device = next(model.parameters()).device
    engine._restore_rng(boundary.rng_state, device)
    cache = engine.clone_cache(boundary.cache)
    engine.compress_knorm(cache, removal)
    return replay_arm_from_cache(
        torch, engine, model, boundary, cache, cap, eos
    )


def phase_b(model, tokenizer, torch, rows):
    import torch as _t  # noqa: F401  (assert same module)

    device = next(model.parameters()).device
    eos = R.eos_ids(model, tokenizer)
    out = []
    for row in rows:
        rec_path = sorted(
            (V4 / "results/113-competitor-v1").glob(f"*-{row['id']}.json")
        )[0]
        saved = json.loads(rec_path.read_text())
        cap = row.get("max_new_tokens", 128)
        prompt_ids = R.tokenize_chat_prompt(tokenizer, row["prompt"], None)
        ref_ids, ref_term, ref_steps = replay_reference(
            torch, engine, model, prompt_ids, cap, eos
        )
        boundary, _ = R.build_last_prompt_boundary(engine, model, prompt_ids)
        arms = {}
        for removal, key in ((0.0, "knorm:0"), (0.1, "knorm:0.1")):
            gen, term, steps = replay_arm(
                torch, engine, model, boundary, removal, cap, eos
            )
            arms[key] = {"ids": gen, "term": term, "steps": steps}
        check = {
            "ref": ref_ids == list(saved["reference"]["token_ids"])
            and ref_term == saved["reference"]["termination_reason"],
            "k0": arms["knorm:0"]["ids"]
            == list(saved["arms"]["knorm:0"]["continuation"]["token_ids"]),
            "k10": arms["knorm:0.1"]["ids"]
            == list(saved["arms"]["knorm:0.1"]["continuation"]["token_ids"]),
        }
        print(
            f"[{row['id']}] ref={check['ref']} k0={check['k0']}"
            f" k10={check['k10']}",
            flush=True,
        )
        out.append(
            {
                "row": row["id"],
                "match": check,
                "ref": {"ids": ref_ids, "term": ref_term, "steps": ref_steps},
                "arms": {k: v for k, v in arms.items()},
            }
        )
        del boundary
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (OUT / "steps.json").write_text(json.dumps(out))
    print("wrote steps.json", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    import torch
    import transformers

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = R.load_model_and_tokenizer(
        MODEL, device, torch.bfloat16, transformers
    )
    model.eval()
    rows, groups = load_rows()
    rows = rows[: args.rows]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "a":
        phase_a(model, tokenizer, torch, rows, groups)
    else:
        phase_b(model, tokenizer, torch, rows)


if __name__ == "__main__":
    main()
