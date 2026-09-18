# Phase C: causal span-ablation probes at critical decision steps.
import importlib
import json
import sys
from pathlib import Path

V4 = Path("/clustergpu/home/jcampo/herald-v4")
ENGINE_ROOT = "/clustergpu/home/jcampo/herald-v3/src"
MODEL = ("/clustergpu/home/jcampo/.cache/huggingface/hub/models--Qwen--"
         "Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
OUT = Path("/tmp/mech")

sys.path.insert(0, str(V4 / "scripts"))
sys.path.insert(0, ENGINE_ROOT)
import run_pair_pilot as R
engine = importlib.import_module("herald_v3.engineering.engine")

TARGETS = [
    ("113-competitor-v1-00-base", "knorm:0.1", 0.1, 22, ("needle",)),
    ("113-competitor-v1-00-base", "ref", 0.0, 22, ("needle_last2",)),
    ("113-competitor-v1-02-planted", "knorm:0.1", 0.1, 20, ("needle", "dist", "both")),
    ("113-competitor-v1-02-planted", "ref", 0.0, 20, ("needle",)),
    ("113-competitor-v1-00-planted", "knorm:0.1", 0.1, 20, ("needle", "dist")),
    ("113-competitor-v1-03-base", "knorm:0.1", 0.1, 20, ("needle",)),
]


def load_all():
    payload = json.loads((V4 / "data/113-competitor-v1/pair-manifest.json").read_text())
    manifest = json.loads((V4 / "data/113-competitor-v1/manifest.json").read_text())
    groups = {r["id"]: r for r in manifest["rows"]}
    steps = {r["row"]: r for r in json.loads((OUT / "steps.json").read_text())}
    return payload["prompts"], groups, steps


def find_span(hay, variants):
    hits = set()
    for variant in variants:
        n = len(variant)
        for s in range(len(hay) - n + 1):
            if hay[s:s + n] == list(variant):
                hits.add((s, s + n))
    return sorted(hits)


def summarize(torch, logits):
    with torch.no_grad():
        lf = logits[0].detach().to(torch.float32)
        prob = torch.softmax(lf, dim=-1)
        top_p, top_i = torch.topk(prob, 5)
        ent = -(prob * torch.log(prob.clamp_min(1e-30))).sum().item()
    return {"argmax": int(top_i[0].item()),
            "top": [[int(i), round(float(p), 6)] for i, p in zip(top_i.tolist(), top_p.tolist())],
            "ent": round(ent, 4)}


def run_target(torch, model, tokenizer, by_id, groups, steps, target):
    device = next(model.parameters()).device
    mid, arm_key, removal, t, conditions = target
    pair_id, variant = mid.rsplit("-", 1)
    row = by_id[mid]
    group = groups[pair_id]
    entity, answer = group["entity"], group["answers"][0]
    distractor = group["distractor"]
    saved = steps[mid]
    src = saved["ref"] if arm_key == "ref" else saved["arms"][arm_key]
    recorded_ids = src["ids"]
    assert len(recorded_ids) > t, (mid, t)
    prompt_ids = R.tokenize_chat_prompt(tokenizer, row["prompt"], None)
    prompt_len = int(prompt_ids.shape[1])
    boundary, _ = R.build_last_prompt_boundary(engine, model, prompt_ids)
    full_ids = [int(v) for v in boundary.prompt_ids[0].tolist()]
    cache_ids = full_ids[:-1]
    needle = "One of the special magic numbers for %s is: %s." % (entity, answer)
    dsent = " The ledger recorded %s among its entries." % distractor
    n_vars = [tokenizer.encode(s, add_special_tokens=False) for s in (needle, " " + needle)]
    d_vars = [tokenizer.encode(s, add_special_tokens=False) for s in (dsent, dsent.lstrip())]
    n_hits = find_span(cache_ids, n_vars)
    d_hits = find_span(cache_ids, d_vars)
    assert len(n_hits) == 1, (mid, n_hits)
    nspan = n_hits[0]
    dspan = d_hits[0] if (variant == "planted" and len(d_hits) == 1) else None
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    digit_hits = find_span(cache_ids[nspan[0]:nspan[1]], [ans_ids])
    assert len(digit_hits) == 1, (mid, digit_hits)
    d0 = nspan[0] + digit_hits[0][0]
    last2 = (d0 + len(ans_ids) - 2, d0 + len(ans_ids))
    dans_ids = tokenizer.encode(distractor, add_special_tokens=False)
    ddigit_hits = find_span(cache_ids[dspan[0]:dspan[1]], [dans_ids]) if dspan else []
    dd0 = dspan[0] + ddigit_hits[0][0] if (dspan and len(ddigit_hits) == 1) else None
    disc_idx = next((i for i, (a, b) in enumerate(zip(answer, distractor)) if a != b), None)
    disc_n = (d0 + disc_idx, d0 + disc_idx + 1) if disc_idx is not None else None
    disc_d = (dd0 + disc_idx, dd0 + disc_idx + 1) if (dd0 is not None and disc_idx is not None) else None
    engine._restore_rng(boundary.rng_state, device)
    cache = engine.clone_cache(boundary.cache)
    ev = engine.compress_knorm(cache, removal)
    kept = []
    for layer in ev.kept_indices:
        hs = layer if isinstance(layer, (list, tuple)) else [layer]
        kept.append([set(int(v) for v in h) for h in hs])
    expected = int(len(cache_ids) * (1.0 - removal))
    assert all(len(s) == expected for layer in kept for s in layer), (mid, "kept-count")
    pending = torch.tensor([[boundary.pending_token_id]], device=device,
                           dtype=boundary.prompt_ids.dtype)
    out, _ = engine._pending_forward(model, pending, cache, boundary.logical_position)
    cache = out.past_key_values
    prefix = recorded_ids[:t]
    for j, tok_id in enumerate(prefix[:t - 1] if t > 1 else []):
        tt = torch.tensor([[tok_id]], device=device, dtype=boundary.prompt_ids.dtype)
        out, _ = engine._pending_forward(model, tt, cache, prompt_len + j)
        cache = out.past_key_values
    if t == 0:
        last_input, logical = pending, boundary.logical_position
    else:
        last_input = torch.tensor([[prefix[-1]]], device=device, dtype=boundary.prompt_ids.dtype)
        logical = prompt_len + t - 1
    spans = {"needle": [nspan],
             "dist": [dspan] if dspan else [],
             "both": [nspan] + ([dspan] if dspan else []),
             "needle_last2": [last2]}
    cond_results = {}
    for cond in ("baseline",) + tuple(conditions):
        probe_cache = engine.clone_cache(cache)
        if cond != "baseline":
            for layer_idx, layer in enumerate(probe_cache.layers):
                vals = layer.values
                assert vals.ndim == 4, vals.shape
                with torch.no_grad():
                    mean = vals.mean(dim=2, keepdim=True)
                    for head_idx in range(vals.shape[1]):
                        coords = set()
                        for (a, b) in spans[cond]:
                            coords.update(p for p in range(a, b) if p in kept[layer_idx][head_idx])
                        coords = sorted(coords)
                        if coords:
                            vals[0, head_idx, coords, :] = mean[0, head_idx, 0, :]
        out, _ = engine._pending_forward(model, last_input, probe_cache, logical)
        cond_results[cond] = summarize(torch, out.logits[:, -1, :])
        del probe_cache
    valid = cond_results["baseline"]["argmax"] == recorded_ids[t]
    print("[%s %s t=%d] recorded=%d valid=%s" % (mid, arm_key, t, recorded_ids[t], valid), flush=True)
    for cond, summ in cond_results.items():
        print("    %s: argmax=%d top=%s ent=%s" % (cond, summ["argmax"], summ["top"][:3], summ["ent"]), flush=True)
    del cache, boundary
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"target": [mid, arm_key, removal, t, list(conditions)], "recorded": recorded_ids[t],
            "valid": valid, "nspan": list(nspan), "dspan": list(dspan) if dspan else None,
            "last2": list(last2), "conditions": cond_results}


def main():
    import torch
    import transformers
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = R.load_model_and_tokenizer(MODEL, device, torch.bfloat16, transformers)
    model.eval()
    rows, groups, steps = load_all()
    by_id = {r["id"]: r for r in rows}
    results = [run_target(torch, model, tokenizer, by_id, groups, steps, tg) for tg in TARGETS]
    (OUT / "probe.json").write_text(json.dumps(results, indent=1))
    print("wrote probe.json valid=%s" % all(r["valid"] for r in results), flush=True)




LAYER_COMBOS = [
    ("113-competitor-v1-02-planted", "knorm:0.1", 0.1, 20, "dist"),
    ("113-competitor-v1-02-planted", "knorm:0.1", 0.1, 20, "needle"),
    ("113-competitor-v1-02-planted", "ref", 0.0, 20, "needle"),
    ("113-competitor-v1-00-base", "knorm:0.1", 0.1, 22, "needle"),
    ("113-competitor-v1-00-base", "ref", 0.0, 22, "needle_last2"),
]


def build_probe_state(torch, model, tokenizer, packed, row_mid, arm_key, removal, t):
    device = next(model.parameters()).device
    by_id, groups_d, steps = packed
    pair_id, variant = row_mid.rsplit("-", 1)
    row = by_id[row_mid]
    group = groups_d[pair_id]
    entity, answer = group["entity"], group["answers"][0]
    distractor = group["distractor"]
    saved = steps[row_mid]
    src = saved["ref"] if arm_key == "ref" else saved["arms"][arm_key]
    recorded_ids = src["ids"]
    prompt_ids = R.tokenize_chat_prompt(tokenizer, row["prompt"], None)
    prompt_len = int(prompt_ids.shape[1])
    boundary, _ = R.build_last_prompt_boundary(engine, model, prompt_ids)
    full_ids = [int(v) for v in boundary.prompt_ids[0].tolist()]
    cache_ids = full_ids[:-1]
    needle = "One of the special magic numbers for %s is: %s." % (entity, answer)
    dsent = " The ledger recorded %s among its entries." % distractor
    n_vars = [tokenizer.encode(s, add_special_tokens=False) for s in (needle, " " + needle)]
    d_vars = [tokenizer.encode(s, add_special_tokens=False) for s in (dsent, dsent.lstrip())]
    nspan = find_span(cache_ids, n_vars)[0]
    d_hits = find_span(cache_ids, d_vars)
    dspan = d_hits[0] if (variant == "planted" and len(d_hits) == 1) else None
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    d0 = nspan[0] + find_span(cache_ids[nspan[0]:nspan[1]], [ans_ids])[0][0]
    last2 = (d0 + len(ans_ids) - 2, d0 + len(ans_ids))
    engine._restore_rng(boundary.rng_state, device)
    cache = engine.clone_cache(boundary.cache)
    ev = engine.compress_knorm(cache, removal)
    kept = []
    kept_order = []
    for layer in ev.kept_indices:
        hs = layer if isinstance(layer, (list, tuple)) else [layer]
        kept.append([set(int(v) for v in h) for h in hs])
        kept_order.append([[int(v) for v in h] for h in hs])
    pending = torch.tensor([[boundary.pending_token_id]], device=device,
                           dtype=boundary.prompt_ids.dtype)
    out, _ = engine._pending_forward(model, pending, cache, boundary.logical_position)
    cache = out.past_key_values
    prefix = recorded_ids[:t]
    for j, tok_id in enumerate(prefix[:t - 1] if t > 1 else []):
        tt = torch.tensor([[tok_id]], device=device, dtype=boundary.prompt_ids.dtype)
        out, _ = engine._pending_forward(model, tt, cache, prompt_len + j)
        cache = out.past_key_values
    if t == 0:
        last_input, logical = pending, boundary.logical_position
    else:
        last_input = torch.tensor([[prefix[-1]]], device=device, dtype=boundary.prompt_ids.dtype)
        logical = prompt_len + t - 1
    dans_ids = tokenizer.encode(distractor, add_special_tokens=False)
    ddigit_hits = find_span(cache_ids[dspan[0]:dspan[1]], [dans_ids]) if dspan else []
    dd0 = dspan[0] + ddigit_hits[0][0] if (dspan and len(ddigit_hits) == 1) else None
    disc_idx = next((i for i, (a, b) in enumerate(zip(answer, distractor)) if a != b), None)
    disc_n = (d0 + disc_idx, d0 + disc_idx + 1) if disc_idx is not None else None
    disc_d = (dd0 + disc_idx, dd0 + disc_idx + 1) if (dd0 is not None and disc_idx is not None) else None
    spans = {"needle": [nspan], "digits": [(d0, d0 + len(ans_ids))], "disc_n": [disc_n] if disc_n else [], "disc_d": [disc_d] if disc_d else [], "dist": [dspan] if dspan else [],
             "needle_last2": [last2]}
    return {"cache": cache, "boundary": boundary, "last_input": last_input, "logical": logical,
            "kept": kept, "kept_order": kept_order, "spans": spans, "recorded": recorded_ids[t], "prompt_len": prompt_len}


def ablate_and_forward(torch, model, state, cond, layers):
    device = next(model.parameters()).device
    probe_cache = engine.clone_cache(state["cache"])
    span_list = state["spans"][cond]
    for layer_idx, layer in enumerate(probe_cache.layers):
        if layers is not None and layer_idx not in layers:
            continue
        vals = layer.values
        with torch.no_grad():
            mean = vals.mean(dim=2, keepdim=True)
            for head_idx in range(vals.shape[1]):
                coords = set()
                for (a, b) in span_list:
                    coords.update(p for p in range(a, b) if p in state["kept"][layer_idx][head_idx])
                coords = sorted(coords)
                if coords:
                    vals[0, head_idx, coords, :] = mean[0, head_idx, 0, :]
    out, _ = engine._pending_forward(model, state["last_input"], probe_cache, state["logical"])
    summ = summarize(torch, out.logits[:, -1, :])
    del probe_cache
    return summ


def run_layerwise(torch, model, tokenizer, rows, groups, steps):
    by_id = {r["id"]: r for r in rows}
    results = []
    for (mid, arm_key, removal, t, cond) in LAYER_COMBOS:
        st = build_probe_state(torch, model, tokenizer, (by_id, groups, steps), mid, arm_key, removal, t)
        base = ablate_and_forward(torch, model, st, cond, layers=set())
        assert base["argmax"] == st["recorded"], (mid, arm_key, t, base)
        per_layer = {}
        for layer in range(len(st["kept"])):
            per_layer[str(layer)] = ablate_and_forward(torch, model, st, cond, layers={layer})
        print("[%s %s t=%d %s] base_top=%s" % (mid, arm_key, t, cond, base["top"][:2]), flush=True)
        for layer in (0, 1, 2, 3, 26, 27):
            s = per_layer[str(layer)]
            print("    L%d: argmax=%d top=%s" % (layer, s["argmax"], s["top"][:2]), flush=True)
        results.append({"combo": [mid, arm_key, removal, t, cond], "recorded": st["recorded"],
                        "baseline": base, "per_layer": per_layer})
        del st
        if next(model.parameters()).device.type == "cuda":
            torch.cuda.empty_cache()
    (OUT / "layerwise.json").write_text(json.dumps(results, indent=1))
    print("wrote layerwise.json", flush=True)


def main_layer():
    import torch
    import transformers
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = R.load_model_and_tokenizer(MODEL, device, torch.bfloat16, transformers)
    model.eval()
    rows, groups, steps = load_all()
    run_layerwise(torch, model, tokenizer, rows, groups, steps)

if __name__ == "__main__":
    import sys as _sys
    main_layer() if len(_sys.argv) > 1 and _sys.argv[1] == "layer" else main()
