# Phase F: manual single-query attention mass per layer (ref vs comp).
# Includes a RoPE self-test: manual q-rotation vs cached K cosine.
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/mech")
sys.path.insert(0, "/clustergpu/home/jcampo/herald-v3/src")
sys.path.insert(0, "/clustergpu/home/jcampo/herald-v4/scripts")
import probe as P
import run_pair_pilot as R
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
import torch
import numpy as np

MODEL = ("/clustergpu/home/jcampo/.cache/huggingface/hub/models--Qwen--"
         "Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
OUT = Path("/tmp/mech")
F_PAIRS = [
    ("113-competitor-v1-02-planted", 20),
    ("113-competitor-v1-00-base", 22),
    ("113-competitor-v1-00-planted", 20),
    ("113-competitor-v1-03-base", 20),
]


def rotate_q(q, cos, sin):
    # GPT-NeoX style: q (..., D)
    d = q.shape[-1]
    q1, q2 = q[..., :d // 2], q[..., d // 2:]
    rot = torch.cat([-q2, q1], dim=-1)
    c = cos.reshape(-1).to(q.device)
    s = sin.reshape(-1).to(q.device)
    return q * c + rot * s


def rope_self_test(torch, model, tokenizer, row):
    device = next(model.parameters()).device
    prompt_ids = R.tokenize_chat_prompt(tokenizer, row["prompt"], None).to(device)
    mask = torch.ones_like(prompt_ids)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, attention_mask=mask, use_cache=True,
                    output_hidden_states=True, return_dict=True)
    nq = model.config.num_attention_heads
    D = model.config.hidden_size // nq
    L = int(prompt_ids.shape[1])
    worst = 1.0
    for li in (0, 13, 27):
        lyr = model.model.layers[li]
        sattn = lyr.self_attn
        with torch.no_grad():
            hn = lyr.input_layernorm(out.hidden_states[li][0].to(device, model.dtype)).to(torch.float32)
        K = out.past_key_values.layers[li].keys[0].to(torch.float32)
        for pos in (0, L // 2, L - 2):
            with torch.no_grad():
                k = sattn.k_proj(hn[pos].to(device, model.dtype)).to(torch.float32)
                dummy = torch.empty(1, 1, model.config.hidden_size, device=device, dtype=model.dtype)
                cos, sin = model.model.rotary_emb(dummy, torch.tensor([[pos]], device=device))
            K4 = k.view(1, 4, 1, D)
            kr, _ = apply_rotary_pos_emb(K4, K4, cos, sin)
            for kh in range(4):
                manual = kr[0, kh, 0]
                cached = K[kh, pos]
                c = float((manual @ cached) / (manual.norm() * cached.norm()).clamp_min(1e-12))
                worst = min(worst, c)
    print("rope self-test worst cosine: %.6f" % worst, flush=True)
    return worst


def hidden_forward(torch, model, st):
    device = next(model.parameters()).device
    cache = st["cache"]
    S = int(cache.layers[0].values.shape[2])
    mask = torch.ones((1, S + 1), device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=st["last_input"], attention_mask=mask,
                    cache_position=torch.tensor([st["logical"]], device=device),
                    past_key_values=cache, use_cache=True,
                    output_hidden_states=True, return_dict=True)
    return out


def layer_masses(torch, np, model, st, hidden, li, spans, L_prompt, cos, sin):
    device = next(model.parameters()).device
    sattn = model.model.layers[li].self_attn
    nq, nkv = model.config.num_attention_heads, model.config.num_key_value_heads
    g = nq // nkv
    D = model.config.hidden_size // nq
    with torch.no_grad():
        hn = model.model.layers[li].input_layernorm(hidden[li].to(device, model.dtype)).to(torch.float32)
        Q = sattn.q_proj(hn.to(device, model.dtype)).to(torch.float32).view(nq, D)
        Qr = rotate_q(Q, cos, sin)
        K = st["cache"].layers[li].keys[0].to(torch.float32)
        W = torch.softmax(torch.einsum("gqd,gkd->gqk", Qr.view(4, 7, 128), K) * sattn.scaling, dim=-1).view(nq, -1)
    W = W.cpu().numpy()
    assert abs(W.sum(axis=1).mean() - 1.0) < 1e-4, W.sum(axis=1).mean()
    order = st["kept_order"][li]
    n_keep = len(order[0])
    ent = float((-(W * np.log(np.clip(W, 1e-30, 1))).sum(axis=1)).mean())
    res = {}
    for name, s in spans.items():
        if not s:
            res[name] = 0.0
            continue
        per_q = []
        for qh in range(nq):
            kh = qh // g
            cols = [j for j, p in enumerate(order[kh]) if p in s]
            m = float(W[qh, cols].sum()) if cols else 0.0
            # appended prefix columns
            base = n_keep
            total_cols = W.shape[1]
            pre = [j for j in range(base, total_cols)
                   if (L_prompt - 1 + (j - base)) in s]
            if pre:
                m += float(W[qh, pre].sum())
            per_q.append(m)
        res[name] = sum(per_q) / len(per_q)
    return res, ent


def main():
    import numpy as np
    import torch
    import transformers
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = R.load_model_and_tokenizer(MODEL, device, torch.bfloat16, transformers)
    model.eval()
    rows, groups, steps = P.load_all()
    by_id = {r["id"]: r for r in rows}
    packed = (by_id, groups, steps)
    worst = rope_self_test(torch, model, tokenizer, rows[0])
    if worst < 0.999:
        print("ROPE SELF-TEST FAILED - aborting", flush=True)
        return
    results = []
    for (mid, t) in F_PAIRS:
        pair_id, variant = mid.rsplit("-", 1)
        group = groups[pair_id]
        L_prompt = None
        arms = {}
        for arm_key, removal in (("ref", 0.0), ("knorm:0.1", 0.1)):
            st = P.build_probe_state(torch, model, tokenizer, packed, mid, arm_key, removal, t)
            L_prompt = st["prompt_len"]
            out = hidden_forward(torch, model, st)
            amax = int(out.logits[0, -1].argmax().item())
            dummy = torch.empty(1, 1, model.config.hidden_size, device=device, dtype=model.dtype)
            cos, sin = model.model.rotary_emb(dummy, torch.tensor([[st["logical"]]], device=device))
            cos = cos.cpu().to(torch.float32)
            sin = sin.cpu().to(torch.float32)
            st["cos"], st["sin"] = cos, sin
            hidden = [h[0, -1].detach().to(torch.float32).cpu() for h in out.hidden_states]
            arms[arm_key] = {"st": st, "hidden": hidden, "reproduced": amax == st["recorded"]}
            print("[%s %s t=%d] reproduced=%s" % (mid, arm_key, t, amax == st["recorded"]), flush=True)
            del out
        nspan = set(range(*arms["ref"]["st"]["spans"]["needle"][0]))
        dlist = arms["ref"]["st"]["spans"]["dist"]
        dspan = set(range(*dlist[0])) if dlist else set()
        L = L_prompt
        d_span = set(range(*arms["ref"]["st"]["spans"]["digits"][0]))
        l2_span = set(range(*arms["ref"]["st"]["spans"]["needle_last2"][0]))
        dn_list = arms["ref"]["st"]["spans"]["disc_n"]
        dd_list = arms["ref"]["st"]["spans"]["disc_d"]
        dn_span = set(range(*dn_list[0])) if dn_list else set()
        dd_span = set(range(*dd_list[0])) if dd_list else set()
        spans = {"needle": nspan, "digits": d_span, "last2": l2_span, "disc_n": dn_span, "disc_d": dd_span, "dist": dspan, "sink": {0, 1, 2, 3},
                 "tail": set(range(L - 65, L - 1)), "prefix": set(range(L - 1, L + 64))}
        pair_res = {"pair": [mid, t], "layers": []}
        cos_list = []
        for li in range(28):
            hr = arms["ref"]["hidden"][li].numpy()
            hc = arms["knorm:0.1"]["hidden"][li].numpy()
            cos = float(hr @ hc / (np.linalg.norm(hr) * np.linalg.norm(hc)))
            cos_list.append(round(cos, 5))
            lm_r, ent_r = layer_masses(torch, np, model, arms["ref"]["st"], arms["ref"]["hidden"], li, spans, L, arms["ref"]["st"]["cos"], arms["ref"]["st"]["sin"])
            lm_c, ent_c = layer_masses(torch, np, model, arms["knorm:0.1"]["st"], arms["knorm:0.1"]["hidden"], li, spans, L, arms["knorm:0.1"]["st"]["cos"], arms["knorm:0.1"]["st"]["sin"])
            pair_res["layers"].append({"cos": round(cos, 5), "ref": lm_r, "comp": lm_c,
                                       "ent_ref": round(ent_r, 4), "ent_comp": round(ent_c, 4)})
        print("[%s t=%d] cosines: %s" % (mid, t, cos_list), flush=True)
        results.append(pair_res)
        del arms
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (OUT / "mass.json").write_text(json.dumps(results, indent=1))
    print("wrote mass.json", flush=True)


if __name__ == "__main__":
    main()
