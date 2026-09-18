"""Phase J: routing-generality campaign. Attention focus at L22/L23 for every
damaged knorm arm (ref vs compressed) at first-digit step and decision step.
Reuses probe.build_probe_state + attent.hidden_forward/layer_masses.
Appends one JSON line per (arm, step) to mass_all.jsonl. OUT: /tmp/mech2.
"""
import importlib
import json
import sys
from pathlib import Path

MECH2 = Path("/tmp/mech2")
V4 = Path("/clustergpu/home/jcampo/herald-v4")
ENGINE_ROOT = "/clustergpu/home/jcampo/herald-v3/src"
MODEL = ("/clustergpu/home/jcampo/.cache/huggingface/hub/models--Qwen--"
         "Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")

sys.path.insert(0, str(MECH2))
sys.path.insert(0, str(V4 / "scripts"))
sys.path.insert(0, ENGINE_ROOT)
import run_pair_pilot as R  # noqa: E402
engine = importlib.import_module("herald_v3.engineering.engine")  # noqa: E402
import replay as RP  # noqa: E402
import probe as P  # noqa: E402
import attent as A  # noqa: E402
import mech2 as M2  # noqa: E402


def main() -> None:
    import numpy as np
    import torch
    import transformers
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = R.load_model_and_tokenizer(MODEL, device, torch.bfloat16, transformers)
    model.eval()
    rows, groups = M2.load_rows(V4 / "data/113-competitor-v1")
    by_id = {r["id"]: r for r in rows}
    dose = {r["row"]: r for r in json.loads((MECH2 / "dose.json").read_text())}
    targets = json.loads((MECH2 / "targets.json").read_text())
    eos = R.eos_ids(model, tokenizer)

    # rope self-test on first row (same gate as attent.py)
    if A.rope_self_test(torch, model, tokenizer, rows[0]) < 0.999:
        print("ROPE SELF-TEST FAILED - aborting", flush=True)
        return

    out = MECH2 / "mass_all.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done.add((d["mid"], d["arm"], d["t"]))
    print("resuming: %d records done" % len(done), flush=True)

    for tg in targets:
        mid, arm, removal = tg["mid"], tg["arm"], tg["removal"]
        row = by_id[mid]
        cap = row.get("max_new_tokens", 128)
        # reference ids by exact deterministic replay
        prompt_ids = R.tokenize_chat_prompt(tokenizer, row["prompt"], None)
        ref_ids, _, _ = RP.replay_reference(torch, engine, model, prompt_ids, cap, eos)
        arm_steps = dose[mid]["arms"][arm]["steps"]
        arm_ids = [s["top"][0][0] for s in arm_steps]
        packed = (by_id, groups, {mid: {"ref": {"ids": list(ref_ids)}, "arms": {arm: {"ids": arm_ids}}}})
        for t in (tg["tA"], tg["tB"]):
            if (mid, arm, t) in done:
                continue
            rec = {"mid": mid, "arm": arm, "t": t}
            try:
                st_r = P.build_probe_state(torch, model, tokenizer, packed, mid, "ref", 0.0, t)
                st_c = P.build_probe_state(torch, model, tokenizer, packed, mid, arm, removal, t)
                out_r = A.hidden_forward(torch, model, st_r)
                out_c = A.hidden_forward(torch, model, st_c)
                ar = int(out_r.logits[0, -1].argmax().item())
                ac = int(out_c.logits[0, -1].argmax().item())
                rec["repro_ref"] = bool(ar == st_r["recorded"])
                rec["repro_comp"] = bool(ac == st_c["recorded"])
                L = st_r["prompt_len"]
                nspan = set(range(*st_r["spans"]["needle"][0]))
                d_span = set(range(*st_r["spans"]["digits"][0]))
                l2 = set(range(*st_r["spans"]["needle_last2"][0]))
                dn = set(range(*st_r["spans"]["disc_n"][0])) if st_r["spans"]["disc_n"] else set()
                dd = set(range(*st_r["spans"]["disc_d"][0])) if st_r["spans"]["disc_d"] else set()
                spans = {"needle": nspan, "digits": d_span, "last2": l2, "disc_n": dn, "disc_d": dd}
                dummy = torch.empty(1, 1, model.config.hidden_size, device=device, dtype=model.dtype)
                for st in (st_r, st_c):
                    cos, sin = model.model.rotary_emb(dummy, torch.tensor([[st["logical"]]], device=device))
                    st["cos"] = cos.cpu().to(torch.float32)
                    st["sin"] = sin.cpu().to(torch.float32)
                hr = [h[0, -1].detach().to(torch.float32).cpu() for h in out_r.hidden_states]
                hc = [h[0, -1].detach().to(torch.float32).cpu() for h in out_c.hidden_states]
                rec["cos"] = [round(float(hh_r.numpy() @ hh_c.numpy() / (np.linalg.norm(hh_r.numpy()) * np.linalg.norm(hh_c.numpy()))), 5)
                              for hh_r, hh_c in zip(hr, hc)]
                rec["layers"] = {}
                for li in (22, 23):
                    m_r, _ = A.layer_masses(torch, np, model, st_r, hr, li, spans, L, st_r["cos"], st_r["sin"])
                    m_c, _ = A.layer_masses(torch, np, model, st_c, hc, li, spans, L, st_c["cos"], st_c["sin"])
                    rec["layers"][str(li)] = {"ref": m_r, "comp": m_c}
                del st_r, st_c, out_r, out_c, hr, hc
            except Exception as e:  # noqa: BLE001 - record and continue
                rec["error"] = repr(e)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print("[%s %s t=%d] %s" % (mid, arm, t, "ERR " + rec["error"] if "error" in rec
                  else "repro=%s/%s L22needle=%.3f/%.3f" % (rec["repro_ref"], rec["repro_comp"],
                  rec["layers"]["22"]["ref"]["needle"], rec["layers"]["22"]["comp"]["needle"])), flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print("campaign done", flush=True)


if __name__ == "__main__":
    main()
