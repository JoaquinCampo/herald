"""Broad-landscape mechanism probes for 113-competitor-v1 (Orion scratch).

Phase G: dose-response -- knorm at 0.0/0.05/0.1/0.25/0.5, damage scored per
  (row, removal) with the numeric-run rule. Validates the extended harness
  by exact match on the 0.0 and 0.1 arms against saved records.
Phase H: H3 needle-pinned Knorm -- same kept budget as 0.1/0.25, but needle
  span positions are force-kept (score +inf). Same amount, different
  victims: adjudicates whether needle positions drive the damage.
Phase I: StreamingLLM-style eviction (4 attention sinks + recent window) at
  matched budgets. First cross-compressor: same failure modes or not?

Mirrors run_pair_pilot (seed 0, bfloat16, sdpa, engine primitives).
Reuses replay.replay_reference / replay_arm_from_cache / capture.
Writes only to --out-dir.
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

NUMERIC_RUN_RE = re.compile(r"[0-9]{5,}")
ESSAY_URL_ID = "1056050270"
SINKS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("g", "h", "h2", "i"), required=True
    )
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--pilot-scripts", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def score_text(text: str, answers: list[str]) -> dict:
    runs = [r for r in NUMERIC_RUN_RE.findall(text) if r != ESSAY_URL_ID]
    return {
        "score": 1.0 if any(a in runs for a in answers) else 0.0,
        "runs": runs,
    }


def load_rows(data_dir: Path):
    payload = json.loads((data_dir / "pair-manifest.json").read_text())
    manifest = json.loads((data_dir / "manifest.json").read_text())
    groups = {row["id"]: row for row in manifest["rows"]}
    return payload["prompts"], groups


def get_needle_span(
    tokenizer, cache_ids: list[int], group: dict
) -> tuple[int, int]:
    entity, answer = group["entity"], group["answers"][0]
    needle = f"One of the special magic numbers for {entity} is: {answer}."
    variants = [
        tokenizer.encode(s, add_special_tokens=False)
        for s in (needle, " " + needle)
    ]
    hits = set()
    for variant in variants:
        n = len(variant)
        for s in range(len(cache_ids) - n + 1):
            if cache_ids[s : s + n] == variant:
                hits.add((s, s + n))
    found = sorted(hits)
    assert len(found) == 1, found
    return found[0]


def gather_heads(x, orders: list, torch):
    """Gather kept columns per head; orders[kh] lists kept positions."""
    parts = []
    for head_idx, kept in enumerate(orders):
        idx = torch.tensor(kept, dtype=torch.long, device=x.device)
        idx = idx.view(1, 1, -1, 1).expand(1, 1, -1, x.shape[-1])
        parts.append(x[:, head_idx : head_idx + 1].gather(2, idx))
    return torch.cat(parts, dim=1).contiguous()


def pinned_compress(
    engine, torch, cache, removal: float, pin: set[int]
) -> list:
    """Knorm topk with pinned positions force-kept; same total budget."""
    device = engine._cache_tensors(cache)[0].device
    engine._sync_device(device)
    kept_all = []
    with torch.no_grad():
        for layer in cache.layers:
            keys, values = layer.keys, layer.values
            seq = int(keys.shape[-2])
            kept_count = int(seq * (1.0 - removal))
            layer_kept = []
            for head_idx in range(keys.shape[1]):
                scores = -keys[0, head_idx].norm(dim=-1)
                for p in pin:
                    if p < seq:
                        scores[p] = float("inf")
                layer_kept.append(
                    scores.topk(kept_count, dim=-1).indices.tolist()
                )
            kept_all.append(layer_kept)
            layer.keys = gather_heads(keys, layer_kept, torch)
            layer.values = gather_heads(values, layer_kept, torch)
    engine._sync_device(device)
    return kept_all


def excise_compress(engine, torch, cache, drop_per_head: list) -> None:
    """Keep every position except the per-head drop sets."""
    device = engine._cache_tensors(cache)[0].device
    engine._sync_device(device)
    with torch.no_grad():
        for layer, heads in zip(cache.layers, drop_per_head, strict=True):
            seq = int(layer.keys.shape[-2])
            orders = [
                [p for p in range(seq) if p not in drop] for drop in heads
            ]
            layer.keys = gather_heads(layer.keys, orders, torch)
            layer.values = gather_heads(layer.values, orders, torch)
    engine._sync_device(device)


def streaming_compress(
    engine, torch, cache, removal: float, sinks: int = SINKS
) -> list:
    """Keep first `sinks` positions plus the most recent window."""
    device = engine._cache_tensors(cache)[0].device
    engine._sync_device(device)
    kept_all = []
    with torch.no_grad():
        for layer in cache.layers:
            keys = layer.keys
            seq = int(keys.shape[-2])
            kept_count = int(seq * (1.0 - removal))
            order = list(range(sinks)) + list(
                range(seq - (kept_count - sinks), seq)
            )
            assert len(order) == kept_count, (len(order), kept_count)
            kept_all.append([list(order) for _ in range(keys.shape[1])])
    with torch.no_grad():
        for layer, heads in zip(cache.layers, kept_all, strict=True):
            layer.keys = gather_heads(layer.keys, heads, torch)
            layer.values = gather_heads(layer.values, heads, torch)
    return kept_all


def replay_pinned_or_streaming(
    replay,
    engine,
    torch,
    model,
    boundary,
    cache,
    cap: int,
    eos,
    tokenizer,
    answers: list[str],
) -> dict:
    gen, term, steps = replay.replay_arm_from_cache(
        torch, engine, model, boundary, cache, cap, eos
    )
    text = tokenizer.decode(gen, skip_special_tokens=True)
    sc = score_text(text, answers)
    return {
        "score": sc["score"],
        "runs": sc["runs"],
        "term": term,
        "len": len(gen),
        "steps": steps,
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.pilot_scripts)
    sys.path.insert(0, args.engine_root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_pair_pilot as R  # noqa: E402

    engine = importlib.import_module("herald_v3.engineering.engine")  # noqa: E402
    import replay as RP  # noqa: E402
    import torch  # noqa: E402
    import transformers  # noqa: E402

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = R.load_model_and_tokenizer(
        args.model, device, torch.bfloat16, transformers
    )
    model.eval()
    rows, groups = load_rows(Path(args.data_dir))
    rows = rows[: args.rows]
    cuda = next(model.parameters()).device.type == "cuda"
    eos = R.eos_ids(model, tokenizer)

    if args.phase == "g":
        report = []
        for row in rows:
            saved = json.loads(
                sorted(Path(args.results_dir).glob(f"*-{row['id']}.json"))[
                    0
                ].read_text()
            )
            cap = row.get("max_new_tokens", 128)
            prompt_ids = R.tokenize_chat_prompt(
                tokenizer, row["prompt"], None
            )
            ref_ids, ref_term, _ = RP.replay_reference(
                torch, engine, model, prompt_ids, cap, eos
            )
            boundary, _ = R.build_last_prompt_boundary(
                engine, model, prompt_ids
            )
            row_out: dict = {
                "row": row["id"],
                "ref_match": ref_ids == list(saved["reference"]["token_ids"]),
                "arms": {},
            }
            print(
                f"[{row['id']}] ref_match={row_out['ref_match']}", flush=True
            )
            for r in (0.0, 0.05, 0.1, 0.25, 0.5):
                engine._restore_rng(
                    boundary.rng_state, engine._model_device(model)
                )
                cache = engine.clone_cache(boundary.cache)
                engine.compress_knorm(cache, r)
                gen, term, steps = RP.replay_arm_from_cache(
                    torch, engine, model, boundary, cache, cap, eos
                )
                text = tokenizer.decode(gen, skip_special_tokens=True)
                sc = score_text(text, row["answers"])
                arm = {
                    "score": sc["score"],
                    "runs": sc["runs"],
                    "term": term,
                    "len": len(gen),
                    "steps": steps,
                }
                if r in (0.0, 0.1):
                    key = "knorm:0" if r == 0.0 else "knorm:0.1"
                    arm["match_saved"] = gen == list(
                        saved["arms"][key]["continuation"]["token_ids"]
                    )
                row_out["arms"][f"knorm:{r}"] = arm
                print(
                    f"[{row['id']} knorm:{r}] score={sc['score']}",
                    f"runs={sc['runs']}",
                    f"match={arm.get('match_saved', 'n/a')}",
                    flush=True,
                )
                del cache
            report.append(row_out)
            del boundary
            if cuda:
                torch.cuda.empty_cache()
        (out_dir / "dose.json").write_text(json.dumps(report))
        print("wrote dose.json", flush=True)
    elif args.phase == "h":
        report = []
        for row in rows:
            pair_id, _ = row["id"].rsplit("-", 1)
            cap = row.get("max_new_tokens", 128)
            prompt_ids = R.tokenize_chat_prompt(
                tokenizer, row["prompt"], None
            )
            boundary, _ = R.build_last_prompt_boundary(
                engine, model, prompt_ids
            )
            cache_ids = [int(v) for v in boundary.prompt_ids[0].tolist()][:-1]
            nspan = get_needle_span(tokenizer, cache_ids, groups[pair_id])
            pin = set(range(*nspan))
            row_out = {"row": row["id"], "needle": list(nspan), "arms": {}}
            for name, r in (("pinned:0.1", 0.1), ("pinned:0.25", 0.25)):
                engine._restore_rng(
                    boundary.rng_state, engine._model_device(model)
                )
                cache = engine.clone_cache(boundary.cache)
                kept = pinned_compress(engine, torch, cache, r, pin)
                kept_ok = all(
                    p in s for layer in kept for s in layer for p in pin
                )
                rec = replay_pinned_or_streaming(
                    RP,
                    engine,
                    torch,
                    model,
                    boundary,
                    cache,
                    cap,
                    eos,
                    tokenizer,
                    row["answers"],
                )
                rec["pinned_kept"] = kept_ok
                row_out["arms"][name] = rec
                print(
                    f"[{row['id']} {name}] pinned={kept_ok}",
                    f"score={rec['score']} runs={rec['runs']}",
                    flush=True,
                )
                del cache
            report.append(row_out)
            del boundary
            if cuda:
                torch.cuda.empty_cache()
        (out_dir / "pinned.json").write_text(json.dumps(report))
        print("wrote pinned.json", flush=True)
    elif args.phase == "h2":
        report = []
        for row in rows:
            pair_id, _ = row["id"].rsplit("-", 1)
            cap = row.get("max_new_tokens", 128)
            prompt_ids = R.tokenize_chat_prompt(
                tokenizer, row["prompt"], None
            )
            boundary, _ = R.build_last_prompt_boundary(
                engine, model, prompt_ids
            )
            cache_ids = [int(v) for v in boundary.prompt_ids[0].tolist()][:-1]
            nspan = get_needle_span(tokenizer, cache_ids, groups[pair_id])
            engine._restore_rng(
                boundary.rng_state, engine._model_device(model)
            )
            probe = engine.clone_cache(boundary.cache)
            ev = engine.compress_knorm(probe, 0.1)
            needle_set = set(range(*nspan))
            drop_global: set[int] = set()
            for layer in ev.kept_indices:
                hs = layer if isinstance(layer, (list, tuple)) else [layer]
                for h in hs:
                    kept_h = set(int(v) for v in h)
                    drop_global.update(
                        p for p in needle_set if p not in kept_h
                    )
            drop_sorted = sorted(drop_global)
            drop_per_head = []
            for layer in ev.kept_indices:
                hs = layer if isinstance(layer, (list, tuple)) else [layer]
                drop_per_head.append([list(drop_sorted) for _ in hs])
            n_drop = len(drop_sorted)
            del probe
            engine._restore_rng(
                boundary.rng_state, engine._model_device(model)
            )
            cache = engine.clone_cache(boundary.cache)
            excise_compress(engine, torch, cache, drop_per_head)
            rec = replay_pinned_or_streaming(
                RP,
                engine,
                torch,
                model,
                boundary,
                cache,
                cap,
                eos,
                tokenizer,
                row["answers"],
            )
            rec["dropped_total"] = n_drop
            report.append(
                {"row": row["id"], "needle": list(nspan), "arm": rec}
            )
            print(
                f"[{row['id']} excise] drop={n_drop} "
                f"score={rec['score']} runs={rec['runs']}",
                flush=True,
            )
            del cache, boundary
            if cuda:
                torch.cuda.empty_cache()
        (out_dir / "excise.json").write_text(json.dumps(report))
        print("wrote excise.json", flush=True)
    else:
        report = []
        for row in rows:
            cap = row.get("max_new_tokens", 128)
            prompt_ids = R.tokenize_chat_prompt(
                tokenizer, row["prompt"], None
            )
            boundary, _ = R.build_last_prompt_boundary(
                engine, model, prompt_ids
            )
            row_out = {"row": row["id"], "arms": {}}
            for name, r in (("stream:0.1", 0.1), ("stream:0.25", 0.25)):
                engine._restore_rng(
                    boundary.rng_state, engine._model_device(model)
                )
                cache = engine.clone_cache(boundary.cache)
                streaming_compress(engine, torch, cache, r)
                rec = replay_pinned_or_streaming(
                    RP,
                    engine,
                    torch,
                    model,
                    boundary,
                    cache,
                    cap,
                    eos,
                    tokenizer,
                    row["answers"],
                )
                row_out["arms"][name] = rec
                print(
                    f"[{row['id']} {name}] score={rec['score']}",
                    f"runs={rec['runs']}",
                    flush=True,
                )
                del cache
            report.append(row_out)
            del boundary
            if cuda:
                torch.cuda.empty_cache()
        (out_dir / "streaming.json").write_text(json.dumps(report))
        print("wrote streaming.json", flush=True)


if __name__ == "__main__":
    main()
