#!/usr/bin/env python3
"""Run the fixed direct last-prompt query salience mechanism diagnostic."""

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pair_pilot as runner  # noqa: E402
from diagnose_needle_rescue import custom_branch, native_masks, score_branch  # noqa: E402
from score_ruler_pilot import OFFICIAL_CONSTANTS, OFFICIAL_EVAL  # noqa: E402

TARGET_IDS = (
    "ruler-ea-dev-v1-niah_single_2-000",
    "ruler-ea-dev-v1-niah_single_2-004",
    "ruler-ea-dev-v1-niah_single_2-011",
    "ruler-ea-dev-v1-niah_single_2-001",
)
FRACTION = 0.10
SINK = 4
SEED = 0


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_prior_record(directories, row_id):
    found = []
    for directory in directories:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.glob("*.json"):
            if path.name == "run.json":
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if record.get("manifest_row", {}).get("id") == row_id:
                found.append((path, record))
    if len(found) != 1:
        raise ValueError(f"expected one prior record for {row_id}, found {len(found)}")
    return found[0]


def _attention_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("model does not expose model.layers")
    return [getattr(layer, "self_attn", None) for layer in layers]


def capture_queries(model, engine, boundary, torch, modeling_qwen2):
    """Probe the uncompressed full cache and capture native Q projection plus RoPE."""
    layers = _attention_layers(model)
    captured = [{"q": None, "cos": None, "sin": None} for _ in layers]
    handles = []

    def make_pre_hook(index):
        def hook(module, args, kwargs):
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) >= 2:
                position_embeddings = args[1]
            if position_embeddings is None or len(position_embeddings) != 2:
                raise RuntimeError("Qwen2Attention did not expose position_embeddings")
            cos, sin = position_embeddings
            captured[index]["cos"] = cos.detach().clone()
            captured[index]["sin"] = sin.detach().clone()

        return hook

    def make_q_hook(index):
        def hook(module, args, output):
            captured[index]["q"] = output.detach().clone()

        return hook

    try:
        for index, layer in enumerate(layers):
            if layer is None or not hasattr(layer, "q_proj"):
                raise ValueError(f"attention layer {index} is not Qwen2 compatible")
            head_dim = int(getattr(layer, "head_dim", 0))
            scaling = float(getattr(layer, "scaling", head_dim**-0.5))
            if not math.isclose(scaling, head_dim**-0.5, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Qwen2 module scaling differs from head_dim**-0.5")
            handles.append(
                layer.register_forward_pre_hook(make_pre_hook(index), with_kwargs=True)
            )
            handles.append(layer.q_proj.register_forward_hook(make_q_hook(index)))
        logits, elapsed, clone_elapsed, baseline_memory, peak_memory = (
            engine._probe_logits(model, boundary, None)
        )
    finally:
        for handle in handles:
            handle.remove()
    q_rotated = []
    transfer_started = time.perf_counter()
    for index, item in enumerate(captured):
        if any(item[key] is None for key in ("q", "cos", "sin")):
            raise RuntimeError(
                f"probe did not capture complete query for layer {index}"
            )
        q_linear = item["q"]
        head_dim = int(getattr(layers[index], "head_dim", q_linear.shape[-1]))
        query_heads = int(q_linear.shape[-1]) // head_dim
        q = q_linear.view(
            q_linear.shape[0], q_linear.shape[1], query_heads, head_dim
        ).transpose(1, 2)
        dummy = torch.zeros_like(q)
        q_rot, _ = modeling_qwen2.apply_rotary_pos_emb(
            q, dummy, item["cos"], item["sin"]
        )
        q_rotated.append(q_rot.detach().float().cpu())
    return (
        logits,
        q_rotated,
        {
            "probe_seconds": elapsed,
            "probe_clone_seconds": clone_elapsed,
            "probe_baseline_memory": baseline_memory,
            "probe_peak_memory": peak_memory,
            "capture_transfer_seconds": time.perf_counter() - transfer_started,
        },
    )


def salience(boundary, q_rotated, torch):
    """Return normalized per-layer KV salience and complete selection audit."""
    tensors = []
    probabilities_all = []
    value_norms_all = []
    audits = []
    started = time.perf_counter()
    for layer_index, (layer, q) in enumerate(
        zip(boundary.cache.layers, q_rotated, strict=True)
    ):
        keys = layer.keys.detach().float().cpu()
        values = layer.values.detach().float().cpu()
        _, kv_heads, length, head_dim = keys.shape
        query_heads = int(q.shape[1])
        if query_heads % kv_heads:
            raise ValueError("query and KV heads do not form GQA groups")
        groups = query_heads // kv_heads
        scaling = float(getattr(layer, "scaling", head_dim**-0.5))
        expanded_keys = keys[0].repeat_interleave(groups, dim=0)
        scores = torch.matmul(q, expanded_keys.transpose(-2, -1)) * scaling
        probabilities = torch.softmax(scores, dim=-1)[0].squeeze(1)
        value_norm = values[0].norm(dim=-1)
        expanded_value_norm = value_norm.repeat_interleave(groups, dim=0)
        per_query = probabilities * expanded_value_norm
        per_kv = per_query.view(kv_heads, groups, length).amax(dim=1)
        normalizer = per_kv.sum(dim=-1, keepdim=True)
        if not bool(torch.isfinite(per_kv).all()) or bool((normalizer <= 0).any()):
            raise ValueError("query salience is nonfinite or has zero normalization")
        normalized = per_kv / normalizer
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("normalized query salience is nonfinite")
        tensors.append(normalized)
        probabilities_all.append(probabilities)
        value_norms_all.append(value_norm)
        audits.append(
            {
                "layer": layer_index,
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "scaling": scaling,
                "length": length,
                "normalization_sums": normalized.sum(dim=-1).tolist(),
            }
        )
    return (
        tensors,
        probabilities_all,
        value_norms_all,
        audits,
        time.perf_counter() - started,
    )


def build_branches(engine, boundary, native, salience_tensors, keep_counts, torch):
    proxy, control, audit = [], [], []
    for li, (layer, base_layer, c) in enumerate(
        zip(boundary.cache.layers, native, salience_tensors, strict=True)
    ):
        keys = layer.keys.detach().float().cpu()
        native_priority = (
            (-layer.keys.norm(dim=-1))[0].detach().to(device="cpu", dtype=torch.float32)
        )
        p_layer, c_layer, a_layer = [], [], []
        for hi, base in enumerate(base_layer):
            base = list(base)
            base_set = set(base)
            evicted = [
                j for j in range(int(keys.shape[-2])) if j not in base_set and j >= SINK
            ]
            retained = [j for j in base if j >= SINK]
            if not evicted or not retained:
                p_layer.append(base)
                c_layer.append(base)
                a_layer.append({"layer": li, "kv_head": hi, "selected": False})
                continue
            evicted_sorted = sorted(evicted, key=lambda j: (-float(c[hi, j]), j))
            retained_sorted = sorted(retained, key=lambda j: (-float(c[hi, j]), j))
            candidate, retained_top = evicted_sorted[0], retained_sorted[0]
            selected = float(c[hi, candidate]) > float(c[hi, retained_top])
            if selected:
                victim = min(retained, key=lambda j: (float(native_priority[hi, j]), j))
                control_candidates = [j for j in evicted if j != candidate]
                if not control_candidates:
                    raise RuntimeError(
                        "selected proxy candidate has no distinct control candidate"
                    )
                control_candidate = min(
                    control_candidates, key=lambda j: (float(c[hi, j]), j)
                )
                slot = base.index(victim)
                p = list(base)
                p[slot] = candidate
                co = list(base)
                co[slot] = control_candidate
            else:
                victim = control_candidate = slot = None
                p = list(base)
                co = list(base)
            p_layer.append(p)
            c_layer.append(co)
            a_layer.append(
                {
                    "layer": li,
                    "kv_head": hi,
                    "selected": selected,
                    "candidate": candidate,
                    "retained_top": retained_top,
                    "victim": victim,
                    "control_candidate": control_candidate,
                    "vacated_slot": slot,
                    "candidate_salience": float(c[hi, candidate]),
                    "retained_top_salience": float(c[hi, retained_top]),
                    "gap": max(
                        0.0, float(c[hi, candidate]) - float(c[hi, retained_top])
                    ),
                }
            )
        proxy.append(p_layer)
        control.append(c_layer)
        audit.append(a_layer)
    return proxy, control, audit


def deterministic_mask_input_proof():
    """Exercise one fixed proxy/control replacement, separately from science data."""
    native = [[0, 1, 2, 3, 4]]
    proxy = [[0, 1, 2, 3, 5]]
    control = [[0, 1, 2, 3, 6]]
    return {
        "native": native,
        "proxy": proxy,
        "control": control,
        "same_vacated_slot": proxy[0].index(5) == control[0].index(6) == 4,
        "same_length": len(native[0]) == len(proxy[0]) == len(control[0]),
        "distinct_replacements": proxy[0][-1] != control[0][-1],
    }


def run_prompt(
    args, row, engine, model, tokenizer, torch, modeling_qwen2, output_path, tensor_path
):
    prompt_ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
    boundary, source_cache = runner.build_last_prompt_boundary(
        engine, model, prompt_ids
    )
    boundary_before = engine.cache_fingerprint(boundary.cache)
    source_before = engine.cache_fingerprint(source_cache)
    if not engine.cache_tensors_equal(
        boundary.cache, source_cache
    ) or not engine.cache_storage_independent(boundary.cache, source_cache):
        raise RuntimeError("boundary/source cache equality or disjointness failed")
    native, keep_counts = native_masks(engine, boundary, FRACTION)
    logits_instrumented, q_rotated, probe_meta = capture_queries(
        model, engine, boundary, torch, modeling_qwen2
    )
    logits_plain, plain_elapsed, plain_clone, plain_base, plain_peak = (
        engine._probe_logits(model, boundary, None)
    )
    plain_meta = {
        "probe_seconds": plain_elapsed,
        "probe_clone_seconds": plain_clone,
        "probe_baseline_memory": plain_base,
        "probe_peak_memory": plain_peak,
    }
    if not torch.equal(logits_instrumented, logits_plain):
        raise RuntimeError(
            "instrumented and uninstrumented reference probe logits differ"
        )
    (
        salience_tensors,
        probabilities_all,
        value_norms_all,
        salience_audit,
        feature_seconds,
    ) = salience(boundary, q_rotated, torch)
    selection_started = time.perf_counter()
    proxy, control, swap_audit = build_branches(
        engine, boundary, native, salience_tensors, keep_counts, torch
    )
    selection_seconds = time.perf_counter() - selection_started
    z = sum(item["gap"] for layer in swap_audit for item in layer if item["selected"])
    eos = runner.eos_ids(model, tokenizer)
    cap = row.get("max_new_tokens", 128)
    ref = engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", 0.0),
    )
    noop = engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", 0.0),
    )
    standard = engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", FRACTION),
    )
    standard_replay, standard_meta = custom_branch(
        engine, model, boundary, native, eos, cap, torch
    )
    proxy_cont, proxy_meta = custom_branch(
        engine, model, boundary, proxy, eos, cap, torch
    )
    control_cont, control_meta = custom_branch(
        engine, model, boundary, control, eos, cap, torch
    )
    branches = {}
    for name, cont in (
        ("reference", ref.continuation),
        ("noop", noop.continuation),
        ("standard", standard.continuation),
        ("standard_replay", standard_replay),
        ("proxy_rescue", proxy_cont),
        ("low_salience_control", control_cont),
    ):
        branches[name] = score_branch(tokenizer, cont, row["answers"])
    ref_score = branches["reference"]["score"]["score_fraction"]
    for name, cont in (
        ("reference", ref.continuation),
        ("noop", noop.continuation),
        ("standard", standard.continuation),
        ("standard_replay", standard_replay),
        ("proxy_rescue", proxy_cont),
        ("low_salience_control", control_cont),
    ):
        branches[name]["signed_loss_vs_reference"] = (
            ref_score - branches[name]["score"]["score_fraction"]
        )
        branches[name]["token_ids"] = list(cont.token_ids)
    tensor_payload = {
        "query_rotated": q_rotated,
        "probabilities_per_query_head": probabilities_all,
        "value_norm_per_kv_head": value_norms_all,
        "salience_normalized": salience_tensors,
        "native_masks": torch.tensor(native, dtype=torch.long),
        "proxy_masks": torch.tensor(proxy, dtype=torch.long),
        "control_masks": torch.tensor(control, dtype=torch.long),
        "selection_audit": swap_audit,
    }
    torch.save(tensor_payload, tensor_path)
    checks = {
        "reference_noop_tokens_equal": ref.continuation.token_ids
        == noop.continuation.token_ids,
        "standard_replay_tokens_equal": standard.continuation.token_ids
        == standard_replay.token_ids,
        "standard_replay_clone_controls": standard_meta["clone_equal_before_mask"]
        and standard_meta["clone_disjoint_before_mask"],
        "proxy_clone_controls": proxy_meta["clone_equal_before_mask"]
        and proxy_meta["clone_disjoint_before_mask"],
        "control_clone_controls": control_meta["clone_equal_before_mask"]
        and control_meta["clone_disjoint_before_mask"],
        "instrumented_plain_logits_equal": bool(
            torch.equal(logits_instrumented, logits_plain)
        ),
        "source_boundary_unchanged": engine.cache_fingerprint(boundary.cache)
        == boundary_before
        and engine.cache_fingerprint(source_cache) == source_before,
        "all_lengths_exact": all(
            len(head) == keep_counts[layer_index]
            for masks in (native, proxy, control)
            for layer_index, layer in enumerate(masks)
            for head in layer
        ),
        "all_masks_unique": all(
            len(h) == len(set(h))
            for layers in (native, proxy, control)
            for layer in layers
            for h in layer
        ),
        "proxy_control_victims_equal": all(
            item["victim"] is None
            or (
                item["victim"] in native[li][item["kv_head"]]
                and proxy[li][item["kv_head"]][item["vacated_slot"]]
                == item["candidate"]
                and control[li][item["kv_head"]][item["vacated_slot"]]
                == item["control_candidate"]
            )
            for li, layer in enumerate(swap_audit)
            for item in layer
        ),
        "selected_count_matches_masks": sum(
            item["selected"] for layer in swap_audit for item in layer
        )
        == sum(
            proxy[li][hi] != native[li][hi]
            for li in range(len(native))
            for hi in range(len(native[li]))
        )
        == sum(
            control[li][hi] != native[li][hi]
            for li in range(len(native))
            for hi in range(len(native[li]))
        ),
        "sink_positions_preserved": all(
            {index for index in head if index < SINK}
            == {index for index in native[li][hi] if index < SINK}
            for masks in (proxy, control)
            for li, layer in enumerate(masks)
            for hi, head in enumerate(layer)
        ),
        "salience_finite_normalized": all(
            bool(torch.isfinite(c).all())
            and bool(torch.allclose(c.sum(-1), torch.ones(c.shape[0]), atol=1e-5))
            for c in salience_tensors
        ),
        "deterministic_mask_input_proof": all(
            deterministic_mask_input_proof().values()
        ),
    }
    prior_checks = {"enabled": not args.skip_prior_check}
    if not args.skip_prior_check:
        _, prior = find_prior_record(args.prior_results, row["id"])
        prior_checks.update(
            {
                "reference_token_ids_match": branches["reference"]["token_ids"]
                == prior["reference"]["token_ids"],
                "standard_token_ids_match": branches["standard"]["token_ids"]
                == prior["arms"]["knorm:0.1"]["continuation"]["token_ids"],
            }
        )
    checks.update(prior_checks)
    checks["all_checks_pass"] = all(
        value for key, value in checks.items() if key != "enabled"
    )
    if not checks["all_checks_pass"]:
        raise RuntimeError(
            "query salience diagnostic gate failed: "
            + json.dumps(
                {key: value for key, value in checks.items() if not value},
                sort_keys=True,
            )
        )
    return {
        "schema_version": "query_salience_diagnostic.v1",
        "status": "completed",
        "manifest_row": row,
        "prompt_token_ids": [int(x) for x in prompt_ids[0].tolist()],
        "prompt_length": int(prompt_ids.shape[1]),
        "boundary": {
            "cache_lengths": list(boundary.cache_lengths),
            "cache_bytes": boundary.cache_bytes,
            "boundary_cache_fingerprint": boundary_before,
            "source_cache_fingerprint": source_before,
        },
        "action": {"name": "knorm", "removal_fraction": FRACTION},
        "native_baseline_masks": native,
        "keep_counts": keep_counts,
        "feature": {
            "z": z,
            "tensor_file": tensor_path.name,
            "salience_audit": salience_audit,
        },
        "selection_audit": swap_audit,
        "branches": branches,
        "probe": {
            "instrumented": probe_meta,
            "plain": plain_meta,
            "logit_exact": True,
        },
        "feature_cost": {
            "salience_seconds": feature_seconds,
            "selection_seconds": selection_seconds,
            "salience_and_selection_seconds": feature_seconds + selection_seconds,
        },
        "deterministic_mask_input_proof": deterministic_mask_input_proof(),
        "branch_meta": {
            "standard_replay": standard_meta,
            "proxy_rescue": proxy_meta,
            "low_salience_control": control_meta,
        },
        "prior_checks": prior_checks,
        "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument(
        "--prior-results",
        nargs="+",
        default=["results/ea-dev-v1-first", "results/ea-dev-v1-rest"],
    )
    parser.add_argument("--skip-prior-check", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    args = parser.parse_args(argv)
    output = runner.ensure_output_dir(args.output)
    engine, engine_root, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    by_id = {r["id"]: r for r in rows}
    requested_ids = tuple(args.ids) if args.ids else TARGET_IDS
    if args.ids is None and any(i not in by_id for i in requested_ids):
        requested_ids = tuple(row["id"] for row in rows)
    selected = [by_id[i] for i in requested_ids]
    import torch
    import transformers
    from transformers.models.qwen2 import modeling_qwen2

    torch.manual_seed(SEED)
    device = runner.choose_device(args.device, torch)
    dtype = runner.choose_dtype(args.dtype, device, torch)
    model, tokenizer = runner.load_model_and_tokenizer(
        args.model, device, dtype, transformers
    )
    run_record = {
        "schema_version": "query_salience_diagnostic.v1",
        "status": "running",
        "ids": list(requested_ids),
        "manifest_path": str(manifest_path),
        "manifest_sha256": runner.sha256_bytes(manifest_path.read_bytes()),
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "diagnostic_source_sha256": sha256_file(Path(__file__).resolve()),
        "official_evaluate_sha256": sha256_file(OFFICIAL_EVAL),
        "official_constants_sha256": sha256_file(OFFICIAL_CONSTANTS),
        "model": runner.model_runtime_identity(
            engine, model, args.model, transformers, torch
        ),
        "device": str(device),
        "dtype": str(dtype),
        "prompts": [],
        "failures": [],
    }
    runner.write_json(output / "run.json", run_record)
    for index, row in enumerate(selected):
        path = output / runner.safe_filename(index, row["id"])
        tensor_path = output / f"{path.stem}.pt"
        try:
            record = run_prompt(
                args,
                row,
                engine,
                model,
                tokenizer,
                torch,
                modeling_qwen2,
                path,
                tensor_path,
            )
            runner.write_json(path, record)
            status = "completed"
        except Exception as error:
            status = "failed"
            record = {
                "schema_version": "query_salience_diagnostic.v1",
                "status": "failed",
                "manifest_row": row,
                "failure": {
                    "type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
            runner.write_json(path, record)
            run_record["failures"].append({"id": row["id"], **record["failure"]})
        run_record["prompts"].append(
            {"id": row["id"], "path": path.name, "status": status}
        )
        runner.write_json(output / "run.json", run_record)
    run_record["status"] = "completed" if not run_record["failures"] else "failed"
    runner.write_json(output / "run.json", run_record)
    return 0 if run_record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
