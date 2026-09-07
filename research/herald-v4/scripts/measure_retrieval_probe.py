#!/usr/bin/env python3
"""Measure one forced-text retrieval probe at the frozen B0 boundary."""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pair_pilot as runner  # noqa: E402
from diagnose_needle_rescue import native_masks  # noqa: E402

TARGET_IDS = (
    "ruler-ea-dev-v1-niah_single_2-000",
    "ruler-ea-dev-v1-niah_single_2-004",
    "ruler-ea-dev-v1-niah_single_2-011",
    "ruler-ea-dev-v1-niah_single_2-001",
)
CUE = (
    "Before answering, identify the information in the prompt that is needed. "
    "The needed information is:"
)
FRACTION = 0.10
SEED = 0


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_prior(directories, row_id):
    found = []
    for directory in directories:
        for path in Path(directory).expanduser().resolve().glob("*.json"):
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


def layers(model):
    values = getattr(getattr(model, "model", None), "layers", None)
    if values is None:
        raise ValueError("model does not expose model.layers")
    return [layer.self_attn for layer in values]


def probe(
    model, engine, boundary, torch, modeling_qwen2, input_ids, cache, capture=True
):
    modules = layers(model)
    captured = [{"q": None, "cos": None, "sin": None} for _ in modules]
    handles = []

    def pre(index):
        def hook(module, args, kwargs):
            embeddings = kwargs.get("position_embeddings")
            if embeddings is None and len(args) >= 2:
                embeddings = args[1]
            if embeddings is None:
                raise RuntimeError("missing native Qwen2 position embeddings")
            captured[index]["cos"] = embeddings[0].detach().clone()
            captured[index]["sin"] = embeddings[1].detach().clone()

        return hook

    def qhook(index):
        def hook(module, args, output):
            captured[index]["q"] = output.detach().clone()

        return hook

    try:
        if capture:
            for index, module in enumerate(modules):
                handles.append(
                    module.register_forward_pre_hook(pre(index), with_kwargs=True)
                )
                handles.append(module.q_proj.register_forward_hook(qhook(index)))
        device = engine._model_device(model)
        n = int(input_ids.shape[1])
        positions = torch.arange(
            boundary.logical_position,
            boundary.logical_position + n,
            device=device,
            dtype=torch.long,
        )
        mask = torch.ones(
            (1, int(cache.get_seq_length()) + n), device=device, dtype=torch.long
        )
        memory_baseline = engine._begin_peak_memory_measurement(device)
        engine._sync_device(device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=mask,
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        engine._sync_device(device)
        elapsed = time.perf_counter() - started
    finally:
        for handle in handles:
            handle.remove()
    if not capture:
        return output.logits[:, -1].detach().clone(), [], {"probe_seconds": elapsed}
    q_rotated = []
    transfer_started = time.perf_counter()
    for index, item in enumerate(captured):
        if any(item[key] is None for key in ("q", "cos", "sin")):
            raise RuntimeError(f"probe did not capture layer {index}")
        module = modules[index]
        head_dim = int(getattr(module, "head_dim", 0))
        query_heads = int(item["q"].shape[-1]) // head_dim
        q = item["q"].view(1, -1, query_heads, head_dim).transpose(1, 2)
        dummy = torch.zeros_like(q)
        rotated, _ = modeling_qwen2.apply_rotary_pos_emb(
            q, dummy, item["cos"], item["sin"]
        )
        q_rotated.append(rotated[:, :, -1].detach().float().cpu())
    return (
        output.logits[:, -1].detach().clone(),
        q_rotated,
        {
            "probe_seconds": elapsed,
            "q_transfer_and_rope_seconds": time.perf_counter() - transfer_started,
            "q_transferred_bytes": sum(q.numel() * q.element_size() for q in q_rotated),
            "probe_memory_baseline_bytes": memory_baseline,
            "probe_peak_allocated_bytes": engine._finish_peak_memory_measurement(device),
        },
    )


def features(boundary, q_rotated, torch):
    probabilities_all, value_norm_all, salience_all, masses = [], [], [], []
    transfer_bytes = 0
    started = time.perf_counter()
    for layer, q in zip(boundary.cache.layers, q_rotated, strict=True):
        keys = layer.keys.detach().float().cpu()[0]
        values = layer.values.detach().float().cpu()[0]
        transfer_bytes += int(layer.keys.numel() * 4 + layer.values.numel() * 4)
        kv_heads, length, head_dim = keys.shape
        query_heads = int(q.shape[1])
        groups = query_heads // kv_heads
        if query_heads % kv_heads:
            raise ValueError("invalid GQA grouping")
        scale = float(getattr(layer, "scaling", head_dim**-0.5))
        expanded_keys = keys.repeat_interleave(groups, dim=0)
        probs = torch.softmax(
            torch.matmul(q[0].unsqueeze(1), expanded_keys.transpose(-2, -1)).squeeze(1)
            * scale,
            dim=-1,
        )
        value_norm = values.norm(dim=-1)
        expanded_norm = value_norm.repeat_interleave(groups, dim=0)
        raw = probs * expanded_norm
        salience = raw.view(kv_heads, groups, length).amax(dim=1)
        normalizer = salience.sum(-1, keepdim=True)
        if not bool(torch.isfinite(salience).all()) or bool((normalizer <= 0).any()):
            raise ValueError("nonfinite or zero salience normalization")
        salience = salience / normalizer
        probabilities_all.append(probs)
        value_norm_all.append(value_norm)
        salience_all.append(salience)
        masses.append((salience, scale))
    return (
        probabilities_all,
        value_norm_all,
        salience_all,
        masses,
        time.perf_counter() - started,
        transfer_bytes,
    )


def scalar_z(salience_all, native, torch):
    started = time.perf_counter()
    values = []
    for layer_index, salience in enumerate(salience_all):
        per_head = []
        for head_index, row in enumerate(salience):
            evicted = set(range(row.shape[-1])) - set(native[layer_index][head_index])
            per_head.append(float(row[list(evicted)].sum()) if evicted else 0.0)
        values.append(per_head)
    flat = [value for layer in values for value in layer]
    return float(sum(flat) / len(flat)), values, time.perf_counter() - started


def run_prompt(args, row, engine, model, tokenizer, torch, modeling_qwen2, tensor_path):
    prompt_ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
    cue_ids = tokenizer(CUE, add_special_tokens=False, return_tensors="pt")["input_ids"]
    boundary, source_cache = runner.build_last_prompt_boundary(
        engine, model, prompt_ids
    )
    before = engine.cache_fingerprint(boundary.cache)
    source_before = engine.cache_fingerprint(source_cache)
    native, keep_counts = native_masks(engine, boundary, FRACTION)
    prior_checks = {"enabled": not args.skip_prior_check}
    prior = None
    if not args.skip_prior_check:
        _, prior = find_prior(args.prior_results, row["id"])
        prior_compression = prior["arms"]["knorm:0.1"]["compression"]
        prior_clone = engine.clone_cache(boundary.cache)
        prior_compact = runner.compact_compression(
            engine, engine.compress_knorm(prior_clone, FRACTION), boundary
        )
        prior_checks["native_mask_hash_match_prior"] = (
            prior_compact["kept_index_hash"] == prior_compression["kept_index_hash"]
        )
        prior_checks["native_lengths_match_prior"] = (
            prior_compact["after_lengths"] == prior_compression["after_lengths"]
        )
    unforced_cache = engine.clone_cache(boundary.cache)
    unforced_logits, unforced_q, unforced_probe = probe(
        model,
        engine,
        boundary,
        torch,
        modeling_qwen2,
        torch.tensor([[boundary.pending_token_id]], dtype=prompt_ids.dtype),
        unforced_cache,
    )
    plain_logits, plain_elapsed, _, _, _ = engine._probe_logits(model, boundary, None)
    if not torch.equal(unforced_logits, plain_logits):
        raise RuntimeError("unforced instrumented/plain logits differ")
    if prior is not None:
        prior_checks["reference_first_token_match"] = int(
            unforced_logits.argmax(dim=-1).item()
        ) == int(prior["reference"]["token_ids"][0])
    forced_cache = engine.clone_cache(boundary.cache)
    forced_ids = torch.cat(
        (torch.tensor([[boundary.pending_token_id]], dtype=prompt_ids.dtype), cue_ids),
        dim=1,
    )
    forced_logits, forced_q, forced_probe = probe(
        model, engine, boundary, torch, modeling_qwen2, forced_ids, forced_cache
    )
    plain_forced_cache = engine.clone_cache(boundary.cache)
    plain_forced_logits, _, _ = probe(
        model,
        engine,
        boundary,
        torch,
        modeling_qwen2,
        forced_ids,
        plain_forced_cache,
        capture=False,
    )
    if not torch.equal(forced_logits, plain_forced_logits):
        raise RuntimeError("forced instrumented/plain logits differ")
    unforced = features(boundary, unforced_q, torch)
    forced = features(boundary, forced_q, torch)
    unforced_z, unforced_mass, unforced_reduction_seconds = scalar_z(
        unforced[2], native, torch
    )
    probe_z, probe_mass, probe_reduction_seconds = scalar_z(forced[2], native, torch)
    torch.save(
        {
            "cue_token_ids": cue_ids[0].tolist(),
            "unforced_q_rotated": unforced_q,
            "probe_q_rotated": forced_q,
            "unforced_probabilities": unforced[0],
            "probe_probabilities": forced[0],
            "unforced_value_norm": unforced[1],
            "probe_value_norm": forced[1],
            "unforced_salience": unforced[2],
            "probe_salience": forced[2],
            "native_masks": torch.tensor(native, dtype=torch.long),
            "unforced_mass_per_head": unforced_mass,
            "probe_mass_per_head": probe_mass,
        },
        tensor_path,
    )
    checks = {
        "source_boundary_unchanged": engine.cache_fingerprint(boundary.cache) == before
        and engine.cache_fingerprint(source_cache) == source_before,
        "source_boundary_equal": engine.cache_tensors_equal(
            boundary.cache, source_cache
        ),
        "source_boundary_disjoint": engine.cache_storage_independent(
            boundary.cache, source_cache
        ),
        "unforced_instrumented_plain_exact": bool(
            torch.equal(unforced_logits, plain_logits)
        ),
        "forced_instrumented_plain_exact": bool(
            torch.equal(forced_logits, plain_forced_logits)
        ),
        "unforced_clone_disjoint": engine.cache_storage_independent(
            boundary.cache, unforced_cache
        ),
        "forced_clone_disjoint": engine.cache_storage_independent(
            boundary.cache, forced_cache
        ),
        "native_lengths_exact": all(
            len(head) == keep_counts[li]
            for li, layer in enumerate(native)
            for head in layer
        ),
        "finite_normalized": all(
            bool(torch.isfinite(x).all())
            and bool(torch.allclose(x.sum(-1), torch.ones(x.shape[0]), atol=1e-5))
            for x in unforced[2] + forced[2]
        ),
    }
    checks.update(prior_checks)
    checks["all_checks_pass"] = all(
        value for key, value in checks.items() if key != "enabled"
    )
    if not checks["all_checks_pass"]:
        raise RuntimeError(
            "retrieval probe gate failed: "
            + json.dumps({k: v for k, v in checks.items() if not v})
        )
    return {
        "schema_version": "retrieval_probe.v1",
        "status": "completed",
        "manifest_row": row,
        "prompt_length": int(prompt_ids.shape[1]),
        "cue": {
            "text": CUE,
            "token_ids": cue_ids[0].tolist(),
            "count": int(cue_ids.shape[1]),
        },
        "boundary": {
            "cache_lengths": list(boundary.cache_lengths),
            "cache_bytes": boundary.cache_bytes,
            "fingerprint": before,
        },
        "native_masks": native,
        "keep_counts": keep_counts,
        "features": {
            "unforced": {"z": unforced_z, "mass_per_head": unforced_mass},
            "probe": {"z": probe_z, "mass_per_head": probe_mass},
            "tensor_file": tensor_path.name,
        },
        "cost": {
            "unforced_probe": unforced_probe,
            "forced_probe": forced_probe,
            "unforced_feature_seconds": unforced[4],
            "probe_feature_seconds": forced[4],
            "unforced_reduction_seconds": unforced_reduction_seconds,
            "probe_reduction_seconds": probe_reduction_seconds,
            "unforced_feature_transfer_bytes": unforced[5],
            "probe_feature_transfer_bytes": forced[5],
            "plain_unforced_probe_seconds": plain_elapsed,
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
    engine, _, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    by_id = {row["id"]: row for row in rows}
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
    run = {
        "schema_version": "retrieval_probe.v1",
        "status": "running",
        "ids": list(requested_ids),
        "manifest_path": str(manifest_path),
        "manifest_sha256": runner.sha256_bytes(manifest_path.read_bytes()),
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "diagnostic_source_sha256": sha256_file(Path(__file__).resolve()),
        "model": runner.model_runtime_identity(
            engine, model, args.model, transformers, torch
        ),
        "prompts": [],
        "failures": [],
    }
    runner.write_json(output / "run.json", run)
    for index, row in enumerate(selected):
        path = output / runner.safe_filename(index, row["id"])
        tensor_path = output / f"{path.stem}.pt"
        try:
            runner.write_json(
                path,
                run_prompt(
                    args,
                    row,
                    engine,
                    model,
                    tokenizer,
                    torch,
                    modeling_qwen2,
                    tensor_path,
                ),
            )
            status = "completed"
        except Exception as error:
            status = "failed"
            runner.write_json(
                path,
                {
                    "schema_version": "retrieval_probe.v1",
                    "status": status,
                    "manifest_row": row,
                    "failure": {
                        "type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
            run["failures"].append({"id": row["id"], "error": str(error)})
        run["prompts"].append({"id": row["id"], "path": path.name, "status": status})
        runner.write_json(output / "run.json", run)
    run["status"] = "completed" if not run["failures"] else "failed"
    runner.write_json(output / "run.json", run)
    return 0 if run["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
