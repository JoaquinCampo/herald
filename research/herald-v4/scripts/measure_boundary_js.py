#!/usr/bin/env python3
"""Measure first-pending-token full-vocabulary JS under Knorm actions."""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pair_pilot as pilot  # noqa: E402

ACTIONS = (0.05, 0.10, 0.20)
SEED = 0


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_for(prior_dirs, row_id):
    matches = []
    for directory in prior_dirs:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"prior result directory is missing: {root}"
            )
        for path in root.glob("*.json"):
            if path.name == "run.json":
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if record.get("manifest_row", {}).get("id") == row_id:
                matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"expected one prior record for {row_id}, found {len(matches)}"
        )
    return matches[0]


def compression_summary(engine, boundary, fraction):
    cache = engine.clone_cache(boundary.cache)
    compression = engine.compress_knorm(cache, fraction)
    compact = pilot.compact_compression(engine, compression, boundary)
    return compression, compact


def timed_probe(engine, model, boundary, action):
    engine._sync_device(boundary.prompt_ids.device)
    start = time.perf_counter()
    result = engine._probe_logits(model, boundary, action)
    engine._sync_device(boundary.prompt_ids.device)
    return result, time.perf_counter() - start


def normalized(logits):
    probabilities = torch.softmax(logits.float(), dim=-1)
    finite = bool(torch.isfinite(probabilities).all().item())
    total = float(probabilities.sum().item())
    return probabilities, finite and abs(total - 1.0) <= 1e-5, total


def raw_tensor_record(actions):
    record = {}
    for action_id, values in actions.items():
        record[action_id] = {
            "logits": values["logits"].detach().cpu().float(),
            "probabilities": values["probabilities"].detach().cpu().float(),
        }
    return record


def run_prompt(args, row, engine, model, tokenizer, output_path):
    partial = {
        "schema_version": "boundary_js_measurement.v1",
        "status": "running",
        "manifest_row": row,
        "stages": [],
    }
    pilot.write_json(output_path, partial)
    prompt_ids = pilot.tokenize_chat_prompt(tokenizer, row["prompt"])
    token_identity = pilot.token_identity_checks(
        tokenizer, row, prompt_ids, row["prompt"]
    )
    boundary, source_cache = pilot.build_last_prompt_boundary(
        engine, model, prompt_ids
    )
    boundary_before = engine.cache_fingerprint(boundary.cache)
    source_before = engine.cache_fingerprint(source_cache)
    if not engine.cache_tensors_equal(boundary.cache, source_cache):
        raise AssertionError("boundary and source cache tensors differ")
    if not engine.cache_storage_independent(boundary.cache, source_cache):
        raise AssertionError("boundary and source cache storage is aliased")
    partial.update(
        {
            "prompt_token_ids": [
                int(value) for value in prompt_ids[0].tolist()
            ],
            "token_identity": token_identity,
            "boundary": boundary.to_dict(),
            "stages": [
                {"name": "boundary", "cache_fingerprint": boundary_before}
            ],
        }
    )
    pilot.write_json(output_path, partial)

    (reference_logits, ref_forward, ref_clone, _, _), ref_total = timed_probe(
        engine,
        model, boundary, None
    )
    (noop_logits, noop_forward, noop_clone, _, _), noop_total = timed_probe(
        engine,
        model, boundary, None
    )
    reference_probabilities, ref_probs_ok, ref_sum = normalized(
        reference_logits
    )
    noop_probabilities, noop_probs_ok, noop_sum = normalized(noop_logits)
    partial["stages"].append(
        {
            "name": "reference_noop_probe",
            "reference_forward_seconds": ref_forward,
            "noop_forward_seconds": noop_forward,
            "reference_clone_seconds": ref_clone,
            "noop_clone_seconds": noop_clone,
        }
    )
    pilot.write_json(output_path, partial)

    prior = (
        None
        if args.skip_prior_check
        else prior_for(args.prior_results, row["id"])
    )
    actions = {}
    raw_actions = {}
    action_checks = {}
    for fraction in ACTIONS:
        action = engine.ActionSpec("knorm", fraction)
        (logits, forward_seconds, clone_seconds, _, _), probe_total = timed_probe(
            engine,
            model, boundary, action
        )
        probe = engine.full_vocabulary_js(reference_logits, logits, action)
        probabilities, probs_ok, prob_sum = normalized(logits)
        compression, compact = compression_summary(engine, boundary, fraction)
        action_id = action.action_id
        raw_actions[action_id] = {
            "logits": logits,
            "probabilities": probabilities,
        }
        actions[action_id] = {
            "action": action.to_dict(),
            "probe": probe.to_dict(),
            "logits_shape": list(logits.shape),
            "probability_sum": prob_sum,
            "probe_total_seconds_synchronized": probe_total,
            "forward_seconds_synchronized": forward_seconds,
            "clone_seconds_synchronized": clone_seconds,
            "compression": compact,
            "kept_index_hash": compact["kept_index_hash"],
            "kept_index_counts_per_layer_head": compact[
                "kept_index_counts_per_layer_head"
            ],
            "after_lengths": compact["after_lengths"],
        }
        action_checks[action_id] = {
            "probabilities_finite_normalized": probs_ok and probe.finite,
            "physical_effect_exact": compact["physical_effect_exact"],
            "lengths_exact": compact["after_lengths"]
            == compact["expected_after_lengths"],
            "source_unchanged": engine.cache_fingerprint(source_cache)
            == source_before,
            "boundary_unchanged": engine.cache_fingerprint(boundary.cache)
            == boundary_before,
        }
        if prior is not None:
            old = prior["arms"][action_id]
            action_checks[action_id].update(
                {
                    "first_argmax_matches_prior": int(logits.argmax().item())
                    == old["continuation"]["token_ids"][0],
                    "mask_hash_matches_prior": compact["kept_index_hash"]
                    == old["compression"]["kept_index_hash"],
                    "lengths_match_prior": compact["after_lengths"]
                    == old["compression"]["after_lengths"],
                }
            )
        partial["actions"] = actions
        partial["action_checks"] = action_checks
        pilot.write_json(output_path, partial)

    raw_path = output_path.with_name(output_path.stem + "-tensors.pt")
    torch.save(
        {
            "reference_logits": reference_logits.detach().cpu().float(),
            "reference_probabilities": reference_probabilities.detach().cpu(),
            "noop_logits": noop_logits.detach().cpu().float(),
            "noop_probabilities": noop_probabilities.detach().cpu(),
            "actions": raw_tensor_record(raw_actions),
            "kept_indices": {
                action_id: compression_summary(engine, boundary, fraction)[
                    0
                ].kept_indices
                for action_id, fraction in (
                    (engine.ActionSpec("knorm", value).action_id, value)
                    for value in ACTIONS
                )
            },
        },
        raw_path,
    )
    checks = {
        "reference_noop_logits_exact": torch.equal(
            reference_logits, noop_logits
        ),
        "reference_probability_finite_normalized": ref_probs_ok,
        "noop_probability_finite_normalized": noop_probs_ok,
        "boundary_source_equal": engine.cache_tensors_equal(
            boundary.cache, source_cache
        ),
        "boundary_source_disjoint": engine.cache_storage_independent(
            boundary.cache, source_cache
        ),
        "source_unchanged": engine.cache_fingerprint(source_cache)
        == source_before,
        "boundary_unchanged": engine.cache_fingerprint(boundary.cache)
        == boundary_before,
        "reference_sum_recorded": abs(ref_sum - 1.0) <= 1e-5,
        "noop_sum_recorded": abs(noop_sum - 1.0) <= 1e-5,
        **{
            f"{action_id}_{key}": value
            for action_id, values in action_checks.items()
            for key, value in values.items()
        },
    }
    if prior is not None:
        checks.update(
            {
                "reference_argmax_matches_prior": int(
                    reference_logits.argmax().item()
                )
                == prior["reference"]["token_ids"][0],
                "noop_argmax_matches_prior": int(noop_logits.argmax().item())
                == prior["reference"]["token_ids"][0],
            }
        )
    checks["all_checks_pass"] = all(checks.values())
    if not checks["all_checks_pass"]:
        raise RuntimeError("boundary JS diagnostic controls failed")
    record = {
        "schema_version": "boundary_js_measurement.v1",
        "status": "completed",
        "manifest_row": row,
        "prompt_token_ids": [int(value) for value in prompt_ids[0].tolist()],
        "token_identity": token_identity,
        "boundary": boundary.to_dict(),
        "reference": {
            "argmax": int(reference_logits.argmax().item()),
            "probability_sum": ref_sum,
            "probe_total_seconds_synchronized": ref_total,
            "forward_seconds_synchronized": ref_forward,
            "clone_seconds_synchronized": ref_clone,
        },
        "noop": {
            "argmax": int(noop_logits.argmax().item()),
            "probability_sum": noop_sum,
            "probe_total_seconds_synchronized": noop_total,
            "forward_seconds_synchronized": noop_forward,
            "clone_seconds_synchronized": noop_clone,
        },
        "actions": actions,
        "raw_tensors_path": raw_path.name,
        "prior_checks_enabled": prior is not None,
        "checks": checks,
    }
    pilot.write_json(output_path, record)
    return record


def run(args):
    output = pilot.ensure_output_dir(args.output)
    engine, engine_root, engine_path = pilot.load_engine(args.engine_root)
    manifest_path, rows = pilot.load_manifest(args.manifest)
    if args.ids:
        selected = set(args.ids)
        rows = [row for row in rows if row["id"] in selected]
    if not rows:
        raise ValueError("no requested manifest rows found")
    import transformers

    torch.manual_seed(SEED)
    device = pilot.choose_device(args.device, torch)
    dtype = pilot.choose_dtype(args.dtype, device, torch)
    model, tokenizer = pilot.load_model_and_tokenizer(
        args.model, device, dtype, transformers
    )
    run_record = {
        "schema_version": "boundary_js_measurement.v1",
        "status": "running",
        "actions": list(ACTIONS),
        "manifest_path": str(manifest_path),
        "manifest_sha256": pilot.sha256_bytes(manifest_path.read_bytes()),
        "engine_path": str(engine_path),
        "engine_sha256": file_sha256(engine_path),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "model": pilot.model_runtime_identity(
            engine, model, args.model, transformers, torch
        ),
        "device": str(device),
        "dtype": str(dtype),
        "prompts": [],
        "failures": [],
    }
    pilot.write_json(output / "run.json", run_record)
    for index, row in enumerate(rows):
        path = output / pilot.safe_filename(index, row["id"])
        try:
            record = run_prompt(args, row, engine, model, tokenizer, path)
            status = "completed"
        except Exception as error:
            record = {}
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
            failure = {
                "type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            record.update({"status": "failed", "failure": failure})
            pilot.write_json(path, record)
            run_record["failures"].append({"id": row["id"], **failure})
            status = "failed"
        run_record["prompts"].append(
            {"id": row["id"], "path": path.name, "status": status}
        )
        pilot.write_json(output / "run.json", run_record)
    run_record["status"] = (
        "completed" if not run_record["failures"] else "failed"
    )
    pilot.write_json(output / "run.json", run_record)
    return 0 if run_record["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument(
        "--prior-results",
        nargs="+",
        default=["results/ea-dev-v1-first", "results/ea-dev-v1-rest"],
    )
    parser.add_argument("--skip-prior-check", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        output = Path(args.output).expanduser().resolve()
        if output.is_dir():
            pilot.write_json(
                output / "failure.json",
                {
                    "status": "failed",
                    "type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        print(
            f"boundary JS failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
