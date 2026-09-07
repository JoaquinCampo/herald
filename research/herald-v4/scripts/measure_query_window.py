#!/usr/bin/env python3
"""Measure full-prompt and final-32 ExpectedAttentionPress statistics."""

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import traceback
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import measure_ea as existing  # noqa: E402
import run_pair_pilot as pilot  # noqa: E402


TARGET_IDS = (
    "ruler-ea-dev-v1-niah_single_2-000",
    "ruler-ea-dev-v1-niah_single_2-004",
    "ruler-ea-dev-v1-niah_single_2-011",
    "ruler-ea-dev-v1-niah_single_2-001",
)
ACTION_FRACTION = 0.10
SINK = 4
FUTURE = 128
WINDOW = 32
CAP = 128
SEED = 0


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def press_config(ExpectedAttentionPress):
    return ExpectedAttentionPress(
        compression_ratio=0.0,
        n_sink=SINK,
        n_future_positions=FUTURE,
        use_covariance=False,
        use_vnorm=True,
        epsilon=0.0,
    )


def make_tail_press(ExpectedAttentionPress, get_prerope_query_states):
    class TailExpectedAttentionPress(ExpectedAttentionPress):
        def get_query_statistics(self, module, hidden_states):
            original_q_len = int(hidden_states.shape[1])
            if original_q_len < WINDOW + SINK:
                raise ValueError(
                    f"tail window needs at least {WINDOW + SINK} prefill rows"
                )
            tail = hidden_states[:, -WINDOW:]
            query_states = get_prerope_query_states(module, tail)
            mu = query_states.mean(dim=2, keepdim=True).squeeze(2)
            self.last_original_q_len = original_q_len
            return self.apply_avg_rope(module, mu, None, original_q_len)

    return TailExpectedAttentionPress


def locate_span(tokenizer, prompt_ids, prompt, answer):
    pattern = re.compile(
        r"One of the special magic numbers for [^:\n]+ is: "
        + re.escape(answer)
        + r"\."
    )
    matches = list(pattern.finditer(prompt))
    if len(matches) != 1 or prompt.count(answer) != 1:
        raise ValueError("needle sentence and answer must occur exactly once")
    sentence = matches[0].group(0)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    start = rendered.find(sentence)
    if start < 0 or rendered.count(sentence) != 1:
        raise ValueError("needle sentence is not unique in rendered chat")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    native_ids = [int(value) for value in prompt_ids[0].tolist()]
    fast_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    if fast_ids != native_ids:
        raise ValueError("fast rendered tokenization differs from native chat")
    end = start + len(sentence)
    positions = [
        index
        for index, (offset_start, offset_end) in enumerate(
            encoded["offset_mapping"][0].tolist()
        )
        if offset_end > start and offset_start < end
    ]
    prefix_length = len(native_ids) - 1
    if not positions or any(index >= prefix_length for index in positions):
        raise ValueError("needle span must be nonempty and entirely in prefix")
    return {
        "answer": answer,
        "sentence": sentence,
        "token_positions": positions,
        "token_ids": [native_ids[index] for index in positions],
        "prefix_length": prefix_length,
        "chat_tokenization_ids_equal": True,
    }


def action_rows(engine, scores, knorm_scores, keys, values, layer, span):
    rows = existing.action_features(
        engine,
        scores,
        knorm_scores,
        keys,
        values,
        layer,
        ACTION_FRACTION,
    )
    length = int(keys.shape[-2])
    keep = int(length * (1.0 - ACTION_FRACTION))
    knorm = keys.norm(dim=-1)
    retained = (-knorm).topk(keep, dim=-1).indices
    mask_digest = hashlib.sha256()
    for head in retained[0].tolist():
        for index in head:
            mask_digest.update(int(index).to_bytes(8, "little"))
    mask_hash = mask_digest.hexdigest()
    removed = torch.ones_like(knorm, dtype=torch.bool)
    removed.scatter_(2, retained, False)
    span_positions = [index for index in span["token_positions"] if index < length]
    if len(span_positions) != len(span["token_positions"]):
        raise ValueError("needle span exceeds the measured cache")
    span_positions = [index for index in span_positions if index >= SINK]
    score_values = scores.float()
    for row in rows:
        head = row["kv_head"]
        score_non_sink = score_values[0, head, SINK:]
        total = float(score_non_sink.sum().item())
        span_mass = float(score_values[0, head, span_positions].sum().item())
        removed_span = removed[0, head, span_positions]
        removed_mass = float(
            score_values[0, head, span_positions][removed_span].sum().item()
        )
        row["oracle_span_mass_non_sink"] = span_mass / total
        row["oracle_removed_span_mass_non_sink"] = removed_mass / total
        row["oracle_span_fraction_non_sink"] = len(span_positions) / max(
            length - SINK, 1
        )
        row["knorm_mask_hash"] = mask_hash
    return rows


def instrument_prefill(
    engine, model, prompt_ids, press, device, span, ExpectedAttentionPress
):
    rows = []
    metadata = {"window_original_q_lengths": [], "full_stats_identity": True}
    hooks = []
    prior_rotary = []
    from kvpress import KnormPress

    knorm_press = KnormPress(compression_ratio=0.0)
    layers = model.model.layers

    def make_hook(layer_index):
        def hook(module, args, kwargs, output):
            hidden_states = kwargs["hidden_states"]
            cache = kwargs["past_key_values"]
            layer_cache = cache.layers[module.layer_idx]
            keys = layer_cache.keys
            values = layer_cache.values
            scores = press.score(module, hidden_states, keys, values, None, kwargs)
            knorm_scores = knorm_press.score(
                module, hidden_states, keys, values, None, kwargs
            )
            rows.extend(
                action_rows(
                    engine,
                    scores,
                    knorm_scores,
                    keys,
                    values,
                    layer_index,
                    span,
                )
            )
            if hasattr(press, "last_original_q_len"):
                metadata["window_original_q_lengths"].append(
                    press.last_original_q_len
                )
            if not metadata["window_original_q_lengths"]:
                baseline = press_config(ExpectedAttentionPress)
                baseline.post_init_from_model(model)
                baseline_stats = baseline.get_query_statistics(
                    module, hidden_states
                )
                current_stats = press.get_query_statistics(
                    module, hidden_states
                )

                def equal_stat(left, right):
                    if left is None or right is None:
                        return left is right
                    return torch.equal(left, right)

                metadata["full_stats_identity"] = all(
                    equal_stat(left, right)
                    for left, right in zip(current_stats, baseline_stats, strict=True)
                )

        return hook

    press.post_init_from_model(model)
    try:
        rotary = model.model.rotary_emb
        for layer_index, layer in enumerate(layers):
            module = layer.self_attn
            had_rotary = hasattr(module, "rotary_emb")
            old_rotary = getattr(module, "rotary_emb", None)
            prior_rotary.append((module, had_rotary, old_rotary))
            if not had_rotary or old_rotary is not rotary:
                module.rotary_emb = rotary
            hooks.append(
                module.register_forward_hook(
                    make_hook(layer_index), with_kwargs=True
                )
            )
        result, wall_seconds = pilot.timed_call(
            engine,
            device,
            lambda: pilot.build_last_prompt_boundary(engine, model, prompt_ids),
        )
    finally:
        for handle in hooks:
            handle.remove()
        for module, had_rotary, old_rotary in prior_rotary:
            if had_rotary:
                module.rotary_emb = old_rotary
            else:
                delattr(module, "rotary_emb")
    boundary, source_cache = result
    return boundary, source_cache, rows, metadata, wall_seconds


def find_prior_record(prior_dirs, row_id):
    matches = []
    for directory in prior_dirs:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"prior result directory is missing: {root}")
        for path in root.glob("*.json"):
            if path.name == "run.json":
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if record.get("manifest_row", {}).get("id") == row_id:
                matches.append((path, record))
    if len(matches) != 1:
        raise ValueError(f"expected one prior record for {row_id}, found {len(matches)}")
    return matches[0]


def continuation_arm(engine, model, boundary, eos, fraction, cap):
    return engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", fraction),
    )


def summarize_rows(rows):
    return existing.summarize(rows)


def run_prompt(args, row, engine, model, tokenizer, device, ExpectedAttentionPress):
    prompt_ids = pilot.tokenize_chat_prompt(tokenizer, row["prompt"])
    span = locate_span(tokenizer, prompt_ids, row["prompt"], row["answers"][0])
    if not args.skip_prior_check:
        _, prior = find_prior_record(args.prior_results, row["id"])
    else:
        prior = None

    plain_boundary, plain_source = pilot.build_last_prompt_boundary(
        engine, model, prompt_ids
    )
    full_press = press_config(ExpectedAttentionPress)
    TailPress = make_tail_press(
        ExpectedAttentionPress,
        __import__(
            "kvpress.presses.expected_attention_press",
            fromlist=["get_prerope_query_states"],
        ).get_prerope_query_states,
    )
    tail_press = press_config(TailPress)
    full_result = instrument_prefill(
        engine,
        model,
        prompt_ids,
        full_press,
        device,
        span,
        ExpectedAttentionPress,
    )
    tail_result = instrument_prefill(
        engine,
        model,
        prompt_ids,
        tail_press,
        device,
        span,
        ExpectedAttentionPress,
    )
    full_boundary, full_source, full_rows, full_meta, full_seconds = full_result
    tail_boundary, tail_source, tail_rows, tail_meta, tail_seconds = tail_result
    if not full_meta["full_stats_identity"]:
        raise AssertionError("full ExpectedAttentionPress statistics changed")
    expected_q_len = int(full_boundary.cache_lengths[0])
    if tail_meta["window_original_q_lengths"] != [expected_q_len] * len(
        model.model.layers
    ):
        raise AssertionError("tail scorer did not preserve original q_len")
    eos = pilot.eos_ids(model, tokenizer)
    cap = row.get("max_new_tokens", CAP)
    plain_cache_before = engine.cache_fingerprint(plain_boundary.cache)
    full_cache_before = engine.cache_fingerprint(full_boundary.cache)
    tail_cache_before = engine.cache_fingerprint(tail_boundary.cache)
    plain_source_before = engine.cache_fingerprint(plain_source)
    full_source_before = engine.cache_fingerprint(full_source)
    tail_source_before = engine.cache_fingerprint(tail_source)
    plain_reference = continuation_arm(
        engine, model, plain_boundary, eos, 0.0, cap
    )
    plain_standard = continuation_arm(
        engine, model, plain_boundary, eos, ACTION_FRACTION, cap
    )
    full_reference = continuation_arm(
        engine, model, full_boundary, eos, 0.0, cap
    )
    full_standard = continuation_arm(
        engine, model, full_boundary, eos, ACTION_FRACTION, cap
    )
    tail_reference = continuation_arm(
        engine, model, tail_boundary, eos, 0.0, cap
    )
    tail_standard = continuation_arm(
        engine, model, tail_boundary, eos, ACTION_FRACTION, cap
    )
    prior_checks = {"enabled": prior is not None}
    if prior is not None:
        prior_checks.update(
            {
                "reference_tokens_match": plain_reference.continuation.token_ids
                == tuple(prior["reference"]["token_ids"]),
                "standard_tokens_match": plain_standard.continuation.token_ids
                == tuple(
                    prior["arms"]["knorm:0.1"]["continuation"]["token_ids"]
                ),
                "reference_termination_match": (
                    plain_reference.continuation.termination_reason
                    == prior["reference"]["termination_reason"]
                ),
                "standard_termination_match": (
                    plain_standard.continuation.termination_reason
                    == prior["arms"]["knorm:0.1"]["continuation"][
                        "termination_reason"
                    ]
                ),
            }
        )
    checks = {
        "plain_full_cache_equal": engine.cache_tensors_equal(
            plain_boundary.cache, full_boundary.cache
        ),
        "plain_tail_cache_equal": engine.cache_tensors_equal(
            plain_boundary.cache, tail_boundary.cache
        ),
        "full_tail_cache_equal": engine.cache_tensors_equal(
            full_boundary.cache, tail_boundary.cache
        ),
        "full_tail_storage_disjoint": engine.cache_storage_independent(
            full_boundary.cache, tail_boundary.cache
        ),
        "plain_full_storage_disjoint": engine.cache_storage_independent(
            plain_boundary.cache, full_boundary.cache
        ),
        "plain_tail_storage_disjoint": engine.cache_storage_independent(
            plain_boundary.cache, tail_boundary.cache
        ),
        "plain_full_source_equal": engine.cache_tensors_equal(
            plain_source, full_source
        ),
        "plain_tail_source_equal": engine.cache_tensors_equal(
            plain_source, tail_source
        ),
        "full_tail_source_equal": engine.cache_tensors_equal(
            full_source, tail_source
        ),
        "full_tail_source_disjoint": engine.cache_storage_independent(
            full_source, tail_source
        ),
        "full_stats_identity": full_meta["full_stats_identity"],
        "tail_original_q_len": all(
            value == expected_q_len
            for value in tail_meta["window_original_q_lengths"]
        ),
        "reference_tokens_stable": plain_reference.continuation.token_ids
        == full_reference.continuation.token_ids
        == tail_reference.continuation.token_ids,
        "standard_tokens_stable": plain_standard.continuation.token_ids
        == full_standard.continuation.token_ids
        == tail_standard.continuation.token_ids,
        "reference_termination_stable": plain_reference.continuation.termination_reason
        == full_reference.continuation.termination_reason
        == tail_reference.continuation.termination_reason,
        "standard_termination_stable": plain_standard.continuation.termination_reason
        == full_standard.continuation.termination_reason
        == tail_standard.continuation.termination_reason,
        "native_mask_hashes_equal": sorted(
            row["knorm_mask_hash"] for row in full_rows
        )
        == sorted(row["knorm_mask_hash"] for row in tail_rows),
        "boundary_unchanged": all(
            before == after
            for before, after in (
                (plain_cache_before, engine.cache_fingerprint(plain_boundary.cache)),
                (full_cache_before, engine.cache_fingerprint(full_boundary.cache)),
                (tail_cache_before, engine.cache_fingerprint(tail_boundary.cache)),
                (plain_source_before, engine.cache_fingerprint(plain_source)),
                (full_source_before, engine.cache_fingerprint(full_source)),
                (tail_source_before, engine.cache_fingerprint(tail_source)),
            )
        ),
        **{
            key: value
            for key, value in prior_checks.items()
            if key != "enabled"
        },
    }
    checks["scores_finite"] = all(
        bool(torch.isfinite(torch.tensor(value, dtype=torch.float64)))
        for rows in (full_rows, tail_rows)
        for row_value in rows
        for value in row_value.values()
        if isinstance(value, (int, float))
    )
    expected_heads = len(model.model.layers) * model.config.num_key_value_heads
    checks["head_counts_exact"] = len(full_rows) == len(tail_rows) == expected_heads
    checks["all_checks_pass"] = all(checks.values())
    return {
        "schema_version": "query_window_measurement.v1",
        "status": "completed",
        "manifest_row": row,
        "prompt_token_ids": [int(item) for item in prompt_ids[0].tolist()],
        "span": span,
        "parameters": {
            "action_fraction": ACTION_FRACTION,
            "n_sink": SINK,
            "n_future_positions": FUTURE,
            "tail_window": WINDOW,
            "use_covariance": False,
            "use_vnorm": True,
        },
        "boundary": {
            "plain": {
                "cache_fingerprint": engine.cache_fingerprint(plain_boundary.cache),
                "source_fingerprint": engine.cache_fingerprint(plain_source),
            },
            "full": {
                "cache_fingerprint": engine.cache_fingerprint(full_boundary.cache),
                "source_fingerprint": engine.cache_fingerprint(full_source),
            },
            "tail": {
                "cache_fingerprint": engine.cache_fingerprint(tail_boundary.cache),
                "source_fingerprint": engine.cache_fingerprint(tail_source),
            },
        },
        "windows": {
            "full_prompt": {
                "prefill_seconds": full_seconds,
                "head_count": len(full_rows),
                "summary": summarize_rows(full_rows),
                "heads": full_rows,
            },
            "tail_32": {
                "prefill_seconds": tail_seconds,
                "head_count": len(tail_rows),
                "summary": summarize_rows(tail_rows),
                "heads": tail_rows,
                "original_q_lengths": tail_meta["window_original_q_lengths"],
            },
        },
        "continuations": {
            label: pilot.continuation_dict(
                arm.continuation,
                tokenizer.decode(list(arm.continuation.token_ids), skip_special_tokens=True),
            )
            for label, arm in (
                ("plain_reference", plain_reference), ("plain_standard", plain_standard),
                ("full_reference", full_reference), ("full_standard", full_standard),
                ("tail_reference", tail_reference), ("tail_standard", tail_standard),
            )
        },
        "prior_checks": prior_checks,
        "checks": checks,
    }


def run(args):
    output = pilot.ensure_output_dir(args.output)
    engine, engine_root, engine_path = pilot.load_engine(args.engine_root)
    manifest_path, rows = pilot.load_manifest(args.manifest)
    by_id = {row["id"]: row for row in rows}
    requested = tuple(args.ids)
    if len(requested) != len(set(requested)) or any(
        row_id not in by_id for row_id in requested
    ):
        raise ValueError("manifest does not contain every requested ID")
    import transformers
    from kvpress import ExpectedAttentionPress

    torch.manual_seed(SEED)
    device = pilot.choose_device(args.device, torch)
    dtype = pilot.choose_dtype(args.dtype, device, torch)
    model, tokenizer = pilot.load_model_and_tokenizer(
        args.model, device, dtype, transformers
    )
    run_record = {
        "schema_version": "query_window_measurement.v1",
        "status": "running",
        "ids": list(requested),
        "manifest_path": str(manifest_path),
        "manifest_sha256": pilot.sha256_bytes(manifest_path.read_bytes()),
        "engine_path": str(engine_path),
        "engine_sha256": file_sha256(engine_path),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "kvpress_version": importlib.metadata.version("kvpress"),
        "model": pilot.model_runtime_identity(
            engine, model, args.model, transformers, torch
        ),
        "device": str(device),
        "dtype": str(dtype),
        "prompts": [],
        "failures": [],
    }
    pilot.write_json(output / "run.json", run_record)
    for index, row_id in enumerate(requested):
        row = by_id[row_id]
        path = output / pilot.safe_filename(index, row_id)
        try:
            record = run_prompt(
                args, row, engine, model, tokenizer, device, ExpectedAttentionPress
            )
            pilot.write_json(path, record)
            if not record["checks"]["all_checks_pass"]:
                raise RuntimeError("query-window diagnostic controls failed")
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
            record.update(
                {
                    "schema_version": "query_window_measurement.v1",
                    "status": "failed",
                    "manifest_row": row,
                    "failure": failure,
                }
            )
            pilot.write_json(path, record)
            run_record["failures"].append({"id": row_id, **failure})
            status = "failed"
        run_record["prompts"].append(
            {"id": row_id, "path": path.name, "status": status}
        )
        pilot.write_json(output / "run.json", run_record)
    run_record["status"] = "completed" if not run_record["failures"] else "failed"
    pilot.write_json(output / "run.json", run_record)
    return 0 if run_record["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", nargs="+", default=list(TARGET_IDS))
    parser.add_argument(
        "--prior-results",
        nargs="+",
        default=["results/ea-dev-v1-first", "results/ea-dev-v1-rest"],
    )
    parser.add_argument("--skip-prior-check", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
                    "schema_version": "query_window_measurement.v1",
                    "status": "failed",
                    "type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        print(f"query-window measurement failed: {type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
