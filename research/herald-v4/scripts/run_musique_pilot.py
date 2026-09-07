#!/usr/bin/env python3
"""Run the bounded MuSiQue answer-only EA quality pilot.

The runner keeps the prompt manifest and model untouched, builds an
independent
plain B0 and score-instrumented B0, applies native ExpectedAttentionPress to a
clone before the pending prompt token, then records raw continuations and all
engineering controls needed to interpret the result.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import prove_ea_boundary as ea  # noqa: E402
import run_pair_pilot as pilot  # noqa: E402

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "data/musique-pilot-v1/manifest.json"
)
DEFAULT_ENGINE_ROOT = Path(
    "/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3/src"
)
SCHEMA = "musique_pilot_runner.v1"


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_rows(path: Path) -> tuple[dict, list[dict]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
        payload = rows
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty list of rows")
    if not all(
        isinstance(row, dict) and row.get("id") and row.get("prompt")
        for row in rows
    ):
        raise ValueError("every row needs id and prompt")
    return (payload if isinstance(payload, dict) else {"prompts": rows}), rows


def select_rows(rows: list[dict], selector: str) -> list[dict]:
    if selector == "first":
        return rows[:1]
    if selector == "all":
        return rows
    wanted = [part.strip() for part in selector.split(",") if part.strip()]
    by_id = {row["id"]: row for row in rows}
    missing = [row_id for row_id in wanted if row_id not in by_id]
    if missing:
        raise ValueError(f"unknown manifest IDs: {missing}")
    return [by_id[row_id] for row_id in wanted]


def answer_score(
    prediction: str, answers: object, scorer_root: Path | None
) -> float | None:
    if scorer_root is None or not isinstance(answers, list) or not answers:
        return None
    sys.path.insert(0, str(scorer_root))
    try:
        from metrics.answer import compute_f1, metric_max_over_ground_truths

        return float(
            metric_max_over_ground_truths(compute_f1, prediction, answers)
        )
    finally:
        if sys.path[0] == str(scorer_root):
            sys.path.pop(0)


def model_context_limit(model: object) -> int | None:
    config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "max_sequence_length"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def continuation_payload(
    tokenizer: object, result: object, wall_seconds: float
) -> dict:
    token_ids = list(result.token_ids)
    return {
        "token_ids": token_ids,
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "termination_reason": result.termination_reason,
        "wall_seconds": wall_seconds,
        "forward_seconds": result.forward_seconds,
        "first_forward_seconds": result.first_forward_seconds,
        "final_cache_lengths": list(result.final_cache_lengths),
        "final_cache_bytes": result.final_cache_bytes,
        "final_cache_fingerprint": result.final_cache_fingerprint,
    }


def run_row(
    args: argparse.Namespace,
    row: dict,
    engine: object,
    model: object,
    tokenizer: object,
    device: object,
    dtype: object,
    scorer_root: Path | None,
) -> dict:
    import torch
    from kvpress import ExpectedAttentionPress

    row_id = row["id"]
    prompt_ids = ea.render_prompt(tokenizer, row["prompt"])
    prompt_length = int(prompt_ids.shape[1])
    cap = int(row.get("max_new_tokens", 64))
    if cap != 64 and not args.technical_fixture:
        raise ValueError(
            f"{row_id}: manifest max_new_tokens must be frozen at 64"
        )
    limit = model_context_limit(model)
    if (
        limit is not None
        and prompt_length + cap > limit
        and not args.technical_fixture
    ):
        raise ValueError(
            f"{row_id}: prompt plus horizon exceeds context limit "
            "without truncation"
        )
    if prompt_length < 2:
        raise ValueError(
            f"{row_id}: prompt must contain at least two rendered tokens"
        )

    press = ExpectedAttentionPress(
        compression_ratio=0.0,
        n_future_positions=ea.N_FUTURE_POSITIONS,
        n_sink=ea.N_SINK,
        use_covariance=ea.USE_COVARIANCE,
        use_vnorm=ea.USE_VNORM,
        epsilon=ea.EPSILON,
    )
    torch.manual_seed(int(row.get("seed", ea.SEED)))
    (plain_pair, plain_seconds) = ea.timed(
        device, lambda: ea.build_boundary(engine, model, prompt_ids)
    )
    plain_boundary, plain_source = plain_pair
    (inst_pair, inst_scores, hook_seconds, inst_seconds) = (
        ea.instrument_prefill(engine, model, prompt_ids, press, device)
    )
    inst_boundary, inst_source = inst_pair

    plain_summary = ea.cache_summary(engine, plain_boundary.cache)
    inst_summary = ea.cache_summary(engine, inst_boundary.cache)
    plain_source_summary = ea.cache_summary(engine, plain_source)
    inst_source_summary = ea.cache_summary(engine, inst_source)
    boundary_equal = engine.cache_tensors_equal(
        plain_boundary.cache, inst_boundary.cache
    )
    source_equal = engine.cache_tensors_equal(plain_source, inst_source)
    state_equal = (
        plain_boundary.prompt_ids.equal(inst_boundary.prompt_ids)
        and plain_boundary.generated_ids == inst_boundary.generated_ids
        and plain_boundary.pending_token_id == inst_boundary.pending_token_id
        and plain_boundary.logical_position == inst_boundary.logical_position
        and plain_boundary.attention_mask.equal(inst_boundary.attention_mask)
        and plain_boundary.cache_lengths == inst_boundary.cache_lengths
        and plain_boundary.cache_bytes == inst_boundary.cache_bytes
    )
    boundary_disjoint = engine.cache_storage_independent(
        plain_boundary.cache, inst_boundary.cache
    )
    source_disjoint = engine.cache_storage_independent(
        plain_source, plain_boundary.cache
    ) and engine.cache_storage_independent(inst_source, inst_boundary.cache)
    scores_finite = all(ea.torch_isfinite(score) for _, score in inst_scores)
    expected_heads = int(model.config.num_key_value_heads)
    score_shape = all(
        tuple(score.shape[:2]) == (1, expected_heads)
        for _, score in inst_scores
    )

    (action_cache, kept_indices), action_seconds = ea.timed(
        device,
        lambda: ea.apply_ea_action(
            engine, plain_boundary.cache, inst_scores, ea.REMOVAL_FRACTION
        ),
    )
    gather_checks = ea.direct_gather_matches(
        engine,
        plain_boundary.cache,
        action_cache,
        inst_scores,
        ea.REMOVAL_FRACTION,
    )
    action_lengths = list(engine.cache_lengths(action_cache))
    expected_keep = int(
        plain_summary["lengths"][0] * (1.0 - ea.REMOVAL_FRACTION)
    )
    action_bytes = engine.cache_nbytes(action_cache)
    expected_bytes = sum(
        (tensor.numel() // int(tensor.shape[-2]))
        * expected_keep
        * tensor.element_size()
        for tensor in engine._cache_tensors(plain_boundary.cache)
    )
    physical = (
        action_lengths == [expected_keep] * len(action_lengths)
        and action_bytes == expected_bytes
        and action_bytes < plain_summary["bytes"]
    )
    action_disjoint = engine.cache_storage_independent(
        plain_boundary.cache, action_cache
    )

    eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        eos_value = getattr(model.config, "eos_token_id", None)
    eos = (
        {int(eos_value)}
        if isinstance(eos_value, int)
        else ({int(x) for x in eos_value} if eos_value is not None else set())
    )
    plain_before = engine.cache_fingerprint(plain_boundary.cache)
    source_before = engine.cache_fingerprint(plain_source)
    inst_source_before = engine.cache_fingerprint(inst_source)
    (uninterrupted, uninterrupted_seconds) = ea.timed(
        device,
        lambda: engine._greedy_from_prompt(
            model, prompt_ids, max_new_tokens=cap, eos_ids=eos
        ),
    )
    (plain_noop, plain_noop_seconds) = ea.timed(
        device,
        lambda: (
            engine.continue_from_boundary(
                model,
                plain_boundary,
                max_new_tokens=cap,
                eos_ids=eos,
                action=engine.ActionSpec("knorm", 0.0),
            ).continuation
        ),
    )
    (instrumented_noop, inst_noop_seconds) = ea.timed(
        device,
        lambda: (
            engine.continue_from_boundary(
                model,
                inst_boundary,
                max_new_tokens=cap,
                eos_ids=eos,
                action=engine.ActionSpec("knorm", 0.0),
            ).continuation
        ),
    )
    (knorm_result, knorm_seconds) = ea.timed(
        device,
        lambda: (
            engine.continue_from_boundary(
                model,
                plain_boundary,
                max_new_tokens=cap,
                eos_ids=eos,
                action=engine.ActionSpec("knorm", ea.REMOVAL_FRACTION),
            ).continuation
        ),
    )
    (ea_result, ea_seconds) = ea.timed(
        device,
        lambda: ea.continuation(
            engine, model, plain_boundary, action_cache, cap, eos
        ),
    )

    no_op_exact = (
        plain_noop.token_ids == instrumented_noop.token_ids
        and plain_noop.termination_reason
        == instrumented_noop.termination_reason
    )
    uninterrupted_parity = (
        uninterrupted.token_ids == plain_noop.token_ids
        and uninterrupted.termination_reason == plain_noop.termination_reason
    )
    unchanged = (
        engine.cache_fingerprint(plain_boundary.cache) == plain_before
        and engine.cache_fingerprint(plain_source) == source_before
        and engine.cache_fingerprint(inst_source) == inst_source_before
    )
    gates = {
        "plain_instrumented_boundary_exact": boundary_equal and source_equal,
        "boundary_state_and_pending_token_exact": state_equal,
        "boundary_source_storage_disjoint": boundary_disjoint
        and source_disjoint
        and action_disjoint,
        "scores_expected_and_finite": len(inst_scores)
        == len(model.model.layers)
        and score_shape
        and scores_finite,
        "native_gather_exact": all(
            item["keys_equal"] and item["values_equal"]
            for item in gather_checks
        ),
        "physical_cache_effect_exact": physical,
        "no_op_continuation_exact": no_op_exact,
        "source_and_reference_unchanged": unchanged,
        "ea_action_continues": len(ea_result.token_ids) > 0,
    }
    return {
        "schema_version": SCHEMA,
        "id": row_id,
        "status": "completed" if all(gates.values()) else "failed",
        "manifest_row": row,
        "prompt": {
            "text": row["prompt"],
            "token_ids": [int(x) for x in prompt_ids[0].tolist()],
            "length": prompt_length,
            "assistant_prefix": None,
            "add_generation_prompt": True,
        },
        "config": {
            "removal_fraction": ea.REMOVAL_FRACTION,
            "n_sink": ea.N_SINK,
            "n_future_positions": ea.N_FUTURE_POSITIONS,
            "use_covariance": ea.USE_COVARIANCE,
            "use_vnorm": ea.USE_VNORM,
            "epsilon": ea.EPSILON,
            "max_new_tokens": cap,
            "seed": int(row.get("seed", ea.SEED)),
            "logical_position_before_pending": (
                plain_boundary.logical_position
            ),
        },
        "prefill": {
            "plain": {
                "seconds": plain_seconds,
                **plain_summary,
                "source": plain_source_summary,
            },
            "instrumented": {
                "seconds": inst_seconds,
                "hook_seconds": hook_seconds,
                **inst_summary,
                "source": inst_source_summary,
            },
        },
        "boundary_state": {
            "plain": plain_boundary.to_dict(),
            "instrumented": inst_boundary.to_dict(),
            "state_equal": state_equal,
        },
        "scores": {
            "layers": [
                {
                    "layer": i,
                    "shape": list(s.shape),
                    "finite": ea.torch_isfinite(s),
                }
                for i, s in inst_scores
            ],
            "layer_count": len(inst_scores),
            "expected_kv_heads": expected_heads,
        },
        "action": {
            "mask_hash": ea.mask_digest(kept_indices),
            "kept_indices": kept_indices,
            "cache": {
                "lengths": action_lengths,
                "bytes": action_bytes,
                "fingerprint": engine.cache_fingerprint(action_cache),
            },
            "expected_keep": expected_keep,
            "expected_bytes": expected_bytes,
            "direct_gather_checks": gather_checks,
            "seconds": action_seconds,
        },
        "continuations": {
            "uninterrupted": {
                "wall_seconds": uninterrupted_seconds,
                **ea.continuation_record(tokenizer, uninterrupted),
            },
            "plain_noop": {
                "wall_seconds": plain_noop_seconds,
                **ea.continuation_record(tokenizer, plain_noop),
            },
            "instrumented_noop": {
                "wall_seconds": inst_noop_seconds,
                **ea.continuation_record(tokenizer, instrumented_noop),
            },
            "knorm_10": {
                "wall_seconds": knorm_seconds,
                **ea.continuation_record(tokenizer, knorm_result),
            },
            "ea_10": {
                "wall_seconds": ea_seconds,
                **ea.continuation_record(tokenizer, ea_result),
            },
            "no_op_exact": no_op_exact,
            "uninterrupted_parity_supplemental": uninterrupted_parity,
        },
        "scores_by_answer": {
            name: answer_score(value["text"], row.get("answers"), scorer_root)
            for name, value in {
                "uninterrupted": ea.continuation_record(
                    tokenizer, uninterrupted
                ),
                "plain_noop": ea.continuation_record(tokenizer, plain_noop),
                "instrumented_noop": ea.continuation_record(
                    tokenizer, instrumented_noop
                ),
                "knorm_10": ea.continuation_record(tokenizer, knorm_result),
                "ea_10": ea.continuation_record(tokenizer, ea_result),
            }.items()
        },
        "quality_reference": {
            "reference_branch": "plain_noop",
            "signed_loss_reference_minus_ea_10": (
                answer_score(
                    tokenizer.decode(
                        list(plain_noop.token_ids), skip_special_tokens=True
                    ),
                    row.get("answers"),
                    scorer_root,
                )
                - answer_score(
                    tokenizer.decode(
                        list(ea_result.token_ids), skip_special_tokens=True
                    ),
                    row.get("answers"),
                    scorer_root,
                )
                if scorer_root is not None
                else None
            ),
            "signed_loss_reference_minus_knorm_10": (
                answer_score(
                    tokenizer.decode(
                        list(plain_noop.token_ids), skip_special_tokens=True
                    ),
                    row.get("answers"),
                    scorer_root,
                )
                - answer_score(
                    tokenizer.decode(
                        list(knorm_result.token_ids), skip_special_tokens=True
                    ),
                    row.get("answers"),
                    scorer_root,
                )
                if scorer_root is not None
                else None
            ),
            "uninterrupted_is_supplemental": True,
        },
        "cache_controls": {
            "plain_instrumented_boundary_equal": boundary_equal,
            "plain_instrumented_source_equal": source_equal,
            "boundary_storage_disjoint": boundary_disjoint,
            "source_storage_disjoint": source_disjoint,
            "action_storage_disjoint": action_disjoint,
            "source_and_reference_unchanged": unchanged,
            "plain_boundary_before": plain_before,
            "plain_boundary_after": engine.cache_fingerprint(
                plain_boundary.cache
            ),
            "plain_source_before": source_before,
            "plain_source_after": engine.cache_fingerprint(plain_source),
            "instrumented_source_before": inst_source_before,
            "instrumented_source_after": engine.cache_fingerprint(
                inst_source
            ),
        },
        "timings": {
            "plain_prefill_seconds": plain_seconds,
            "instrumented_prefill_seconds": inst_seconds,
            "instrumented_hook_seconds": hook_seconds,
            "action_seconds": action_seconds,
            "uninterrupted_seconds": uninterrupted_seconds,
            "plain_noop_seconds": plain_noop_seconds,
            "instrumented_noop_seconds": inst_noop_seconds,
            "knorm_seconds": knorm_seconds,
            "ea_seconds": ea_seconds,
        },
        "gates": gates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", default=str(DEFAULT_ENGINE_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ids",
        default="first",
        help="first, all, or comma-separated manifest IDs",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "bfloat16", "float16"),
    )
    parser.add_argument(
        "--technical-fixture",
        action="store_true",
        help=(
            "allow tiny fixture horizons shorter than the frozen "
            "64-token pilot"
        ),
    )
    args = parser.parse_args(argv)
    import torch
    import transformers

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest, rows = read_rows(manifest_path)
    selected = select_rows(rows, args.ids)
    write_json(
        output / "partial.json",
        {
            "schema_version": SCHEMA,
            "status": "running",
            "selected_ids": [r["id"] for r in selected],
            "completed": [],
        },
    )
    fixture_dir = output / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, fixture_dir / "manifest.json")
    scorer_root = None
    candidate = (
        manifest.get("scorer_root") if isinstance(manifest, dict) else None
    )
    if candidate:
        scorer_root = Path(candidate).expanduser().resolve()
    if scorer_root is None:
        candidate = manifest_path.parent / "provenance/official_scorer"
        if candidate.is_dir():
            scorer_root = candidate
    if scorer_root is not None and scorer_root.is_dir():
        shutil.copytree(
            scorer_root, fixture_dir / "official_scorer", dirs_exist_ok=True
        )
        scorer_root = fixture_dir / "official_scorer"

    engine, engine_root, engine_path = ea.load_engine(args.engine_root)
    device = torch.device(args.device)
    dtype = pilot.choose_dtype(args.dtype, device, torch)
    model, tokenizer = pilot.load_model_and_tokenizer(
        args.model, device, dtype, transformers
    )
    records = []
    status = "completed"
    for index, row in enumerate(selected):
        try:
            record = run_row(
                args,
                row,
                engine,
                model,
                tokenizer,
                device,
                dtype,
                scorer_root,
            )
        except Exception as exc:
            status = "failed"
            record = {
                "schema_version": SCHEMA,
                "id": row["id"],
                "status": "failed",
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        records.append(record)
        write_json(output / f"{index:04d}-{row['id']}.json", record)
        partial = {
            "schema_version": SCHEMA,
            "status": status if record["status"] == "failed" else "running",
            "selected_ids": [r["id"] for r in selected],
            "completed": [r["id"] for r in records],
            "records": [r["id"] + ":" + r["status"] for r in records],
        }
        write_json(output / "partial.json", partial)
        if record["status"] == "failed":
            status = "failed"
            break
    if len(records) != len(selected):
        status = "failed"
    run = {
        "schema_version": SCHEMA,
        "status": status,
        "selected_ids": [r["id"] for r in selected],
        "completed_ids": [r["id"] for r in records],
        "remaining_ids": [r["id"] for r in selected[len(records) :]],
        "manifest": {
            "path": str(manifest_path),
            "sha256": digest_file(manifest_path),
        },
        "engine": {
            "root": str(engine_root),
            "path": str(engine_path),
            "sha256": digest_file(engine_path),
        },
        "model": str(Path(args.model).expanduser().resolve()),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "kvpress": importlib.metadata.version("kvpress"),
            "device": str(device),
            "dtype": str(dtype),
        },
        "script_sha256": digest_file(Path(__file__).resolve()),
        "finished_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output / "run.json", run)
    write_json(
        output / "failures.json",
        [r for r in records if r["status"] == "failed"],
    )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
