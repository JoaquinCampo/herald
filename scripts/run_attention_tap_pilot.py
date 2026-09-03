"""Run the locked, label-blind H5-R0 paired AttentionTap pilot."""

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from herald import attention_features as attention_features_module
from herald import attention_replay as attention_replay_module
from herald import features as features_module
from herald import generate as generate_module
from herald import ifeval as ifeval_module
from herald.attention_features import AttentionTap, tap_feature_names
from herald.attention_replay import (
    PILOT_ROSTER,
    TAP_LAYER_INDICES,
    AttentionReplayError,
    audit_arrays,
    classify_decision,
    hash_prompt_ids,
    load_prefix_hash_sidecar,
    prefix_hash,
    read_legacy_identity_prefix,
    sha256_file,
    validate_hashes,
    validate_protocol_lock,
    write_float16_array,
    write_report,
)
from herald.config import MODELS
from herald.features import FEATURE_NAMES
from herald.generate import LoadedModel, generate_reference, load_model
from herald.ifeval import load_ifeval_exact
from herald.storage import safe_id
from herald.tasks import PromptRecord

MAX_NEW_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class ArmRun:
    ids: tuple[int, ...]
    prompt_ids: tuple[int, ...]
    values: np.ndarray
    feature_names: tuple[str, ...] | None
    elapsed_seconds: float
    peak_allocated_bytes: int
    oom: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/herald/attention_features.py",
    )
    parser.add_argument(
        "--generate-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/herald/generate.py",
    )
    parser.add_argument(
        "--ifeval-source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/herald/ifeval.py",
    )
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--legacy-reference-dir", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-prefix-sidecar", type=Path, required=True)
    parser.add_argument("--legacy-prefix-manifest", type=Path, required=True)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--legacy-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-id", default=MODELS["llama"])
    parser.add_argument("--device", default="cuda")
    return parser


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AttentionReplayError(
            f"cannot read JSON object {path}"
        ) from error
    if not isinstance(value, dict):
        raise AttentionReplayError(f"JSON file is not an object: {path}")
    return value


def _validate_roster(lock: dict[str, object]) -> tuple[str, ...]:
    raw = lock.get("pilot_roster")
    if not isinstance(raw, list):
        raise AttentionReplayError("protocol lock pilot roster is missing")
    roster: list[tuple[int, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise AttentionReplayError(
                "protocol lock pilot roster is malformed"
            )
        fold = item.get("fold")
        prompt_id = item.get("prompt_id")
        if (
            isinstance(fold, bool)
            or not isinstance(fold, int)
            or not isinstance(prompt_id, str)
        ):
            raise AttentionReplayError(
                "protocol lock pilot roster is malformed"
            )
        roster.append((fold, prompt_id))
    if tuple(roster) != PILOT_ROSTER:
        raise AttentionReplayError("protocol lock pilot roster is not exact")
    prompt_ids = tuple(prompt_id for _, prompt_id in roster)
    return prompt_ids


def _validate_generation_config(
    lock: dict[str, object], args: argparse.Namespace
) -> None:
    expected = {
        "model_key": "llama",
        "model_id": MODELS["llama"],
        "dtype": "bfloat16",
        "device": "cuda",
        "attn_implementation": "sdpa",
        "batch_size": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tap_layer_indices": list(TAP_LAYER_INDICES),
    }
    if lock.get("generation") != expected:
        raise AttentionReplayError(
            "protocol lock generation config is not frozen"
        )
    if args.device != "cuda":
        raise AttentionReplayError("pilot device must be cuda")
    if args.model_id != MODELS["llama"]:
        raise AttentionReplayError("pilot model identity is not frozen")


def _validate_bound_paths(
    lock: dict[str, object], args: argparse.Namespace
) -> None:
    paths = lock.get("paths")
    if not isinstance(paths, dict):
        raise AttentionReplayError("protocol lock paths are missing")
    if args.source_oof is None or args.legacy_config is None:
        raise AttentionReplayError("OOF and legacy config paths are required")
    expected = {
        "attention_source": args.attention_source,
        "generate_source": args.generate_source,
        "ifeval_source": args.ifeval_source,
        "protocol_lock": args.protocol_lock,
        "source_oof": args.source_oof,
        "legacy_config": args.legacy_config,
        "legacy_reference_dir": args.legacy_reference_dir,
        "legacy_manifest": args.legacy_manifest,
        "legacy_prefix_sidecar": args.legacy_prefix_sidecar,
        "legacy_prefix_manifest": args.legacy_prefix_manifest,
        "output_root": args.output_root,
    }
    for name, path in expected.items():
        if paths.get(name) != str(path.resolve()):
            raise AttentionReplayError(
                f"protocol lock path binding is invalid: {name}"
            )
    module_paths = {
        "attention_source": attention_features_module.__file__,
        "generate_source": generate_module.__file__,
        "ifeval_source": ifeval_module.__file__,
        "attention_replay_source": attention_replay_module.__file__,
        "features_source": features_module.__file__,
        "script_source": __file__,
    }
    for name, actual in module_paths.items():
        if actual is None or paths.get(name) != str(Path(actual).resolve()):
            raise AttentionReplayError(
                f"source path is not the imported module: {name}"
            )


def _validate_input_hashes(
    lock: dict[str, object], args: argparse.Namespace
) -> dict[str, str]:
    value = lock.get("input_hashes")
    if not isinstance(value, dict):
        raise AttentionReplayError("protocol lock input hashes are missing")
    required = {
        "source_oof",
        "legacy_config",
        "legacy_manifest",
        "prefix_sidecar",
        "prefix_manifest",
        "attention_source",
        "generate_source",
        "ifeval_source",
        "attention_replay_source",
        "features_source",
        "script",
    }
    declared: dict[str, str] = {}
    for raw_name, raw_hash in value.items():
        if not isinstance(raw_name, str) or not raw_name.endswith("_sha256"):
            raise AttentionReplayError("protocol hash name is not canonical")
        name = raw_name.removesuffix("_sha256")
        if (
            name in declared
            or not isinstance(raw_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", raw_hash)
        ):
            raise AttentionReplayError(
                f"protocol hash declaration is malformed: {raw_name}"
            )
        declared[name] = raw_hash
    if set(declared) != required or "protocol_lock_sha256" in value:
        raise AttentionReplayError("protocol hash names are not exact")
    if args.source_oof is None or args.legacy_config is None:
        raise AttentionReplayError("OOF and legacy config paths are required")
    paths = {
        "source_oof": args.source_oof,
        "legacy_config": args.legacy_config,
        "legacy_manifest": args.legacy_manifest,
        "prefix_sidecar": args.legacy_prefix_sidecar,
        "prefix_manifest": args.legacy_prefix_manifest,
        "attention_source": args.attention_source,
        "generate_source": args.generate_source,
        "ifeval_source": args.ifeval_source,
        "attention_replay_source": Path(
            attention_replay_module.__file__ or ""
        ),
        "features_source": Path(features_module.__file__ or ""),
        "script": Path(__file__),
    }
    return validate_hashes(paths, declared)


def _validate_legacy_manifest(
    path: Path, root: Path, prompt_ids: tuple[str, ...]
) -> dict[str, tuple[Path, Path]]:
    manifest = _load_json(path)
    raw_files = manifest.get("artifact_files")
    if not isinstance(raw_files, list):
        raise AttentionReplayError(
            "legacy manifest artifact_files are missing"
        )
    entries: dict[str, tuple[str, int]] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            raise AttentionReplayError("legacy manifest entry is malformed")
        if set(item) != {"path", "sha256", "size"}:
            raise AttentionReplayError(
                "legacy manifest entry keys are not exact"
            )
        manifest_name = item["path"]
        digest = item["sha256"]
        size = item["size"]
        if (
            not isinstance(manifest_name, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise AttentionReplayError("legacy manifest entry is malformed")
        if manifest_name in entries:
            raise AttentionReplayError(
                "legacy manifest contains duplicate paths"
            )
        entries[manifest_name] = (digest, size)
    if not entries:
        raise AttentionReplayError("legacy manifest has no files")
    result: dict[str, tuple[Path, Path]] = {}
    for prompt_id in prompt_ids:
        expected_names = (f"{prompt_id}.json", f"{safe_id(prompt_id)}.npy")
        selected: list[Path] = []
        for expected_name in expected_names:
            matches = [
                (manifest_name, digest, size)
                for manifest_name, (digest, size) in entries.items()
                if Path(manifest_name).name == expected_name
            ]
            if len(matches) != 1:
                raise AttentionReplayError(
                    "legacy manifest does not uniquely identify "
                    f"{expected_name}"
                )
            manifest_name, digest, size = matches[0]
            candidate = (root / manifest_name).resolve()
            if root.resolve() not in candidate.parents:
                raise AttentionReplayError(
                    "legacy manifest path escapes reference directory"
                )
            if not candidate.is_file() or candidate.stat().st_size != size:
                raise AttentionReplayError(
                    f"legacy file size mismatch: {manifest_name}"
                )
            if sha256_file(candidate) != digest:
                raise AttentionReplayError(
                    f"legacy file hash mismatch: {manifest_name}"
                )
            selected.append(candidate)
        result[prompt_id] = (selected[0], selected[1])
    return result


def _generate_arm(
    lm: LoadedModel, record: PromptRecord, *, tapped: bool
) -> ArmRun:
    tap: AttentionTap | None = None
    if tapped:
        tap = AttentionTap(lm.model, list(TAP_LAYER_INDICES))
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        runs = generate_reference(
            lm,
            [record],
            MAX_NEW_TOKENS,
            tap=tap,
            decode_text=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        run = runs[0]
        # Match the immutable sweep persistence boundary exactly.
        values = np.asarray(run.features, dtype=np.float16)
        names = (
            tuple(run.feature_names)
            if run.feature_names is not None
            else None
        )
        if tapped and names is None:
            raise AttentionReplayError(
                "tapped run did not capture feature names"
            )
        peak = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        )
        return ArmRun(
            tuple(int(value) for value in run.gen_ids),
            tuple(int(value) for value in run.prompt_input_ids),
            values,
            names,
            elapsed,
            peak,
        )
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        return ArmRun(
            (), (), np.empty((0, 0), dtype=np.float16), None, 0.0, 0, oom=True
        )
    finally:
        if tap is not None:
            tap.remove()


def _warmup(lm: LoadedModel, *, tapped: bool) -> None:
    record = PromptRecord(
        task="ifeval",
        prompt_id="synthetic-warmup",
        messages=[
            {"role": "user", "content": "Reply with four short tokens."}
        ],
        gold={},
    )
    tap: AttentionTap | None = None
    if tapped:
        tap = AttentionTap(lm.model, list(TAP_LAYER_INDICES))
    try:
        generate_reference(lm, [record], 4, tap=tap, decode_text=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        arm = "tapped" if tapped else "control"
        raise AttentionReplayError(
            f"{arm} warmup CUDA OOM retires AttentionTap"
        ) from error
    finally:
        if tap is not None:
            tap.remove()


def _hash_ids(values: tuple[int, ...]) -> str:
    return hashlib.sha256(
        np.asarray(values, dtype=np.int64).tobytes()
    ).hexdigest()


def _prefix_hashes_match(
    sidecar: dict[tuple[str, int], str],
    prompt_id: str,
    control: ArmRun,
) -> bool:
    records = [
        (position, digest)
        for (sidecar_prompt, position), digest in sidecar.items()
        if sidecar_prompt == prompt_id
    ]
    if not records:
        raise AttentionReplayError(f"sidecar has no boundary for {prompt_id}")
    for position, digest in records:
        try:
            if (
                prefix_hash(control.prompt_ids, control.ids, position)
                != digest
            ):
                return False
        except AttentionReplayError:
            return False
    return True


def _tapped_control_ratio(
    generation_times: dict[str, float], *, oom: bool
) -> float | None:
    control_time = generation_times["control"]
    if oom or control_time <= 0:
        return None
    return generation_times["tapped"] / control_time


def main() -> None:
    args = _parser().parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite pilot output root: {args.output_root}"
        )
    lock = _load_json(args.protocol_lock)
    validate_protocol_lock(lock)
    _validate_generation_config(lock, args)
    _validate_bound_paths(lock, args)
    prompt_ids = _validate_roster(lock)
    hashes = _validate_input_hashes(lock, args)
    manifest_files = _validate_legacy_manifest(
        args.legacy_manifest, args.legacy_reference_dir, prompt_ids
    )
    records = load_ifeval_exact(prompt_ids)
    if tuple(record.prompt_id for record in records) != prompt_ids:
        raise AttentionReplayError(
            "exact IFEval loader returned the wrong order"
        )
    identities = {
        prompt_id: read_legacy_identity_prefix(
            manifest_files[prompt_id][0], prompt_id
        )
        for prompt_id in prompt_ids
    }
    sidecar = load_prefix_hash_sidecar(args.legacy_prefix_sidecar, prompt_ids)
    lm = load_model(
        "llama",
        dtype="bfloat16",
        device="cuda",
        attn_implementation="sdpa",
        model_id=args.model_id,
    )
    _warmup(lm, tapped=False)
    _warmup(lm, tapped=True)
    generation_times = {"control": 0.0, "tapped": 0.0}
    oom = False
    array_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    arm_runs: list[tuple[ArmRun, ArmRun]] = []
    audits: list[dict[str, object]] = []
    prompt_reports: dict[str, dict[str, object]] = {}
    for fold, prompt_id in PILOT_ROSTER:
        record = records[fold]
        if record.prompt_id != prompt_id:
            raise AttentionReplayError(
                "roster order changed during generation"
            )
        first_tapped = fold % 2 == 1
        first = _generate_arm(lm, record, tapped=first_tapped)
        second = _generate_arm(lm, record, tapped=not first_tapped)
        control = first if not first_tapped else second
        tapped = first if first_tapped else second
        legacy = np.load(manifest_files[prompt_id][1], allow_pickle=False)
        identity = identities[prompt_id]
        prefix_match = _prefix_hashes_match(sidecar, prompt_id, control)
        audit = audit_arrays(
            control.values,
            tapped.values,
            legacy,
            expected_gen_ids=identity.gen_ids,
            control_gen_ids=control.ids,
            tapped_gen_ids=tapped.ids,
            expected_prompt_ids=identity.prompt_input_ids,
            control_prompt_ids=control.prompt_ids,
            tapped_prompt_ids=tapped.prompt_ids,
            feature_names=tapped.feature_names,
        )
        audit["prefix_hashes_match"] = prefix_match
        audit["no_oom"] = not first.oom and not second.oom
        audit["cost_ok"] = False
        audits.append(audit)
        oom = oom or first.oom or second.oom
        generation_times["control"] += control.elapsed_seconds
        generation_times["tapped"] += tapped.elapsed_seconds
        array_pairs.append((control.values, tapped.values))
        arm_runs.append((control, tapped))

    ratio = _tapped_control_ratio(generation_times, oom=oom)
    cost_ok = not oom and ratio is not None and ratio <= 2.0
    for (control_values, tapped_values), audit, (
        fold,
        prompt_id,
    ), runs in zip(array_pairs, audits, PILOT_ROSTER, arm_runs, strict=True):
        control_run, tapped_run = runs
        audit["cost_ok"] = cost_ok
        prompt_reports[f"fold_{fold}"] = {
            "fold": fold,
            "control_rows": int(control_values.shape[0]),
            "tapped_rows": int(tapped_values.shape[0]),
            "control_time_seconds": control_run.elapsed_seconds,
            "tapped_time_seconds": tapped_run.elapsed_seconds,
            "control_peak_allocated_bytes": control_run.peak_allocated_bytes,
            "tapped_peak_allocated_bytes": tapped_run.peak_allocated_bytes,
            **audit,
            "legacy_prompt_ids_sha256": _hash_ids(
                identities[prompt_id].prompt_input_ids
            ),
            "legacy_gen_ids_sha256": _hash_ids(identities[prompt_id].gen_ids),
            "control_prompt_ids_sha256": _hash_ids(control_run.prompt_ids),
            "control_gen_ids_sha256": _hash_ids(control_run.ids),
            "tapped_prompt_ids_sha256": _hash_ids(tapped_run.prompt_ids),
            "tapped_gen_ids_sha256": _hash_ids(tapped_run.ids),
        }
    decision = classify_decision(audits, aggregate_cost_ok=cost_ok)
    args.output_root.mkdir(parents=True)
    for (fold, prompt_id), (control_values, tapped_values) in zip(
        PILOT_ROSTER, array_pairs, strict=True
    ):
        stem = safe_id(prompt_id)
        control_path = args.output_root / f"{stem}.control.npy"
        tapped_path = args.output_root / f"{stem}.tapped.npy"
        write_float16_array(control_path, control_values)
        write_float16_array(tapped_path, tapped_values)
        prompt_reports[f"fold_{fold}"]["control_array_sha256"] = sha256_file(
            control_path
        )
        prompt_reports[f"fold_{fold}"]["tapped_array_sha256"] = sha256_file(
            tapped_path
        )
    feature_names = tuple([*FEATURE_NAMES, *tap_feature_names()])
    feature_names_hash = hashlib.sha256(
        json.dumps(list(feature_names), separators=(",", ":")).encode()
    ).hexdigest()
    report: dict[str, object] = {
        "schema_version": "herald.attention_tap_pilot_report.v1",
        "decision": decision,
        "pilot_count": len(prompt_ids),
        "pilot_prompt_ids_sha256": hash_prompt_ids(prompt_ids),
        "feature_names_sha256": feature_names_hash,
        "tap_layers": list(TAP_LAYER_INDICES),
        "tap_feature_count": len(feature_names) - len(FEATURE_NAMES),
        "control_time_seconds": generation_times["control"],
        "tapped_time_seconds": generation_times["tapped"],
        "tapped_control_time_ratio": ratio,
        "aggregate_cost_ok": cost_ok,
        "prompt_audits": prompt_reports,
        "no_oom": not oom,
        "input_hashes": {
            f"{name}_sha256": digest for name, digest in hashes.items()
        },
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
    }
    write_report(args.output_root / "report.json", report)


if __name__ == "__main__":
    main()
