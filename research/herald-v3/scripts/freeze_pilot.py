#!/usr/bin/env python3
"""Freeze the outcome-blind HERALD v3 pilot roster and protocol.

This script reads only the official IFEval JSONL, the generated exposure
ledger, and the already accepted engineering prompt manifest.  It never
reads generation outcomes and it fails closed if those sources drift.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_PROMPTS_SPEC = importlib.util.spec_from_file_location(
    "herald_v3_engineering_prompts",
    PROJECT_ROOT / "src/herald_v3/engineering/prompts.py",
)
if _PROMPTS_SPEC is None or _PROMPTS_SPEC.loader is None:
    raise ImportError("cannot load the prompt manifest module")
_PROMPTS_MODULE = importlib.util.module_from_spec(_PROMPTS_SPEC)
sys.modules[_PROMPTS_SPEC.name] = _PROMPTS_MODULE
_PROMPTS_SPEC.loader.exec_module(_PROMPTS_MODULE)
EngineeringPrompt = _PROMPTS_MODULE.EngineeringPrompt
PromptManifest = _PROMPTS_MODULE.PromptManifest
load_prompt_manifest = _PROMPTS_MODULE.load_prompt_manifest
write_prompt_manifest = _PROMPTS_MODULE.write_prompt_manifest

DEFAULT_OFFICIAL = (
    PROJECT_ROOT / "results/engineering/official-scorer/input_data.jsonl"
)
DEFAULT_LEDGER = PROJECT_ROOT / "data/exposure-ledger.json"
DEFAULT_ENGINEERING = PROJECT_ROOT / "data/engineering-prompts.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/pilot-v1"
EXPOSURE_BUILDER = PROJECT_ROOT / "scripts/build_exposure_ledger.py"
RESEARCH_PLAN = PROJECT_ROOT / "docs/research/research-plan.md"
SEED_STRING = "herald-v3-js-pilot-v1:0"
OWNER_EXCLUDED = frozenset({288, 2337, 3224, 3750})
ROSTER_SIZE = 160
PRIMARY_TARGET = 120
FOLD_COUNT = 5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--engineering", type=Path, default=DEFAULT_ENGINEERING
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    result = freeze(
        official_path=args.official,
        ledger_path=args.ledger,
        engineering_path=args.engineering,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def freeze(
    *,
    official_path: Path,
    ledger_path: Path,
    engineering_path: Path,
    output_path: Path,
) -> dict[str, object]:
    official_bytes = _read(official_path)
    ledger_bytes = _read(ledger_path)
    engineering_bytes = _read(engineering_path)
    builder_bytes = _read(EXPOSURE_BUILDER)
    research_plan_bytes = _read(RESEARCH_PLAN)
    official_rows = _read_official(official_bytes)
    ledger = _read_json(ledger_bytes, ledger_path)

    _assert_ledger(ledger)
    engineering_manifest = load_prompt_manifest(engineering_path, limit=8)
    system_message = _system_message(engineering_manifest)
    official_by_key = {cast(int, row["key"]): row for row in official_rows}
    ledger_prompts = _ledger_prompts(ledger)
    remaining = [
        row
        for row in ledger_prompts
        if not cast(bool, row["exposed"])
        and cast(int, row["official_key"]) not in OWNER_EXCLUDED
    ]
    if len(remaining) != 216:
        raise ValueError(
            f"expected 216 singleton candidates, got {len(remaining)}"
        )
    if any(
        cast(int, row["official_key"]) not in official_by_key
        for row in remaining
    ):
        raise ValueError(
            "ledger contains a prompt missing from official JSONL"
        )
    if any(
        cast(int, row["official_key"]) in _remaining_near_duplicates(ledger)
        for row in remaining
    ):
        raise ValueError("remaining candidates are not all singleton prompts")

    ranked = sorted(remaining, key=_selection_key)
    selected = ranked[:ROSTER_SIZE]
    holdout = ranked[ROSTER_SIZE:]
    if len(selected) != ROSTER_SIZE or len(holdout) != 56:
        raise ValueError(
            "pilot roster and holdout sizes do not match the lock"
        )

    prompts = tuple(
        _make_prompt(row, official_by_key, system_message) for row in selected
    )
    manifest = PromptManifest(
        source_path=str(official_path),
        source_sha256=_sha256(official_bytes),
        source_bytes=len(official_bytes),
        official_dataset_path=str(official_path),
        official_dataset_sha256=_sha256(official_bytes),
        official_dataset_bytes=len(official_bytes),
        official_dataset_rows=len(official_rows),
        selection_rule=(
            "remaining official rows after exposure ledger and owner "
            f"exclusions; rank sha256({SEED_STRING}|"
            "normalized_prompt_utf8_sha256), "
            f"take first {ROSTER_SIZE}; fold=integer(sha256(fold|"
            f"normalized_prompt_utf8_sha256)[:8],16) mod {FOLD_COUNT}"
        ),
        prompts=prompts,
    )
    _assert_manifest(manifest, selected, ledger, official_by_key)

    output_path.mkdir(parents=True, exist_ok=True)
    review = _review_document(
        official_path=official_path,
        official_bytes=official_bytes,
        ledger_path=ledger_path,
        ledger_bytes=ledger_bytes,
        engineering_path=engineering_path,
        engineering_bytes=engineering_bytes,
        engineering_manifest=engineering_manifest,
        ledger=ledger,
        selected=selected,
        holdout=holdout,
    )
    review_path = output_path / "exposure-review.json"
    review_path.write_text(_dump(review), encoding="utf-8")
    prompts_path = output_path / "prompts.json"
    write_prompt_manifest(manifest, prompts_path)
    roundtripped = load_prompt_manifest(prompts_path, limit=ROSTER_SIZE)
    if roundtripped.fingerprint != manifest.fingerprint:
        raise ValueError("written prompt manifest does not round trip")
    lock = _lock_document(
        official_path=official_path,
        official_bytes=official_bytes,
        ledger_path=ledger_path,
        ledger_bytes=ledger_bytes,
        engineering_path=engineering_path,
        engineering_bytes=engineering_bytes,
        engineering_manifest=engineering_manifest,
        builder_bytes=builder_bytes,
        research_plan_bytes=research_plan_bytes,
        review_path=review_path,
        prompts_path=prompts_path,
        manifest=manifest,
        selected=selected,
        holdout=holdout,
    )
    lock_path = output_path / "lock.json"
    lock_path.write_text(_dump(lock), encoding="utf-8")
    return {
        "output": str(output_path),
        "selected": len(selected),
        "primary_target": PRIMARY_TARGET,
        "holdout": len(holdout),
        "manifest_sha256": manifest.fingerprint,
        "lock_sha256": _sha256(lock_path.read_bytes()),
    }


def _make_prompt(
    ledger_row: dict[str, object],
    official_by_key: dict[int, dict[str, object]],
    system_message: str,
) -> EngineeringPrompt:
    key = cast(int, ledger_row["official_key"])
    official = official_by_key[key]
    user = cast(str, official["prompt"])
    prompt_text = (
        f"<|im_start|>system\n{system_message}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    normalized_hash = cast(str, ledger_row["normalized_prompt_utf8_sha256"])
    fold_hash = hashlib.sha256(
        f"fold|{normalized_hash}".encode("ascii")
    ).hexdigest()
    split_hash = hashlib.sha256(
        f"split|{normalized_hash}".encode("ascii")
    ).hexdigest()
    fold = int(fold_hash[:8], 16) % FOLD_COUNT
    messages = (
        {"role": "system", "content": system_message},
        {"role": "user", "content": user},
    )
    ids = cast(list[str], official["instruction_id_list"])
    raw_kwargs = cast(list[dict[str, object]], official["kwargs"])
    return EngineeringPrompt(
        prompt_id=f"ifeval_{key}",
        key=key,
        fold=fold,
        fold_hash=fold_hash,
        split_hash=split_hash,
        prompt_text=prompt_text,
        user_prompt=user,
        messages=messages,
        instruction_id_list=tuple(ids),
        kwargs=tuple(dict(item) for item in raw_kwargs),
        prompt_text_sha256=_sha256(prompt_text.encode("utf-8")),
        user_prompt_sha256=_sha256(user.encode("utf-8")),
    )


def _review_document(
    *,
    official_path: Path,
    official_bytes: bytes,
    ledger_path: Path,
    ledger_bytes: bytes,
    engineering_path: Path,
    engineering_bytes: bytes,
    engineering_manifest: PromptManifest,
    ledger: dict[str, object],
    selected: list[dict[str, object]],
    holdout: list[dict[str, object]],
) -> dict[str, object]:
    selected_keys = [cast(int, row["official_key"]) for row in selected]
    holdout_keys = [cast(int, row["official_key"]) for row in holdout]
    return {
        "schema_version": "herald_v3.pilot_exposure_review.v1",
        "outcomes_read": False,
        "selection_decision": (
            "owner-approved exclusions then deterministic hash rank"
        ),
        "owner_excluded_keys": sorted(OWNER_EXCLUDED),
        "owner_excluded_reason": (
            "explicit owner review flag; excluded before ranking"
        ),
        "ledger_counts": ledger["counts"],
        "candidate_counts": {
            "remaining_official": 220,
            "owner_excluded": 4,
            "singleton_remaining": 216,
            "selected_roster": len(selected),
            "selected_primary_target": PRIMARY_TARGET,
            "selected_roster_reserve": len(selected) - PRIMARY_TARGET,
            "unselected_holdout": len(holdout),
        },
        "selected_keys": selected_keys,
        "unselected_holdout_keys": holdout_keys,
        "sources": {
            "official_jsonl": _source_record(official_path, official_bytes),
            "exposure_ledger": _source_record(ledger_path, ledger_bytes),
            "engineering_manifest": _source_record(
                engineering_path, engineering_bytes
            ),
            "engineering_template_dataset": _dataset_record(
                engineering_manifest
            ),
        },
        "near_duplicate_policy": {
            "threshold": 0.70,
            "remaining_near_duplicate_keys_excluded": sorted(OWNER_EXCLUDED),
            "selected_prompts_are_singletons": True,
            "method": (
                "use final exposure ledger candidates; no outcome-based "
                "filtering"
            ),
        },
        "selection": {
            "seed_string": SEED_STRING,
            "rank_expression": (
                "sha256(seed_string + '|' + normalized_prompt_utf8_sha256)"
            ),
            "take": ROSTER_SIZE,
            "holdout": 56,
        },
    }


def _lock_document(
    *,
    official_path: Path,
    official_bytes: bytes,
    ledger_path: Path,
    ledger_bytes: bytes,
    engineering_path: Path,
    engineering_bytes: bytes,
    engineering_manifest: PromptManifest,
    builder_bytes: bytes,
    research_plan_bytes: bytes,
    review_path: Path,
    prompts_path: Path,
    manifest: PromptManifest,
    selected: list[dict[str, object]],
    holdout: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": "herald_v3.pilot_lock.v1",
        "status": "frozen_roster_pending_owner_launch",
        "outcomes_collected": False,
        "pilot_roster": {
            "candidate_count": len(selected),
            "primary_target_count": PRIMARY_TARGET,
            "roster_reserve_count": len(selected) - PRIMARY_TARGET,
            "unselected_singleton_holdout_count": len(holdout),
            "prompt_manifest": str(prompts_path),
            "prompt_manifest_sha256": _sha256(prompts_path.read_bytes()),
            "manifest_fingerprint": manifest.fingerprint,
            "fold_count": FOLD_COUNT,
            "fold_rule": (
                "sha256('fold|' + normalized_prompt_utf8_sha256) first 8 hex "
                "mod 5"
            ),
            "selection_seed": SEED_STRING,
        },
        "generation": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "checkpoint_snapshot": "a09a35458c702b33eeacc393d103063234e8bc28",
            "tokenizer_snapshot": "a09a35458c702b33eeacc393d103063234e8bc28",
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "decoding": "greedy",
            "seed": 0,
            "total_new_token_budget": 1024,
            "eos_token_ids": [151643, 151645],
            "chat_template": (
                "Qwen2.5 im_start/im_end template, system and user messages "
                "preserved from engineering manifest"
            ),
            "decision_boundary": {
                "committed_output_tokens": 32,
                "pending_token_output_index": 31,
                "first_affected_prediction_output_index": 32,
                "action_once_before_pending_token_forward": True,
            },
        },
        "actions": {
            "reference": "none",
            "candidates": [
                {"name": "knorm_025", "removal_fraction": 0.25},
                {"name": "knorm_050", "removal_fraction": 0.50},
            ],
            "implementation": (
                "direct live-cache Knorm transform, one-time at state32"
            ),
        },
        "target": {
            "primary": "signed loose IFEval fraction difference d = q0 - qa",
            "strict_sensitivity": (
                "repeat the same analysis with strict IFEval fractions"
            ),
            "retain_zero_positive_negative": True,
            "prompt_equal_action_meaning": True,
            "no_op_excluded_from_prediction_metrics": True,
        },
        "feature_sets": {
            "B0": [
                "removal_fraction",
                "prompt_token_count",
                "decision_index",
                "pre_action_cache_size",
            ],
            "B1": [
                "B0",
                "uncompressed_next_distribution_entropy",
                "uncompressed_top_two_margin",
            ],
            "B2": [
                "B1",
                "compressed_entropy_delta",
                "compressed_margin_delta",
                "argmax_match",
            ],
            "B3": ["B2", "full_vocabulary_js_divergence"],
            "predictors_exclude": [
                "outcomes",
                "future_tokens",
                "final_scores",
                "instruction_annotations",
            ],
        },
        "evaluation": {
            "folds": 5,
            "grouping": (
                "prompt singleton groups, all actions retained together"
            ),
            "standardization": (
                "StandardScaler fit on each training fold only"
            ),
            "model": "Ridge(alpha=1)",
            "tuning": "none",
            "prediction_clip": [-1.0, 1.0],
            "weighting": "equal prompt weighting, mean of two action errors",
            "baseline": "action-wise training mean",
            "primary_metric": "out-of-fold MSE",
            "diagnostics": [
                "MAE",
                "mean_signed_bias",
                "action_specific_errors",
                "positive_zero_negative_subsets",
            ],
            "bootstrap": {
                "replicates": 2000,
                "unit": "prompt cluster",
                "preserve_cluster_multiplicity": True,
                "seed": 0,
                "interval": "percentile",
                "confidence_level": 0.95,
                "interval_scope": (
                    "conditional on fitted out-of-fold models; no refitting "
                    "or training uncertainty"
                ),
            },
            "information_floor": (
                "at least 20 distinct prompts with nonzero d for either "
                "action"
            ),
            "go_criterion": (
                "B3 point MSE gain at least 5 percent against every "
                "comparator "
                "and paired intervals exclude zero"
            ),
        },
        "provenance": {
            "official_jsonl": _source_record(official_path, official_bytes),
            "exposure_ledger": _source_record(ledger_path, ledger_bytes),
            "engineering_manifest": _source_record(
                engineering_path, engineering_bytes
            ),
            "engineering_template_dataset": _dataset_record(
                engineering_manifest
            ),
            "exposure_review": _source_record(
                review_path, review_path.read_bytes()
            ),
            "exposure_ledger_builder": _source_record(
                EXPOSURE_BUILDER, builder_bytes
            ),
            "research_plan": _source_record(
                RESEARCH_PLAN, research_plan_bytes
            ),
            "source_correction": {
                "key": 2785,
                "evidence": (
                    "results/engineering/official-scorer/"
                    "dataset-differences.json"
                ),
                "difference": (
                    "Arrow prompt says at least one placeholder; pinned "
                    "official JSONL says at least 3 placeholders."
                ),
                "instruction_ids_and_kwargs_equal": True,
                "pilot_prompt_source": "pinned official JSONL",
                "engineering_template_dataset_role": (
                    "template provenance only; never the pilot prompt source"
                ),
            },
            "no_outcome_statement": (
                "This lock was built before generation and reads no "
                "outcomes, "
                "scores, or features."
            ),
        },
    }


def _assert_manifest(
    manifest: PromptManifest,
    selected: list[dict[str, object]],
    ledger: dict[str, object],
    official_by_key: dict[int, dict[str, object]],
) -> None:
    if len(manifest.prompts) != ROSTER_SIZE:
        raise ValueError("manifest does not contain exactly 160 prompts")
    selected_keys = {prompt.key for prompt in manifest.prompts}
    expected_keys = {cast(int, row["official_key"]) for row in selected}
    if selected_keys != expected_keys:
        raise ValueError("manifest keys differ from ranked roster")
    exposed = {
        cast(int, row["official_key"])
        for row in cast(list[dict[str, object]], ledger["prompts"])
        if cast(bool, row["exposed"])
    }
    if selected_keys & exposed or selected_keys & OWNER_EXCLUDED:
        raise ValueError("manifest includes exposed or owner-excluded prompt")
    if {prompt.fold for prompt in manifest.prompts} != set(range(FOLD_COUNT)):
        raise ValueError("manifest does not cover all five folds")
    if (
        len({prompt.user_prompt_sha256 for prompt in manifest.prompts})
        != ROSTER_SIZE
    ):
        raise ValueError("manifest contains duplicate prompt text")
    for prompt in manifest.prompts:
        official = official_by_key[prompt.key]
        if prompt.user_prompt != official["prompt"]:
            raise ValueError(
                "manifest prompt text differs from official JSONL at "
                f"key {prompt.key}"
            )
        if (
            list(prompt.instruction_id_list)
            != official["instruction_id_list"]
        ):
            raise ValueError(
                "manifest instruction IDs differ from official JSONL at "
                f"key {prompt.key}"
            )
        if list(prompt.kwargs) != official["kwargs"]:
            raise ValueError(
                "manifest kwargs differ from official JSONL at "
                f"key {prompt.key}"
            )
    roundtrip = manifest.to_dict()
    if roundtrip["manifest_sha256"] != manifest.fingerprint:
        raise ValueError("manifest fingerprint is not reproducible")


def _assert_ledger(ledger: dict[str, object]) -> None:
    counts = cast(dict[str, object], ledger["counts"])
    required = {
        "official_ifeval": 541,
        "exposed_union_ids": 321,
        "remaining_official_ids": 220,
        "near_duplicate_candidates": 12,
    }
    if any(counts.get(key) != value for key, value in required.items()):
        raise ValueError(f"exposure ledger counts drifted: {counts}")
    scope = cast(dict[str, object], ledger["scope"])
    if any(
        scope.get(key) for key in ("outcomes_read", "pilot_roster_selected")
    ):
        raise ValueError("exposure ledger is outcome-contaminated")


def _remaining_near_duplicates(ledger: dict[str, object]) -> set[int]:
    result: set[int] = set()
    for item in cast(
        list[dict[str, object]], ledger["near_duplicate_candidates"]
    ):
        if item["pair_kind"] == "remaining_vs_remaining":
            result.update(
                (cast(int, item["left_key"]), cast(int, item["right_key"]))
            )
    return result


def _selection_key(row: dict[str, object]) -> tuple[str, int]:
    digest = hashlib.sha256(
        f"{SEED_STRING}|{row['normalized_prompt_utf8_sha256']}".encode(
            "ascii"
        )
    ).hexdigest()
    return digest, cast(int, row["official_key"])


def _ledger_prompts(ledger: dict[str, object]) -> list[dict[str, object]]:
    prompts = ledger.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("exposure ledger prompts are malformed")
    return [cast(dict[str, object], row) for row in prompts]


def _system_message(manifest: PromptManifest) -> str:
    if not manifest.prompts:
        raise ValueError("engineering manifest is empty")
    messages = manifest.prompts[0].messages
    systems = [
        item["content"] for item in messages if item["role"] == "system"
    ]
    if len(systems) != 1:
        raise ValueError(
            "engineering manifest does not have one system message"
        )
    return systems[0]


def _read_official(raw: bytes) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"official row {line_number} is not an object")
        if not isinstance(value.get("key"), int) or not isinstance(
            value.get("prompt"), str
        ):
            raise ValueError(f"official row {line_number} is malformed")
        ids = value.get("instruction_id_list")
        kwargs = value.get("kwargs")
        if (
            not isinstance(ids, list)
            or not isinstance(kwargs, list)
            or len(ids) != len(kwargs)
        ):
            raise ValueError(
                f"official row {line_number} has malformed scorer metadata"
            )
        rows.append(value)
    if (
        len(rows) != 541
        or len({cast(int, row["key"]) for row in rows}) != 541
    ):
        raise ValueError("official IFEval JSONL must contain 541 unique rows")
    return rows


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise FileNotFoundError(
            f"required pilot source is missing: {path}"
        ) from error


def _read_json(raw: bytes, path: Path) -> dict[str, object]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _source_record(path: Path, raw: bytes) -> dict[str, object]:
    return {"path": str(path), "bytes": len(raw), "sha256": _sha256(raw)}


def _dataset_record(manifest: PromptManifest) -> dict[str, object]:
    return {
        "path": manifest.official_dataset_path,
        "bytes": manifest.official_dataset_bytes,
        "sha256": manifest.official_dataset_sha256,
        "rows": manifest.official_dataset_rows,
    }


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _dump(value: object) -> str:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
