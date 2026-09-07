#!/usr/bin/env python3
"""Build a reproducible IFEval exposure inventory from named local sources.

The output is an inventory for owner review. It does not choose a pilot
roster, filter prompts, read outcomes, or claim that any remaining prompt is
fresh. Source paths are intentionally explicit so a missing or changed source
fails loudly instead of silently changing the ledger.
"""

import argparse
import difflib
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL = (
    PROJECT_ROOT / "results/engineering/official-scorer/input_data.jsonl"
)
DEFAULT_OLD_REFERENCES = Path(
    "/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v2/"
    "results/sweep_tap/llama/ifeval/references"
)
DEFAULT_CURRENT_STATE = Path(
    "/Users/joaquincamponario/orca/workspaces/herald-v2/cero/results/"
    "recovered/current-state-damage-v1"
)
DEFAULT_HIDDEN_STATE = Path(
    "/Users/joaquincamponario/orca/workspaces/herald-v2/cero/results/"
    "recovered/quality-risk-v1/audit-2026-09-04"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/exposure-ledger.json"
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.70
_IFEVAL_ID_RE = re.compile(r"^ifeval[-_](?P<key>[1-9][0-9]*)$")


def main(argv: list[str] | None = None) -> int:
    """Build the ledger and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument(
        "--old-references", type=Path, default=DEFAULT_OLD_REFERENCES
    )
    parser.add_argument(
        "--current-state", type=Path, default=DEFAULT_CURRENT_STATE
    )
    parser.add_argument(
        "--hidden-state", type=Path, default=DEFAULT_HIDDEN_STATE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
        help="SequenceMatcher ratio threshold for review candidates",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.near_duplicate_threshold <= 1.0:
        parser.error("--near-duplicate-threshold must be between 0 and 1")

    ledger = build_ledger(
        official_path=args.official,
        old_references=args.old_references,
        current_state=args.current_state,
        hidden_state=args.hidden_state,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    counts = ledger["counts"]
    print(json.dumps({"output": str(args.output), "counts": counts}))
    return 0


def build_ledger(
    *,
    official_path: Path,
    old_references: Path,
    current_state: Path,
    hidden_state: Path,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, object]:
    """Read the declared sources and return the deterministic ledger."""
    if not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be between 0 and 1")

    official_rows, official_file = _read_official(official_path)
    source_files = [official_file]
    provenance: dict[int, list[dict[str, str]]] = {}

    old_ids, old_files = _read_old_references(old_references, provenance)
    source_files.extend(old_files)

    current_dev_ids, current_dev_file = _read_prompt_list(
        current_state / "development_prompts.json",
        "v2_current_state_development",
        provenance,
    )
    current_confirmation_ids, current_confirmation_file = _read_prompt_list(
        current_state / "confirmation_prompts.json",
        "v2_current_state_confirmation",
        provenance,
    )
    source_files.extend((current_dev_file, current_confirmation_file))

    hidden_dev_ids, hidden_dev_file = _read_hidden_inputs(
        hidden_state / "hidden_state_development" / "inputs.json",
        "v2_quality_risk_hidden_state_development",
        provenance,
    )
    hidden_moderate_ids, hidden_moderate_file = _read_hidden_inputs(
        hidden_state / "hidden_state_moderate" / "inputs.json",
        "v2_quality_risk_hidden_state_moderate",
        provenance,
    )
    source_files.extend((hidden_dev_file, hidden_moderate_file))

    official_by_key = {cast(int, row["key"]): row for row in official_rows}
    official_keys: set[int] = set(official_by_key)
    current_ids = current_dev_ids | current_confirmation_ids
    exposed_ids = old_ids | current_ids
    remaining_ids = official_keys - exposed_ids
    _assert_documented_counts(
        official_keys=official_keys,
        old_ids=old_ids,
        current_dev_ids=current_dev_ids,
        current_confirmation_ids=current_confirmation_ids,
        current_ids=current_ids,
        hidden_dev_ids=hidden_dev_ids,
        hidden_moderate_ids=hidden_moderate_ids,
        exposed_ids=exposed_ids,
        remaining_ids=remaining_ids,
    )

    prompt_ledger = []
    for row in official_rows:
        key = cast(int, row["key"])
        prompt = cast(str, row["prompt"])
        records = sorted(
            provenance.get(key, []),
            key=lambda item: (item["source_label"], item["source_id"]),
        )
        prompt_ledger.append(
            {
                "official_key": key,
                "official_prompt_id": f"ifeval_{key}",
                "prompt_utf8_sha256": _sha256_text(prompt),
                "normalized_prompt_utf8_sha256": _sha256_text(
                    _normalize_prompt(prompt)
                ),
                "instruction_ids": list(
                    cast(list[str], row["instruction_id_list"])
                ),
                "instruction_count": len(
                    cast(list[str], row["instruction_id_list"])
                ),
                "exposed": key in exposed_ids,
                "source_labels": sorted(
                    {item["source_label"] for item in records}
                ),
                "source_records": records,
            }
        )

    near_duplicates = _near_duplicates(
        official_by_key,
        exposed_ids=exposed_ids,
        remaining_ids=remaining_ids,
        threshold=near_duplicate_threshold,
    )
    key_sets = {
        "official": sorted(official_keys),
        "exposed": sorted(exposed_ids),
        "remaining": sorted(remaining_ids),
    }
    return {
        "schema_version": "herald_v3.exposure_ledger.v1",
        "builder": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "python": sys.version,
        },
        "scope": {
            "purpose": "exposure inventory for owner review",
            "outcomes_read": False,
            "pilot_roster_selected": False,
            "freshness_decision_made": False,
        },
        "source_files": source_files,
        "counts": {
            "official_ifeval": len(official_keys),
            "old_reference_ids": len(old_ids),
            "current_state_development_ids": len(current_dev_ids),
            "current_state_confirmation_ids": len(current_confirmation_ids),
            "current_state_union_ids": len(current_ids),
            "old_current_overlap_ids": len(old_ids & current_ids),
            "exposed_union_ids": len(exposed_ids),
            "remaining_official_ids": len(remaining_ids),
            "hidden_state_development_ids": len(hidden_dev_ids),
            "hidden_state_moderate_ids": len(hidden_moderate_ids),
            "near_duplicate_candidates": len(near_duplicates),
        },
        "assertions": {
            "official_ifeval_is_541": len(official_keys) == 541,
            "exposed_union_is_321": len(exposed_ids) == 321,
            "remaining_is_220": len(remaining_ids) == 220,
            "hidden_collections_identical": hidden_dev_ids
            == hidden_moderate_ids,
            "hidden_collections_are_current_subset": hidden_dev_ids
            <= current_ids,
        },
        "key_sets": key_sets,
        "near_duplicate_method": {
            "library": "difflib.SequenceMatcher",
            "normalization": (
                "unicodedata.normalize('NFKC'), collapse whitespace, strip"
            ),
            "comparison": (
                "normalized official text, remaining IDs against exposed IDs"
            ),
            "threshold": near_duplicate_threshold,
            "autojunk": False,
            "use": "review candidates only; no prompt is filtered",
        },
        "near_duplicate_candidates": near_duplicates,
        "prompts": prompt_ledger,
    }


def _read_official(
    path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    lines = _read_text(path).splitlines()
    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines, start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"official row {index} is not an object")
        key = value.get("key")
        prompt = value.get("prompt")
        ids = value.get("instruction_id_list")
        kwargs = value.get("kwargs")
        if (
            isinstance(key, bool)
            or not isinstance(key, int)
            or not isinstance(prompt, str)
            or not isinstance(ids, list)
            or not isinstance(kwargs, list)
            or len(ids) != len(kwargs)
            or not all(isinstance(item, str) for item in ids)
        ):
            raise ValueError(f"official row {index} is malformed")
        rows.append(
            {
                "key": key,
                "prompt": prompt,
                "instruction_id_list": ids,
                "kwargs": kwargs,
            }
        )
    if len({cast(int, row["key"]) for row in rows}) != len(rows):
        raise ValueError("official input_data.jsonl contains duplicate keys")
    return rows, _file_record(path, "official_ifeval_input_data")


def _read_old_references(
    directory: Path,
    provenance: dict[int, list[dict[str, str]]],
) -> tuple[set[int], list[dict[str, object]]]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"no old reference JSON files under {directory}"
        )
    ids: set[int] = set()
    records: list[dict[str, object]] = []
    for path in files:
        value = _read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"old reference is not an object: {path}")
        source_id = value.get("prompt_id")
        if not isinstance(source_id, str):
            raise ValueError(f"old reference has no prompt_id: {path}")
        key = _parse_ifeval_id(source_id, path)
        if key in ids:
            raise ValueError(f"duplicate old reference key {key}")
        ids.add(key)
        provenance.setdefault(key, []).append(
            {
                "source_label": "v2_sweep_tap_ifeval_references",
                "source_id": source_id,
            }
        )
        records.append(
            _file_record(
                path,
                "v2_sweep_tap_ifeval_reference",
                record_count=1,
                extra={"source_id": source_id},
            )
        )
    return ids, records


def _read_prompt_list(
    path: Path,
    source_label: str,
    provenance: dict[int, list[dict[str, str]]],
) -> tuple[set[int], dict[str, object]]:
    value = _read_json(path)
    if not isinstance(value, list):
        raise ValueError(f"prompt source is not a list: {path}")
    ids: set[int] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict) or row.get("task") != "ifeval":
            continue
        source_id = row.get("prompt_id")
        if not isinstance(source_id, str):
            raise ValueError(f"IFEval row {index} has no prompt_id: {path}")
        key = _parse_ifeval_id(source_id, path)
        if key in ids:
            raise ValueError(f"duplicate IFEval key {key} in {path}")
        ids.add(key)
        provenance.setdefault(key, []).append(
            {"source_label": source_label, "source_id": source_id}
        )
    return ids, _file_record(
        path,
        source_label,
        record_count=len(value),
        extra={"ifeval_ids": sorted(ids)},
    )


def _read_hidden_inputs(
    path: Path,
    source_label: str,
    provenance: dict[int, list[dict[str, str]]],
) -> tuple[set[int], dict[str, object]]:
    value = _read_json(path)
    if not isinstance(value, dict) or not isinstance(
        value.get("prompts"), list
    ):
        raise ValueError(f"hidden-state inputs are malformed: {path}")
    ids: set[int] = set()
    for index, row in enumerate(value["prompts"]):
        if not isinstance(row, dict) or row.get("task") != "ifeval":
            continue
        source_id = row.get("prompt_id")
        if not isinstance(source_id, str):
            raise ValueError(f"hidden IFEval row {index} has no prompt_id")
        key = _parse_ifeval_id(source_id, path)
        if key in ids:
            raise ValueError(f"duplicate hidden IFEval key {key} in {path}")
        ids.add(key)
        provenance.setdefault(key, []).append(
            {"source_label": source_label, "source_id": source_id}
        )
    return ids, _file_record(
        path,
        source_label,
        record_count=len(value["prompts"]),
        extra={"ifeval_ids": sorted(ids)},
    )


def _assert_documented_counts(
    *,
    official_keys: set[int],
    old_ids: set[int],
    current_dev_ids: set[int],
    current_confirmation_ids: set[int],
    current_ids: set[int],
    hidden_dev_ids: set[int],
    hidden_moderate_ids: set[int],
    exposed_ids: set[int],
    remaining_ids: set[int],
) -> None:
    if len(official_keys) != 541:
        raise AssertionError(
            f"expected 541 official keys, got {len(official_keys)}"
        )
    if not old_ids <= official_keys:
        raise AssertionError(
            "old reference IDs include unknown official keys"
        )
    if not current_ids <= official_keys:
        raise AssertionError(
            "current-state IDs include unknown official keys"
        )
    if len(old_ids) != 200:
        raise AssertionError(f"expected 200 old IDs, got {len(old_ids)}")
    if len(current_dev_ids) != 133 or len(current_confirmation_ids) != 67:
        raise AssertionError(
            "current-state development/confirmation counts changed"
        )
    if current_dev_ids & current_confirmation_ids:
        raise AssertionError(
            "current-state development and confirmation overlap"
        )
    if len(current_ids) != 200:
        raise AssertionError(
            f"expected 200 current-state IDs, got {len(current_ids)}"
        )
    if len(old_ids & current_ids) != 79:
        raise AssertionError(
            "old/current overlap is not the documented 79 IDs"
        )
    if hidden_dev_ids != hidden_moderate_ids or len(hidden_dev_ids) != 80:
        raise AssertionError(
            "hidden-state collections are not identical 80-ID sets"
        )
    if not hidden_dev_ids <= current_ids:
        raise AssertionError(
            "hidden-state IDs are not a current-state subset"
        )
    if len(exposed_ids) != 321:
        raise AssertionError(
            f"expected 321 exposed IDs, got {len(exposed_ids)}"
        )
    if len(remaining_ids) != 220:
        raise AssertionError(
            f"expected 220 remaining IDs, got {len(remaining_ids)}"
        )


def _near_duplicates(
    official_by_key: dict[int, dict[str, object]],
    *,
    exposed_ids: set[int],
    remaining_ids: set[int],
    threshold: float,
) -> list[dict[str, object]]:
    normalized = {
        key: _normalize_prompt(str(row["prompt"]))
        for key, row in official_by_key.items()
    }
    candidates: list[dict[str, object]] = []
    comparisons = (
        ("remaining_vs_exposed", sorted(remaining_ids), sorted(exposed_ids)),
        (
            "remaining_vs_remaining",
            sorted(remaining_ids),
            sorted(remaining_ids),
        ),
    )
    for pair_kind, left_keys, right_keys in comparisons:
        for left_key in left_keys:
            left_prompt = normalized[left_key]
            for right_key in right_keys:
                if (
                    pair_kind == "remaining_vs_remaining"
                    and right_key <= left_key
                ):
                    continue
                right_prompt = normalized[right_key]
                matcher = difflib.SequenceMatcher(
                    None, left_prompt, right_prompt, autojunk=False
                )
                if matcher.real_quick_ratio() < threshold:
                    continue
                score = matcher.ratio()
                if score < threshold:
                    continue
                candidates.append(
                    {
                        "pair_kind": pair_kind,
                        "left_key": left_key,
                        "left_prompt_id": f"ifeval_{left_key}",
                        "right_key": right_key,
                        "right_prompt_id": f"ifeval_{right_key}",
                        "sequence_matcher_ratio": score,
                    }
                )
    candidates.sort(
        key=lambda item: (
            -cast(float, item["sequence_matcher_ratio"]),
            str(item["pair_kind"]),
            cast(int, item["left_key"]),
            cast(int, item["right_key"]),
        )
    )
    return candidates


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", prompt)).strip()


def _parse_ifeval_id(value: str, path: Path) -> int:
    match = _IFEVAL_ID_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"malformed IFEval prompt ID {value!r} in {path}")
    return int(match.group("key"))


def _read_json(path: Path) -> object:
    return json.loads(_read_text(path))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(
            f"required exposure source is missing: {path}"
        ) from error


def _file_record(
    path: Path,
    label: str,
    *,
    record_count: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "label": label,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if record_count is not None:
        result["record_count"] = record_count
    if extra:
        result.update(extra)
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
