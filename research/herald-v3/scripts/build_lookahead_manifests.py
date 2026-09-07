#!/usr/bin/env python3
"""Derive fixed lookahead inputs from the accepted pilot, without fitting."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import freeze_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    text = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    if path.exists() and path.read_text() != text:
        raise ValueError(
            f"refusing to replace different frozen content: {path}"
        )
    path.write_text(text)


def build(output: Path) -> dict[str, object]:
    old = ROOT / "data/pilot-v1"
    collection = ROOT / "results/pilot-v1"
    checkpoint = read(collection / "checkpoint.json")
    review = read(old / "exposure-review.json")
    ledger = read(ROOT / "data/exposure-ledger.json")
    official_path = pilot.DEFAULT_OFFICIAL
    official = pilot._read_official(official_path.read_bytes())
    original = pilot.load_prompt_manifest(old / "prompts.json", limit=160)
    if not (
        checkpoint["status"] == "completed" and checkpoint["passed"] is True
    ):
        raise ValueError("pilot collection is not complete and passed")
    entries = checkpoint["ledger"]
    if len(entries) != 140 or checkpoint["accepted_eligible"] != 120:
        raise ValueError("unexpected pilot ledger size")
    if [e["prompt_id"] for e in entries] != [
        p.prompt_id for p in original.prompts[:140]
    ]:
        raise ValueError("pilot ledger is not the exact first 140 candidates")
    accepted = [e for e in entries if e["status"] == "accepted"]
    if len(accepted) != 120 or any(
        e["status"] not in {"accepted", "ineligible_early_eos"}
        for e in entries
    ):
        raise ValueError("invalid acceptance ledger")
    training_ids = [e["prompt_id"] for e in accepted]
    remaining = [
        r
        for r in pilot._ledger_prompts(ledger)
        if not r["exposed"] and r["official_key"] not in pilot.OWNER_EXCLUDED
    ]
    ranked = sorted(remaining, key=pilot._selection_key)
    keys = [r["official_key"] for r in ranked]
    if (
        keys != review["selected_keys"] + review["unselected_holdout_keys"]
        or len(keys) != 216
    ):
        raise ValueError("original seeded216 ranking does not reproduce")
    official_by_key = {row["key"]: row for row in official}
    system = pilot._system_message(original)
    rebuilt = tuple(
        pilot._make_prompt(row, official_by_key, system) for row in ranked
    )
    if [p.to_dict() for p in rebuilt[:160]] != [
        p.to_dict() for p in original.prompts
    ]:
        raise ValueError("official prompts or source metadata drifted")
    train = tuple(
        p for p in original.prompts if p.prompt_id in set(training_ids)
    )
    test = rebuilt[140:]
    if (
        len(train) != 120
        or len(test) != 76
        or set(training_ids) & {p.prompt_id for p in test}
    ):
        raise ValueError("invalid train/test partition")
    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name, prompts, rule in [
        (
            "train-prompts.json",
            train,
            (
                "all120accepted pilot prompts in original seeded order, "
                "exposed training"
            ),
        ),
        (
            "test-prompts.json",
            test,
            (
                "original seeded216 rank positions140..215, "
                "all76unused candidates, no replacement"
            ),
        ),
    ]:
        manifest = replace(original, prompts=prompts, selection_rule=rule)
        write(output / name, manifest.to_dict())
        files[name] = sha(output / name)
    reuse_rows = []
    for entry in accepted:
        prompt_id = entry["prompt_id"]
        directory = collection / prompt_id
        artifacts_path = directory / "artifacts.json"
        if sha(artifacts_path) != entry["artifacts_sha256"]:
            raise ValueError(f"artifact index drift: {prompt_id}")
        artifacts = read(artifacts_path)
        for name, digest in artifacts["sha256"].items():
            if Path(name).name != name or sha(directory / name) != digest:
                raise ValueError(f"artifact drift: {prompt_id}/{name}")
        run = read(directory / "run.json")
        if run["passed"] is not True or len(run["results"]) != 1:
            raise ValueError(f"invalid training run: {prompt_id}")
        result = run["results"][0]
        if (
            result["prompt"]["prompt_id"] != prompt_id
            or not result["acceptance"]["passed"]
        ):
            raise ValueError(f"invalid training result: {prompt_id}")
        cache_hashes = {
            arm["compression"]["before_fingerprint"]
            for arm in result["acceptance"]["action_arms"]
        }
        if len(cache_hashes) != 1:
            raise ValueError(
                f"training cache identity disagrees: {prompt_id}"
            )
        boundary_cache_fingerprint = next(iter(cache_hashes))
        reuse_rows.append(
            {
                "prompt_id": prompt_id,
                "artifact_directory": prompt_id,
                "artifacts_sha256": sha(artifacts_path),
                "files_sha256": artifacts["sha256"],
                "prompt": result["prompt"],
                "tokenization": result["tokenization"],
                "boundary": result["acceptance"]["boundary"],
                "boundary_cache_fingerprint": boundary_cache_fingerprint,
                "configuration": run["configuration"],
                "model": run["model"],
                "environment": run["environment"],
                "source_manifest": run["source_manifest"],
            }
        )
    reuse = {
        "schema_version": "herald_v3.lookahead_training_reuse.v1",
        "training_prompt_ids": training_ids,
        "expected_prompt_count": 120,
        "expected_action_rows": 240,
        "collection_root_relative_to_project": "results/pilot-v1",
        "checkpoint_sha256": sha(collection / "checkpoint.json"),
        "experiment_lock_sha256": sha(collection / "experiment-lock.json"),
        "pilot_manifest_sha256": sha(old / "prompts.json"),
        "pilot_design_lock_sha256": sha(old / "lock.json"),
        "rows": reuse_rows,
        "labels_embedded": False,
        "mismatch_policy": (
            "stop whole study before fitting; no subset or replacement"
        ),
    }
    write(output / "training-reuse.json", reuse)
    files["training-reuse.json"] = sha(output / "training-reuse.json")
    summary = {
        "train_count": 120,
        "test_candidate_count": 76,
        "files_sha256": files,
        "collection_approved": False,
    }
    write(output / "manifest-build.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/lookahead-v1"
    )
    print(json.dumps(build(parser.parse_args().output), sort_keys=True))
