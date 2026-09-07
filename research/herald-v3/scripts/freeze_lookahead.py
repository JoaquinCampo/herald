#!/usr/bin/env python3
"""Freeze the reviewed H8 protocol after implementation is final."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import build_lookahead_manifests as manifests
import evaluate_lookahead as evaluator

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPTS = (
    "build_lookahead_manifests.py",
    "freeze_lookahead.py",
    "collect_lookahead.py",
    "evaluate_lookahead.py",
    "collect_lookahead_outcomes.py",
    "verify_official_scores.py",
)


def freeze(output: Path) -> dict[str, object]:
    manifests.build(output)
    train = manifests.read(output / "train-prompts.json")
    test = manifests.read(output / "test-prompts.json")
    reuse = manifests.read(output / "training-reuse.json")
    first = reuse["rows"][0]
    source_paths = sorted(
        (ROOT / "src/herald_v3/engineering").rglob("*.py")
    ) + [ROOT / "scripts" / name for name in SOURCE_SCRIPTS]
    source_hashes = {
        str(path.relative_to(ROOT)): manifests.sha(path)
        for path in source_paths
    }
    official = ROOT / "results/engineering/official-scorer"
    scorer = manifests.read(official / "source-manifest.json")
    for name, entry in scorer["files"].items():
        path = (
            official / name
            if name == "input_data.jsonl"
            else official / "instruction_following_eval" / name
        )
        if manifests.sha(path) != entry["sha256"]:
            raise ValueError(f"official scorer source drift: {name}")
    prompt_refs = {}
    for phase, document in (("train", train), ("test", test)):
        path = output / f"{phase}-prompts.json"
        prompt_refs[phase] = {
            "path": path.name,
            "file_sha256": manifests.sha(path),
            "manifest_sha256": document["manifest_sha256"],
            "prompt_ids": [p["prompt_id"] for p in document["prompts"]],
        }
    columns = list(evaluator.BLOCK_FEATURES["B4_L"])
    lock = {
        "schema_version": "herald_v3.lookahead_protocol_lock.v1",
        "status": "frozen_implementation_pending_owner_collection_approval",
        "manifests": prompt_refs,
        "training_reuse": {
            "path": "training-reuse.json",
            "sha256": manifests.sha(output / "training-reuse.json"),
        },
        "boundary": {
            "committed_output_tokens": 32,
            "first_affected_prediction_output_index": 32,
            "pending_token_output_index": 31,
        },
        "actions": ["knorm:0.25", "knorm:0.5"],
        "configuration": {
            **first["configuration"],
            "max_lookahead_steps": 8,
        },
        "features": {
            "columns": columns,
            "dtypes": {name: "float64" for name in columns},
            "model_columns": {
                name: list(values)
                for name, values in evaluator.BLOCK_FEATURES.items()
            },
            "mean_delayed_js": "mean(JS_1..JS_(L-1)) if L>1 else0",
            "lookahead_steps": "L=1..8, stop after reference EOS",
            "has_delayed": "float(L>1)",
            "reference_prefix": (
                "same freshly generated reference greedy inputs "
                "for both actions"
            ),
            "missing_values": "forbidden",
        },
        "estimator": {
            "scaler": "StandardScaler",
            "scaler_parameters": {"with_mean": True, "with_std": True},
            "regressor": "Ridge",
            "ridge_alpha": 1.0,
            "ridge_parameters": {
                "alpha": 1.0,
                "fit_intercept": True,
                "copy_X": True,
                "max_iter": None,
                "tol": 0.0001,
                "solver": "auto",
                "positive": False,
                "random_state": None,
            },
            "prediction_clip": [-1.0, 1.0],
            "targets": ["loose", "strict"],
            "train_prompt_count": 120,
            "train_action_rows": 240,
            "training": "once on all240 rows, no test transforms or refit",
            "baseline": "action_mean over all120 training prompts per action",
        },
        "evaluation": {
            "primary": "prompt_equal_MSE_loose",
            "sensitivity": "strict fit with same specification",
            "bootstrap": {
                "replicates": 2000,
                "seed": 0,
                "quantiles": [0.025, 0.975],
                "unit": "paired prompt with multiplicity",
            },
            "minimum_nonzero_loose_prompts": 20,
            "minimum_relative_gain_each_comparator": 0.05,
            "positive_interval_lower_bound": 0.0,
            "zero_comparator_mse": "skill=null, beaten=false",
            "statuses": [
                "operational_integrity_failure",
                "inconclusive_information_floor",
                "negative",
                "positive_exploratory",
            ],
            "test_candidate_count": 76,
            "early_eos": "immutable ledger, no replacement",
        },
        "model": first["model"],
        "tokenizer": manifests.read(
            ROOT / "data/lookahead-v1/tokenizer-identity.json"
        ),
        "generation_environment": first["environment"],
        "software": {
            "generation": first["environment"]["packages"],
            "evaluation": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scikit-learn")
            },
        },
        "source_manifest": source_hashes,
        "scorer_provenance": scorer,
        "prediction_seal_schema": evaluator.SEAL_SCHEMA_VERSION,
        "outcome_collection_schema": "herald_v3.lookahead_outcomes.v1",
        "outcomes_collected": False,
        "proposal_sha256": manifests.sha(
            ROOT / "docs/research/lookahead-prediction-proposal.md"
        ),
        "roster_derivation_sha256": manifests.sha(
            ROOT / "data/lookahead-v1/roster-derivation.json"
        ),
    }
    target = output / "protocol-lock.json"
    manifests.write(target, lock)
    return {
        "path": str(target),
        "sha256": manifests.sha(target),
        "collection_approved": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/lookahead-v1"
    )
    print(json.dumps(freeze(parser.parse_args().output), sort_keys=True))
