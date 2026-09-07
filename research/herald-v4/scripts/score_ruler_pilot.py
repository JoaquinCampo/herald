#!/usr/bin/env python3
"""Apply the pinned NVIDIA/RULER synthetic scorer to pilot predictions."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RULER = ROOT / "vendor" / "ruler"
OFFICIAL_EVAL = RULER / "scripts" / "eval" / "evaluate.py"
OFFICIAL_CONSTANTS = RULER / "scripts" / "eval" / "synthetic" / "constants.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned RULER source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_official_postprocess() -> Any:
    source = OFFICIAL_EVAL.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(OFFICIAL_EVAL))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "postprocess_pred"
    )
    namespace = {"re": re}
    compiled = compile(ast.Module(body=[function], type_ignores=[]), str(OFFICIAL_EVAL), "exec")
    exec(compiled, namespace)
    return namespace["postprocess_pred"]


_OFFICIAL_CONSTANTS = _load_module(OFFICIAL_CONSTANTS, "ruler_pilot_official_constants")
_OFFICIAL_POSTPROCESS = _load_official_postprocess()


def score_prediction(prediction: str, answers: list[str]) -> dict[str, Any]:
    """Return exact pinned RULER postprocessing, score, and answer hits."""
    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if not answers:
        raise ValueError("answers must contain at least one reference")
    postprocessed = _OFFICIAL_POSTPROCESS(prediction, {})
    pass_vector = [answer.lower() in postprocessed.lower() for answer in answers]
    score = _OFFICIAL_CONSTANTS.string_match_all([postprocessed], [answers])
    return {
        "raw_prediction": prediction,
        "postprocessed_prediction": postprocessed,
        "score": score,
        "score_fraction": score / 100.0,
        "pass_vector": pass_vector,
        "null": postprocessed == "",
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return rows
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def score_file(manifest_path: Path, predictions_path: Path, output_path: Path) -> None:
    manifest = _read_rows(manifest_path)
    predictions = _read_rows(predictions_path)
    by_id = {}
    for row in predictions:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in by_id:
            raise ValueError("Predictions need unique string id fields")
        by_id[row_id] = row

    output = []
    for row in manifest:
        row_id = row["id"]
        prediction_row = by_id.get(row_id)
        if prediction_row is None:
            output.append(
                {
                    "id": row_id,
                    "task": row["task"],
                    "status": "missing_prediction",
                    "raw_prediction": None,
                    "postprocessed_prediction": None,
                    "score": None,
                    "score_fraction": None,
                    "pass_vector": None,
                }
            )
            continue
        prediction = prediction_row.get("prediction", prediction_row.get("pred"))
        if prediction is None:
            prediction = prediction_row.get("output")
        try:
            scored = score_prediction(prediction, row["answers"])
            output.append({"id": row_id, "task": row["task"], "status": "scored", **scored})
        except Exception as exc:
            output.append(
                {
                    "id": row_id,
                    "task": row["task"],
                    "status": "scoring_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_prediction": prediction,
                    "postprocessed_prediction": None,
                    "score": None,
                    "score_fraction": None,
                    "pass_vector": None,
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_pair_run(run_dir: Path, manifest_path: Path, output_path: Path) -> None:
    """Score each reference/action arm in a frozen ``run_pair_pilot`` output."""
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    manifest = _read_rows(manifest_path)
    prompt_entries = {item["id"]: item for item in run.get("prompts", [])}
    scorer = {
        "evaluate_sha256": _sha256(OFFICIAL_EVAL),
        "constants_sha256": _sha256(OFFICIAL_CONSTANTS),
    }
    output = []
    for row in manifest:
        row_id = row["id"]
        entry = prompt_entries.get(row_id)
        if entry is None:
            output.append(
                {
                    "id": row_id,
                    "task": row["task"],
                    "status": "missing_prompt_record",
                    "actions": [],
                    "scorer": scorer,
                }
            )
            continue
        prompt_path = run_dir / entry["path"]
        if not prompt_path.is_file():
            output.append(
                {
                    "id": row_id,
                    "task": row["task"],
                    "status": "missing_prompt_file",
                    "path": str(prompt_path),
                    "actions": [],
                    "scorer": scorer,
                }
            )
            continue
        record = json.loads(prompt_path.read_text(encoding="utf-8"))
        recorded_id = record.get("manifest_row", {}).get("id")
        if recorded_id != row_id:
            raise ValueError(f"manifest_row.id mismatch for {row_id}: {recorded_id}")
        if record.get("status") != "completed":
            output.append(
                {
                    "id": row_id,
                    "task": row["task"],
                    "status": "run_failed",
                    "failure": record.get("failure"),
                    "actions": [],
                    "scorer": scorer,
                }
            )
            continue
        reference = record.get("reference")
        reference_text = (
            reference.get("text") if isinstance(reference, dict) else None
        )
        if not isinstance(reference_text, str):
            output.append(
                {
                    "id": row_id,
                    "task": row["task"],
                    "status": "missing_reference",
                    "actions": [],
                    "scorer": scorer,
                }
            )
            continue
        reference_scoring = score_prediction(reference_text, row["answers"])
        scored_actions = []
        arms = record.get("arms")
        if not isinstance(arms, dict):
            arms = {}
        for action_id, arm in arms.items():
            if not isinstance(arm, dict):
                arm = {}
            action = arm.get("action", {"action_id": action_id})
            continuation = arm.get("continuation")
            action_text = (
                continuation.get("text")
                if isinstance(continuation, dict)
                else None
            )
            base = {
                "action_id": action_id,
                "action": action,
                "reference_scoring": reference_scoring,
            }
            if not isinstance(action_text, str):
                scored_actions.append(
                    {
                        **base,
                        "status": "missing_action_continuation",
                        "action_scoring": None,
                        "q_ref": reference_scoring["score_fraction"],
                        "q_action": None,
                        "signed_d": None,
                    }
                )
                continue
            try:
                action_scoring = score_prediction(action_text, row["answers"])
                q_ref = reference_scoring["score_fraction"]
                q_action = action_scoring["score_fraction"]
                scored_actions.append(
                    {
                        **base,
                        "status": "scored",
                        "action_scoring": action_scoring,
                        "q_ref": q_ref,
                        "q_action": q_action,
                        "signed_d": q_ref - q_action,
                    }
                )
            except Exception as exc:
                scored_actions.append(
                    {
                        **base,
                        "status": "scoring_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "action_scoring": None,
                        "q_ref": reference_scoring["score_fraction"],
                        "q_action": None,
                        "signed_d": None,
                    }
                )
        output.append(
            {
                "id": row_id,
                "task": row["task"],
                "status": "scored" if scored_actions else "missing_actions",
                "actions": scored_actions,
                "scorer": scorer,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--predictions", type=Path, required=False)
    parser.add_argument("--run-dir", type=Path, required=False)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        assert score_prediction("answer 123", ["123"])["pass_vector"] == [True]
        assert score_prediction("answer", ["123"])["pass_vector"] == [False]
        assert score_prediction("123 and 456", ["123", "456"])["score"] == 100.0
        print("official RULER scorer examples passed")
    if args.run_dir is not None:
        if args.manifest is None:
            parser.error("--manifest is required with --run-dir")
        output = args.output or args.run_dir / "scores.jsonl"
        score_pair_run(args.run_dir, args.manifest, output)
        print(f"Wrote {output}")
        return
    if args.manifest is None or args.predictions is None:
        if args.self_check:
            return
        parser.error("--manifest and --predictions are required unless --self-check is used")
    output = args.output or args.predictions.with_name("scores.jsonl")
    score_file(args.manifest, args.predictions, output)
    print(f"Wrote {output}")
    print(f"postprocess_sha256={_sha256(OFFICIAL_EVAL)}")
    print(f"constants_sha256={_sha256(OFFICIAL_CONSTANTS)}")


if __name__ == "__main__":
    main()
