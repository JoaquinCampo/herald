"""Compare every saved response to an independently pinned IFEval package.

Put the official-scorer directory on PYTHONPATH alongside src when running.
The official package is retained with its source manifest in run evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path

from instruction_following_eval import evaluation_lib

from herald_v3.engineering.scoring import score_ifeval_gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_bytes = args.run.read_bytes()
    run = json.loads(run_bytes)
    official_root = Path(evaluation_lib.__file__).resolve().parent.parent
    official_inputs = {
        item.prompt: item
        for item in evaluation_lib.read_prompt_list(
            official_root / "input_data.jsonl"
        )
    }
    checks = []
    for row in run["results"]:
        if "outputs" not in row:
            continue
        prompt = row["prompt"]
        gold = {
            "prompt": prompt["user_prompt"],
            "instruction_id_list": prompt["instruction_id_list"],
            "kwargs": prompt["kwargs"],
        }
        example = official_inputs[prompt["user_prompt"]]
        assert example.key == prompt["key"]
        assert example.instruction_id_list == prompt["instruction_id_list"]
        assert example.kwargs == [
            {key: value for key, value in kwargs.items() if value is not None}
            for kwargs in prompt["kwargs"]
        ]
        outputs = row["outputs"]
        responses = {"uninterrupted": outputs["uninterrupted"]}
        responses.update(
            {
                f"noop_{i}": value
                for i, value in enumerate(outputs["noop_forks"])
            }
        )
        for group in ("actions", "reverse_actions"):
            responses.update(
                {
                    f"{group}/{key}": value
                    for key, value in outputs[group].items()
                }
            )
        for name, response in responses.items():
            text = response["text"]
            ours = score_ifeval_gold(text, gold).to_dict()
            record = {"prompt_id": prompt["prompt_id"], "arm": name}
            for mode in ("strict", "loose"):
                oracle = getattr(
                    evaluation_lib, f"test_instruction_following_{mode}"
                )(example, {example.prompt: text}).follow_instruction_list
                record[mode] = oracle
                assert oracle == ours[f"{mode}_pass"], record
            if name == "uninterrupted":
                assert ours == row["scores"]["reference"], record
            elif name.startswith("actions/"):
                key = name.split("/", 1)[1]
                pair = row["scores"]["actions"][key]
                assert ours == pair["action"], record
                for mode in ("strict", "loose"):
                    assert pair[f"d_{mode}"] == (
                        pair["reference"][mode] - ours[mode]
                    ), record
            checks.append(record)
    assert checks, "No generated responses found"
    report = {
        "passed": True,
        "run_sha256": hashlib.sha256(run_bytes).hexdigest(),
        "official_source": json.loads(
            (official_root / "source-manifest.json").read_text()
        ),
        "response_count": len(checks),
        "checks": checks,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "response_count": len(checks)}))


if __name__ == "__main__":
    main()
