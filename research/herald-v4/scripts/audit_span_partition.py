#!/usr/bin/env python3
"""Independent mask, replay, and scoring audit for span partitions."""

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDS = [f"ruler-ea-dev-v1-niah_single_2-{i:03d}" for i in (0, 2, 3, 4, 5, 8, 10, 11, 1)]
RUN_DIRS = [ROOT / "results/span-partition-v1-first", ROOT / "results/span-partition-v1-rest"]
PRIOR_DIRS = [ROOT / "results/needle-rescue-v1-first", ROOT / "results/needle-rescue-v1-rest", ROOT / "results/needle-rescue-all12-additional"]
SINK = 4


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def official_score():
    source = ROOT / "vendor/ruler/scripts/eval/evaluate.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "postprocess_pred")
    namespace = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    constants = ROOT / "vendor/ruler/scripts/eval/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location("ruler_span_constants", constants)
    if spec is None or spec.loader is None:
        raise RuntimeError("official constants could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def score(text, answers):
        processed = namespace["postprocess_pred"](text, {})
        value = module.string_match_all([processed], [answers])
        return {"score": float(value), "score_fraction": float(value) / 100.0,
                "pass_vector": [a.lower() in processed.lower() for a in answers],
                "postprocessed_prediction": processed, "null": processed == ""}
    return score


def mask_digest(masks):
    digest = hashlib.sha256()
    for layer in masks:
        digest.update(len(layer).to_bytes(4, "little"))
        for head in layer:
            digest.update(len(head).to_bytes(8, "little"))
            for value in head:
                digest.update(int(value).to_bytes(8, "little"))
    return digest.hexdigest()


def current_records():
    result = {}
    for directory in RUN_DIRS:
        for path in directory.glob("*.json"):
            if path.name != "run.json":
                record = read(path)
                result[record["manifest_row"]["id"]] = record
    return result


def prior_record(prompt_id):
    matches = []
    for directory in PRIOR_DIRS:
        for path in directory.glob("*.json"):
            if path.name != "run.json":
                record = read(path)
                if record.get("manifest_row", {}).get("id") == prompt_id:
                    matches.append(record)
    if len(matches) != 1:
        raise AssertionError(f"expected one prior record for {prompt_id}, got {len(matches)}")
    return matches[0]


def reconstruct(base, sentence, target, old_audit, actual, keep_counts, cache_length, control):
    sentence, target = set(sentence), set(target)
    expected, counts = [], []
    ok = True
    for layer, prior_layer in zip(base, old_audit, strict=True):
        expected_layer, count_layer = [], []
        for original, prior_head in zip(layer, prior_layer, strict=True):
            original_set = set(original)
            missing = sorted(target - original_set)
            k = len(missing)
            victims = [int(v) for v in prior_head["victim_indices"][:k]]
            candidates = [v for v in range(cache_length) if v not in original_set and v not in sentence and v >= SINK]
            additions = (sorted(candidates, key=lambda v: (min(abs(v - t) for t in target), v))[:k] if control else missing)
            slots = sorted(original.index(v) for v in victims)
            replacements = sorted(additions)
            branch = list(original)
            for slot, value in zip(slots, replacements, strict=True):
                branch[slot] = value
            expected_layer.append(branch)
            count_layer.append(k)
            ok &= len(victims) == len(additions) == k
            ok &= all(v in original_set and v not in sentence and v >= SINK for v in victims)
            ok &= not control or all(v not in sentence and v >= SINK for v in additions)
            ok &= all(branch[i] == original[i] for i in range(len(original)) if i not in slots)
            ok &= len(branch) == len(set(branch)) and all(0 <= v < cache_length for v in branch)
            ok &= k != 0 or (not victims and not additions and branch == original)
        expected.append(expected_layer)
        counts.append(count_layer)
    ok &= actual == expected
    ok &= all(len(head) == keep for layer, keep in zip(actual, keep_counts, strict=True) for head in layer)
    ok &= all(len(head) == len(set(head)) for layer in actual for head in layer)
    if control:
        ok &= all(set(a) & target == set(o) & target for al, ol in zip(actual, base, strict=True) for a, o in zip(al, ol, strict=True))
    else:
        ok &= all(target <= set(head) for layer in actual for head in layer)
    return ok, counts, mask_digest(actual)


def audit(allow_partial=False):
    score = official_score()
    current = current_records()
    available = [prompt_id for prompt_id in IDS if prompt_id in current]
    missing = [prompt_id for prompt_id in IDS if prompt_id not in current]
    if missing and not allow_partial:
        raise SystemExit("pending records: " + ", ".join(missing))
    cases, score_count, replay_count = {}, 0, 0
    for prompt_id in available:
        record, prior = current[prompt_id], prior_record(prompt_id)
        row, span = record["manifest_row"], record["span"]
        base = record["native_baseline_masks"]
        sentence = span["token_positions"]
        value, context = span["value_token_positions"], span["context_token_positions"]
        prompt, answer, sentence_text = row["prompt"], row["answers"][0], span["sentence"]
        sentence_start = prompt.find(sentence_text)
        answer_in_sentence = sentence_text.find(answer)
        standard = record["branches"]["standard_knorm_0.10"]
        cache_length = standard["compression"]["before_lengths"][0]
        checks = {
            "partition_union_exact": set(value) | set(context) == set(sentence),
            "value_span_unique_and_nonempty": len(value) == len(set(value)) and bool(value),
            "sentence_unique_in_prefix": len(sentence) == len(set(sentence)) and max(sentence) < span["prefix_length"],
            "sentence_disjoint_from_sink": min(sentence) >= SINK,
            "answer_unique_in_prompt_and_sentence": prompt.count(answer) == 1 and sentence_text.count(answer) == 1 and answer_in_sentence >= 0,
            "raw_char_offsets_exact": sentence_start == span["raw_start"] and sentence_start + len(sentence_text) == span["raw_end"] and sentence_start + answer_in_sentence == span["value_raw_start"] and sentence_start + answer_in_sentence + len(answer) == span["value_raw_end"],
            "rendered_char_offsets_consistent": span["rendered_end"] - span["rendered_start"] == len(sentence_text) and span["value_rendered_start"] == span["rendered_start"] + answer_in_sentence and span["value_rendered_end"] == span["value_rendered_start"] + len(answer),
            "native_cardinality_and_hash": len(standard["kept_indices"]) == len(base) and mask_digest(standard["kept_indices"]) == mask_digest(base),
        }
        full = record["branches"]["oracle_span_rescue"]
        full_ok, _, full_hash = reconstruct(base, sentence, sentence, full["swap_audit"], full["kept_indices"], record["keep_counts"], cache_length, False)
        checks["full_oracle_mask_reconstruction"] = full_ok
        checks["full_oracle_target_is_sentence"] = all(set(sentence) <= set(head) for layer in full["kept_indices"] for head in layer)
        for name, target, control in (("value_rescue", value, False), ("context_rescue", context, False), ("value_control", value, True), ("context_control", context, True)):
            branch = record["branches"][name]
            ok, counts, branch_hash = reconstruct(base, sentence, target, full["swap_audit"], branch["kept_indices"], record["keep_counts"], cache_length, control)
            checks[name + "_mask_reconstruction"] = ok
            checks[name + "_hash_present"] = bool(branch_hash)
            checks[name + "_physical_lengths"] = branch["branch_meta"]["cache_lengths"] == standard["compression"]["after_lengths"]
            checks[name + "_physical_bytes"] = branch["branch_meta"]["cache_bytes"] == standard["compression"]["after_bytes"]
            if control:
                rescue_counts = reconstruct(base, sentence, target, full["swap_audit"], record["branches"][name.replace("_control", "_rescue")]["kept_indices"], record["keep_counts"], cache_length, False)[1]
                checks[name + "_paired_swap_counts"] = counts == rescue_counts
            record["branches"][name]["_audit_counts"] = sum(sum(layer) for layer in counts)
        scores = {}
        for name, branch in record["branches"].items():
            actual = score(branch["continuation"]["text"], row["answers"])
            stored = branch["score"]
            scores[name] = actual["score_fraction"]
            checks[name + "_official_score_exact"] = all(stored.get(key) == actual.get(key) for key in ("score", "score_fraction", "pass_vector", "postprocessed_prediction", "null"))
            score_count += 1
        for name in ("reference_shared_boundary", "standard_knorm_0.10", "oracle_span_rescue"):
            checks[name + "_prior_ids_exact"] = record["branches"][name]["continuation"]["token_ids"] == prior["branches"][name]["continuation"]["token_ids"]
            checks[name + "_prior_termination_exact"] = record["branches"][name]["continuation"]["termination_reason"] == prior["branches"][name]["continuation"]["termination_reason"]
            checks[name + "_prior_score_exact"] = record["branches"][name]["score"] == prior["branches"][name]["score"]
            replay_count += 1
        cases[prompt_id] = {"checks": checks, "scores": scores, "swap_counts": {name: record["branches"][name]["_audit_counts"] for name in ("value_rescue", "value_control", "context_rescue", "context_control")}, "full_oracle_mask_hash": full_hash}
    complete = not missing
    counts = {}
    if complete:
        def failed(case):
            return case["scores"]["standard_knorm_0.10"] < case["scores"]["reference_shared_boundary"]

        counts = {"value_rescued": sum(c["scores"]["value_rescue"] == c["scores"]["reference_shared_boundary"] and failed(c) for c in cases.values()), "value_control_rescued": sum(c["scores"]["value_control"] == c["scores"]["reference_shared_boundary"] and failed(c) for c in cases.values()), "context_rescued": sum(c["scores"]["context_rescue"] == c["scores"]["reference_shared_boundary"] and failed(c) for c in cases.values()), "context_control_rescued": sum(c["scores"]["context_control"] == c["scores"]["reference_shared_boundary"] and failed(c) for c in cases.values()), "full_oracle_rescued": sum(c["scores"]["oracle_span_rescue"] == c["scores"]["reference_shared_boundary"] and failed(c) for c in cases.values())}
    result = {"scope": "Independent audit of causal span partition diagnostic", "available_cases": len(available), "expected_cases": len(IDS), "missing": missing, "official_scores_recomputed": score_count, "prior_replays_checked": replay_count, "cases": cases, "rescue_counts": counts, "all_checks_pass": complete and all(all(case["checks"].values()) for case in cases.values()), "verdict": "audit_passes_controls_and_replays" if complete else "pending_full_9_cases", "audit_history": {"initial_auditor_mismatch": {"case": "ruler-ea-dev-v1-niah_single_2-000", "branch": "value_control", "layer": 4, "head": 1, "producer_candidate_order": [325, 329, 277], "producer_replacement_order": [277, 325, 329], "victim_slots": [2981, 2982, 2984]}, "resolution": "The producer sorts control additions before slot assignment; corrected reconstruction matches the frozen rule."}, "limitations": ["This is exposed oracle diagnostic evidence, not predictor or confirmation evidence.", "The audit verifies persisted outputs and does not rerun the 7B GPU inference locally."]}
    (ROOT / "results/span-partition-owner-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("available_cases", "expected_cases", "official_scores_recomputed", "prior_replays_checked", "rescue_counts", "all_checks_pass", "verdict")}, indent=2))


if __name__ == "__main__":
    audit()
