#!/usr/bin/env python3
"""Analyze the frozen EA development rows with grouped, fold-local models."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ACTIONS = (0.05, 0.10, 0.20)
FOLDS = (0, 1, 2, 3)
FEATURES = (
    "severity",
    "cache_length",
    "removed_position_mean_normalized",
    "removed_last128_fraction",
    "removed_sink_fraction",
    "removed_knorm_mass",
    "removed_vnorm_mass",
    "ea_removed_excess_non_sink",
)


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _prompt_id(value):
    if isinstance(value, dict):
        for key in ("prompt_id", "id", "group_id"):
            if value.get(key) is not None:
                return str(value[key])
    if value is None:
        return None
    return str(value)


def _fraction(value):
    if isinstance(value, dict):
        for key in ("removal_fraction", "fraction", "value"):
            if key in value:
                return _fraction(value[key])
    try:
        result = float(value)
    except (TypeError, ValueError):
        text = str(value).lower()
        if ":" in text:
            return _fraction(text.rsplit(":", 1)[1])
        return None
    return result


def _action_fraction(action):
    if isinstance(action, dict):
        return _fraction(action)
    return _fraction(action)


def _action_map(container):
    """Map action keys and action ids to their removal fraction."""
    result = {}
    if isinstance(container, dict):
        items = container.items()
    elif isinstance(container, list):
        items = ((str(index), item) for index, item in enumerate(container))
    else:
        return result
    for key, item in items:
        fraction = _action_fraction(item)
        if fraction is None:
            fraction = _action_fraction(key)
        if fraction is None:
            continue
        result[str(key)] = fraction
        if isinstance(item, dict):
            for name in ("action_id", "id", "name"):
                if item.get(name) is not None:
                    result[str(item[name])] = fraction
    return result


def _resolve_action(container, key):
    direct = _action_map(container)
    if str(key) in direct:
        return direct[str(key)]
    return _fraction(key)


def _manifest_rows(path):
    payload = _json(path)
    if isinstance(payload, dict):
        for key in ("prompts", "rows", "manifest"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(
            "manifest must be a JSON list or contain a prompts/rows list"
        )
    rows = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("manifest entries must be objects")
        prompt_id = _prompt_id(item)
        if prompt_id is None:
            raise ValueError("manifest entry has no id")
        if prompt_id in rows:
            raise ValueError(f"duplicate manifest prompt id: {prompt_id}")
        fold = item.get("development_fold", item.get("fold"))
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError(
                f"manifest fold is not an integer for {prompt_id}"
            )
        rows[prompt_id] = {
            "prompt_id": prompt_id,
            "task": str(item.get("task", item.get("task_id", "unknown"))),
            "fold": fold,
        }
    return rows


def _run_records(run_dirs, manifest_ids):
    records = {}
    failures = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        paths = sorted(root.glob("*.json"))
        for path in paths:
            if path.name == "run.json":
                continue
            try:
                payload = _json(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(
                    {"source": str(path), "reason": f"invalid_json: {exc}"}
                )
                continue
            prompt_id = _prompt_id(payload.get("prompt_id")) or _prompt_id(
                payload.get("manifest_row")
            )
            if prompt_id is None:
                continue
            if prompt_id not in manifest_ids:
                failures.append(
                    {
                        "prompt_id": prompt_id,
                        "source": str(path),
                        "reason": "unknown_prompt",
                    }
                )
                continue
            if prompt_id in records:
                raise ValueError(
                    f"duplicate run record for prompt {prompt_id}"
                )
            status = payload.get("status", "unknown")
            if status != "completed":
                failures.append(
                    {
                        "prompt_id": prompt_id,
                        "source": str(path),
                        "reason": f"run_status:{status}",
                    }
                )
            checks = payload.get("checks")
            feature_checks = payload.get("feature_checks")
            integrity_ok = (
                isinstance(checks, dict)
                and bool(checks)
                and all(value is True for value in checks.values())
                and isinstance(feature_checks, dict)
                and bool(feature_checks)
                and all(
                    isinstance(item, dict) for item in feature_checks.values()
                )
                and all(
                    item.get("head_count_exact") is True
                    and item.get("all_numeric_values_finite") is True
                    for item in feature_checks.values()
                )
            )
            if not integrity_ok:
                failures.append(
                    {
                        "prompt_id": prompt_id,
                        "source": str(path),
                        "reason": "record_integrity_checks_failed",
                    }
                )
            records[prompt_id] = {
                "payload": payload,
                "source": str(path),
                "status": status,
                "integrity_ok": integrity_ok,
            }
    return records, failures


def _score_rows(run_dirs, manifest_ids):
    scores = {}
    failures = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "scores.jsonl"
        if not path.exists():
            failures.append(
                {"source": str(path), "reason": "missing_scores_file"}
            )
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    {
                        "source": str(path),
                        "line": line_number,
                        "reason": f"invalid_json: {exc}",
                    }
                )
                continue
            prompt_id = _prompt_id(
                payload.get("prompt_id", payload.get("id"))
            )
            if prompt_id is None:
                prompt_id = _prompt_id(payload.get("manifest_row"))
            if prompt_id is None:
                failures.append(
                    {
                        "source": str(path),
                        "line": line_number,
                        "reason": "missing_prompt_id",
                    }
                )
                continue
            if prompt_id not in manifest_ids:
                failures.append(
                    {
                        "prompt_id": prompt_id,
                        "source": str(path),
                        "reason": "unknown_prompt",
                    }
                )
                continue
            if prompt_id in scores:
                raise ValueError(
                    f"duplicate score row for prompt {prompt_id}"
                )
            scores[prompt_id] = {"payload": payload, "source": str(path)}
    return scores, failures


def _head_features(record, fraction):
    payload = record["payload"]
    features = payload.get("features", {})
    selected = None
    for key, value in features.items():
        if (
            abs(
                (_resolve_action(payload.get("actions"), key) or -99.0)
                - fraction
            )
            < 1e-8
        ):
            selected = value
            break
    if selected is None:
        raise ValueError(f"no EA feature block for action {fraction}")
    heads = selected.get("heads") if isinstance(selected, dict) else None
    if not isinstance(heads, list) or not heads:
        raise ValueError(f"missing EA feature heads for action {fraction}")
    names = (
        "cache_length",
        "removed_position_mean_normalized",
        "removed_last128_fraction",
        "removed_sink_fraction",
        "removed_knorm_mass",
        "removed_vnorm_mass",
        "ea_removed_excess_non_sink",
    )
    means = {}
    for name in names:
        values = [
            _number(head.get(name), f"{fraction}:{name}") for head in heads
        ]
        means[name] = float(np.mean(values))
    means["severity"] = fraction
    return means


def _score_actions(score_payload):
    result = {}
    for action in score_payload.get("actions", []):
        if not isinstance(action, dict):
            continue
        fraction = _action_fraction(action.get("action", action))
        if fraction is None:
            fraction = _action_fraction(action.get("action_id"))
        if fraction is None or all(
            abs(fraction - wanted) > 1e-8 for wanted in ACTIONS
        ):
            continue
        signed = action.get("signed_d")
        if signed is None:
            continue
        status = action.get("status", score_payload.get("status"))
        if status != "scored":
            raise ValueError(
                f"score status is {status!r} for action {fraction}"
            )
        if fraction in result:
            raise ValueError(f"duplicate score for action {fraction}")
        result[fraction] = _number(signed, f"signed_d:{fraction}")
    return result


def _materialize(manifest, records, scores):
    rows = []
    missing_records = []
    missing_scores = []
    incomplete = []
    for prompt_id, meta in manifest.items():
        record = records.get(prompt_id)
        score = scores.get(prompt_id)
        if record is None:
            missing_records.append(prompt_id)
        if score is None:
            missing_scores.append(prompt_id)
        if (
            record is None
            or score is None
            or record["status"] != "completed"
            or not record["integrity_ok"]
        ):
            continue
        try:
            labels = _score_actions(score["payload"])
            prompt_rows = []
            for fraction in ACTIONS:
                features = _head_features(record, fraction)
                if fraction not in labels:
                    raise ValueError(f"missing score for action {fraction}")
                prompt_rows.append(
                    {
                        **meta,
                        **features,
                        "signed_loss": labels[fraction],
                        "action": fraction,
                    }
                )
            rows.extend(prompt_rows)
        except (KeyError, TypeError, ValueError) as exc:
            incomplete.append({"prompt_id": prompt_id, "reason": str(exc)})
    return rows, {
        "missing_records": missing_records,
        "missing_scores": missing_scores,
        "incomplete_prompts": incomplete,
    }


def _metric(rows, prediction_key):
    by_prompt = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)
    if not by_prompt:
        return {
            "mae": None,
            "signed_bias": None,
            "prompt_count": 0,
            "action_count": 0,
        }
    prompt_mae = [
        float(
            np.mean(
                [abs(r[prediction_key] - r["signed_loss"]) for r in group]
            )
        )
        for group in by_prompt.values()
    ]
    prompt_bias = [
        float(np.mean([r[prediction_key] - r["signed_loss"] for r in group]))
        for group in by_prompt.values()
    ]
    return {
        "mae": float(np.mean(prompt_mae)),
        "signed_bias": float(np.mean(prompt_bias)),
        "prompt_count": len(by_prompt),
        "action_count": sum(len(group) for group in by_prompt.values()),
    }


def _spearman(rows, prediction_key):
    result = {}
    for task in sorted({row["task"] for row in rows}):
        subset = [row for row in rows if row["task"] == task]
        if len(subset) < 2:
            result[task] = {"rho": None, "n": len(subset)}
            continue
        correlation = spearmanr(
            [r[prediction_key] for r in subset],
            [r["signed_loss"] for r in subset],
        )
        rho = float(correlation.statistic)
        result[task] = {
            "rho": rho if math.isfinite(rho) else None,
            "n": len(subset),
        }
    return result


def _fit_oof(rows, feature_names):
    by_fold = {}
    predictions = []
    folds_seen = sorted({row["fold"] for row in rows})
    for fold in folds_seen:
        train = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        if not train or not test:
            raise ValueError(f"fold {fold} has empty train or test set")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            np.asarray(
                [[row[name] for name in feature_names] for row in train],
                dtype=float,
            )
        )
        x_test = scaler.transform(
            np.asarray(
                [[row[name] for name in feature_names] for row in test],
                dtype=float,
            )
        )
        model = Ridge(alpha=1.0)
        model.fit(
            x_train,
            np.asarray([row["signed_loss"] for row in train], dtype=float),
        )
        fold_predictions = model.predict(x_test)
        for row, prediction in zip(test, fold_predictions, strict=False):
            copy = dict(row)
            copy["prediction"] = float(prediction)
            predictions.append(copy)
        by_fold[str(fold)] = {
            "train_prompts": len({r["prompt_id"] for r in train}),
            "test_prompts": len({r["prompt_id"] for r in test}),
        }
    return predictions, by_fold


def analyze(manifest_path, run_dirs, output_dir):
    manifest = _manifest_rows(manifest_path)
    records, record_failures = _run_records(run_dirs, set(manifest))
    scores, score_failures = _score_rows(run_dirs, set(manifest))
    rows, coverage = _materialize(manifest, records, scores)
    complete_prompts = sorted({row["prompt_id"] for row in rows})
    by_prompt = {
        prompt_id: sum(row["prompt_id"] == prompt_id for row in rows)
        for prompt_id in complete_prompts
    }
    eligible_ids = sorted(
        prompt_id
        for prompt_id, count in by_prompt.items()
        if count == len(ACTIONS)
    )
    eligible = [row for row in rows if row["prompt_id"] in eligible_ids]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    model_metrics = {}
    model_spearman = {}
    if eligible and {row["fold"] for row in eligible} == set(FOLDS):
        for model_name, feature_names in (
            ("baseline", FEATURES[:-1]),
            ("ea", FEATURES),
        ):
            predicted, _fold_info = _fit_oof(eligible, feature_names)
            model_metrics[model_name] = {
                "pooled": _metric(predicted, "prediction"),
                "folds": {},
            }
            model_spearman[model_name] = _spearman(predicted, "prediction")
            for fold in FOLDS:
                model_metrics[model_name]["folds"][str(fold)] = _metric(
                    [r for r in predicted if r["fold"] == fold], "prediction"
                )
            for row in predicted:
                prediction_rows.append(
                    {
                        "prompt_id": row["prompt_id"],
                        "task": row["task"],
                        "fold": row["fold"],
                        "action": row["action"],
                        "signed_loss": row["signed_loss"],
                        f"{model_name}_prediction": row["prediction"],
                    }
                )
    table_path = output / "predictions.csv"
    fields = [
        "prompt_id",
        "task",
        "fold",
        "action",
        "signed_loss",
        "baseline_prediction",
        "ea_prediction",
    ]
    merged = {}
    for row in prediction_rows:
        key = (row["prompt_id"], row["action"])
        merged.setdefault(
            key, {key_name: row.get(key_name) for key_name in fields[:-2]}
        )
        for key_name in fields[-2:]:
            if key_name in row:
                merged[key][key_name] = row[key_name]
    table_rows = list(merged.values())
    for row in table_rows:
        if not all(
            key in row and math.isfinite(float(row[key]))
            for key in ("baseline_prediction", "ea_prediction")
        ):
            raise ValueError("merged predictions are incomplete or nonfinite")
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in table_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    baseline_mae = (
        model_metrics.get("baseline", {}).get("pooled", {}).get("mae")
    )
    ea_mae = model_metrics.get("ea", {}).get("pooled", {}).get("mae")
    gain = (
        None
        if baseline_mae in (None, 0) or ea_mae is None
        else (baseline_mae - ea_mae) / baseline_mae
    )
    wins = None
    if model_metrics:
        wins = sum(
            model_metrics["ea"]["folds"][str(fold)]["mae"]
            < model_metrics["baseline"]["folds"][str(fold)]["mae"]
            for fold in FOLDS
        )
    controls = {
        "manifest_ids_unique": len(manifest) == len(set(manifest)),
        "all_manifest_ids_have_run_record": not coverage["missing_records"],
        "all_manifest_ids_have_score_row": not coverage["missing_scores"],
        "no_record_failures": not record_failures,
        "no_score_failures": not score_failures,
        "all_prompts_have_three_actions": len(eligible_ids) == len(manifest),
        "all_four_folds_present": {meta["fold"] for meta in manifest.values()}
        == set(FOLDS),
        "all_models_fit": bool(model_metrics)
        and set(model_metrics) == {"baseline", "ea"},
    }
    go = {
        "ten_percent_pooled_mae_gain": gain is not None and gain >= 0.10,
        "three_fold_mae_wins": wins is not None and wins >= 3,
        "positive_spearman_each_task": bool(model_spearman.get("ea"))
        and all(
            item["rho"] is not None and item["rho"] > 0
            for item in model_spearman["ea"].values()
        ),
        "all_controls": all(controls.values()),
    }
    summary = {
        "schema_version": "ea_development_analysis.v1",
        "manifest": str(manifest_path),
        "run_dirs": [str(path) for path in run_dirs],
        "expected_prompt_count": len(manifest),
        "eligible_prompt_count": len(eligible_ids),
        "coverage": {
            **coverage,
            "record_failures": record_failures,
            "score_failures": score_failures,
            "observed_prompt_ids": sorted(set(records) | set(scores)),
        },
        "features": {"baseline": list(FEATURES[:-1]), "ea": list(FEATURES)},
        "metrics": model_metrics,
        "spearman_oof_signed_loss_by_task": model_spearman,
        "pooled_ea_mae_gain": gain,
        "ea_fold_wins": wins,
        "controls": controls,
        "go": go,
        "artifacts": {
            "predictions_jsonl": str(prediction_path),
            "predictions_csv": str(table_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--run-dir", required=True, action="append", type=Path
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.manifest, args.run_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "go": summary["go"],
            },
            sort_keys=True,
        )
    )
    return 0 if all(summary["go"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
