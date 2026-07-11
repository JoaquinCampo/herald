# pyright: reportAttributeAccessIssue=false, reportCallIssue=false

"""Evaluate the live grace-window run against the mission's three goals.

Goal 1, REPLAY FIDELITY: per compressor, live mean savings and mean
quality cost over the test groups must fall inside the replay's
cluster-bootstrap 95% CI at the same frozen alarm and theta
(fidelity_targets.json).

Goal 2, FRONTIER DOMINANCE: live savings meet or exceed the static
point-rule result (crossfit OOF scores, point tau at eps 0.03,
evaluated through the locked evaluate_frozen_tau) on the same split,
at a realized live cost within budget.

Goal 3, MEASURED LEDGERS: wall-clock overhead of the live episode vs
the uncompressed baseline run of the same prompt, plus peak memory.

Diagnostic: per-(group, s) agreement between live alarm scores and the
replay's recorded scores, restricted to grid points before the live
reference's first divergence from the recorded reference (token-tie
flips make later comparisons trajectory-confounded; see experiment log
entry live_ref_divergence).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from herald.controller_metrics import evaluate_frozen_tau  # noqa: E402
from herald.fleet_selection import calibrate_tau  # noqa: E402
from herald.grace_replay import bootstrap_group_ci  # noqa: E402

STATIC_COMPRESSORS = (
    "expected_attention",
    "knorm",
    "streaming_llm",
)
PARQUET = "results/predictor/switch_dataset_attn.parquet"
ROBUSTNESS = Path("results/predictor/robustness")
EPS = 0.03


def _json_object(text: str, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {source}")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        return _json_object(path.read_text(), source=str(path))
    except OSError as error:
        raise ValueError(f"could not read JSON file {path}") from error


def _finite_float(value: object, *, source: str) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str | int | float):
        raise ValueError(f"expected numeric {source}, got {value!r}")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(
            f"expected numeric {source}, got {value!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"expected finite {source}, got {value!r}")
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as stream:
            for line in stream:
                line = line.strip()
                if line:
                    rows.append(_json_object(line, source=str(path)))
    except OSError as error:
        raise ValueError(f"could not read JSONL file {path}") from error
    return rows


def compressors_from_targets(targets: dict[str, Any]) -> tuple[str, ...]:
    """Return the only compressors valid for this frozen fidelity run."""
    raw = targets.get("compressors")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "fidelity targets must contain non-empty compressors"
        )
    return tuple(sorted(str(compressor) for compressor in raw))


def static_point_baseline(compressor: str) -> dict[str, float] | None:
    """Static point rule at eps 0.03 on canonical split 0, through the
    locked evaluator (feat+hyb_k16 scores from the robustness run)."""
    npz_path = ROBUSTNESS / f"xgb_{compressor}_s0_feat+hyb_k16.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    df = pd.read_parquet(PARQUET)
    df = df[df["task"] == "ifeval"].reset_index(drop=True)
    df3 = df[df["compressor"].isin(STATIC_COMPRESSORS)].reset_index(drop=True)
    rows_value = cast(Any, df3).to_dict(orient="records")
    if not isinstance(rows_value, list):
        raise RuntimeError("pandas did not produce record rows")
    rows = cast(list[dict[str, Any]], rows_value)
    train = [rows[i] for i in d["train_idx"]]
    test = [rows[i] for i in d["test_idx"]]
    cal = calibrate_tau(train, list(d["oof"]), method="point", epsilon=EPS)
    test_scored = [
        {**r, "predicted_dq": _finite_float(s, source="static score")}
        for r, s in zip(test, d["test_scores"], strict=True)
    ]
    report = evaluate_frozen_tau(
        test_scored,
        prediction_key="predicted_dq",
        tau=cal.tau,
        epsilon=EPS,
    )
    return {
        "tau": _finite_float(cal.tau, source="static tau"),
        "savings": _finite_float(
            report["mean_savings"], source="static savings"
        ),
        "cost": _finite_float(report["mean_cost"], source="static cost"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", default="results/live_controller")
    ap.add_argument(
        "--targets",
        default="results/predictor/alarm_bundle/fidelity_targets.json",
    )
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    targets = _read_json_object(Path(args.targets))
    compressors = compressors_from_targets(targets)
    episodes = load_jsonl(Path(args.live_dir) / "episodes.jsonl")
    baselines = {
        b["prompt_id"]: b
        for b in load_jsonl(Path(args.live_dir) / "baseline.jsonl")
    }
    by_comp: dict[str, list[dict[str, Any]]] = {}
    for ep in episodes:
        by_comp.setdefault(ep["compressor"], []).append(ep)

    for comp in compressors:
        eps_list = by_comp.get(comp, [])
        tgt = targets["compressors"][comp]
        rep = tgt["replay_test"]
        n_expected = rep["n_groups"]
        if not eps_list:
            print(f"\n=== {comp}: no live episodes yet ===")
            continue
        sav = np.array([e["savings"] for e in eps_list], dtype=float)
        cost = np.array([e["dq_live"] for e in eps_list], dtype=float)
        prompts = [e["prompt_id"] for e in eps_list]
        # Live-internal protocol: the no-compression counterfactual of
        # a live episode is the LIVE reference (never-commit cost is 0
        # by construction, exactly as in the replay); savings use the
        # live reference length from the baseline pass.
        live_cost = []
        live_sav = []
        for e in eps_list:
            b = baselines.get(e["prompt_id"])
            if e["commit_s"] is None:
                live_cost.append(0.0)
                live_sav.append(0.0)
            else:
                q_ref = (
                    b["q_ref_live"] if b is not None else e["q_ref_recorded"]
                )
                live_cost.append(
                    _finite_float(q_ref, source="live reference quality")
                    - _finite_float(e["q_live"], source="live quality")
                )
                denom = (
                    _finite_float(
                        b["ref_len"], source="live reference length"
                    )
                    if b is not None
                    else None
                )
                live_sav.append(
                    max(0.0, 1.0 - e["commit_s"] / denom)
                    if denom
                    else math.nan
                )
        cost_li = np.array(live_cost, dtype=float)
        sav_li = np.array(live_sav, dtype=float)
        live_sav_ci = bootstrap_group_ci(sav, prompts)
        live_cost_ci = bootstrap_group_ci(cost, prompts)
        lo_s, hi_s = rep["ci95_savings"]
        lo_c, hi_c = rep["ci95_cost"]
        g1_sav = lo_s <= sav.mean() <= hi_s
        g1_cost = lo_c <= cost.mean() <= hi_c

        print(f"\n=== {comp} ({len(eps_list)}/{n_expected} groups) ===")
        print(
            f"GOAL1 savings: live {sav.mean():.4f} "
            f"(live CI [{live_sav_ci[0]:.4f},{live_sav_ci[1]:.4f}]) "
            f"vs replay {rep['mean_savings']:.4f} "
            f"CI [{lo_s:.4f},{hi_s:.4f}] -> "
            f"{'PASS' if g1_sav else 'FAIL'}"
        )
        print(
            f"GOAL1 cost:    live {cost.mean():.4f} "
            f"(live CI [{live_cost_ci[0]:.4f},{live_cost_ci[1]:.4f}]) "
            f"vs replay {rep['mean_cost']:.4f} "
            f"CI [{lo_c:.4f},{hi_c:.4f}] -> "
            f"{'PASS' if g1_cost else 'FAIL'}"
        )
        li_sav_ok = lo_s <= sav_li.mean() <= hi_s
        li_cost_ok = lo_c <= cost_li.mean() <= hi_c
        ci_sav_li = bootstrap_group_ci(sav_li, prompts)
        ci_cost_li = bootstrap_group_ci(cost_li, prompts)
        print(
            f"GOAL1 (live-internal baseline) savings "
            f"{sav_li.mean():.4f} "
            f"[{ci_sav_li[0]:.4f},{ci_sav_li[1]:.4f}] -> "
            f"{'PASS' if li_sav_ok else 'FAIL'}; cost "
            f"{cost_li.mean():.4f} "
            f"[{ci_cost_li[0]:.4f},{ci_cost_li[1]:.4f}] -> "
            f"{'PASS' if li_cost_ok else 'FAIL'}"
        )

        if not args.skip_static:
            static = static_point_baseline(comp)
            if static is not None:
                dominates = (
                    sav.mean() >= static["savings"] - 1e-9
                    and cost.mean() <= EPS
                )
                print(
                    f"GOAL2 static point eps={EPS}: savings "
                    f"{static['savings']:.4f} cost "
                    f"{static['cost']:.4f}; live "
                    f"{sav.mean():.4f} at cost {cost.mean():.4f} -> "
                    f"{'PASS' if dominates else 'FAIL'}"
                )

        walls = []
        attempt_walls = []
        revert_walls = []
        for e in eps_list:
            b = baselines.get(e["prompt_id"])
            if b is not None and b["wall_s"] > 0:
                walls.append(e["total_wall_s"] / b["wall_s"] - 1.0)
                attempt_walls.append(
                    sum(a["wall_s"] for a in e["attempts"]) / b["wall_s"]
                )
                # Pure controller waste: reverted attempts (prefill +
                # k tokens + alarm CPU). The commit attempt's wall is
                # dominated by the continuation decode, which is the
                # output itself, not overhead.
                revert_walls.append(
                    sum(
                        a["wall_s"]
                        for a in e["attempts"]
                        if not a["committed"]
                    )
                    / b["wall_s"]
                )
        overhead = (
            _finite_float(np.mean(walls), source="wall overhead")
            if walls
            else math.nan
        )
        attempt_share = (
            _finite_float(np.mean(attempt_walls), source="attempt wall share")
            if attempt_walls
            else math.nan
        )
        revert_share = (
            _finite_float(np.mean(revert_walls), source="revert wall share")
            if revert_walls
            else math.nan
        )
        peak_gb = max(e["peak_mem_bytes"] for e in eps_list) / 1e9
        tok_over = []
        for e in eps_list:
            rec_len = e.get("ref_vs_recorded", {}).get("recorded_len")
            if rec_len:
                reverts = sum(1 for a in e["attempts"] if not a["committed"])
                tok_over.append(2.0 * reverts / rec_len)
        token_overhead = (
            _finite_float(np.mean(tok_over), source="token overhead")
            if tok_over
            else math.nan
        )
        print(
            f"GOAL3 wall overhead vs uncompressed: "
            f"{overhead * 100:.1f}% "
            f"({'PASS' if overhead <= 0.15 else 'FAIL'}); "
            f"revert-wall overhead {revert_share * 100:.1f}% "
            f"({'PASS' if revert_share <= 0.15 else 'FAIL'}); "
            f"attempt wall share {attempt_share * 100:.1f}%; "
            f"token overhead {token_overhead * 100:.1f}%; "
            f"peak mem {peak_gb:.1f} GB"
        )

        # score-agreement diagnostic (pre-divergence grid points only)
        tgt_scores = {
            (
                g["prompt_id"],
                round(_finite_float(g["ratio"], source="target ratio"), 4),
            ): dict(zip(g["attempt_s"], g["scores"], strict=True))
            for g in tgt["groups"]
        }
        diffs = []
        n_cmp = 0
        for e in eps_list:
            key = (
                e["prompt_id"],
                round(_finite_float(e["ratio"], source="live ratio"), 4),
            )
            gmap = tgt_scores.get(key)
            if gmap is None:
                continue
            div = e.get("ref_vs_recorded", {}).get("divergence_index")
            limit = div if div is not None else 10**9
            for a in e["attempts"]:
                if a["s"] <= limit and a["s"] in gmap:
                    diffs.append(abs(a["score"] - gmap[a["s"]]))
                    n_cmp += 1
        if diffs:
            arr = np.array(diffs)
            print(
                f"DIAG score agreement (pre-divergence, n={n_cmp}): "
                f"median |d|={np.median(arr):.4f} "
                f"p90={np.quantile(arr, 0.9):.4f} "
                f"max={arr.max():.4f}"
            )


if __name__ == "__main__":
    main()
