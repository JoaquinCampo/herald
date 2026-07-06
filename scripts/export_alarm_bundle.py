"""Freeze the grace-window alarm bundles and replay fidelity targets.

Reruns the funded AIMD feasibility protocol (canonical split 0,
feat_plus_alarm variant, k=2, eps 0.03, point theta) and persists the
deployable pieces the live controller needs:

- per compressor: an AlarmBundle (3-seed XGB ensemble trained on the
  full train side, point theta calibrated on crossfit OOF replay, the
  exact feature column order) under
  results/predictor/alarm_bundle/{compressor}/;
- fidelity_targets.json: the test-side replay outcome at the frozen
  theta (means + cluster-bootstrap 95% CIs + the perfect-alarm oracle)
  and per-(prompt_id, ratio) attempt records (grid s values, alarm
  scores, commit point), which the live run is compared against.

The printed summary lines must reproduce the feasibility log
(split 0, k=2, eps 0.03, point): expected_attention sav=0.7986
cost=0.0254; knorm sav=0.1229 cost=-0.0018; streaming_llm sav=0.3739
cost=0.0100. Zero GPU; runs locally in minutes.
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, "src")

from herald.features import FEATURE_NAMES  # noqa: E402
from herald.grace_replay import (  # noqa: E402
    bootstrap_group_ci,
    calibrate_theta,
    oracle_replay,
    replay,
    replay_matrices,
)
from herald.grace_window import AlarmBundle, hyb_feature_names  # noqa: E402
from herald.switch_baselines import (  # noqa: E402
    leave_one_compressor_splits,
)
from herald.switch_risk import featurize  # noqa: E402

COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
PARQUET = "results/predictor/switch_dataset_attn.parquet"
NPZ = "results/predictor/hybrid_streams_ifeval.npz"
OUT_DIR = Path("results/predictor/alarm_bundle")
SPLIT_SEED = 0
N_FOLDS = 5
SEEDS = (0, 1, 2)
K = 2
EPSILON = 0.03
BASE = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "nthread": -1,
}


def hyb_summaries(
    blocks: np.ndarray,
    lengths: np.ndarray,
    trailing: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str]]:
    """Vectorized training-side block summaries (feasibility verbatim;
    the live per-run equivalent is grace_window.hybrid_block_summary,
    pinned to this construction by tests)."""
    n = blocks.shape[0]
    b = blocks[:, :k].astype(np.float32)
    m = np.minimum(lengths, k)
    step = np.arange(k)[None, :, None]
    b = np.where(step < m[:, None, None], b, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(b, axis=1)
        mn = np.nanmin(b, axis=1)
        mx = np.nanmax(b, axis=1)
    step0 = b[:, 0]
    last = b[np.arange(n), np.maximum(m - 1, 0)]
    slope = (last - step0) / np.maximum(m - 1, 1).astype(np.float32)[:, None]
    dtrail = mean - trailing
    parts = [step0, mean, mn, mx, slope, dtrail]
    for p in parts:
        p[m == 0] = np.nan
    return np.concatenate(parts, axis=1), hyb_feature_names(k)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df = df[df["task"] == "ifeval"].reset_index(drop=True)
    data = np.load(NPZ, allow_pickle=True)
    mask = df["compressor"].isin(COMPRESSORS).to_numpy()
    df3 = df[mask].reset_index(drop=True)
    rows = df3.to_dict("records")
    feat_cols = sorted(
        c
        for c in df3.columns
        if c.startswith("feat__") and not c.startswith("feat__attn_")
    )
    blocks = data["blocks"][mask]
    lengths = data["lengths"][mask]
    trailing = data["trailing"][mask]

    mat, names = hyb_summaries(blocks, lengths, trailing, K)
    cols = feat_cols + names

    splits = leave_one_compressor_splits(
        rows,
        compressors=COMPRESSORS,
        seed=SPLIT_SEED,
        test_group_fraction=0.25,
    )
    targets: dict[str, Any] = {
        "split_seed": SPLIT_SEED,
        "k": K,
        "epsilon": EPSILON,
        "theta_method": "point",
        "variant": "feat_plus_alarm",
        "parquet": PARQUET,
        "streams_npz": NPZ,
        "compressors": {},
    }
    for split in splits:
        c = split.heldout_compressor
        t0 = time.time()
        test_pids = {r["prompt_id"] for r in split.test}
        comp_idx = [
            i for i, r in enumerate(rows) if r["compressor"] == c
        ]
        train_sel = [
            i for i in comp_idx if rows[i]["prompt_id"] not in test_pids
        ]
        test_sel = [
            i for i in comp_idx if rows[i]["prompt_id"] in test_pids
        ]
        train = [rows[i] for i in train_sel]
        test_rows = [rows[i] for i in test_sel]
        train_pids = sorted({r["prompt_id"] for r in train})
        rng = random.Random(SPLIT_SEED)
        pids = train_pids[:]
        rng.shuffle(pids)
        folds = [set(pids[f::N_FOLDS]) for f in range(N_FOLDS)]
        pid_arr = np.asarray([r["prompt_id"] for r in train])
        y_all = np.asarray(
            [float(r["dq"]) > 0 for r in train], dtype=np.float32
        )

        work = [dict(r) for r in train]
        work_te = [dict(r) for r in test_rows]
        for lst, sel in ((work, train_sel), (work_te, test_sel)):
            for r, i in zip(lst, sel, strict=True):
                for j, name in enumerate(names):
                    v = float(mat[i, j])
                    r[name] = None if np.isnan(v) else v
        x_all = featurize(work, cols)
        x_te = featurize(work_te, cols)

        oof_acc = np.zeros(len(train))
        te_acc = np.zeros(len(test_rows))
        boosters = []
        for seed in SEEDS:
            params = {**BASE, "seed": seed}
            oof = np.full(len(train), np.nan)
            for fold_pids in folds:
                te_m = np.isin(pid_arr, list(fold_pids))
                tr_m = ~te_m
                clf = xgb.train(
                    params,
                    xgb.DMatrix(x_all[tr_m], label=y_all[tr_m]),
                    num_boost_round=300,
                )
                oof[te_m] = clf.predict(xgb.DMatrix(x_all[te_m]))
            final = xgb.train(
                params,
                xgb.DMatrix(x_all, label=y_all),
                num_boost_round=300,
            )
            boosters.append(final)
            oof_acc += oof
            te_acc += final.predict(xgb.DMatrix(x_te))
        oof_s = oof_acc / len(SEEDS)
        te_s = te_acc / len(SEEDS)

        tr_mats = replay_matrices(train, list(oof_s))
        te_mats = replay_matrices(test_rows, list(te_s))
        theta = calibrate_theta(tr_mats, list(oof_s), epsilon=EPSILON, k=K)
        out = replay(te_mats, theta=theta, k=K)
        osav, oover = oracle_replay(te_mats, k=K)

        ci_sav = bootstrap_group_ci(out.savings, te_mats.prompt_ids)
        ci_cost = bootstrap_group_ci(out.cost, te_mats.prompt_ids)
        ci_over = bootstrap_group_ci(out.overhead, te_mats.prompt_ids)

        bundle = AlarmBundle(
            compressor=c,
            k=K,
            epsilon=EPSILON,
            theta=float(theta),
            feature_cols=cols,
            boosters=boosters,
            meta={
                "split_seed": SPLIT_SEED,
                "variant": "feat_plus_alarm",
                "theta_method": "point",
                "n_train_rows": len(train),
                "n_train_prompts": len(train_pids),
                "xgb_params": BASE,
                "num_boost_round": 300,
                "oof_mean_savings_at_theta": float(
                    replay(tr_mats, theta=theta, k=K).savings.mean()
                ),
            },
        )
        bundle.save(OUT_DIR / c)

        groups = []
        for i, (pid, ratio) in enumerate(te_mats.keys):
            v = int(te_mats.valid[i])
            ci = int(out.commit_index[i])
            groups.append(
                {
                    "prompt_id": pid,
                    "ratio": ratio,
                    "ref_len": float(te_mats.ref_len[i]),
                    "attempt_s": te_mats.s[i, :v].tolist(),
                    "scores": te_mats.pred[i, :v].tolist(),
                    "commit_s": (
                        int(te_mats.s[i, ci]) if ci >= 0 else None
                    ),
                    "savings": float(out.savings[i]),
                    "cost": float(out.cost[i]),
                    "overhead": float(out.overhead[i]),
                }
            )
        targets["compressors"][c] = {
            "theta": float(theta),
            "test_prompt_ids": sorted(test_pids),
            "replay_test": {
                "n_groups": len(te_mats.keys),
                "mean_savings": float(out.savings.mean()),
                "ci95_savings": list(ci_sav),
                "mean_cost": float(out.cost.mean()),
                "ci95_cost": list(ci_cost),
                "mean_overhead": float(out.overhead.mean()),
                "ci95_overhead": list(ci_over),
                "oracle_savings": float(osav.mean()),
                "oracle_overhead": float(oover.mean()),
            },
            "groups": groups,
        }
        print(
            f"BUNDLE {c:20s} k={K} eps={EPSILON:.2f} "
            f"theta={theta:.6f} "
            f"sav={out.savings.mean():.4f} "
            f"cost={out.cost.mean():.4f} "
            f"overhead={out.overhead.mean():.3f} "
            f"oracle={osav.mean():.4f} "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    (OUT_DIR / "fidelity_targets.json").write_text(
        json.dumps(targets, indent=1)
    )
    print(f"wrote {OUT_DIR / 'fidelity_targets.json'}")


if __name__ == "__main__":
    main()
