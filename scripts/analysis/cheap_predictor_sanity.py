"""Phase 0 analysis #8: cheap-features sanity predictor.

Aggregate Tier 0 token features (entropy, top1_prob, top5_prob,
h_alts, avg_logp, delta_h, kl_div, top10_jaccard, eff_vocab_size,
tail_mass, logit_range) per run via mean / max / p95 / EWMA (alpha
0.1), then fit logistic regression with leave-one-prompt-out CV
predicting a damage label.

Label: damage = rouge_l < median(rouge_l) across all 114 successful
compressed runs. Median split is preferred over baseline_correct
AND compressed_wrong because (a) baseline answer-extraction yields
36/114 nulls, breaking that label; (b) the binary correct label is
near-degenerate (only 3/114 compressed correct). Median split gives
57/57 balance and lets the predictor find ANY trajectory-level
signal that separates more-damaged runs from less-damaged ones.

Chance baselines for the median-split label:
  AUROC chance = 0.5
  AUPRC chance = 57/114 = 0.5

Output: results/phase0/analysis/cheap_predictor_sanity.json
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

TIER0_COLS = [
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "delta_h",
    "kl_div",
    "top10_jaccard",
    "eff_vocab_size",
    "tail_mass",
    "logit_range",
]


def _ewma(arr: np.ndarray, alpha: float = 0.1) -> float:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    s = float(arr[0])
    for x in arr[1:]:
        s = alpha * float(x) + (1 - alpha) * s
    return s


def _aggregate_run(tokens: pl.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in TIER0_COLS:
        if c not in tokens.columns:
            continue
        arr = tokens[c].to_numpy().astype(np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            for tag in ("mean", "max", "p95", "ewma"):
                out[f"{c}__{tag}"] = float("nan")
            continue
        out[f"{c}__mean"] = float(np.mean(finite))
        out[f"{c}__max"] = float(np.max(finite))
        out[f"{c}__p95"] = float(np.percentile(finite, 95))
        out[f"{c}__ewma"] = _ewma(finite, alpha=0.1)
    return out


def main() -> None:
    root = Path("results/phase0")
    runs = pl.read_parquet(root / "final" / "runs.parquet")
    seq = pl.read_parquet(root / "metrics" / "sequence_metrics.parquet")

    compr = runs.filter(pl.col("press") != "none").select(
        ["run_id", "prompt_id", "press", "compression_ratio"]
    )
    df = compr.join(
        seq.select(["run_id", "rouge_l"]), on="run_id", how="inner"
    )

    rows: list[dict[str, float | str]] = []
    for r in df.iter_rows(named=True):
        rid = r["run_id"]
        press = r["press"]
        ratio = r["compression_ratio"]
        # tokens.parquet path
        tok_path = (
            root
            / "final"
            / "tokens"
            / f"press={press}"
            / f"ratio={ratio:.4f}"
            / f"{rid}.parquet"
        )
        if not tok_path.exists():
            continue
        tok = pl.read_parquet(tok_path)
        agg = _aggregate_run(tok)
        agg["run_id"] = rid
        agg["prompt_id"] = r["prompt_id"]
        agg["press"] = press
        agg["compression_ratio"] = ratio
        agg["rouge_l"] = r["rouge_l"]
        rows.append(agg)

    feat = pl.DataFrame(rows)
    median_rouge = feat["rouge_l"].median()
    feat = feat.with_columns(
        (pl.col("rouge_l") < median_rouge).alias("damage"),
    )

    feature_cols = [
        c
        for c in feat.columns
        if c
        not in {
            "run_id",
            "prompt_id",
            "press",
            "compression_ratio",
            "rouge_l",
            "damage",
        }
    ]

    # Drop columns that are all-NaN; impute remaining NaN with column
    # median (fit-time only). Phase 0 has ~512 tokens per run so most
    # columns are well-populated, but lookback_ratio is NaN.
    arr = feat.select(feature_cols).to_numpy().astype(np.float64)
    keep = ~np.all(np.isnan(arr), axis=0)
    feature_cols = [c for c, k in zip(feature_cols, keep) if k]
    arr = arr[:, keep]
    col_med = np.nanmedian(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_med, inds[1])

    y = feat["damage"].cast(pl.Int8).to_numpy()
    groups = feat["prompt_id"].to_numpy()
    prompts = sorted(set(groups))

    logo = LeaveOneGroupOut()
    fold_records: list[dict] = []
    oof_preds = np.full(len(y), np.nan)

    for fold, (tr, te) in enumerate(logo.split(arr, y, groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(arr[tr])
        Xte = scaler.transform(arr[te])
        # If a fold has only one class in train, log + skip
        if len(set(y[tr])) < 2:
            fold_records.append(
                {
                    "fold": fold,
                    "held_out_prompt": str(groups[te][0]),
                    "n_test": int(len(te)),
                    "skipped": "single_class_in_train",
                }
            )
            continue
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(Xtr, y[tr])
        p = clf.predict_proba(Xte)[:, 1]
        oof_preds[te] = p
        fold_records.append(
            {
                "fold": fold,
                "held_out_prompt": str(groups[te][0]),
                "n_test": int(len(te)),
                "n_pos_test": int(int(y[te].sum())),
                "n_pos_train": int(int(y[tr].sum())),
                "skipped": None,
            }
        )

    mask = ~np.isnan(oof_preds)
    if mask.sum() == 0 or len(set(y[mask])) < 2:
        result = {"error": "no usable predictions across folds"}
    else:
        auroc = float(roc_auc_score(y[mask], oof_preds[mask]))
        auprc = float(average_precision_score(y[mask], oof_preds[mask]))
        result = {
            "n_runs": int(len(y)),
            "n_runs_predicted": int(mask.sum()),
            "n_features": int(arr.shape[1]),
            "feature_cols": feature_cols,
            "label": "rouge_l < median(rouge_l) across compressed runs",
            "median_rouge_l_split": float(median_rouge),
            "n_positive": int(int(y.sum())),
            "n_negative": int(int((1 - y).sum())),
            "n_prompts": int(len(prompts)),
            "cv_strategy": "leave-one-prompt-out",
            "auroc_oof": auroc,
            "auprc_oof": auprc,
            "auroc_chance": 0.5,
            "auprc_chance": float(y.mean()),
            "lift_auroc": auroc - 0.5,
            "lift_auprc": auprc - float(y.mean()),
            "fold_records": fold_records,
        }

    out_path = root / "analysis" / "cheap_predictor_sanity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    for k in (
        "n_runs",
        "n_runs_predicted",
        "n_features",
        "n_positive",
        "n_negative",
        "n_prompts",
        "auroc_oof",
        "auprc_oof",
        "auroc_chance",
        "auprc_chance",
        "lift_auroc",
        "lift_auprc",
    ):
        if k in result:
            v = result[k]
            print(f"{k}: {v if not isinstance(v, float) else f'{v:.4f}'}")


if __name__ == "__main__":
    main()
