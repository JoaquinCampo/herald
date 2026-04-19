"""XGBoost hazard predictor training and evaluation."""

import json
import math
from pathlib import Path

import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from herald.features import build_dataset

DEFAULT_HORIZONS = [1, 5, 10, 25, 50]

XGB_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
}


def _eval_metrics(
    y_true: list[int],
    y_pred: list[float],
) -> dict[str, float | None]:
    """Compute AUROC and AUPRC, handling degenerate cases."""
    if len(set(y_true)) > 1:
        return {
            "auroc": round(roc_auc_score(y_true, y_pred), 4),
            "auprc": round(average_precision_score(y_true, y_pred), 4),
        }
    return {"auroc": None, "auprc": None}


def _mean_std(
    values: list[float | None],
) -> tuple[float | None, float | None]:
    """Mean and std of non-None values."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None
    mean = sum(valid) / len(valid)
    if len(valid) < 2:
        return round(mean, 4), None
    var = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    return round(mean, 4), round(math.sqrt(var), 4)


def _cv_metrics(
    X: list[list[float]],
    y: list[int],
    run_ids: list[str],
    feat_names: list[str],
    pre_onset: list[bool] | None = None,
    n_splits: int = 5,
) -> dict[str, object]:
    """Run GroupKFold CV and return per-fold + aggregate metrics.

    Trains on all tokens but reports two evaluation tracks:
    - overall: all validation tokens
    - pre_onset: only tokens before catastrophe onset
      (the real early-warning metric)

    Returns fold metrics, aggregates, and best_iteration (median).
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_aurocs: list[float | None] = []
    fold_auprcs: list[float | None] = []
    fold_pre_aurocs: list[float | None] = []
    fold_pre_auprcs: list[float | None] = []
    best_iters: list[int] = []

    for train_idx, val_idx in gkf.split(X, y, groups=run_ids):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        n_pos = sum(y_train)
        n_neg = len(y_train) - n_pos
        params = {
            **XGB_PARAMS,
            "scale_pos_weight": float(n_neg / max(n_pos, 1)),
        }

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )

        y_pred = model.predict(dval).tolist()

        # Overall metrics
        ev = _eval_metrics(y_val, y_pred)
        fold_aurocs.append(ev["auroc"])
        fold_auprcs.append(ev["auprc"])
        best_iters.append(model.best_iteration)

        # Pre-onset metrics (early-warning evaluation)
        if pre_onset is not None:
            val_pre = [pre_onset[i] for i in val_idx]
            y_val_pre = [y_val[j] for j, p in enumerate(val_pre) if p]
            y_pred_pre = [y_pred[j] for j, p in enumerate(val_pre) if p]
            ev_pre = _eval_metrics(y_val_pre, y_pred_pre)
            fold_pre_aurocs.append(ev_pre["auroc"])
            fold_pre_auprcs.append(ev_pre["auprc"])

    auroc_mean, auroc_std = _mean_std(fold_aurocs)
    auprc_mean, auprc_std = _mean_std(fold_auprcs)

    result: dict[str, object] = {
        "fold_aurocs": fold_aurocs,
        "fold_auprcs": fold_auprcs,
        "auroc_mean": auroc_mean,
        "auroc_std": auroc_std,
        "auprc_mean": auprc_mean,
        "auprc_std": auprc_std,
        "best_iteration": int(sorted(best_iters)[len(best_iters) // 2]),
    }

    if fold_pre_aurocs:
        pre_auroc_mean, pre_auroc_std = _mean_std(fold_pre_aurocs)
        pre_auprc_mean, pre_auprc_std = _mean_std(fold_pre_auprcs)
        result["pre_onset"] = {
            "fold_aurocs": fold_pre_aurocs,
            "fold_auprcs": fold_pre_auprcs,
            "auroc_mean": pre_auroc_mean,
            "auroc_std": pre_auroc_std,
            "auprc_mean": pre_auprc_mean,
            "auprc_std": pre_auprc_std,
        }

    return result


def _train_single_horizon(
    X: list[list[float]],
    y: list[int],
    run_ids: list[str],
    feat_names: list[str],
    pre_onset: list[bool],
    horizon: int,
    output_dir: Path,
) -> dict[str, object]:
    """Train one XGBoost model for a single horizon value.

    Runs 5-fold GroupKFold CV for metrics, then retrains
    on full data for the saved model.
    """
    n_pos = sum(y)
    n_total = len(y)
    event_rate = round(n_pos / max(n_total, 1), 6)

    # 5-fold CV for evaluation metrics
    cv = _cv_metrics(X, y, run_ids, feat_names, pre_onset)

    # Retrain on full data for the production model
    n_neg = n_total - n_pos
    params = {
        **XGB_PARAMS,
        "scale_pos_weight": float(n_neg / max(n_pos, 1)),
    }
    best_round = max(cv["best_iteration"], 10)  # type: ignore[call-overload]

    dtrain = xgb.DMatrix(X, label=y, feature_names=feat_names)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_round,
        verbose_eval=0,
    )

    model_path = output_dir / f"hazard_H{horizon}.json"
    model.save_model(str(model_path))
    logger.info(f"H={horizon}: model saved to {model_path}")

    metrics: dict[str, object] = {
        "horizon": horizon,
        "n_samples": n_total,
        "event_rate": event_rate,
        **cv,
    }
    return metrics


# ---------------------------------------------------------------
# Rolling entropy baseline
# ---------------------------------------------------------------

BASELINE_FEATURE = "entropy_mean_8"


def _baseline_cv(
    X: list[list[float]],
    y: list[int],
    run_ids: list[str],
    feat_names: list[str],
    pre_onset: list[bool] | None = None,
    n_splits: int = 5,
) -> dict[str, object]:
    """Baseline: use rolling entropy mean as the sole predictor.

    No training; just uses entropy_mean_8 as the score
    (higher entropy = more likely catastrophe). Evaluated on
    the same GroupKFold splits for fair comparison with XGBoost.
    """
    if BASELINE_FEATURE not in feat_names:
        logger.warning(
            f"Baseline feature {BASELINE_FEATURE!r} not found in feat_names"
        )
        return {}

    feat_idx = feat_names.index(BASELINE_FEATURE)

    gkf = GroupKFold(n_splits=n_splits)
    fold_aurocs: list[float | None] = []
    fold_auprcs: list[float | None] = []
    fold_pre_aurocs: list[float | None] = []
    fold_pre_auprcs: list[float | None] = []

    for _, val_idx in gkf.split(X, y, groups=run_ids):
        y_val = [y[i] for i in val_idx]
        scores = [X[i][feat_idx] for i in val_idx]

        ev = _eval_metrics(y_val, scores)
        fold_aurocs.append(ev["auroc"])
        fold_auprcs.append(ev["auprc"])

        if pre_onset is not None:
            val_pre = [pre_onset[i] for i in val_idx]
            y_val_pre = [y_val[j] for j, p in enumerate(val_pre) if p]
            scores_pre = [scores[j] for j, p in enumerate(val_pre) if p]
            ev_pre = _eval_metrics(y_val_pre, scores_pre)
            fold_pre_aurocs.append(ev_pre["auroc"])
            fold_pre_auprcs.append(ev_pre["auprc"])

    auroc_mean, auroc_std = _mean_std(fold_aurocs)
    auprc_mean, auprc_std = _mean_std(fold_auprcs)

    result: dict[str, object] = {
        "feature": BASELINE_FEATURE,
        "fold_aurocs": fold_aurocs,
        "fold_auprcs": fold_auprcs,
        "auroc_mean": auroc_mean,
        "auroc_std": auroc_std,
        "auprc_mean": auprc_mean,
        "auprc_std": auprc_std,
    }

    if fold_pre_aurocs:
        pre_auroc_mean, pre_auroc_std = _mean_std(fold_pre_aurocs)
        pre_auprc_mean, pre_auprc_std = _mean_std(fold_pre_auprcs)
        result["pre_onset"] = {
            "fold_aurocs": fold_pre_aurocs,
            "fold_auprcs": fold_pre_auprcs,
            "auroc_mean": pre_auroc_mean,
            "auroc_std": pre_auroc_std,
            "auprc_mean": pre_auprc_mean,
            "auprc_std": pre_auprc_std,
        }

    return result


# ---------------------------------------------------------------
# Leave-one-compressor-out cross-validation
# ---------------------------------------------------------------


def _loco_cv(
    X: list[list[float]],
    y: list[int],
    press_ids: list[str],
    feat_names: list[str],
) -> list[dict[str, object]]:
    """Leave-one-compressor-out CV.

    For each unique press (excluding 'none'), train on all other
    compressors and evaluate on the held-out one. Tests whether
    the predictor generalises across compression methods.
    """
    presses = sorted({p for p in press_ids if p != "none"})
    if len(presses) < 2:
        return []

    folds: list[dict[str, object]] = []

    for held_out in presses:
        train_x = [X[i] for i in range(len(X)) if press_ids[i] != held_out]
        train_y = [y[i] for i in range(len(y)) if press_ids[i] != held_out]
        val_x = [X[i] for i in range(len(X)) if press_ids[i] == held_out]
        val_y = [y[i] for i in range(len(y)) if press_ids[i] == held_out]

        if not train_x or not val_x:
            continue
        if len(set(train_y)) < 2:
            continue

        n_pos = sum(train_y)
        n_neg = len(train_y) - n_pos
        params = {
            **XGB_PARAMS,
            "scale_pos_weight": float(n_neg / max(n_pos, 1)),
        }

        dtrain = xgb.DMatrix(train_x, label=train_y, feature_names=feat_names)
        dval = xgb.DMatrix(val_x, label=val_y, feature_names=feat_names)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=0,
        )

        y_pred = model.predict(dval).tolist()
        ev = _eval_metrics(val_y, y_pred)

        folds.append(
            {
                "held_out_press": held_out,
                "n_train": len(train_y),
                "n_val": len(val_y),
                "event_rate_val": round(sum(val_y) / max(len(val_y), 1), 4),
                **ev,
            }
        )

        logger.info(
            f"  LOCO {held_out}: "
            f"AUROC={ev.get('auroc')}, AUPRC={ev.get('auprc')}"
        )

    return folds


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------


def train_predictor(
    results_dir: Path,
    output_dir: Path,
    horizons: list[int] | None = None,
) -> dict[str, object]:
    """Train XGBoost hazard predictors for multiple horizons.

    Returns dict with per-horizon metrics + LOCO CV results.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, object] = {}

    for horizon in horizons:
        logger.info(f"Building dataset for H={horizon}...")
        ds = build_dataset(results_dir, horizon=horizon)

        if not ds.X:
            logger.warning(f"H={horizon}: no data in {results_dir}")
            continue

        n_pos = sum(ds.y)
        logger.info(
            f"H={horizon}: {len(ds.X)} samples, "
            f"{n_pos} positive "
            f"({n_pos / len(ds.X):.2%})"
        )

        metrics = _train_single_horizon(
            ds.X,
            ds.y,
            ds.run_ids,
            ds.feat_names,
            ds.pre_onset,
            horizon,
            output_dir,
        )

        # Rolling entropy baseline
        logger.info(f"H={horizon}: running baseline...")
        baseline = _baseline_cv(
            ds.X, ds.y, ds.run_ids, ds.feat_names, ds.pre_onset
        )
        if baseline:
            metrics["baseline"] = baseline
            logger.info(
                f"H={horizon} baseline: "
                f"AUROC={baseline.get('auroc_mean')}, "
                f"AUPRC={baseline.get('auprc_mean')}"
            )
            bl_pre = baseline.get("pre_onset")
            if isinstance(bl_pre, dict):
                logger.info(
                    f"H={horizon} baseline pre-onset: "
                    f"AUROC={bl_pre.get('auroc_mean')}, "
                    f"AUPRC={bl_pre.get('auprc_mean')}"
                )

        # Leave-one-compressor-out CV
        logger.info(f"H={horizon}: running LOCO CV...")
        loco = _loco_cv(ds.X, ds.y, ds.press_ids, ds.feat_names)
        if loco:
            aurocs: list[float] = [
                f["auroc"]  # type: ignore[misc]
                for f in loco
                if f.get("auroc") is not None
            ]
            mean_auroc = (
                round(sum(aurocs) / len(aurocs), 4) if aurocs else None
            )
            metrics["loco_cv"] = {
                "folds": loco,
                "mean_auroc": mean_auroc,
            }
            logger.info(f"H={horizon}: LOCO mean AUROC={mean_auroc}")

        all_metrics[f"H{horizon}"] = metrics

        logger.info(
            f"H={horizon}: "
            f"AUROC={metrics.get('auroc_mean')}, "
            f"AUPRC={metrics.get('auprc_mean')}"
        )
        pre = metrics.get("pre_onset")
        if isinstance(pre, dict):
            logger.info(
                f"H={horizon} pre-onset: "
                f"AUROC={pre.get('auroc_mean')}, "
                f"AUPRC={pre.get('auprc_mean')}"
            )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    logger.info(f"All metrics saved to {metrics_path}")

    return all_metrics
