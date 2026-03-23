"""XGBoost hazard predictor training and evaluation."""

import json
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


def _train_single_horizon(
    X: list[list[float]],
    y: list[int],
    run_ids: list[str],
    feat_names: list[str],
    horizon: int,
    output_dir: Path,
) -> dict[str, object]:
    """Train one XGBoost model for a single horizon value."""
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups=run_ids))
    train_idx, val_idx = splits[0]

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
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    y_pred = model.predict(dval).tolist()

    metrics: dict[str, object] = {
        "horizon": horizon,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "event_rate_train": round(n_pos / max(len(y_train), 1), 6),
        "event_rate_val": round(sum(y_val) / max(len(y_val), 1), 6),
        "best_iteration": model.best_iteration,
        **_eval_metrics(y_val, y_pred),
    }

    model_path = output_dir / f"hazard_H{horizon}.json"
    model.save_model(str(model_path))
    logger.info(f"H={horizon}: model saved to {model_path}")

    return metrics


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
        X, y, run_ids, press_ids, feat_names = build_dataset(
            results_dir, horizon=horizon
        )

        if not X:
            logger.warning(f"H={horizon}: no data found in {results_dir}")
            continue

        logger.info(
            f"H={horizon}: {len(X)} samples, "
            f"{sum(y)} positive ({sum(y) / len(X):.2%})"
        )

        metrics = _train_single_horizon(
            X, y, run_ids, feat_names, horizon, output_dir
        )

        # Leave-one-compressor-out CV
        logger.info(f"H={horizon}: running LOCO CV...")
        loco = _loco_cv(X, y, press_ids, feat_names)
        if loco:
            aurocs = [f["auroc"] for f in loco if f.get("auroc") is not None]
            mean_auroc = (
                round(sum(aurocs) / len(aurocs), 4)  # type: ignore[arg-type]
                if aurocs
                else None
            )
            metrics["loco_cv"] = {
                "folds": loco,
                "mean_auroc": mean_auroc,
            }
            logger.info(f"H={horizon}: LOCO mean AUROC={mean_auroc}")

        all_metrics[f"H{horizon}"] = metrics

        logger.info(
            f"H={horizon}: AUROC={metrics.get('auroc')}, "
            f"AUPRC={metrics.get('auprc')}"
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    logger.info(f"All metrics saved to {metrics_path}")

    return all_metrics
