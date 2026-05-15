"""Causal GRU sequence predictor for the loop-specialized K=16 task.

Locked architecture (pre-stated in
`gold/phase-2d-streaming-online-results.md` §6.3):
  - 1-layer causal GRU, hidden_size=64, dropout=0.1
  - input = same 511-dim per-segment feature vector as XGBoost
    loop-specialist (extended set + press one-hot +
    position_in_budget)
  - per-segment sigmoid output (streaming online by construction)
  - BCE with pos_weight = (1 - pos_rate) / pos_rate from train fold
  - AdamW lr=1e-3 weight_decay=1e-4
  - batch_size=32 runs, max_epochs=20, grad clip 1.0
  - early-stop on val AUPRC, patience=3
  - same 5-fold GroupKFold(prompt_id) and loop-only filter as
    `train_phase2_v2_xgb_loop.py`

Outputs (compatible with existing run-level analyzer):
  {output-dir}/scores_fold{i}.parquet  cols: run_id, seg_idx, y,
                                              score
  {output-dir}/model_fold{i}.pt
  {output-dir}/summary.json

Usage:
  uv run python scripts/train_phase2_v2_seq_loop.py \\
    --input results/phase2_v2/segments_k16_ext.parquet \\
    --output-dir results/phase2_v2/seq_ext_loop \\
    [--folds 0]   # smoke fold-only mode
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

LEAK_OR_META = {
    "run_id",
    "prompt_id",
    "press",
    "looping_onset",
    "non_termination_onset",
    "first_onset",
    "num_tokens_generated",
    "relative_progress",
    "y",
}


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    uniq_g = np.unique(groups)
    vals: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq_g, size=len(uniq_g), replace=True)
        mask = np.isin(groups, sampled)
        if y_true[mask].sum() == 0 or y_true[mask].sum() == mask.sum():
            continue
        vals.append(metric_fn(y_true[mask], y_score[mask]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(vals)),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


class SegmentSeqDataset(Dataset):
    """One sample = one run's full pre-onset segment sequence.

    Returns (X, y, mask, run_id, seg_idx) where X is (T, F), y is
    (T,), mask is (T,) with 1 for real segments and 0 for padding.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        run_ids: np.ndarray,
    ) -> None:
        self.feature_cols = feature_cols
        self.run_ids = list(run_ids)
        self.by_run: dict[object, pd.DataFrame] = {}
        sub = df[df["run_id"].isin(set(run_ids))]
        for rid, g in sub.groupby("run_id", sort=False):
            self.by_run[rid] = g.sort_values("seg_idx").reset_index(
                drop=True
            )

    def __len__(self) -> int:
        return len(self.run_ids)

    def __getitem__(self, idx: int):
        rid = self.run_ids[idx]
        g = self.by_run[rid]
        x = g[self.feature_cols].to_numpy(dtype=np.float32)
        y = g["y"].to_numpy(dtype=np.float32)
        seg_idx = g["seg_idx"].to_numpy(dtype=np.int64)
        return x, y, seg_idx, str(rid)


def collate_pad(batch):
    xs, ys, seg_idxs, rids = zip(*batch, strict=True)
    T = max(x.shape[0] for x in xs)
    F = xs[0].shape[1]
    B = len(xs)
    X = np.zeros((B, T, F), dtype=np.float32)
    Y = np.zeros((B, T), dtype=np.float32)
    M = np.zeros((B, T), dtype=np.float32)
    SI = -np.ones((B, T), dtype=np.int64)
    for i, (x, y, si) in enumerate(zip(xs, ys, seg_idxs, strict=True)):
        t = x.shape[0]
        X[i, :t] = x
        Y[i, :t] = y
        M[i, :t] = 1.0
        SI[i, :t] = si
    return (
        torch.from_numpy(X),
        torch.from_numpy(Y),
        torch.from_numpy(M),
        torch.from_numpy(SI),
        list(rids),
    )


class CausalGRUPredictor(nn.Module):
    def __init__(
        self, n_features: int, hidden: int = 64, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        h = self.drop(h)
        return self.head(h).squeeze(-1)


def train_fold(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_runs: np.ndarray,
    val_runs: np.ndarray,
    pos_weight: float,
    args,
    device: torch.device,
) -> tuple[CausalGRUPredictor, dict]:
    train_ds = SegmentSeqDataset(df, feature_cols, train_runs)
    val_ds = SegmentSeqDataset(df, feature_cols, val_runs)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pad,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pad,
        drop_last=False,
    )
    model = CausalGRUPredictor(
        n_features=len(feature_cols),
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    best_auprc = -1.0
    best_state: dict | None = None
    bad_epochs = 0
    history: list[dict] = []
    for epoch in range(args.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_seg_n = 0
        for X, Y, M, _SI, _rids in train_dl:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(X)
            losses = bce(logits, Y) * M
            denom = M.sum().clamp_min(1.0)
            loss = losses.sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += float(loss.detach().cpu()) * float(
                denom.cpu()
            )
            train_seg_n += int(denom.cpu())

        model.eval()
        val_y: list[float] = []
        val_s: list[float] = []
        with torch.no_grad():
            for X, Y, M, _SI, _rids in val_dl:
                X = X.to(device, non_blocking=True)
                logits = model(X)
                p = torch.sigmoid(logits).cpu().numpy()
                yn = Y.numpy()
                mn = M.numpy()
                B, T = mn.shape
                for b in range(B):
                    for t in range(T):
                        if mn[b, t] > 0:
                            val_y.append(float(yn[b, t]))
                            val_s.append(float(p[b, t]))
        val_y_arr = np.asarray(val_y, dtype=int)
        val_s_arr = np.asarray(val_s, dtype=float)
        if val_y_arr.sum() == 0 or val_y_arr.sum() == len(val_y_arr):
            val_auprc = float("nan")
            val_auroc = float("nan")
        else:
            val_auprc = float(
                average_precision_score(val_y_arr, val_s_arr)
            )
            val_auroc = float(roc_auc_score(val_y_arr, val_s_arr))
        train_loss = train_loss_sum / max(train_seg_n, 1)
        logger.info(
            "epoch {}: train_loss={:.5f}  val_auroc={:.4f}  "
            "val_auprc={:.4f}",
            epoch,
            train_loss,
            val_auroc,
            val_auprc,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
            }
        )
        if not np.isnan(val_auprc) and val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info(
                    "early stop @ epoch {} (best val_auprc={:.4f})",
                    epoch,
                    best_auprc,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val_auprc": best_auprc}


def predict_fold(
    model: CausalGRUPredictor,
    df: pd.DataFrame,
    feature_cols: list[str],
    val_runs: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    val_ds = SegmentSeqDataset(df, feature_cols, val_runs)
    dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad,
    )
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for X, Y, M, SI, rids in dl:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            p = torch.sigmoid(logits).cpu().numpy()
            yn = Y.numpy()
            mn = M.numpy()
            sin = SI.numpy()
            B, T = mn.shape
            for b in range(B):
                rid = rids[b]
                for t in range(T):
                    if mn[b, t] > 0:
                        rows.append(
                            {
                                "run_id": rid,
                                "seg_idx": int(sin[b, t]),
                                "y": int(yn[b, t]),
                                "score": float(p[b, t]),
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("results/phase2_v2/segments_k16_ext.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2_v2/seq_ext_loop"),
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument(
        "--folds",
        type=str,
        default="all",
        help="comma-list of fold ids, or 'all' (smoke: '0')",
    )
    ap.add_argument("--max-budget", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("device={}", device)

    logger.info("Reading {}", args.input)
    df = pd.read_parquet(args.input)
    logger.info("rows={} cols={}", *df.shape)

    run_onsets = df.groupby("run_id", sort=False).first()
    loop = run_onsets["looping_onset"]
    nt = run_onsets["non_termination_onset"]
    loop_first = loop.notna() & (nt.isna() | (nt > loop))
    clean = loop.isna() & nt.isna()
    keep_runs = run_onsets.index[loop_first | clean]
    n_loop = int(loop_first.sum())
    n_clean = int(clean.sum())
    n_nt = int(((nt.notna()) & (loop.isna() | (loop > nt))).sum())
    logger.info(
        "Run filter: loop-first={} clean={} nt-first-dropped={}",
        n_loop,
        n_clean,
        n_nt,
    )
    df = df[df["run_id"].isin(set(keep_runs))].reset_index(drop=True)
    logger.info("After run filter: rows={}", len(df))

    df["position_in_budget"] = (
        df["seg_end_tok"].astype(float) / float(args.max_budget)
    ).clip(0.0, 1.0)
    presses = sorted(df["press"].unique())
    for p in presses:
        df[f"press_{p}"] = (df["press"] == p).astype(np.float32)

    feature_cols = [
        c
        for c in df.columns
        if c not in LEAK_OR_META and not c.startswith("press_")
    ]
    feature_cols += [f"press_{p}" for p in presses]
    feature_cols = sorted(set(feature_cols))
    logger.info("n_features={}", len(feature_cols))

    df = df.sort_values(["run_id", "seg_idx"]).reset_index(drop=True)
    df[feature_cols] = (
        df[feature_cols].astype(np.float32).fillna(0.0)
    )

    run_first = df.groupby("run_id", sort=False).first().reset_index()
    run_to_pid = dict(
        zip(
            run_first["run_id"].tolist(),
            run_first["prompt_id"].tolist(),
            strict=True,
        )
    )
    unique_runs = run_first["run_id"].to_numpy()
    unique_groups = run_first["prompt_id"].to_numpy()

    if args.folds == "all":
        fold_ids = list(range(args.n_splits))
    else:
        fold_ids = [int(s) for s in args.folds.split(",") if s.strip()]
    logger.info("training folds: {}", fold_ids)

    gkf = GroupKFold(n_splits=args.n_splits)
    splits = list(
        gkf.split(
            unique_runs,
            np.zeros(len(unique_runs)),
            unique_groups,
        )
    )

    fold_summary: list[dict] = []
    for fold_i in fold_ids:
        logger.info("=== fold {} ===", fold_i)
        tr_idx, te_idx = splits[fold_i]
        train_runs = unique_runs[tr_idx]
        val_runs = unique_runs[te_idx]
        y_train_seg = df[df["run_id"].isin(set(train_runs))][
            "y"
        ].to_numpy(dtype=int)
        pos_rate = y_train_seg.mean() if len(y_train_seg) else 0.0
        pos_weight = (
            (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        )
        logger.info(
            "fold {}: n_train_runs={} n_val_runs={}  "
            "pos_rate={:.4%} pos_weight={:.2f}",
            fold_i,
            len(train_runs),
            len(val_runs),
            pos_rate,
            pos_weight,
        )

        model, train_log = train_fold(
            df,
            feature_cols,
            train_runs,
            val_runs,
            pos_weight,
            args,
            device,
        )

        scores_df = predict_fold(
            model, df, feature_cols, val_runs, device, args.batch_size
        )
        scores_df.to_parquet(
            args.output_dir / f"scores_fold{fold_i}.parquet",
            index=False,
        )
        torch.save(
            model.state_dict(),
            args.output_dir / f"model_fold{fold_i}.pt",
        )

        y_arr = scores_df["y"].to_numpy(dtype=int)
        s_arr = scores_df["score"].to_numpy(dtype=float)
        groups_arr = np.asarray(
            [run_to_pid[r] for r in scores_df["run_id"]]
        )
        seg_auroc = float(roc_auc_score(y_arr, s_arr))
        seg_auprc = float(average_precision_score(y_arr, s_arr))
        bs_au = cluster_bootstrap_ci(
            y_arr,
            s_arr,
            groups_arr,
            roc_auc_score,
            args.n_boot,
        )
        bs_pr = cluster_bootstrap_ci(
            y_arr,
            s_arr,
            groups_arr,
            average_precision_score,
            args.n_boot,
        )
        agg = scores_df.groupby("run_id", sort=False).agg(
            run_score=("score", "max"),
            run_label=("y", "max"),
        )
        run_y = agg["run_label"].to_numpy(dtype=int)
        run_s = agg["run_score"].to_numpy(dtype=float)
        run_auroc = float(roc_auc_score(run_y, run_s))
        run_auprc = float(average_precision_score(run_y, run_s))
        logger.info(
            "fold {}: SEG AUROC={:.4f} (CI {:.4f}-{:.4f})  "
            "AUPRC={:.4f} (CI {:.4f}-{:.4f})  "
            "RUN AUROC={:.4f}  AUPRC={:.4f}",
            fold_i,
            seg_auroc,
            bs_au[1],
            bs_au[2],
            seg_auprc,
            bs_pr[1],
            bs_pr[2],
            run_auroc,
            run_auprc,
        )
        fold_summary.append(
            {
                "fold": int(fold_i),
                "n_train_runs": int(len(train_runs)),
                "n_val_runs": int(len(val_runs)),
                "n_val_segments": int(len(scores_df)),
                "pos_rate_train": float(pos_rate),
                "pos_weight": float(pos_weight),
                "seg_auroc": seg_auroc,
                "seg_auroc_ci_low": bs_au[1],
                "seg_auroc_ci_high": bs_au[2],
                "seg_auprc": seg_auprc,
                "seg_auprc_ci_low": bs_pr[1],
                "seg_auprc_ci_high": bs_pr[2],
                "run_auroc": run_auroc,
                "run_auprc": run_auprc,
                "best_val_auprc": train_log["best_val_auprc"],
                "n_epochs_used": len(train_log["history"]),
            }
        )

    summary = {
        "args": {
            "n_splits": args.n_splits,
            "folds": fold_ids,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "max_budget": args.max_budget,
            "n_features": len(feature_cols),
            "seed": args.seed,
        },
        "n_loop_first_runs": n_loop,
        "n_clean_runs": n_clean,
        "n_nt_first_runs_dropped": n_nt,
        "feature_cols": feature_cols,
        "fold_summary": fold_summary,
    }
    if len(fold_summary) >= 1:
        summary["seg_auroc_mean"] = float(
            np.mean([s["seg_auroc"] for s in fold_summary])
        )
        summary["seg_auroc_std"] = float(
            np.std([s["seg_auroc"] for s in fold_summary])
        )
        summary["run_auroc_mean"] = float(
            np.mean([s["run_auroc"] for s in fold_summary])
        )
        summary["run_auroc_std"] = float(
            np.std([s["run_auroc"] for s in fold_summary])
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("Wrote {}", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
