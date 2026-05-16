"""HERALD v1 evaluation: load OOF predictions, compute metrics.

For each (split_kind, horizon):
  - per-token Spearman (cluster bootstrap over prompt_id)
  - per-run Spearman vs run_damage targets with three aggregations
    (max, p95, mean); CI clustered by prompt_id
  - cross-cell retention ratios for transfer splits
  - per-run ECE on max-aggregate

Outputs `results/phase3/eval/metrics__{split_kind}__h{H}.json` plus
a consolidated `results/phase3/eval/summary.json`.
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl

from herald.regression_metrics import (
    clustered_spearman_ci,
    ece_quantile,
    per_run_aggregates,
    safe_spearman,
)

RUN_TARGETS = (
    "rouge_l_drop", "sum_js", "sum_kl", "char_edit_ratio",
)


def _eval_one(
    preds_path: Path,
    run_damage_path: Path,
    n_boot: int,
    seed: int,
    tokens_path: Path | None = None,
    cross_horizons: tuple[int, ...] = (5, 10, 25, 50),
) -> dict:
    meta = json.loads(
        preds_path.with_suffix(".meta.json").read_text())
    horizon = int(meta["horizon"])
    kind = meta["split_kind"]
    label = meta["label"]

    df = pl.read_parquet(preds_path)
    print(f"\n=== {kind} h={horizon}  n_tokens={df.height:,} ===")

    # Per-token Spearman (clustered by prompt_id)
    t0 = time.time()
    rho_token = clustered_spearman_ci(
        df["pred_raw"].to_numpy(),
        df["y_raw"].to_numpy(),
        df["prompt_id"].to_numpy(),
        n_boot=n_boot, seed=seed,
    )
    print(f"  ρ_token(pred, {label}) = {rho_token['rho']:.4f} "
          f"[{rho_token['lo']:.4f}, {rho_token['hi']:.4f}]  "
          f"({time.time()-t0:.0f}s)")

    # Cross-horizon per-token: check whether same model satisfies
    # the per-token bar at other horizons (avoids retraining).
    cross_horizon: dict[str, dict] = {}
    if tokens_path is not None:
        other_h = [h for h in cross_horizons if h != horizon]
        if other_h:
            cross_cols = [f"future_sum_js_{h}" for h in other_h]
            t0 = time.time()
            extra = pl.read_parquet(tokens_path).select(
                ["run_id", "token_pos"] + cross_cols)
            merged = df.select(
                ["run_id", "token_pos", "pred_raw", "prompt_id"]
            ).join(extra, on=["run_id", "token_pos"], how="left")
            for h in other_h:
                col = f"future_sum_js_{h}"
                sub = merged.drop_nulls(subset=[col])
                if sub.height < 1000:
                    continue
                ci = clustered_spearman_ci(
                    sub["pred_raw"].to_numpy(),
                    sub[col].to_numpy(),
                    sub["prompt_id"].to_numpy(),
                    n_boot=n_boot, seed=seed,
                )
                cross_horizon[col] = ci
                print(f"  ρ_token(pred, {col:<18s}) = "
                      f"{ci['rho']:.4f} [{ci['lo']:.4f}, "
                      f"{ci['hi']:.4f}]  n={sub.height:,}")
            print(f"  (cross-horizon eval {time.time()-t0:.0f}s)")

    # Per-run aggregations + run_damage join
    per_run = per_run_aggregates(df)
    rd = pl.read_parquet(run_damage_path).select(
        ["run_id"] + list(RUN_TARGETS))
    joined = per_run.join(rd, on="run_id", how="inner")
    print(f"  n_runs={joined.height:,}")

    run_results: dict[str, dict] = {}
    for agg in ("pred_max", "pred_p95", "pred_mean"):
        for tgt in RUN_TARGETS + ("y_max",):
            sub = joined.drop_nulls(subset=[agg, tgt])
            if sub.height < 10:
                run_results[f"{agg}__{tgt}"] = {
                    "rho": float("nan"),
                    "lo": float("nan"), "hi": float("nan"),
                    "n": int(sub.height),
                }
                continue
            ci = clustered_spearman_ci(
                sub[agg].to_numpy(),
                sub[tgt].to_numpy(),
                sub["prompt_id"].to_numpy(),
                n_boot=n_boot, seed=seed,
            )
            ci["n"] = int(sub.height)
            run_results[f"{agg}__{tgt}"] = ci
            print(f"  ρ_run({agg:>9s} → {tgt:<16s}) = "
                  f"{ci['rho']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]"
                  f"  n={sub.height}")

    # ECE on pred_max vs y_max
    ece = ece_quantile(
        joined["pred_max"].to_numpy(),
        joined["y_max"].to_numpy(),
        n_bins=10,
    )
    print(f"  ECE(pred_max → y_max) = {ece:.4f}")

    # Per-fold retention for transfer splits.
    per_fold: dict[str, dict] = {}
    folds = sorted(df["fold"].unique().to_list())
    if kind != "prompt_group" or len(folds) > 1:
        for fold in folds:
            sub = df.filter(pl.col("fold") == fold)
            if sub.height < 1000:
                continue
            rho_f_token = safe_spearman(
                sub["pred_raw"].to_numpy(),
                sub["y_raw"].to_numpy())
            pr = per_run_aggregates(sub).join(
                rd, on="run_id", how="inner")
            rho_f_sum_js = safe_spearman(
                pr["pred_max"].to_numpy(),
                pr["sum_js"].to_numpy()
            ) if pr.height > 10 else float("nan")
            per_fold[fold] = {
                "rho_token": rho_f_token,
                "rho_run_max_vs_sum_js": rho_f_sum_js,
                "n_tokens": int(sub.height),
                "n_runs": int(pr.height),
            }
            print(f"  [{fold:<24s}] ρ_tok={rho_f_token:.4f}  "
                  f"ρ_run_sum_js={rho_f_sum_js:.4f}")

    return {
        "split_kind": kind,
        "horizon": horizon,
        "label": label,
        "n_tokens": int(df.height),
        "n_runs": int(joined.height),
        "rho_token": rho_token,
        "rho_token_cross_horizon": cross_horizon,
        "rho_run": run_results,
        "ece_max_vs_y_max": ece,
        "per_fold": per_fold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds-dir", type=Path,
        default=Path("results/phase3/preds"))
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("results/phase3/eval"))
    parser.add_argument(
        "--run-damage-path", type=Path,
        default=Path("results/phase1/metrics/run_damage.parquet"))
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument(
        "--tokens-path", type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
        help="Used to evaluate the same model against other "
             "horizon labels.")
    parser.add_argument(
        "--skip-cross-horizon", action="store_true",
        help="Skip the cross-horizon eval (saves ~2GB RAM).")
    parser.add_argument(
        "--only", type=str, default=None,
        help="If set, only eval files matching this substring "
             "(e.g., 'prompt_group__h25').")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.preds_dir.glob("*.parquet"))
    print(f"[eval] {len(paths)} prediction files")

    summary = []
    tokens_path = (
        None if args.skip_cross_horizon else args.tokens_path)
    for p in paths:
        if args.only and args.only not in p.name:
            continue
        if p.suffix == ".parquet" and not p.name.endswith(
                ".meta.parquet"):
            r = _eval_one(
                p, args.run_damage_path, args.n_boot, args.seed,
                tokens_path=tokens_path)
            stem = p.stem
            (args.out_dir / f"metrics__{stem}.json").write_text(
                json.dumps(r, indent=2))
            summary.append(r)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n[done] wrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
