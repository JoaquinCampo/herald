"""Phase 2 token-level dataset builder (CPU-only).

Reads the rsynced Phase 1 final artifacts plus run_damage.parquet,
emits one row per generated token in a compressed run, with:

- cheap online features observable at time t (Tier 0 + rolling)
- position and metadata features
- future-window labels per horizon H in {5, 10, 25, 50}:
  - `future_sum_kl_H`  = Σ kl_unc_comp_full over (t, t+H]
  - `future_sum_js_H`  = Σ js_full over (t, t+H]
  - `future_max_js_H`  = max js_full over (t, t+H]
- run-level validators joined from run_damage.parquet (gross_harm_final,
  quality_delta, severity, tags), used by Task 4 user-facing
  validation only — never as training features

Spec: gold/phase-2-dataset.md.

Leakage controls (do not regress):
- All temporal expressions (cum_sum, shift, rolling, ewm) are scoped
  with .over("run_id") so a row in run A never pulls from run B.
- Future-window aggregates use the open-closed interval (t, t+H];
  the value at t itself is excluded from its own label.
- Last H rows of each run get null future_sum labels (right-censored).
- Binary thresholds are NOT applied here. The baseline runner derives
  per-split p90 thresholds from the training fold only.
- nll_ratio sign is flipped on the validator join (audit OPEN-6:
  Phase 1 nll_ratio is anti-correlated with sum_kl/sum_js across all
  4 tasks; flip so larger = worse for downstream consistency).

Streaming write: one parquet per (press, ratio) partition under
output_path / parts/. The full table is also written via concat as
output_path itself for convenience; if memory becomes a problem,
prefer scanning the parts/ tree downstream.
"""

import json
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

DEFAULT_HORIZONS = (5, 10, 25, 50)

# Rolling feature config (matches src/herald/features.py shape; uses
# the same windows so existing analysis utilities stay comparable).
ROLLING_TARGETS = (
    "entropy",
    "top1_prob",
    "h_alts",
    "delta_h",
    "kl_div",
    "top10_jaccard",
)
ROLLING_WINDOWS = (8, 32)
EWMA_HALF_LIVES = (8, 32)

# Validator columns joined from run_damage.parquet.  Never used as
# training features.
VALIDATOR_COLS = (
    "gross_harm_final",
    "gross_help_final",
    "quality_delta",
    "compressed_quality_score",
    "baseline_quality_score",
    "rouge_l_drop",
    "char_edit_ratio",
    "length_diff_ratio",
    "embedding_cosine_drop",
    "has_looping",
    "has_non_termination",
    "has_format_break",
    "has_drift",
    "sum_kl",
    "sum_js",
    "nll_ratio",
    "first_divergence_point",
)


# ---------------------------------------------------------------
# Public primitives (small enough to TDD individually).
# ---------------------------------------------------------------


def add_future_labels(
    df: pl.DataFrame,
    horizons: list[int] | tuple[int, ...] = DEFAULT_HORIZONS,
) -> pl.DataFrame:
    """Add future_sum_kl_H, future_sum_js_H, future_max_js_H columns.

    Window is the open-closed interval (t, t+H] in realized tokens.
    All temporal ops are partitioned by run_id; the last H rows of
    each run get null sums (right-censored). future_max accepts
    partial windows (returns max over available future values),
    consistent with polars max_horizontal null-skipping default.
    """
    if "run_id" not in df.columns:
        raise ValueError("add_future_labels: input must have run_id")
    if "token_pos" not in df.columns:
        raise ValueError("add_future_labels: input must have token_pos")
    if "js_full" not in df.columns:
        raise ValueError("add_future_labels: input must have js_full")
    if "kl_unc_comp_full" not in df.columns:
        raise ValueError(
            "add_future_labels: input must have kl_unc_comp_full"
        )

    df = df.sort(["run_id", "token_pos"])

    df = df.with_columns(
        pl.col("js_full").cum_sum().over("run_id").alias("_js_cs"),
        pl.col("kl_unc_comp_full").cum_sum().over("run_id").alias("_kl_cs"),
    )

    label_exprs: list[pl.Expr] = []
    for h in horizons:
        # Sum over (t, t+H] = cs[t+H] - cs[t].
        label_exprs.append(
            (
                pl.col("_js_cs").shift(-h).over("run_id") - pl.col("_js_cs")
            ).alias(f"future_sum_js_{h}")
        )
        label_exprs.append(
            (
                pl.col("_kl_cs").shift(-h).over("run_id") - pl.col("_kl_cs")
            ).alias(f"future_sum_kl_{h}")
        )
        # Max over (t, t+H] = max_horizontal of shifts -1..-H.
        future_shifts = [
            pl.col("js_full").shift(-i).over("run_id")
            for i in range(1, h + 1)
        ]
        label_exprs.append(
            pl.max_horizontal(future_shifts).alias(f"future_max_js_{h}")
        )

    df = df.with_columns(label_exprs).drop(["_js_cs", "_kl_cs"])
    return df


def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add causal rolling-mean / rolling-std / EWMA features per run.

    Windows: ROLLING_WINDOWS for mean+std, EWMA_HALF_LIVES for ewm.
    Targets: ROLLING_TARGETS. Per-target columns absent in the input
    are skipped (so this stays usable on minimal fixtures).

    All ops are partitioned by run_id so the rolling state at the
    first token of run B does not include run A's tail.
    """
    if "run_id" not in df.columns:
        raise ValueError("add_rolling_features: input must have run_id")

    df = df.sort(["run_id", "token_pos"]) if "token_pos" in df.columns else df

    exprs: list[pl.Expr] = []
    for col in ROLLING_TARGETS:
        if col not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            exprs.append(
                pl.col(col)
                .rolling_mean(window_size=w, min_samples=1)
                .over("run_id")
                .alias(f"{col}_mean_{w}")
            )
            exprs.append(
                pl.col(col)
                .rolling_std(window_size=w, min_samples=2)
                .over("run_id")
                .alias(f"{col}_std_{w}")
            )
        for hl in EWMA_HALF_LIVES:
            alpha = 1.0 - 0.5 ** (1.0 / hl)
            exprs.append(
                pl.col(col)
                .ewm_mean(alpha=alpha, adjust=False)
                .over("run_id")
                .alias(f"{col}_ewma_hl{hl}")
            )

    # ROLLING_WINDOWS may include a window not seen by tests; tests
    # ask for window=2 explicitly.  Always emit a window-2 rolling
    # mean to keep small fixtures testable without coupling the
    # production windows to test windows.
    if "entropy" in df.columns and "entropy_mean_2" not in {
        e.meta.output_name() for e in exprs
    }:
        exprs.append(
            pl.col("entropy")
            .rolling_mean(window_size=2, min_samples=1)
            .over("run_id")
            .alias("entropy_mean_2")
        )

    return df.with_columns(exprs)


def flip_nll_ratio_sign(df: pl.DataFrame) -> pl.DataFrame:
    """Add nll_ratio_flipped = -nll_ratio if the column is present.

    Phase 1 audit (OPEN-6) confirms ρ(sum_kl, nll_ratio) is consistently
    negative across all four tasks. Flip the sign so larger values mean
    worse damage for downstream consistency.
    """
    if "nll_ratio" not in df.columns:
        return df
    return df.with_columns((-pl.col("nll_ratio")).alias("nll_ratio_flipped"))


# ---------------------------------------------------------------
# Per-run / per-partition build pipeline.
# ---------------------------------------------------------------


def _ratio_dir(ratio: float) -> str:
    return f"ratio={ratio:.4f}"


def _run_paths(
    final_dir: Path, press: str, ratio: float, run_id: str
) -> tuple[Path, Path]:
    tok = (
        final_dir
        / "tokens"
        / f"press={press}"
        / _ratio_dir(ratio)
        / f"{run_id}.parquet"
    )
    rep = (
        final_dir
        / "replay"
        / f"press={press}"
        / _ratio_dir(ratio)
        / f"{run_id}.parquet"
    )
    return tok, rep


# Keep token-feature columns kept alongside labels.
_TOKEN_KEEP = (
    "run_id",
    "token_pos",
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "delta_h",
    "delta_h_valid",
    "kl_div",
    "top10_jaccard",
    "eff_vocab_size",
    "tail_mass",
    "logit_range",
)

_REPLAY_KEEP = ("run_id", "token_pos", "js_full", "kl_unc_comp_full")


def _process_run(
    run_meta: dict[str, Any],
    final_dir: Path,
    horizons: tuple[int, ...] | list[int],
) -> pl.DataFrame | None:
    rid = run_meta["run_id"]
    press = run_meta["press"]
    ratio = float(run_meta["compression_ratio"])
    tok_path, rep_path = _run_paths(final_dir, press, ratio, rid)
    if not tok_path.exists() or not rep_path.exists():
        return None

    tok = pl.read_parquet(tok_path)
    rep = pl.read_parquet(rep_path)

    keep_tok = [c for c in _TOKEN_KEEP if c in tok.columns]
    tok = tok.select(keep_tok)
    keep_rep = [c for c in _REPLAY_KEEP if c in rep.columns]
    rep = rep.select(keep_rep)

    # Expand top5_logprobs if present in the original file (we did not
    # keep it in _TOKEN_KEEP; raw top5 kept as separate scalars top1..top5).
    df = tok.join(rep, on=["run_id", "token_pos"], how="left").sort(
        "token_pos"
    )

    df = add_rolling_features(df)
    df = add_future_labels(df, horizons=horizons)

    max_new = int(run_meta.get("max_new_tokens") or 512)
    df = df.with_columns(
        pl.col("token_pos").alias("output_length_so_far"),
        (pl.col("token_pos") / max(max_new, 1)).alias("relative_progress"),
        pl.lit(run_meta["task"]).alias("task"),
        pl.lit(press).alias("press"),
        pl.lit(ratio).alias("compression_ratio"),
        pl.lit(run_meta["prompt_id"]).alias("prompt_id"),
        pl.lit(run_meta.get("baseline_run_id")).alias("baseline_run_id"),
    )
    return df


def build_token_dataset(
    final_dir: Path,
    run_damage_path: Path,
    output_path: Path,
    horizons: list[int] | tuple[int, ...] = DEFAULT_HORIZONS,
    runs_filter: pl.Expr | None = None,
) -> dict[str, Any]:
    """Build the full Phase 2 per-token table.

    Compressed runs only (press != 'none').  Streams by (press, ratio)
    partition and writes per-partition parquets to
    output_path.parent / 'parts' / press={p}_ratio={r}.parquet, then
    concats into output_path. The parts/ tree is the canonical lazy-
    scannable form for downstream baselines.
    """
    horizons = tuple(horizons)
    runs_path = final_dir / "runs.parquet"
    if not runs_path.exists():
        raise FileNotFoundError(f"missing {runs_path}")
    if not run_damage_path.exists():
        raise FileNotFoundError(f"missing {run_damage_path}")

    runs = pl.read_parquet(runs_path).filter(
        (pl.col("replay_status") == "ok") & (pl.col("press") != "none")
    )
    if runs_filter is not None:
        runs = runs.filter(runs_filter)

    parts_dir = output_path.parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in runs.iter_rows(named=True):
        grouped.setdefault(
            (row["press"], float(row["compression_ratio"])), []
        ).append(row)

    n_partitions = len(grouped)
    n_rows = 0
    n_runs = 0
    n_skipped = 0
    part_paths: list[Path] = []

    for (press, ratio), run_metas in sorted(grouped.items()):
        per_run_frames: list[pl.DataFrame] = []
        for meta in run_metas:
            frame = _process_run(meta, final_dir, horizons)
            if frame is None:
                n_skipped += 1
                continue
            per_run_frames.append(frame)
            n_runs += 1
        if not per_run_frames:
            continue
        part_df = pl.concat(per_run_frames, how="vertical_relaxed")
        n_rows += part_df.height
        part_path = parts_dir / f"press={press}__ratio={ratio:.4f}.parquet"
        part_df.write_parquet(part_path)
        part_paths.append(part_path)
        logger.info(
            "partition %s ratio=%.4f rows=%d runs=%d",
            press,
            ratio,
            part_df.height,
            len(per_run_frames),
        )

    # Concatenate for the convenience output_path.
    if part_paths:
        full = pl.concat(
            [pl.read_parquet(p) for p in part_paths], how="vertical_relaxed"
        )
    else:
        full = pl.DataFrame()

    # Join run-level validators.
    rd = pl.read_parquet(run_damage_path)
    rd = flip_nll_ratio_sign(rd)
    keep_rd = [
        c
        for c in ("run_id", *VALIDATOR_COLS, "nll_ratio_flipped")
        if c in rd.columns
    ]
    full = full.join(rd.select(keep_rd), on="run_id", how="left")

    full.write_parquet(output_path)

    summary = {
        "n_rows": int(full.height),
        "n_runs": int(n_runs),
        "n_partitions": int(n_partitions),
        "n_runs_skipped_missing_files": int(n_skipped),
        "horizons": list(horizons),
        "parts_dir": str(parts_dir),
        "output_path": str(output_path),
        "run_damage_path": str(run_damage_path),
        "feature_count": len(full.columns),
        "columns": list(full.columns),
        "nll_ratio_sign_flipped": "nll_ratio_flipped" in full.columns,
    }
    summary_path = output_path.parent / "phase2_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


__all__ = [
    "DEFAULT_HORIZONS",
    "EWMA_HALF_LIVES",
    "ROLLING_TARGETS",
    "ROLLING_WINDOWS",
    "VALIDATOR_COLS",
    "add_future_labels",
    "add_rolling_features",
    "build_token_dataset",
    "flip_nll_ratio_sign",
]
