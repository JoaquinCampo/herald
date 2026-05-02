"""Failure-onset event-study aggregation.

Aligns catastrophic compressed runs at estimated failure onset
and produces per-feature trajectory aggregates around that onset.
The figure this produces is the canonical HERALD "what does
compression collapse look like at the logit level around onset?"
view.

Phase 0 deliverable is smoke-only: 134 runs, 1 task, 2 presses.
Final science claims wait for Phase 1.

Schema assumption: per-run `catastrophe_onsets` is not in
`runs.parquet` (see `metrics/io.py:RUNS_SCHEMA`), so onsets are
re-derived from `generated_token_ids` via `detect_looping_onset`
and the labeling.py non-termination proxy.
"""

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from herald.detectors import detect_looping_onset
from herald.labeling import DEFAULT_NT_ONSET_FRAC

DEFAULT_FEATURES: tuple[str, ...] = (
    "entropy",
    "top1_prob",
    "top1_top2_margin",
    "kl_div",
    "h_alts",
    "delta_h",
    "top10_jaccard",
    "tail_mass",
)
DEFAULT_WINDOW_BEFORE = 200
DEFAULT_WINDOW_AFTER = 200
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_PLOT_DPI = 120


@dataclass(frozen=True)
class EventStudyConfig:
    window_before: int = DEFAULT_WINDOW_BEFORE
    window_after: int = DEFAULT_WINDOW_AFTER
    features: tuple[str, ...] = DEFAULT_FEATURES
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    rouge_threshold: float | None = None
    seed: int = 0


@dataclass(frozen=True)
class OnsetRecord:
    run_id: str
    press: str
    compression_ratio: float
    onset_token: int
    onset_source: str
    n_tokens: int


@dataclass
class EventStudyResult:
    summary: dict[str, Any]
    event_df: pl.DataFrame = field(default_factory=pl.DataFrame)
    pooled: pl.DataFrame = field(default_factory=pl.DataFrame)
    by_press: pl.DataFrame = field(default_factory=pl.DataFrame)
    by_onset_source: pl.DataFrame = field(default_factory=pl.DataFrame)


def derive_onset(
    catastrophes: list[str],
    token_ids: list[int],
    max_new_tokens: int,
    nt_onset_frac: float,
) -> tuple[int | None, str]:
    """Pragmatic Phase 0 onset.

    Priority:
    - looping: window-based onset from detect_looping_onset
    - non_termination: proxy at nt_onset_frac * max_new_tokens
    - otherwise: None
    """
    if "looping" in catastrophes:
        onset = detect_looping_onset(token_ids)
        if onset is not None:
            return onset, "looping"
    if "non_termination" in catastrophes and token_ids:
        proxy = int(nt_onset_frac * max_new_tokens)
        proxy = min(proxy, len(token_ids) - 1)
        proxy = max(proxy, 0)
        return proxy, "non_termination_proxy"
    return None, "none"


def select_catastrophic_runs(
    runs: pl.DataFrame,
    sequence_metrics: pl.DataFrame | None = None,
    rouge_threshold: float | None = None,
) -> pl.DataFrame:
    """Compressed runs with at least one trainable catastrophe.

    Compressed = press != 'none'. Trainable = looping or
    non_termination present in the catastrophes list. If
    rouge_threshold is supplied alongside sequence_metrics,
    additionally gate on rouge_l < threshold (or null, which we
    treat as severely degraded).
    """
    df = runs.filter(pl.col("press") != "none")
    df = df.with_columns(
        pl.col("catastrophes").list.contains("looping").alias("_has_looping"),
        pl.col("catastrophes")
        .list.contains("non_termination")
        .alias("_has_nt"),
    )
    df = df.filter(pl.col("_has_looping") | pl.col("_has_nt"))
    if rouge_threshold is not None and sequence_metrics is not None:
        df = df.join(
            sequence_metrics.select(["run_id", "rouge_l"]),
            on="run_id",
            how="left",
        )
        df = df.filter(
            pl.col("rouge_l").is_null()
            | (pl.col("rouge_l") < rouge_threshold)
        )
    return df.drop(["_has_looping", "_has_nt"])


def load_tokens_for_runs(
    tokens_root: Path,
    run_ids: Iterable[str],
) -> pl.DataFrame:
    """Read every partition under tokens_root and filter to run_ids."""
    files = sorted(tokens_root.rglob("*.parquet"))
    if not files:
        return pl.DataFrame()
    df = pl.concat(
        [pl.read_parquet(p) for p in files], how="vertical_relaxed"
    )
    return df.filter(pl.col("run_id").is_in(list(run_ids)))


def _add_derived_features(tokens: pl.DataFrame) -> pl.DataFrame:
    """Add cheap derived features that are not in the raw schema.

    top1_top2_margin = logprob_0 - logprob_1, where the column is
    the per-token list `top5_logprobs`. NaN if the list has fewer
    than two entries.
    """
    if "top1_top2_margin" in tokens.columns:
        return tokens
    if "top5_logprobs" not in tokens.columns:
        return tokens
    return tokens.with_columns(
        pl.when(pl.col("top5_logprobs").list.len() >= 2)
        .then(
            pl.col("top5_logprobs").list.get(0)
            - pl.col("top5_logprobs").list.get(1)
        )
        .otherwise(float("nan"))
        .cast(pl.Float32)
        .alias("top1_top2_margin")
    )


def collect_onsets(
    runs: pl.DataFrame,
    cfg: EventStudyConfig,
) -> tuple[list[OnsetRecord], dict[str, int]]:
    """Walk runs and derive an onset for each, recording exclusion
    counts.
    """
    onsets: list[OnsetRecord] = []
    excluded: dict[str, int] = {
        "no_token_ids": 0,
        "no_onset_derivable": 0,
    }
    for row in runs.iter_rows(named=True):
        token_ids = list(row.get("generated_token_ids") or [])
        if not token_ids:
            excluded["no_token_ids"] += 1
            continue
        onset, source = derive_onset(
            list(row.get("catastrophes") or []),
            token_ids,
            int(row.get("max_new_tokens") or 512),
            cfg.nt_onset_frac,
        )
        if onset is None:
            excluded["no_onset_derivable"] += 1
            continue
        onsets.append(
            OnsetRecord(
                run_id=row["run_id"],
                press=row["press"],
                compression_ratio=float(row["compression_ratio"]),
                onset_token=onset,
                onset_source=source,
                n_tokens=len(token_ids),
            )
        )
    return onsets, excluded


def build_event_dataframe(
    onsets: list[OnsetRecord],
    tokens: pl.DataFrame,
    cfg: EventStudyConfig,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Construct the long event-aligned dataframe.

    Returns (event_df, used_features, missing_features). event_df
    columns: run_id, press, compression_ratio, onset_source,
    onset_token, relative_pos, feature, value.
    """
    if not onsets or tokens.is_empty():
        return pl.DataFrame(), [], list(cfg.features)

    tokens = _add_derived_features(tokens)

    onset_df = pl.DataFrame(
        {
            "run_id": [o.run_id for o in onsets],
            "press": [o.press for o in onsets],
            "compression_ratio": [o.compression_ratio for o in onsets],
            "onset_token": [o.onset_token for o in onsets],
            "onset_source": [o.onset_source for o in onsets],
        }
    )

    used = [f for f in cfg.features if f in tokens.columns]
    missing = [f for f in cfg.features if f not in tokens.columns]
    if not used:
        return pl.DataFrame(), used, missing

    df = tokens.join(onset_df, on="run_id", how="inner")
    df = df.with_columns(
        (pl.col("token_pos") - pl.col("onset_token")).alias("relative_pos")
    )
    df = df.filter(
        pl.col("relative_pos").is_between(
            -cfg.window_before, cfg.window_after
        )
    )
    keep_cols = [
        "run_id",
        "press",
        "compression_ratio",
        "onset_source",
        "onset_token",
        "relative_pos",
    ] + used
    df = df.select(keep_cols)
    long = df.unpivot(
        index=[
            "run_id",
            "press",
            "compression_ratio",
            "onset_source",
            "onset_token",
            "relative_pos",
        ],
        on=used,
        variable_name="feature",
        value_name="value",
    )
    return long, used, missing


def _bootstrap_ci(
    vals: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean.

    Falls back to (mean, mean) when the sample is too small to
    bootstrap meaningfully.
    """
    if vals.size < 2 or n_boot <= 0:
        m = float(np.mean(vals))
        return m, m
    idx = rng.integers(0, vals.size, (n_boot, vals.size))
    boot_means = np.mean(vals[idx], axis=1)
    return (
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def aggregate(
    event_df: pl.DataFrame,
    cfg: EventStudyConfig,
    group: str | None = None,
) -> pl.DataFrame:
    """Per (group?, feature, relative_pos): n, mean, median, CI95."""
    if event_df.is_empty():
        return pl.DataFrame()

    rng = np.random.default_rng(cfg.seed)
    keys = ["feature", "relative_pos"]
    if group is not None:
        keys = [group] + keys

    rows: list[dict[str, Any]] = []
    for key_vals, sub in event_df.group_by(keys, maintain_order=True):
        vals = sub.get_column("value").drop_nulls().to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean = float(np.mean(vals))
        median = float(np.median(vals))
        ci_lo, ci_hi = _bootstrap_ci(vals, cfg.n_bootstrap, rng)
        record: dict[str, Any] = {
            "n": int(vals.size),
            "mean": mean,
            "median": median,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
        if group is not None:
            record[group] = key_vals[0]
            record["feature"] = key_vals[1]
            record["relative_pos"] = int(key_vals[2])
        else:
            record["feature"] = key_vals[0]
            record["relative_pos"] = int(key_vals[1])
        rows.append(record)
    return pl.DataFrame(rows)


def plot_event_study(
    agg: pl.DataFrame,
    output_path: Path,
    title: str,
    group_col: str | None = None,
) -> None:
    """One subplot per feature; mean line with bootstrap CI band.

    Pure CPU. No-op (creates an empty file) if agg is empty.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if agg.is_empty():
        # Touch a placeholder so callers can rely on the path existing.
        output_path.write_bytes(b"")
        return

    features = sorted(agg.get_column("feature").unique().to_list())
    n = len(features)
    cols = min(3, n) if n > 0 else 1
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 3.0 * rows), squeeze=False
    )

    for i, feat in enumerate(features):
        ax = axes[i // cols][i % cols]
        sub = agg.filter(pl.col("feature") == feat).sort("relative_pos")
        if group_col is not None and group_col in sub.columns:
            for key_vals, g in sub.group_by(group_col, maintain_order=True):
                label = str(key_vals[0])
                xs = g.get_column("relative_pos").to_numpy()
                m = g.get_column("mean").to_numpy()
                lo = g.get_column("ci_lo").to_numpy()
                hi = g.get_column("ci_hi").to_numpy()
                (line,) = ax.plot(xs, m, label=label, linewidth=1.2)
                ax.fill_between(
                    xs, lo, hi, alpha=0.15, color=line.get_color()
                )
            ax.legend(fontsize=7)
        else:
            xs = sub.get_column("relative_pos").to_numpy()
            m = sub.get_column("mean").to_numpy()
            lo = sub.get_column("ci_lo").to_numpy()
            hi = sub.get_column("ci_hi").to_numpy()
            ax.plot(xs, m, color="C3", linewidth=1.2, label="catastrophic")
            ax.fill_between(xs, lo, hi, alpha=0.18, color="C3")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("relative_pos (tokens from onset)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.2)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def run_event_study(
    input_root: Path,
    output_dir: Path,
    cfg: EventStudyConfig,
) -> EventStudyResult:
    """Top-level Phase 0 entry point.

    Reads runs.parquet + token partitions from `input_root` and
    writes event_study.parquet, summary JSON, and figures to
    `output_dir`.
    """
    runs_path = input_root / "final" / "runs.parquet"
    tokens_root = input_root / "final" / "tokens"
    seq_path = input_root / "metrics" / "sequence_metrics.parquet"

    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "runs_path": str(runs_path),
        "tokens_root": str(tokens_root),
        "sequence_metrics_path": str(seq_path),
        "window_before": cfg.window_before,
        "window_after": cfg.window_after,
        "features_requested": list(cfg.features),
        "nt_onset_frac": cfg.nt_onset_frac,
        "rouge_threshold": cfg.rouge_threshold,
        "n_bootstrap": cfg.n_bootstrap,
        "schema_note": (
            "catastrophe_onsets is not present in runs.parquet; "
            "onsets are re-derived from generated_token_ids via "
            "detect_looping_onset (looping) and the labeling "
            "non-termination proxy."
        ),
        "phase0_caveat": (
            "Phase 0 smoke output. No control group is emitted: "
            "Phase 0 lacks a clean compressed-non-catastrophic "
            "stratum. Final science claims wait for Phase 1."
        ),
    }

    blockers: list[str] = []
    if not runs_path.exists():
        blockers.append(f"missing runs parquet: {runs_path}")
    if not tokens_root.exists():
        blockers.append(f"missing tokens directory: {tokens_root}")
    if blockers:
        summary["blockers"] = blockers
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "event_study_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return EventStudyResult(summary=summary)

    runs = pl.read_parquet(runs_path)
    seq = pl.read_parquet(seq_path) if seq_path.exists() else None
    summary["n_total_runs"] = runs.height

    catastrophic = select_catastrophic_runs(
        runs,
        sequence_metrics=seq,
        rouge_threshold=cfg.rouge_threshold,
    )
    summary["n_catastrophic_runs_selected"] = catastrophic.height

    onsets, excluded = collect_onsets(catastrophic, cfg)
    summary["excluded_counts"] = excluded
    summary["n_runs_with_onset"] = len(onsets)

    onset_source_counts: dict[str, int] = {}
    for o in onsets:
        onset_source_counts[o.onset_source] = (
            onset_source_counts.get(o.onset_source, 0) + 1
        )
    summary["onset_source_counts"] = onset_source_counts

    if not onsets:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "event_study_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return EventStudyResult(summary=summary)

    tokens = load_tokens_for_runs(tokens_root, [o.run_id for o in onsets])
    summary["n_token_rows_loaded"] = tokens.height

    event_df, used, missing = build_event_dataframe(onsets, tokens, cfg)
    summary["features_used"] = used
    summary["features_missing"] = missing
    summary["n_event_rows"] = event_df.height

    pooled = aggregate(event_df, cfg, group=None)
    by_press = aggregate(event_df, cfg, group="press")
    by_source = aggregate(event_df, cfg, group="onset_source")

    summary["grouping"] = {
        "pooled": pooled.height,
        "by_press": by_press.height,
        "by_onset_source": by_source.height,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    event_df.write_parquet(output_dir / "event_study.parquet")
    pooled.write_parquet(output_dir / "event_study_agg_pooled.parquet")
    by_press.write_parquet(output_dir / "event_study_agg_by_press.parquet")
    by_source.write_parquet(
        output_dir / "event_study_agg_by_onset_source.parquet"
    )

    # Headline figure: stratified by onset_source so the proxy
    # cohort never silently dilutes the looping cohort.
    plot_event_study(
        by_source,
        output_dir / "event_study_features.png",
        title=(
            "Event study around failure onset (Phase 0 smoke; "
            "stratified by onset source)"
        ),
        group_col="onset_source",
    )
    plot_event_study(
        by_press,
        output_dir / "event_study_features_by_press.png",
        title="Event study by press (Phase 0 smoke)",
        group_col="press",
    )

    summary["artifacts"] = {
        "event_study_parquet": str(output_dir / "event_study.parquet"),
        "agg_pooled_parquet": str(
            output_dir / "event_study_agg_pooled.parquet"
        ),
        "agg_by_press_parquet": str(
            output_dir / "event_study_agg_by_press.parquet"
        ),
        "agg_by_onset_source_parquet": str(
            output_dir / "event_study_agg_by_onset_source.parquet"
        ),
        "headline_png": str(output_dir / "event_study_features.png"),
        "by_press_png": str(output_dir / "event_study_features_by_press.png"),
    }
    summary_path = output_dir / "event_study_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    return EventStudyResult(
        summary=summary,
        event_df=event_df,
        pooled=pooled,
        by_press=by_press,
        by_onset_source=by_source,
    )
