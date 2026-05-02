"""Per-feature lead-time analysis around failure onset.

Quantifies how many tokens before catastrophic onset each cheap
online feature begins to separate catastrophic compressed runs from
healthy compressed controls. CPU-only; reuses event-study primitives
for run selection, onset derivation, and token alignment.

Method
------
1. Catastrophic runs: same selection as the event study (looping or
   non_termination tag, with optional rouge gating).
2. Healthy controls: compressed runs (press != 'none') with neither
   looping nor non_termination tags, matched to catastrophic runs by
   (task, press, compression_ratio).
3. Virtual control onset: per (task, press, ratio) stratum, controls
   are assigned a synthetic onset equal to the median catastrophic
   onset in that stratum (clipped to the control's own length). This
   keeps relative-position alignment apples-to-apples without faking
   a real failure point in healthy runs.
4. At every relative_pos in [-window_before, +window_after] (strided),
   gather catastrophic-vs-control feature values and compute AUROC
   with bootstrap CIs over runs.
5. Lead time = earliest negative relative position where AUROC >=
   threshold, CI lower bound > 0.5, and the condition persists for at
   least `persistence` consecutive evaluated positions.

Phase 0 caveat: the compressed-non-catastrophic stratum on Phase 0
is small (single task, two presses, three ratios). The summary JSON
flags `control_limited=True` when matched control coverage falls
below a configurable per-stratum minimum.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from herald.analysis.event_study import (
    DEFAULT_FEATURES,
    OnsetRecord,
    _add_derived_features,
    collect_onsets,
    load_tokens_for_runs,
    select_catastrophic_runs,
)
from herald.labeling import DEFAULT_NT_ONSET_FRAC

DEFAULT_WINDOW_BEFORE = 200
DEFAULT_WINDOW_AFTER = 50
DEFAULT_POSITION_STRIDE = 1
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_AUROC_THRESHOLD = 0.65
DEFAULT_CI_LOWER_THRESHOLD = 0.5
DEFAULT_PERSISTENCE = 5
DEFAULT_MIN_PER_STRATUM = 3
DEFAULT_MIN_WELL_COVERED_STRATA = 3
DEFAULT_PLOT_DPI = 120

CATASTROPHE_TAGS = ("looping", "non_termination")


@dataclass(frozen=True)
class LeadTimeConfig:
    window_before: int = DEFAULT_WINDOW_BEFORE
    window_after: int = DEFAULT_WINDOW_AFTER
    position_stride: int = DEFAULT_POSITION_STRIDE
    features: tuple[str, ...] = DEFAULT_FEATURES
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    rouge_threshold: float | None = None
    auroc_threshold: float = DEFAULT_AUROC_THRESHOLD
    ci_lower_threshold: float = DEFAULT_CI_LOWER_THRESHOLD
    persistence: int = DEFAULT_PERSISTENCE
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM
    min_well_covered_strata: int = DEFAULT_MIN_WELL_COVERED_STRATA
    seed: int = 0


@dataclass(frozen=True)
class ControlRecord:
    run_id: str
    task: str
    press: str
    compression_ratio: float
    n_tokens: int
    virtual_onset: int


@dataclass
class LeadTimeResult:
    summary: dict[str, Any]
    by_feature: pl.DataFrame = field(default_factory=pl.DataFrame)
    by_feature_press: pl.DataFrame = field(default_factory=pl.DataFrame)


def select_healthy_controls(runs: pl.DataFrame) -> pl.DataFrame:
    """Compressed runs without any catastrophe tag of interest.

    Compressed = press != 'none'. Healthy = catastrophes list does
    not contain 'looping' nor 'non_termination'. (Other tags such
    as 'wrong_answer' are tolerated; they are answer-quality signals
    rather than trajectory-collapse signals, and excluding them
    would shrink the control pool below usability.)
    """
    df = runs.filter(pl.col("press") != "none")
    df = df.with_columns(
        pl.col("catastrophes").list.contains("looping").alias("_has_loop"),
        pl.col("catastrophes")
        .list.contains("non_termination")
        .alias("_has_nt"),
    )
    df = df.filter(~(pl.col("_has_loop") | pl.col("_has_nt")))
    return df.drop(["_has_loop", "_has_nt"])


def _stratum_key(
    task: str, press: str, ratio: float
) -> tuple[str, str, float]:
    # Round to a stable 4-decimal float so e.g. 0.875 == 0.8750.
    return task, press, round(float(ratio), 4)


def assign_virtual_onsets(
    onsets: list[OnsetRecord],
    controls: pl.DataFrame,
    runs: pl.DataFrame,
    cfg: LeadTimeConfig,
) -> tuple[list[ControlRecord], dict[str, Any]]:
    """Pair healthy controls to catastrophic runs by stratum.

    For each (task, press, ratio) stratum that has at least one
    catastrophic run with a derivable onset, controls in the same
    stratum are assigned a virtual onset equal to the median
    catastrophic onset, clipped to the control's own length.

    Strata without catastrophic onsets, or without any control,
    contribute zero pairs. Counts are returned for the summary.
    """
    runs_by_id = {row["run_id"]: row for row in runs.iter_rows(named=True)}
    cat_strata: dict[tuple[str, str, float], list[int]] = {}
    for o in onsets:
        run = runs_by_id.get(o.run_id)
        task = str(run.get("task")) if run else "unknown"
        key = _stratum_key(task, o.press, o.compression_ratio)
        cat_strata.setdefault(key, []).append(o.onset_token)

    stratum_median: dict[tuple[str, str, float], int] = {
        k: int(np.median(v)) for k, v in cat_strata.items() if v
    }

    matched: list[ControlRecord] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    n_dropped_no_cat = 0
    n_dropped_short = 0
    for row in controls.iter_rows(named=True):
        token_ids = list(row.get("generated_token_ids") or [])
        if not token_ids:
            continue
        task = str(row.get("task") or "unknown")
        press = str(row.get("press"))
        ratio = float(row.get("compression_ratio") or 0.0)
        key = _stratum_key(task, press, ratio)
        if key not in stratum_median:
            n_dropped_no_cat += 1
            continue
        v_onset = stratum_median[key]
        # Clip to a position the control actually reaches.
        v_onset = min(v_onset, len(token_ids) - 1)
        v_onset = max(v_onset, 0)
        # Drop trivially short controls (no usable window at all).
        if len(token_ids) < 2:
            n_dropped_short += 1
            continue
        matched.append(
            ControlRecord(
                run_id=row["run_id"],
                task=task,
                press=press,
                compression_ratio=ratio,
                n_tokens=len(token_ids),
                virtual_onset=v_onset,
            )
        )
        skey = f"{task}|{press}|{round(ratio, 4)}"
        bucket = stratum_counts.setdefault(
            skey, {"n_cat": len(cat_strata[key]), "n_ctrl": 0}
        )
        bucket["n_ctrl"] += 1

    info: dict[str, Any] = {
        "n_strata": len(cat_strata),
        "n_strata_with_controls": sum(
            1 for b in stratum_counts.values() if b["n_ctrl"] > 0
        ),
        "stratum_counts": stratum_counts,
        "controls_dropped_no_catastrophic_in_stratum": n_dropped_no_cat,
        "controls_dropped_too_short": n_dropped_short,
    }
    return matched, info


def build_aligned_long(
    onsets: list[OnsetRecord],
    controls: list[ControlRecord],
    tokens: pl.DataFrame,
    cfg: LeadTimeConfig,
    runs: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Long dataframe: one row per (run, relative_pos, feature).

    Adds a `label` column: 1 for catastrophic, 0 for healthy control.
    Filters to relative positions stepped by cfg.position_stride
    inside the [-window_before, +window_after] window.
    """
    if tokens.is_empty():
        return pl.DataFrame(), [], list(cfg.features)

    tokens = _add_derived_features(tokens)
    used = [f for f in cfg.features if f in tokens.columns]
    missing = [f for f in cfg.features if f not in tokens.columns]
    if not used:
        return pl.DataFrame(), used, missing

    runs_task = {
        r["run_id"]: str(r.get("task") or "unknown")
        for r in runs.iter_rows(named=True)
    }

    cat_rows = [
        {
            "run_id": o.run_id,
            "task": runs_task.get(o.run_id, "unknown"),
            "press": o.press,
            "compression_ratio": float(o.compression_ratio),
            "onset_token": o.onset_token,
            "label": 1,
        }
        for o in onsets
    ]
    ctrl_rows = [
        {
            "run_id": c.run_id,
            "task": c.task,
            "press": c.press,
            "compression_ratio": c.compression_ratio,
            "onset_token": c.virtual_onset,
            "label": 0,
        }
        for c in controls
    ]
    if not cat_rows and not ctrl_rows:
        return pl.DataFrame(), used, missing

    onset_df = pl.DataFrame(cat_rows + ctrl_rows)

    df = tokens.join(onset_df, on="run_id", how="inner")
    df = df.with_columns(
        (pl.col("token_pos") - pl.col("onset_token")).alias("relative_pos")
    )
    df = df.filter(
        pl.col("relative_pos").is_between(
            -cfg.window_before, cfg.window_after
        )
    )
    if cfg.position_stride > 1:
        df = df.filter((pl.col("relative_pos") % cfg.position_stride) == 0)

    keep_cols = [
        "run_id",
        "task",
        "press",
        "compression_ratio",
        "label",
        "onset_token",
        "relative_pos",
    ] + used
    df = df.select(keep_cols)
    long = df.unpivot(
        index=[
            "run_id",
            "task",
            "press",
            "compression_ratio",
            "label",
            "onset_token",
            "relative_pos",
        ],
        on=used,
        variable_name="feature",
        value_name="value",
    )
    long = long.filter(pl.col("value").is_finite())
    return long, used, missing


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Directional AUROC that returns None when degenerate.

    Convention: score is expected to be pre-oriented so that higher
    scores correspond to catastrophic. Directional AUROC means the
    null distribution is symmetric around 0.5 (no upward fold), so
    the spec's "AUROC >= 0.65 AND CI lower > 0.5" thresholds keep
    their literal frequentist meaning. Per-feature orientation is
    chosen up the call stack (see `infer_feature_direction`) by
    inspecting the pooled cat-vs-ctrl mean difference.
    """
    if labels.size == 0 or scores.size == 0:
        return None
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos < 1 or neg < 1:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def infer_feature_direction(long: pl.DataFrame) -> dict[str, int]:
    """Per-feature sign so that pre-oriented scores rise with cat.

    For each feature, compute the pooled-mean difference
    `mean(values | label=1) - mean(values | label=0)`. If positive,
    direction is +1 (no flip); otherwise -1 (multiply scores by -1
    before AUROC). Features with degenerate or empty pools default
    to +1.
    """
    directions: dict[str, int] = {}
    if long.is_empty():
        return directions
    grouped = long.group_by("feature").agg(
        pl.col("value").filter(pl.col("label") == 1).mean().alias("mean_cat"),
        pl.col("value")
        .filter(pl.col("label") == 0)
        .mean()
        .alias("mean_ctrl"),
    )
    for row in grouped.iter_rows(named=True):
        mc = row["mean_cat"]
        mu = row["mean_ctrl"]
        if mc is None or mu is None:
            directions[row["feature"]] = 1
            continue
        directions[row["feature"]] = 1 if float(mc) >= float(mu) else -1
    return directions


def compute_auroc_with_ci(
    long: pl.DataFrame,
    cfg: LeadTimeConfig,
    group: str | None = None,
    directions: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Per (group?, feature, relative_pos): AUROC + bootstrap CI.

    Bootstrap resamples are run-level (resample run_ids with
    replacement once, then reuse those resamples across all
    relative_pos within the same iteration). This preserves
    within-run dependence.
    """
    if long.is_empty():
        return pl.DataFrame()

    rng = np.random.default_rng(cfg.seed)
    keys = ["feature", "relative_pos"]
    if group is not None:
        keys = [group] + keys
    directions = directions or {}

    rows: list[dict[str, Any]] = []
    for key_vals, sub in long.group_by(keys, maintain_order=True):
        labels = sub.get_column("label").to_numpy().astype(np.int8)
        scores = sub.get_column("value").to_numpy().astype(np.float64)
        run_ids = sub.get_column("run_id").to_numpy()
        feat_name = key_vals[1] if group is not None else key_vals[0]
        sign = directions.get(str(feat_name), 1)
        if sign != 1:
            scores = sign * scores
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        auc = _safe_auroc(labels, scores)
        ci_lo: float | None = None
        ci_hi: float | None = None
        if auc is not None and cfg.n_bootstrap > 0:
            unique_runs = np.unique(run_ids)
            run_to_idx: dict[str, np.ndarray] = {
                r: np.where(run_ids == r)[0] for r in unique_runs
            }
            boot: list[float] = []
            for _ in range(cfg.n_bootstrap):
                sampled = rng.choice(
                    unique_runs, size=unique_runs.size, replace=True
                )
                idx = np.concatenate([run_to_idx[r] for r in sampled])
                bauc = _safe_auroc(labels[idx], scores[idx])
                if bauc is not None:
                    boot.append(bauc)
            if boot:
                ci_lo = float(np.percentile(boot, 2.5))
                ci_hi = float(np.percentile(boot, 97.5))

        record: dict[str, Any] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": auc,
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


def compute_lead_time(
    by_feature: pl.DataFrame,
    cfg: LeadTimeConfig,
) -> dict[str, int | None]:
    """Earliest negative rel_pos meeting the persistence condition.

    Returns a dict feature -> lead_time_tokens (tokens before onset)
    or None if no qualifying position exists.
    """
    if by_feature.is_empty():
        return {}

    out: dict[str, int | None] = {}
    for feat in sorted(by_feature["feature"].unique().to_list()):
        sub = by_feature.filter(pl.col("feature") == feat).sort(
            "relative_pos"
        )
        rps = sub["relative_pos"].to_numpy()
        aurocs = sub["auroc"].to_numpy()
        ci_lows = sub["ci_lo"].to_numpy()

        # Build the per-position pass mask.
        passes = np.zeros_like(rps, dtype=bool)
        for i in range(len(rps)):
            if rps[i] > 0:
                continue
            a = aurocs[i]
            cl = ci_lows[i]
            if a is None or cl is None:
                continue
            if not (np.isfinite(a) and np.isfinite(cl)):
                continue
            if a >= cfg.auroc_threshold and cl > cfg.ci_lower_threshold:
                passes[i] = True

        # Persistence: earliest i where passes[i:i+persistence] all True
        # AND rps[i+persistence-1] <= 0 (don't claim lead-time using
        # post-onset positions).
        lead: int | None = None
        for i in range(len(rps) - cfg.persistence + 1):
            window = passes[i : i + cfg.persistence]
            if window.all() and rps[i + cfg.persistence - 1] <= 0:
                lead = int(-rps[i])
                break
        out[feat] = lead
    return out


def _draw_panel(
    ax: Any,
    sub: pl.DataFrame,
    cfg: LeadTimeConfig,
    lead_tokens: int | None,
    feat: str,
    group_col: str | None = None,
) -> None:
    if group_col is not None and group_col in sub.columns:
        for key_vals, g in sub.group_by(group_col, maintain_order=True):
            label = str(key_vals[0])
            xs = g["relative_pos"].to_numpy()
            m = g["auroc"].to_numpy(allow_copy=True)
            lo = g["ci_lo"].to_numpy(allow_copy=True)
            hi = g["ci_hi"].to_numpy(allow_copy=True)
            (line,) = ax.plot(xs, m, label=label, linewidth=1.2)
            ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
        ax.legend(fontsize=7)
    else:
        xs = sub["relative_pos"].to_numpy()
        m = sub["auroc"].to_numpy(allow_copy=True)
        lo = sub["ci_lo"].to_numpy(allow_copy=True)
        hi = sub["ci_hi"].to_numpy(allow_copy=True)
        ax.plot(xs, m, color="C0", linewidth=1.2, label="cat vs ctrl")
        ax.fill_between(xs, lo, hi, alpha=0.18, color="C0")

    ax.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(
        cfg.auroc_threshold,
        color="red",
        linestyle=":",
        alpha=0.5,
        linewidth=0.8,
    )
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4, linewidth=0.6)
    if lead_tokens is not None:
        ax.axvline(
            -lead_tokens,
            color="green",
            linestyle="--",
            alpha=0.6,
            linewidth=1.0,
        )
        title = f"{feat}  (lead={lead_tokens} tok)"
    else:
        title = f"{feat}  (lead=N/A)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("relative_pos (tokens from onset)", fontsize=8)
    ax.set_ylabel("AUROC", fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.2)


def plot_lead_time(
    by_feature: pl.DataFrame,
    output_path: Path,
    title: str,
    cfg: LeadTimeConfig,
    lead_times: dict[str, int | None],
    group_col: str | None = None,
) -> None:
    """One subplot per feature; AUROC vs relative_pos with CI band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if by_feature.is_empty():
        output_path.write_bytes(b"")
        return

    feats = sorted(by_feature["feature"].unique().to_list())
    n = len(feats)
    cols = min(3, n) if n > 0 else 1
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.0 * cols, 3.0 * rows), squeeze=False
    )
    for i, feat in enumerate(feats):
        ax = axes[i // cols][i % cols]
        sub = by_feature.filter(pl.col("feature") == feat).sort(
            "relative_pos"
        )
        _draw_panel(
            ax, sub, cfg, lead_times.get(feat), feat, group_col=group_col
        )
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def _stratum_summary(
    matching_info: dict[str, Any],
    cfg: LeadTimeConfig,
) -> tuple[bool, list[str]]:
    """Decide whether the run is control-limited.

    Control-limited when fewer than `min_well_covered_strata`
    (task, press, ratio) cells have both >= min_per_stratum
    catastrophic AND >= min_per_stratum control runs.
    """
    notes: list[str] = []
    sc = matching_info.get("stratum_counts", {})
    well_covered = [
        k
        for k, v in sc.items()
        if v["n_cat"] >= cfg.min_per_stratum
        and v["n_ctrl"] >= cfg.min_per_stratum
    ]
    if len(well_covered) < cfg.min_well_covered_strata:
        notes.append(
            f"{len(well_covered)} of {len(sc)} strata have >= "
            f"{cfg.min_per_stratum} cat AND >= "
            f"{cfg.min_per_stratum} ctrl runs; required >= "
            f"{cfg.min_well_covered_strata} for non-smoke claims. "
            "Lead-time numbers are control-limited."
        )
        return True, notes
    if len(well_covered) < len(sc):
        notes.append(
            f"{len(well_covered)} of {len(sc)} strata have "
            f">= {cfg.min_per_stratum} cat & ctrl; remaining strata "
            "contribute under-powered AUROC."
        )
    return False, notes


def run_lead_time(
    input_root: Path,
    output_dir: Path,
    cfg: LeadTimeConfig,
) -> LeadTimeResult:
    """Top-level CPU entry point.

    Reads runs.parquet + token partitions from `input_root` and
    writes by-feature parquet, summary JSON, and figures to
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
        "position_stride": cfg.position_stride,
        "features_requested": list(cfg.features),
        "nt_onset_frac": cfg.nt_onset_frac,
        "rouge_threshold": cfg.rouge_threshold,
        "n_bootstrap": cfg.n_bootstrap,
        "auroc_threshold": cfg.auroc_threshold,
        "ci_lower_threshold": cfg.ci_lower_threshold,
        "persistence": cfg.persistence,
        "min_per_stratum": cfg.min_per_stratum,
        "matching_strategy": (
            "Pair healthy compressed controls with catastrophic runs "
            "by (task, press, compression_ratio). Within each "
            "stratum, controls receive a virtual onset equal to the "
            "median catastrophic onset (clipped to the control's "
            "length). Strata with no catastrophic onset, or no "
            "control, contribute zero pairs. Per-feature AUROC at "
            "each relative_pos pools across all matched pairs, "
            "including strata that fall below min_per_stratum; "
            "control_limited and per-stratum counts in the summary "
            "let downstream code re-weight or filter."
        ),
        "control_definition": (
            "press != 'none' AND no 'looping' tag AND no "
            "'non_termination' tag in catastrophes."
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
        (output_dir / "lead_time_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return LeadTimeResult(summary=summary)

    runs = pl.read_parquet(runs_path)
    seq = pl.read_parquet(seq_path) if seq_path.exists() else None
    summary["n_total_runs"] = runs.height

    # Catastrophic side: reuse event_study selection + onset derivation.
    from herald.analysis.event_study import EventStudyConfig

    es_cfg = EventStudyConfig(
        window_before=cfg.window_before,
        window_after=cfg.window_after,
        features=cfg.features,
        nt_onset_frac=cfg.nt_onset_frac,
        n_bootstrap=cfg.n_bootstrap,
        rouge_threshold=cfg.rouge_threshold,
        seed=cfg.seed,
    )
    catastrophic = select_catastrophic_runs(
        runs, sequence_metrics=seq, rouge_threshold=cfg.rouge_threshold
    )
    summary["n_catastrophic_runs_selected"] = catastrophic.height

    onsets, excluded = collect_onsets(catastrophic, es_cfg)
    summary["excluded_counts"] = excluded
    summary["n_runs_with_onset"] = len(onsets)

    onset_source_counts: dict[str, int] = {}
    for o in onsets:
        onset_source_counts[o.onset_source] = (
            onset_source_counts.get(o.onset_source, 0) + 1
        )
    summary["onset_source_counts"] = onset_source_counts

    # Control side.
    healthy = select_healthy_controls(runs)
    summary["n_healthy_compressed_runs"] = healthy.height

    matched_controls, match_info = assign_virtual_onsets(
        onsets, healthy, runs, cfg
    )
    summary["matched_control_count"] = len(matched_controls)
    summary["matching_info"] = match_info

    control_limited, caveats = _stratum_summary(match_info, cfg)
    summary["control_limited"] = control_limited
    if control_limited:
        caveats.append(
            "Phase 0 control coverage is structurally thin (single "
            "task, two presses, three ratios). Treat lead-time "
            "numbers as smoke-only until Phase 1 broadens coverage."
        )
    summary["caveats"] = caveats

    if not onsets or not matched_controls:
        summary["status"] = "insufficient_pairs"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "lead_time_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        # Touch empty artifacts so the script's contract holds.
        (output_dir / "lead_time_by_feature.parquet").write_bytes(b"")
        (output_dir / "lead_time_curves.png").write_bytes(b"")
        return LeadTimeResult(summary=summary)

    all_run_ids: list[str] = [o.run_id for o in onsets] + [
        c.run_id for c in matched_controls
    ]
    tokens = load_tokens_for_runs(tokens_root, all_run_ids)
    summary["n_token_rows_loaded"] = tokens.height

    long, used, missing = build_aligned_long(
        onsets, matched_controls, tokens, cfg, runs
    )
    summary["features_used"] = used
    summary["features_missing"] = missing
    summary["n_aligned_rows"] = long.height

    directions = infer_feature_direction(long)
    summary["feature_directions"] = directions
    by_feature = compute_auroc_with_ci(
        long, cfg, group=None, directions=directions
    )
    by_feature_press = compute_auroc_with_ci(
        long, cfg, group="press", directions=directions
    )
    summary["grouping"] = {
        "by_feature": by_feature.height,
        "by_feature_press": by_feature_press.height,
    }

    lead_times = compute_lead_time(by_feature, cfg)
    summary["lead_time_tokens"] = {k: v for k, v in lead_times.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    if not by_feature.is_empty():
        # Convert None to NaN for parquet round-trip stability.
        by_feature.write_parquet(output_dir / "lead_time_by_feature.parquet")
    else:
        (output_dir / "lead_time_by_feature.parquet").write_bytes(b"")
    if not by_feature_press.is_empty():
        by_feature_press.write_parquet(
            output_dir / "lead_time_by_feature_press.parquet"
        )

    plot_lead_time(
        by_feature,
        output_dir / "lead_time_curves.png",
        title=(
            "Lead-time AUROC: catastrophic vs healthy compressed "
            f"(thr={cfg.auroc_threshold:.2f}, "
            f"persistence={cfg.persistence})"
        ),
        cfg=cfg,
        lead_times=lead_times,
    )
    plot_lead_time(
        by_feature_press,
        output_dir / "lead_time_curves_by_press.png",
        title="Lead-time AUROC by press",
        cfg=cfg,
        lead_times=lead_times,
        group_col="press",
    )

    summary["artifacts"] = {
        "by_feature_parquet": str(
            output_dir / "lead_time_by_feature.parquet"
        ),
        "by_feature_press_parquet": str(
            output_dir / "lead_time_by_feature_press.parquet"
        ),
        "headline_png": str(output_dir / "lead_time_curves.png"),
        "by_press_png": str(output_dir / "lead_time_curves_by_press.png"),
    }

    summary_path = output_dir / "lead_time_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    return LeadTimeResult(
        summary=summary,
        by_feature=by_feature,
        by_feature_press=by_feature_press,
    )


__all__ = [
    "DEFAULT_AUROC_THRESHOLD",
    "DEFAULT_CI_LOWER_THRESHOLD",
    "DEFAULT_FEATURES",
    "DEFAULT_MIN_PER_STRATUM",
    "DEFAULT_MIN_WELL_COVERED_STRATA",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_PERSISTENCE",
    "DEFAULT_POSITION_STRIDE",
    "DEFAULT_WINDOW_AFTER",
    "DEFAULT_WINDOW_BEFORE",
    "ControlRecord",
    "LeadTimeConfig",
    "LeadTimeResult",
    "assign_virtual_onsets",
    "build_aligned_long",
    "compute_auroc_with_ci",
    "compute_lead_time",
    "infer_feature_direction",
    "plot_lead_time",
    "run_lead_time",
    "select_healthy_controls",
]
