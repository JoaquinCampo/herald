"""Classical analysis helpers for switch-level damage datasets."""

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def numeric_summary(
    values: Iterable[object],
) -> dict[str, float | int | None]:
    """Return robust scalar summaries for finite numeric values."""
    arr = _finite_array(values)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    q = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "max": float(q[4]),
    }


def group_counts(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[str, int]:
    """Count rows by a tuple of key values, serialized as a stable string."""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[_group_key(row, keys)] += 1
    return dict(sorted(counts.items()))


def label_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize damage labels for a row subset."""
    dq = _finite_array(row.get("dq") for row in rows)
    out: dict[str, Any] = numeric_summary(dq)
    if dq.size == 0:
        out.update(
            {
                "frac_damage": None,
                "frac_lift": None,
                "frac_zero": None,
                "frac_major_damage": None,
            }
        )
        return out
    out.update(
        {
            "frac_damage": float((dq > 0.0).mean()),
            "frac_lift": float((dq < 0.0).mean()),
            "frac_zero": float((dq == 0.0).mean()),
            "frac_major_damage": float((dq >= 0.5).mean()),
        }
    )
    return out


def grouped_label_summary(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Summarize labels by group."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, keys)].append(row)
    return {
        key: label_summary(group_rows)
        for key, group_rows in sorted(groups.items())
    }


def cell_mean_r2(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> float | None:
    """Return variance fraction in ``dq`` explained by group means."""
    y = _finite_array(row.get("dq") for row in rows)
    if y.size < 2:
        return None
    total_var = float(y.var())
    if total_var == 0.0:
        return None

    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _as_float(row.get("dq"))
        if value is None:
            continue
        groups[_group_key(row, keys)].append(value)

    resid: list[float] = []
    for vals in groups.values():
        mean = float(np.mean(vals))
        resid.extend(v - mean for v in vals)
    if not resid:
        return None
    return float(1.0 - np.var(np.asarray(resid)) / total_var)


def feature_summaries(
    rows: Sequence[dict[str, Any]], feature_cols: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Return missingness and numeric summaries for feature columns."""
    out: dict[str, dict[str, Any]] = {}
    n = len(rows)
    for col in feature_cols:
        raw = [row.get(col) for row in rows]
        finite = _finite_array(raw)
        summary = numeric_summary(finite)
        summary["missing_frac"] = (
            1.0 - float(finite.size / n) if n > 0 else None
        )
        out[col] = summary
    return out


def top_feature_correlations(
    rows: Sequence[dict[str, Any]],
    feature_cols: Sequence[str],
    *,
    target: str = "dq",
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Rank features by absolute Pearson correlation with ``target``."""
    scored: list[dict[str, Any]] = []
    y_raw = [row.get(target) for row in rows]
    for col in feature_cols:
        x_raw = [row.get(col) for row in rows]
        corr, n = pearson_pairwise(x_raw, y_raw)
        if corr is None:
            continue
        scored.append({"feature": col, "corr": corr, "n": n})
    scored.sort(key=lambda r: abs(float(r["corr"])), reverse=True)
    return scored[:top_n]


def residualize_by_cells(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> list[float | None]:
    """Return ``dq`` minus the mean ``dq`` for each metadata cell."""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _as_float(row.get("dq"))
        if value is not None:
            groups[_group_key(row, keys)].append(value)
    means = {key: float(np.mean(vals)) for key, vals in groups.items()}

    residuals: list[float | None] = []
    for row in rows:
        value = _as_float(row.get("dq"))
        if value is None:
            residuals.append(None)
            continue
        residuals.append(value - means[_group_key(row, keys)])
    return residuals


def build_eda_summary(
    rows: Sequence[dict[str, Any]],
    *,
    feature_prefix: str = "feat__",
    top_n_features: int = 20,
) -> dict[str, Any]:
    """Build the classical EDA summary for a switch dataset."""
    feature_cols = (
        sorted(k for k in rows[0] if k.startswith(feature_prefix))
        if rows
        else []
    )
    cell_keys = ("task", "compressor", "ratio", "s")
    residuals = residualize_by_cells(rows, cell_keys)
    residual_rows = [
        {**row, "dq_resid_cell": resid}
        for row, resid in zip(rows, residuals, strict=True)
    ]

    return {
        "inventory": {
            "n_rows": len(rows),
            "n_features": len(feature_cols),
            "by_task": group_counts(rows, ("task",)),
            "by_task_compressor": group_counts(rows, ("task", "compressor")),
            "by_task_ratio": group_counts(rows, ("task", "ratio")),
        },
        "labels": {
            "overall": label_summary(rows),
            "by_task": grouped_label_summary(rows, ("task",)),
            "by_task_compressor_ratio": grouped_label_summary(
                rows, ("task", "compressor", "ratio")
            ),
        },
        "cell_structure": {
            "r2_task_compressor_ratio_s": cell_mean_r2(rows, cell_keys),
            "r2_compressor_ratio_s": cell_mean_r2(
                rows, ("compressor", "ratio", "s")
            ),
            "r2_task_compressor_ratio": cell_mean_r2(
                rows, ("task", "compressor", "ratio")
            ),
        },
        "feature_sanity": feature_summaries(rows, feature_cols),
        "feature_vs_damage": {
            "top_corr_dq": top_feature_correlations(
                rows, feature_cols, target="dq", top_n=top_n_features
            ),
            "top_corr_dq_resid_cell": top_feature_correlations(
                residual_rows,
                feature_cols,
                target="dq_resid_cell",
                top_n=top_n_features,
            ),
        },
    }


def markdown_report(summary: dict[str, Any]) -> str:
    """Render a concise Markdown report from ``build_eda_summary``."""
    inv = summary["inventory"]
    labels = summary["labels"]["overall"]
    cell = summary["cell_structure"]
    lines = [
        "# Switch Dataset EDA",
        "",
        "## Inventory",
        "",
        f"- Rows: {inv['n_rows']}",
        f"- Feature columns: {inv['n_features']}",
        "",
        "### Rows By Task",
        "",
        "| Task | Rows |",
        "|---|---:|",
    ]
    for key, n in inv["by_task"].items():
        lines.append(f"| {key} | {n} |")

    lines.extend(
        [
            "",
            "## Label Summary",
            "",
            f"- Mean dq: {_fmt(labels['mean'])}",
            f"- Std dq: {_fmt(labels['std'])}",
            f"- Damage fraction: {_fmt(labels['frac_damage'])}",
            f"- Lift fraction: {_fmt(labels['frac_lift'])}",
            f"- Zero fraction: {_fmt(labels['frac_zero'])}",
            "",
            "## Cell Structure",
            "",
            (
                "- R2 from task, compressor, ratio, s cell means: "
                f"{_fmt(cell['r2_task_compressor_ratio_s'])}"
            ),
            (
                "- R2 from compressor, ratio, s cell means: "
                f"{_fmt(cell['r2_compressor_ratio_s'])}"
            ),
            (
                "- R2 from task, compressor, ratio cell means: "
                f"{_fmt(cell['r2_task_compressor_ratio'])}"
            ),
            "",
            "## Top Feature Correlations With dq",
            "",
            "| Feature | Corr | N |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["feature_vs_damage"]["top_corr_dq"][:10]:
        lines.append(
            f"| {row['feature']} | {_fmt(row['corr'])} | {row['n']} |"
        )

    lines.extend(
        [
            "",
            "## Top Feature Correlations With Cell Residual dq",
            "",
            "| Feature | Corr | N |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["feature_vs_damage"]["top_corr_dq_resid_cell"][:10]:
        lines.append(
            f"| {row['feature']} | {_fmt(row['corr'])} | {row['n']} |"
        )
    lines.append("")
    return "\n".join(lines)


def pearson_pairwise(
    xs: Sequence[object], ys: Sequence[object]
) -> tuple[float | None, int]:
    """Pearson correlation after dropping non-finite pairs."""
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys, strict=True):
        xf = _as_float(x)
        yf = _as_float(y)
        if xf is not None and yf is not None:
            pairs.append((xf, yf))
    if len(pairs) < 3:
        return None, len(pairs)
    arr = np.asarray(pairs, dtype=np.float64)
    x = arr[:, 0]
    y = arr[:, 1]
    x_std = float(x.std())
    y_std = float(y.std())
    if x_std == 0.0 or y_std == 0.0:
        return None, len(pairs)
    corr = float(np.mean((x - x.mean()) * (y - y.mean())) / (x_std * y_std))
    return corr, len(pairs)


def _group_key(row: dict[str, Any], keys: Sequence[str]) -> str:
    return "|".join(str(row.get(k)) for k in keys)


def _finite_array(values: Iterable[object]) -> np.ndarray:
    out: list[float] = []
    for value in values:
        f = _as_float(value)
        if f is not None:
            out.append(f)
    return np.asarray(out, dtype=np.float64)


def _as_float(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _fmt(value: object) -> str:
    f = _as_float(value)
    if f is None:
        return "NA"
    return f"{f:.4f}"
