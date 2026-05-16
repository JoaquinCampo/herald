"""Build Figure 1 for the HERALD paper.

The figure is intentionally data-backed rather than freehand. The left
panel uses the qualitative example already selected for the manuscript
(`gsm8k_79`, random press, compression ratio 0.75). The right panel
plots its per-token HERALD hazard trajectory against a scaled rolling
entropy trace from the released token features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUALITATIVE = ROOT / "models" / "analysis" / "qualitative.json"
DEFAULT_TOKENS = ROOT / "dataset_release" / "tokens" / "random.parquet"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "figure1"

BLUE = "#2563eb"
RED = "#dc2626"
DARK_RED = "#991b1b"
GRAY = "#6b7280"
DARK = "#111827"
LIGHT = "#f8fafc"
BORDER = "#cbd5e1"


def load_example(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    for item in data["catastrophic"]:
        if (
            item["prompt_id"] == "gsm8k_79"
            and item["press"] == "random"
            and float(item["compression_ratio"]) == 0.75
        ):
            return item
    raise ValueError("Could not find gsm8k_79 random 0.75 in qualitative JSON")


def load_entropy(path: Path) -> np.ndarray:
    import polars as pl

    df = (
        pl.read_parquet(path)
        .filter(
            (pl.col("prompt_id") == "gsm8k_79")
            & (pl.col("compression_ratio") == 0.75)
        )
        .sort("token_idx")
    )
    if df.is_empty():
        raise ValueError("Could not find entropy rows for gsm8k_79 random 0.75")
    return df.get_column("entropy").to_numpy()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def scale01(values: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(values, [5, 95])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = BORDER,
    linewidth: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def add_hazard_bar(
    ax: plt.Axes,
    x: float,
    y: float,
    height: float,
    color: str,
    fill_frac: float,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            0.028,
            height,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=GRAY,
            linewidth=0.8,
        )
    )
    ax.add_patch(
        Rectangle(
            (x + 0.003, y + 0.004),
            0.022,
            max(0.0, height * fill_frac - 0.008),
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0.0,
        )
    )


def draw_left_panel(ax: plt.Axes, example: dict[str, Any]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.0,
        0.985,
        "A  Single compressed generation",
        fontsize=12.5,
        fontweight="bold",
        color=DARK,
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.94,
        "Qwen2.5-7B-Instruct, random press, ratio 0.75",
        fontsize=8.7,
        color=GRAY,
        transform=ax.transAxes,
    )

    add_box(ax, (0.0, 0.755), 1.0, 0.145, facecolor=LIGHT)
    ax.text(
        0.035,
        0.865,
        "GSM8K problem",
        fontsize=8.8,
        fontstyle="italic",
        color=GRAY,
        transform=ax.transAxes,
    )
    problem = (
        "Goldy bought 20 sacks of rice and gave 3 sacks to her cousin "
        "and 5 sacks to her brother. If each sack has 25 kg, how many "
        "kg did she give to her cousin and brother?"
    )
    ax.text(
        0.035,
        0.825,
        fill(problem, width=60),
        fontsize=8.8,
        color=DARK,
        va="top",
        transform=ax.transAxes,
        linespacing=1.35,
    )
    ax.text(
        0.94,
        0.865,
        "ground truth: 175 kg",
        fontsize=7.8,
        color=GRAY,
        ha="right",
        transform=ax.transAxes,
    )

    # Three output states, compressed into manuscript-readable excerpts.
    states = [
        {
            "xy": (0.0, 0.565),
            "height": 0.13,
            "face": "#ffffff",
            "bar": BLUE,
            "fill": 0.18,
            "label": "HERALD low",
            "label_color": BLUE,
            "text": (
                "Total sacks given = 3 + 5 = 8. Total weight = 8 x 25 "
                "= 200 kg. Therefore the total weight is boxed 200 kg."
            ),
        },
        {
            "xy": (0.0, 0.36),
            "height": 0.145,
            "face": "#fff7f7",
            "bar": "#ef4444",
            "fill": 0.58,
            "label": "HERALD rising",
            "label_color": RED,
            "text": (
                "However, the question asks for the 3 sacks to her cousin "
                "and the 5 sacks to her brother. The correct answer "
                "should be boxed 75 kg and 125 kg."
            ),
        },
        {
            "xy": (0.0, 0.075),
            "height": 0.205,
            "face": "#fef2f2",
            "bar": RED,
            "fill": 0.95,
            "label": "looping",
            "label_color": DARK_RED,
            "text": (
                "But since the question asks..., the final answer is "
                "boxed 75 kg and 125 kg.\n"
                "So, the final answer is boxed 75 kg and 125 kg.\n"
                "Therefore, the final answer is boxed 75 kg and 125 kg.\n"
                "So, the final answer is boxed 75 kg and 125 kg."
            ),
        },
    ]

    for state in states:
        x, y = state["xy"]
        add_box(ax, (x, y), 1.0, state["height"], facecolor=state["face"])
        add_hazard_bar(
            ax,
            x + 0.02,
            y + 0.018,
            state["height"] - 0.036,
            state["bar"],
            state["fill"],
        )
        row_text = state["text"]
        if state["label"] != "looping":
            row_text = fill(row_text, width=64)
        ax.text(
            x + 0.085,
            y + state["height"] - 0.035,
            row_text,
            fontsize=8.4,
            color=state["label_color"] if state["label"] == "looping" else DARK,
            va="top",
            transform=ax.transAxes,
            linespacing=1.3,
        )
        ax.text(
            x + 0.98,
            y + state["height"] / 2,
            state["label"],
            fontsize=8.5,
            color=state["label_color"],
            ha="right",
            va="center",
            fontweight="bold" if state["label"] == "looping" else "normal",
            transform=ax.transAxes,
        )

    onset = int(example["derived_onset"])
    ax.plot(
        [0.0, 1.0],
        [0.322, 0.322],
        linestyle=(0, (3, 3)),
        color=RED,
        linewidth=1.0,
        transform=ax.transAxes,
    )
    ax.text(
        0.97,
        0.333,
        f"detected onset: token {onset}",
        fontsize=8.0,
        color=RED,
        ha="right",
        transform=ax.transAxes,
    )
    ax.text(
        0.78,
        0.055,
        "generation reaches the token budget without EOS",
        fontsize=8.0,
        color=GRAY,
        ha="center",
        fontstyle="italic",
        transform=ax.transAxes,
    )


def draw_right_panel(
    ax: plt.Axes,
    example: dict[str, Any],
    entropy: np.ndarray,
) -> None:
    hazards = np.asarray(example["hazards"], dtype=float)
    onset = int(example["derived_onset"])
    rel = np.arange(len(hazards)) - onset

    hazard_smoothed = rolling_mean(hazards, 8)
    entropy_smoothed = rolling_mean(entropy[: len(hazards)], 24)
    entropy_scaled = scale01(entropy_smoothed)

    mask = (rel >= -280) & (rel <= 80)
    x = rel[mask]
    hz = hazard_smoothed[mask]
    ent = entropy_scaled[mask]

    threshold = 0.3
    candidates = np.where((rel < 0) & (np.arange(len(rel)) > 10) & (hazard_smoothed >= threshold))[0]
    first_alarm = int(candidates[0]) if len(candidates) else None

    ax.plot(x, hz, color=RED, linewidth=2.2, label="HERALD hazard")
    ax.plot(
        x,
        ent,
        color=GRAY,
        linewidth=1.35,
        linestyle="--",
        alpha=0.7,
        label="rolling entropy, scaled",
    )
    ax.axvline(0, color=DARK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.axhline(threshold, color=RED, linewidth=1.0, linestyle=(0, (2, 3)), alpha=0.65)
    if first_alarm is not None:
        alarm_x = int(first_alarm - onset)
        alarm_y = float(hazard_smoothed[first_alarm])
        ax.scatter([alarm_x], [alarm_y], s=36, color=RED, zorder=5)
        ax.annotate(
            f"first alarm: {onset - first_alarm} tokens early",
            xy=(alarm_x, alarm_y),
            xytext=(-225, 0.78),
            fontsize=8.3,
            color=DARK_RED,
            arrowprops={
                "arrowstyle": "->",
                "color": DARK_RED,
                "linewidth": 0.9,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    ax.text(
        0,
        0.97,
        "onset",
        color=DARK,
        fontsize=8.5,
        ha="left",
        va="top",
        rotation=90,
    )
    ax.text(
        0.0,
        1.05,
        "B  Warning signal before visible failure",
        fontsize=12.5,
        fontweight="bold",
        color=DARK,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        1.0,
        "Same sequence, aligned so failure onset is token 0",
        fontsize=8.7,
        color=GRAY,
        transform=ax.transAxes,
    )

    ax.set_xlim(-280, 80)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("tokens relative to onset", fontsize=9)
    ax.set_ylabel("signal value", fontsize=9)
    ax.set_yticks([0.0, 0.3, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.3", "0.5", "1"])
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower right",
        fontsize=8.4,
        frameon=True,
        framealpha=0.95,
        edgecolor=BORDER,
    )


def build_figure(
    qualitative_path: Path,
    tokens_path: Path,
    output_base: Path,
) -> list[Path]:
    example = load_example(qualitative_path)
    entropy = load_entropy(tokens_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(11.6, 5.2), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.12, 1.0],
        left=0.035,
        right=0.985,
        bottom=0.08,
        top=0.93,
        wspace=0.18,
    )
    left = fig.add_subplot(gs[0, 0])
    right = fig.add_subplot(gs[0, 1])

    draw_left_panel(left, example)
    draw_right_panel(right, example, entropy)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_base.with_suffix(".png"),
        output_base.with_suffix(".pdf"),
    ]
    fig.savefig(outputs[0], dpi=300)
    fig.savefig(outputs[1])
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualitative", type=Path, default=DEFAULT_QUALITATIVE)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_figure(args.qualitative, args.tokens, args.output)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
