#!/usr/bin/env python3
"""
Plot TruthfulQA MC1/MC2 vs steering strength with 95% CI bands.

Usage:
    python -m steering.plot_truthfulness \
        --results outputs/steering/*truthfulness*/results.csv \
        --output outputs/steering/figures/truthfulness_vs_alpha.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TruthfulQA MC1/MC2 means with 95% CI across seeds."
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="One or more truthfulness results.csv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/steering/figures/truthfulness_vs_alpha.pdf",
        help="Output figure path (.pdf/.png/.svg)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Truthfulness vs Steering Strength",
        help="Figure title",
    )
    return parser.parse_args()


def _z95_ci_half_width(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    std = float(np.std(values, ddof=1))
    sem = std / math.sqrt(n)
    return 1.96 * sem


def _load_rows(csv_paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            required = {"variant", "alpha", "mc1_accuracy", "mc2_accuracy"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{path} is missing required columns: {required}")
            rows.extend(reader)
    return rows


def _aggregate_truthfulness(
    rows: list[dict[str, str]],
) -> tuple[
    dict[str, dict[float, tuple[float, float, int]]],
    dict[str, dict[float, tuple[float, float, int]]],
]:
    grouped_mc1: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_mc2: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        variant = row["variant"]
        alpha = float(row["alpha"])
        mc1 = float(row["mc1_accuracy"])
        mc2 = float(row["mc2_accuracy"])
        grouped_mc1[variant][alpha].append(mc1)
        grouped_mc2[variant][alpha].append(mc2)

    def summarize(
        grouped: dict[str, dict[float, list[float]]],
    ) -> dict[str, dict[float, tuple[float, float, int]]]:
        out: dict[str, dict[float, tuple[float, float, int]]] = {}
        for variant, alpha_map in grouped.items():
            out[variant] = {}
            for alpha, vals in alpha_map.items():
                mean = float(np.mean(vals))
                ci = _z95_ci_half_width(vals)
                out[variant][alpha] = (mean, ci, len(vals))
        return out

    return summarize(grouped_mc1), summarize(grouped_mc2)


def _style_plot():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.6,
        }
    )


def _adaptive_limits(lower_vals: list[float], upper_vals: list[float]) -> tuple[float, float]:
    y_min = min(lower_vals)
    y_max = max(upper_vals)
    span = max(y_max - y_min, 1e-6)
    pad = 0.08 * span
    return y_min - pad, y_max + pad


def plot_truthfulness(
    mc1_summary: dict[str, dict[float, tuple[float, float, int]]],
    mc2_summary: dict[str, dict[float, tuple[float, float, int]]],
    output_path: Path,
    title: str,
):
    _style_plot()
    fig, ax1 = plt.subplots(figsize=(3.3, 3.3), constrained_layout=True)
    ax2 = ax1.twinx()

    colors = plt.get_cmap("viridis")
    variants = sorted(set(mc1_summary.keys()) | set(mc2_summary.keys()))
    color_positions = np.linspace(0.15, 0.9, len(variants)) if variants else np.array([0.5])
    legend_handles: list[Line2D] = []
    all_mc1_lower: list[float] = []
    all_mc1_upper: list[float] = []
    all_mc2_lower: list[float] = []
    all_mc2_upper: list[float] = []

    for idx, variant in enumerate(variants):
        color = colors(float(color_positions[idx]))
        mc1_alpha_map = mc1_summary.get(variant, {})
        mc2_alpha_map = mc2_summary.get(variant, {})

        if mc1_alpha_map:
            xs = sorted(mc1_alpha_map.keys())
            means = [mc1_alpha_map[x][0] for x in xs]
            cis = [mc1_alpha_map[x][1] for x in xs]
            ax1.plot(
                xs,
                means,
                marker="o",
                markersize=3.0,
                linestyle="-",
                color=color,
            )
            ax1.fill_between(
                xs,
                np.array(means) - np.array(cis),
                np.array(means) + np.array(cis),
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            all_mc1_lower.extend((np.array(means) - np.array(cis)).tolist())
            all_mc1_upper.extend((np.array(means) + np.array(cis)).tolist())

        if mc2_alpha_map:
            xs = sorted(mc2_alpha_map.keys())
            means = [mc2_alpha_map[x][0] for x in xs]
            cis = [mc2_alpha_map[x][1] for x in xs]
            ax2.plot(
                xs,
                means,
                marker="s",
                markersize=2.8,
                linestyle="--",
                color=color,
            )
            ax2.fill_between(
                xs,
                np.array(means) - np.array(cis),
                np.array(means) + np.array(cis),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            all_mc2_lower.extend((np.array(means) - np.array(cis)).tolist())
            all_mc2_upper.extend((np.array(means) + np.array(cis)).tolist())

        legend_handles.extend(
            [
                Line2D([0], [0], color=color, linestyle="-", marker="o", markersize=3.2, label=f"{variant} MC1"),
                Line2D([0], [0], color=color, linestyle="--", marker="s", markersize=3.0, label=f"{variant} MC2"),
            ]
        )

    ax1.set_title(title)
    ax1.set_xlabel("Steering strength (alpha)")
    ax1.set_ylabel("MC1 accuracy")
    ax2.set_ylabel("MC2 accuracy")
    if all_mc1_lower and all_mc1_upper:
        y1_low, y1_high = _adaptive_limits(all_mc1_lower, all_mc1_upper)
        ax1.set_ylim(y1_low, y1_high)
    if all_mc2_lower and all_mc2_upper:
        y2_low, y2_high = _adaptive_limits(all_mc2_lower, all_mc2_upper)
        ax2.set_ylim(y2_low, y2_high)

    ax1.legend(handles=legend_handles, loc="best", frameon=True, framealpha=0.9, edgecolor="0.8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def main():
    args = parse_args()
    rows = _load_rows(args.results)
    mc1_summary, mc2_summary = _aggregate_truthfulness(rows)
    if not mc1_summary and not mc2_summary:
        raise ValueError("No truthfulness rows found in provided CSV files.")
    plot_truthfulness(mc1_summary, mc2_summary, Path(args.output), args.title)


if __name__ == "__main__":
    main()
