#!/usr/bin/env python3
"""
Plot CLS toxicity vs steering strength with 95% CI bands.

Usage:
    python -m steering.plot_cls_toxicity \
        --results outputs/steering/*toxicity*/results.csv \
        --output outputs/steering/figures/cls_toxicity_vs_alpha.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CLS toxicity mean with 95% CI across seeds."
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="One or more toxicity results.csv files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/steering/figures/cls_toxicity_vs_alpha.pdf",
        help="Output figure path (.pdf/.png/.svg)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="CLS Toxicity vs Steering Strength",
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
            required = {"variant", "alpha", "cls_toxicity"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"{path} is missing required columns: {required}")
            rows.extend(reader)
    return rows


def _aggregate_cls_toxicity(
    rows: list[dict[str, str]],
) -> dict[str, dict[float, tuple[float, float, int]]]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        variant = row["variant"]
        alpha = float(row["alpha"])
        score = float(row["cls_toxicity"])
        grouped[variant][alpha].append(score)

    summary: dict[str, dict[float, tuple[float, float, int]]] = {}
    for variant, alpha_map in grouped.items():
        summary[variant] = {}
        for alpha, vals in alpha_map.items():
            mean = float(np.mean(vals))
            ci = _z95_ci_half_width(vals)
            summary[variant][alpha] = (mean, ci, len(vals))
    return summary


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


def plot_cls_toxicity(
    summary: dict[str, dict[float, tuple[float, float, int]]],
    output_path: Path,
    title: str,
):
    _style_plot()
    fig, ax = plt.subplots(figsize=(3.3, 3.3), constrained_layout=True)

    variants = sorted(summary.keys())
    color_positions = np.linspace(0.15, 0.9, len(variants)) if variants else np.array([0.5])
    colors = plt.get_cmap("viridis")
    all_lower: list[float] = []
    all_upper: list[float] = []
    for idx, variant in enumerate(variants):
        alpha_map = summary[variant]
        xs = sorted(alpha_map.keys())
        means = [alpha_map[x][0] for x in xs]
        cis = [alpha_map[x][1] for x in xs]
        ns = [alpha_map[x][2] for x in xs]
        color = colors(float(color_positions[idx]))
        label = f"{variant}"

        ax.plot(xs, means, marker="o", markersize=3.2, color=color, label=label)
        lower = np.array(means) - np.array(cis)
        upper = np.array(means) + np.array(cis)
        ax.fill_between(xs, lower, upper, color=color, alpha=0.20, linewidth=0)
        all_lower.extend(lower.tolist())
        all_upper.extend(upper.tolist())

    ax.set_title(title)
    ax.set_xlabel("Steering strength (alpha)")
    ax.set_ylabel("CLS toxicity")
    y_low, y_high = _adaptive_limits(all_lower, all_upper)
    ax.set_ylim(y_low, y_high)
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="0.8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def main():
    args = parse_args()
    rows = _load_rows(args.results)
    summary = _aggregate_cls_toxicity(rows)
    if not summary:
        raise ValueError("No toxicity rows found in provided CSV files.")
    plot_cls_toxicity(summary, Path(args.output), args.title)


if __name__ == "__main__":
    main()
