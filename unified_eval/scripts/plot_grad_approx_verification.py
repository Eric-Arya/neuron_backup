#!/usr/bin/env python3
"""Plot target-gradient and Fisher approximation verification paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUMMARY = Path(
    "results/grad_approx_verify_k12000_c024_c048/summary.json"
)
DEFAULT_OUTPUT = Path("figures/grad_approx_verification_k12000_c024_c048")
COLORS = {"c=.24": "#0072B2", "c=.48": "#D55E00"}


def short_label(label: str) -> str:
    if "c0p24" in label:
        return "c=.24"
    if "c0p48" in label:
        return "c=.48"
    raise ValueError(f"Unknown verification direction: {label}")


def rows_by_label(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(short_label(str(row["label"])), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: float(row["t"]))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    target = rows_by_label(summary["target"]["rows"])
    general = rows_by_label(summary["general_cost"]["rows"])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "svg.hashsalt": "grad-approx-verification-k12000-c024-c048",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))

    ax = axes[0, 0]
    for label, rows in target.items():
        color = COLORS[label]
        t = np.array([row["t"] for row in rows])
        actual = np.array([row["actual_change"] for row in rows])
        predicted = np.array([row["linear_predicted_change"] for row in rows])
        low = np.array([row["actual_change_ci95"][0] for row in rows])
        high = np.array([row["actual_change_ci95"][1] for row in rows])
        ax.plot(t, actual, marker="o", color=color, label=f"{label} actual")
        ax.fill_between(t, low, high, color=color, alpha=0.13, linewidth=0)
        ax.plot(t, predicted, linestyle="--", color=color, label=f"{label} linear")
    ax.set_title("(a) Target objective: actual versus first order")
    ax.set_xlabel("Path fraction t")
    ax.set_ylabel("Change in mean target log probability")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)

    ax = axes[0, 1]
    for label, rows in target.items():
        rows = [row for row in rows if row["t"] > 0]
        t = np.array([row["t"] for row in rows])
        ratio = np.array([row["actual_over_linear"] for row in rows])
        ax.plot(t, ratio, marker="o", color=COLORS[label], label=label)
    ax.axhspan(0.8, 1.25, color="#009E73", alpha=0.10, label="calibration band")
    ax.axhline(1.0, color="black", linewidth=0.9, alpha=0.6)
    ax.set_xscale("log")
    ax.set_title("(b) First-order calibration ratio")
    ax.set_xlabel("Path fraction t (log scale)")
    ax.set_ylabel("Actual target change / linear prediction")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    for label, rows in general.items():
        rows = [row for row in rows if row["t"] > 0]
        color = COLORS[label]
        t = np.array([row["t"] for row in rows])
        actual = np.array([row["actual_mean_sequence_kl"] for row in rows])
        full = np.array([row["full_directional_prediction"] for row in rows])
        diagonal = np.array([row["training_diagonal_prediction"] for row in rows])
        ax.plot(t, actual, marker="o", color=color, label=f"{label} actual")
        ax.plot(t, full, linestyle="-.", color=color, label=f"{label} full F")
        ax.plot(t, diagonal, linestyle=":", color=color, label=f"{label} diagonal F")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("(c) General-distribution KL predictions")
    ax.set_xlabel("Path fraction t (log scale)")
    ax.set_ylabel("Mean sequence KL (log scale)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 1]
    for label, rows in general.items():
        rows = [row for row in rows if row["t"] >= 0.02]
        color = COLORS[label]
        t = np.array([row["t"] for row in rows])
        full_ratio = np.array([row["actual_over_full_directional"] for row in rows])
        diagonal_ratio = np.array([row["actual_over_training_diagonal"] for row in rows])
        ax.plot(t, full_ratio, marker="o", color=color, label=f"{label} full F")
        ax.plot(
            t,
            diagonal_ratio,
            marker="s",
            linestyle="--",
            color=color,
            label=f"{label} diagonal F",
        )
    ax.axhspan(0.8, 1.25, color="#009E73", alpha=0.10, label="calibration band")
    ax.axhline(1.0, color="black", linewidth=0.9, alpha=0.6)
    ax.set_xscale("log")
    ax.set_title("(d) KL calibration ratio (numerically stable range)")
    ax.set_xlabel("Path fraction t (log scale)")
    ax.set_ylabel("Actual KL / predicted KL")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    figure.suptitle(
        "Finite-radius verification for positive-only Fisher-guided Grad (k=12,000)",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output.with_suffix(f".{extension}")
        figure.savefig(path, bbox_inches="tight")
        print(path)


if __name__ == "__main__":
    main()
