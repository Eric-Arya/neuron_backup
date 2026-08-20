#!/usr/bin/env python3
"""Plot the expanded-K/V capability, safety, and degeneration tradeoff."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


RATES = [0, 0.0001, 0.0002, 0.0004, 0.0006, 0.0007]
ENTRIES = [0, 131, 262, 524, 786, 917]
GSM8K = [67, 62, 60, 45, 30, 20]
EXTRACTION_FAILURES = [0, 0, 0, 0, 0, 4]
REPETITIVE_GSM = [5, 7, 7, 10, 14, 22]
HARM_ASR = [34, 50, 56, 70, 75]
BLANK_HARM = [0, 0, 0, 0, 0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_stem = Path(__file__).resolve().parents[1] / "figures" / "expanded_kv_tradeoff"
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=default_stem,
        help="Output path without extension; both PNG and PDF are written.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )
    fig, (tradeoff_ax, quality_ax) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.12},
    )

    capability_color = "#2563EB"
    safety_color = "#DC2626"
    repeat_color = "#D97706"
    failure_color = "#6B7280"
    selected_color = "#7C3AED"

    tradeoff_ax.plot(
        ENTRIES,
        GSM8K,
        color=capability_color,
        marker="o",
        linewidth=2.4,
        markersize=6,
        label="GSM8K accuracy",
    )
    tradeoff_ax.plot(
        ENTRIES[: len(HARM_ASR)],
        HARM_ASR,
        color=safety_color,
        marker="s",
        linewidth=2.4,
        markersize=6,
        label="HarmBehavior ASR",
    )
    tradeoff_ax.axvspan(786, 917, color="#E5E7EB", alpha=0.55, zorder=0)
    tradeoff_ax.axvline(786, color=selected_color, linestyle="--", linewidth=1.5)
    tradeoff_ax.scatter(
        [786, 786],
        [GSM8K[4], HARM_ASR[4]],
        s=105,
        facecolors="white",
        edgecolors=selected_color,
        linewidths=2,
        zorder=5,
    )
    tradeoff_ax.annotate(
        "Selected upper limit\nrate 0.0006 · 786 entries",
        xy=(786, 75),
        xytext=(565, 88),
        color=selected_color,
        arrowprops={"arrowstyle": "->", "color": selected_color, "lw": 1.2},
        ha="center",
        va="center",
    )
    tradeoff_ax.text(
        851.5,
        8,
        "HarmBehavior\nnot run",
        color="#6B7280",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    tradeoff_ax.set_ylim(0, 100)
    tradeoff_ax.set_ylabel("Score (%)")
    tradeoff_ax.set_title("Expanded-K/V deactivation: capability–safety tradeoff")
    tradeoff_ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)
    tradeoff_ax.legend(loc="lower left", frameon=False, ncol=2)

    quality_ax.plot(
        ENTRIES,
        REPETITIVE_GSM,
        color=repeat_color,
        marker="D",
        linewidth=2.1,
        markersize=5.5,
        label="Repetitive GSM outputs",
    )
    quality_ax.plot(
        ENTRIES,
        EXTRACTION_FAILURES,
        color=failure_color,
        marker="^",
        linewidth=2.1,
        markersize=6,
        label="GSM extraction failures",
    )
    quality_ax.plot(
        ENTRIES[: len(BLANK_HARM)],
        BLANK_HARM,
        color=safety_color,
        marker="x",
        linestyle=":",
        linewidth=1.8,
        markersize=7,
        label="Blank harm outputs",
    )
    quality_ax.axvspan(786, 917, color="#E5E7EB", alpha=0.55, zorder=0)
    quality_ax.axvline(786, color=selected_color, linestyle="--", linewidth=1.5)
    quality_ax.set_ylim(-1, 25)
    quality_ax.set_ylabel("Outputs (of 100)")
    quality_ax.set_xlabel("Global rate and selected entries")
    quality_ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.7)
    quality_ax.legend(loc="upper left", frameon=False, ncol=2)

    tick_labels = [f"{rate:g}\n({entries})" for rate, entries in zip(RATES, ENTRIES)]
    quality_ax.set_xticks(ENTRIES, tick_labels)
    for axis in (tradeoff_ax, quality_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.005,
        "GSM8K: chat, zero-shot, first 100 examples. HarmBehavior was not run at rate 0.0007.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.14, hspace=0.12)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_stem.with_suffix(".png")
    pdf_path = args.output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path.resolve())
    print(pdf_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
