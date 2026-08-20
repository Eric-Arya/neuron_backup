#!/usr/bin/env python3
"""Plot matched capability/HarmBench trade-offs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "results" / "ifeval_harmbench_tradeoff.csv"
OUTPUT_DIR = ROOT / "figures"

METHODS = ("SN-Tune", "IA3-SFT", "Grad (off-policy)", "Grad (on-policy)")
MAIN_ONLY_METHODS = ("SN-Tune", "IA3-SFT", "IA3 guide patch", "Grad (on-policy)")
COLORS = {
    "SN-Tune": "#0072B2",
    "IA3-SFT": "#CC79A7",
    "IA3 guide patch": "#E69F00",
    "Grad (off-policy)": "#D55E00",
    "Grad (on-policy)": "#009E73",
}
MARKERS = {
    "SN-Tune": "o",
    "IA3-SFT": "^",
    "IA3 guide patch": "P",
    "Grad (off-policy)": "s",
    "Grad (on-policy)": "D",
}


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def coordinates(
    row: dict[str, str],
    x_field: str = "ifeval_strict_prompt_accuracy_pct",
) -> tuple[float, float]:
    return (
        float(row[x_field]),
        float(row["harmbench_asr_pct"]),
    )


def style_axis(
    ax: plt.Axes,
    xlabel: str = "IFEval strict prompt accuracy (%)  →",
) -> None:
    ax.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-4, 71)
    ax.set_xlabel(xlabel)


def annotate_rows(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    labels: dict[tuple[str, str], str],
    offsets: dict[tuple[str, str], tuple[int, int]],
    x_field: str = "ifeval_strict_prompt_accuracy_pct",
) -> None:
    for row in rows:
        key = (row["method"], row["setting"])
        if key not in labels:
            continue
        ax.annotate(
            labels[key],
            coordinates(row, x_field),
            xytext=offsets[key],
            textcoords="offset points",
            color=COLORS[row["method"]],
            fontsize=8.8,
        )


def main() -> None:
    rows = load_rows()
    baseline = next(row for row in rows if row["method"] == "Baseline")
    base_x, base_y = coordinates(baseline)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    fig, (main_ax, other_ax) = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.2),
        sharey=True,
        gridspec_kw={"width_ratios": (1.0, 1.08)},
        constrained_layout=True,
    )

    for ax in (main_ax, other_ax):
        style_axis(ax)
        ax.scatter(
            base_x,
            base_y,
            s=90,
            marker="*",
            color="#222222",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )

    main_ax.set_xlim(50, 75)
    other_ax.set_xlim(17, 72.5)
    main_ax.set_ylabel("HarmBench attack success rate (%)  ↓")
    main_ax.set_title("Main trajectories\nGrad: strength = 1, candidate pool = 20,000")
    other_ax.set_title("Other measured settings\nGrad: candidate pool = 4,000")

    # Only rows explicitly marked as main are connected. Thus each Grad line
    # holds strength and ranking construction fixed while K changes.
    for method in METHODS:
        method_rows = [
            row
            for row in rows
            if row["method"] == method and row["plot_role"] == "main"
        ]
        xs, ys = zip(*(coordinates(row) for row in method_rows), strict=True)
        main_ax.plot(
            [base_x, *xs],
            [base_y, *ys],
            color=COLORS[method],
            linewidth=2,
            zorder=2,
        )
        main_ax.scatter(
            xs,
            ys,
            color=COLORS[method],
            marker=MARKERS[method],
            s=52,
            zorder=3,
        )

    main_rows = [row for row in rows if row["plot_role"] == "main"]
    main_labels = {
        ("SN-Tune", "alpha=1"): r"$\alpha=1$",
        ("SN-Tune", "alpha=4"): r"$\alpha=4$",
        ("SN-Tune", "alpha=6"): r"$\alpha=6$",
        ("SN-Tune", "alpha=8"): r"$\alpha=8$",
        ("IA3-SFT", "alpha=1"): r"$\alpha=1$",
        ("IA3-SFT", "alpha=2"): r"$\alpha=2$",
        ("IA3-SFT", "alpha=3"): r"$\alpha=3$",
        ("IA3-SFT", "alpha=3.5"): r"$\alpha=3.5$",
        ("IA3 guide patch", "K=40000"): r"$K=40\mathrm{k}$",
        ("IA3 guide patch", "K=80000"): r"$K=80\mathrm{k}$",
        ("IA3 guide patch", "K=160000"): r"$K=160\mathrm{k}$",
        ("IA3 guide patch", "K=320000"): r"$K=320\mathrm{k}$",
        ("Grad (off-policy)", "K=1000, strength=1"): r"$K=1000$",
        ("Grad (off-policy)", "K=2000, strength=1"): r"$K=2000$",
        ("Grad (off-policy)", "K=4000, strength=1"): r"$K=4000$",
        ("Grad (on-policy)", "K=1000, strength=1"): r"$K=1000$",
        ("Grad (on-policy)", "K=2000, strength=1"): r"$K=2000$",
        ("Grad (on-policy)", "K=4000, strength=1"): r"$K=4000$",
        ("Grad (on-policy)", "K=4000, strength=0.4"): r"$s=0.4$",
        ("Grad (on-policy)", "K=4000, strength=0.5"): r"$s=0.5$",
        ("Grad (on-policy)", "K=4000, strength=0.6"): r"$s=0.6$",
        ("Grad (on-policy)", "K=4000, strength=0.75"): r"$s=0.75$",
        ("Grad (on-policy)", "K=4000, strength=0.85"): r"$s=0.85$",
    }
    main_offsets = {
        ("SN-Tune", "alpha=1"): (-50, -18),
        ("SN-Tune", "alpha=4"): (10, 11),
        ("SN-Tune", "alpha=6"): (8, 8),
        ("SN-Tune", "alpha=8"): (8, 8),
        ("IA3-SFT", "alpha=1"): (-5, -20),
        ("IA3-SFT", "alpha=2"): (8, 7),
        ("IA3-SFT", "alpha=3"): (-48, 7),
        ("IA3-SFT", "alpha=3.5"): (8, -14),
        ("IA3 guide patch", "K=40000"): (-18, 9),
        ("IA3 guide patch", "K=80000"): (-50, -14),
        ("IA3 guide patch", "K=160000"): (-18, 10),
        ("IA3 guide patch", "K=320000"): (38, -2),
        ("Grad (off-policy)", "K=1000, strength=1"): (8, -16),
        ("Grad (off-policy)", "K=2000, strength=1"): (-52, -18),
        ("Grad (off-policy)", "K=4000, strength=1"): (8, 10),
        ("Grad (on-policy)", "K=1000, strength=1"): (10, -17),
        ("Grad (on-policy)", "K=2000, strength=1"): (8, -18),
        ("Grad (on-policy)", "K=4000, strength=1"): (8, -18),
        ("Grad (on-policy)", "K=4000, strength=0.4"): (7, 7),
        ("Grad (on-policy)", "K=4000, strength=0.5"): (-38, -16),
        ("Grad (on-policy)", "K=4000, strength=0.6"): (7, 7),
        ("Grad (on-policy)", "K=4000, strength=0.75"): (7, 7),
        ("Grad (on-policy)", "K=4000, strength=0.85"): (7, 7),
    }
    annotate_rows(main_ax, main_rows, main_labels, main_offsets)
    main_ax.annotate(
        "Baseline",
        (base_x, base_y),
        xytext=(-62, 8),
        textcoords="offset points",
        color="#222222",
        fontsize=9,
    )

    separate_rows = [row for row in rows if row["plot_role"] == "separate"]
    for method in ("Grad (off-policy)", "Grad (on-policy)"):
        method_rows = [row for row in separate_rows if row["method"] == method]
        xs, ys = zip(*(coordinates(row) for row in method_rows), strict=True)
        other_ax.scatter(
            xs,
            ys,
            facecolors="none",
            edgecolors=COLORS[method],
            marker=MARKERS[method],
            linewidths=1.7,
            s=64,
            zorder=3,
        )

    separate_labels = {
        ("Grad (off-policy)", "K=1000, strength=0.5"): r"$K=1000,\ s=0.5$",
        ("Grad (off-policy)", "K=1000, strength=1"): r"$K=1000,\ s=1$",
        ("Grad (off-policy)", "K=1000, strength=1.5"): r"$K=1000,\ s=1.5$",
        ("Grad (on-policy)", "K=1000, strength=1"): r"$K=1000,\ s=1$",
        ("Grad (on-policy)", "K=1000, strength=1.5"): r"$K=1000,\ s=1.5$",
        ("Grad (on-policy)", "K=1943, strength=2"): r"$K=1943,\ s=2$",
        ("Grad (on-policy)", "K=1943, strength=2.5"): r"$K=1943,\ s=2.5$",
    }
    separate_offsets = {
        ("Grad (off-policy)", "K=1000, strength=0.5"): (-70, 8),
        ("Grad (off-policy)", "K=1000, strength=1"): (8, -18),
        ("Grad (off-policy)", "K=1000, strength=1.5"): (8, 8),
        ("Grad (on-policy)", "K=1000, strength=1"): (8, 8),
        ("Grad (on-policy)", "K=1000, strength=1.5"): (-78, 9),
        ("Grad (on-policy)", "K=1943, strength=2"): (8, 8),
        ("Grad (on-policy)", "K=1943, strength=2.5"): (8, 8),
    }
    annotate_rows(other_ax, separate_rows, separate_labels, separate_offsets)
    other_ax.annotate(
        "Baseline",
        (base_x, base_y),
        xytext=(-62, 8),
        textcoords="offset points",
        color="#222222",
        fontsize=9,
    )

    legend_handles = [
        Line2D([], [], marker="*", linestyle="none", color="#222222", label="Unmodified"),
        Line2D([], [], marker="o", color=COLORS["SN-Tune"], label="SN-Tune"),
        Line2D([], [], marker="^", color=COLORS["IA3-SFT"], label="IA3-SFT"),
        Line2D(
            [],
            [],
            marker="s",
            color=COLORS["Grad (off-policy)"],
            label="Grad (off-policy)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color=COLORS["Grad (on-policy)"],
            label="Grad (on-policy)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            label="Unconnected measured setting",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=6,
    )
    fig.suptitle("Safety–instruction-following trade-off", fontsize=14)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"ifeval_harmbench_tradeoff.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    # Publication-friendly view containing only the controlled main
    # trajectories. The complete two-panel figure above remains available.
    main_only_fig, main_only_ax = plt.subplots(
        figsize=(8.2, 5.6),
        constrained_layout=True,
    )
    style_axis(main_only_ax)
    main_only_ax.set_xlim(50, 75)
    main_only_ax.set_ylabel("HarmBench attack success rate (%)  ↓")
    main_only_ax.set_title(
        "Main safety–instruction-following trajectories\n"
        "On-policy Grad: solid K sweep at s=1; dashed K=4000 strength sweep"
    )
    main_only_ax.scatter(
        base_x,
        base_y,
        s=105,
        marker="*",
        color="#222222",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    for method in MAIN_ONLY_METHODS:
        method_rows = [
            row
            for row in rows
            if row["method"] == method
            and row["plot_role"] in ("main", "main_patch")
        ]
        xs, ys = zip(*(coordinates(row) for row in method_rows), strict=True)
        main_only_ax.plot(
            [base_x, *xs],
            [base_y, *ys],
            color=COLORS[method],
            marker=MARKERS[method],
            markevery=range(1, len(xs) + 1),
            markersize=7,
            linewidth=2.3,
            zorder=2,
        )
    strength_order = {
        "K=4000, strength=0.4": 0.4,
        "K=4000, strength=0.5": 0.5,
        "K=4000, strength=0.6": 0.6,
        "K=4000, strength=0.75": 0.75,
        "K=4000, strength=0.85": 0.85,
        "K=4000, strength=1": 1.0,
    }
    strength_rows = sorted(
        [
            row
            for row in rows
            if row["method"] == "Grad (on-policy)"
            and row["setting"] in strength_order
            and row["ranking_candidate_pool"] == "20000"
        ],
        key=lambda row: strength_order[row["setting"]],
    )
    strength_xs, strength_ys = zip(
        *(coordinates(row) for row in strength_rows), strict=True
    )
    main_only_ax.plot(
        [base_x, *strength_xs],
        [base_y, *strength_ys],
        color=COLORS["Grad (on-policy)"],
        linestyle="--",
        marker=MARKERS["Grad (on-policy)"],
        markerfacecolor="white",
        markersize=6.5,
        linewidth=1.8,
        zorder=3,
    )
    main_only_rows = [
        row
        for row in rows
        if row["method"] in MAIN_ONLY_METHODS
        and row["plot_role"] in ("main", "main_patch")
    ]
    annotate_rows(main_only_ax, main_only_rows, main_labels, main_offsets)
    annotate_rows(main_only_ax, strength_rows[:-1], main_labels, main_offsets)
    main_only_ax.annotate(
        "Baseline",
        (base_x, base_y),
        xytext=(-62, 8),
        textcoords="offset points",
        color="#222222",
        fontsize=9,
    )
    main_only_legend_handles = [
        Line2D([], [], marker="*", linestyle="none", color="#222222", label="Unmodified"),
        Line2D([], [], marker="o", color=COLORS["SN-Tune"], label="SN-Tune"),
        Line2D([], [], marker="^", color=COLORS["IA3-SFT"], label="IA3-SFT"),
        Line2D(
            [],
            [],
            marker="P",
            color=COLORS["IA3 guide patch"],
            label="IA3 guide patch",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color=COLORS["Grad (on-policy)"],
            label="Grad (on-policy)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            markerfacecolor="white",
            linestyle="--",
            color=COLORS["Grad (on-policy)"],
            label="Grad K=4000 strength sweep",
        ),
    ]
    main_only_ax.legend(
        handles=main_only_legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
    )
    for extension in ("png", "pdf", "svg"):
        main_only_fig.savefig(
            OUTPUT_DIR / f"ifeval_harmbench_tradeoff_main_only.{extension}",
            bbox_inches="tight",
        )
    plt.close(main_only_fig)

    math_field = "math500_l1_l3_n100_accuracy_pct"
    math_base_x, math_base_y = coordinates(baseline, math_field)
    math_fig, math_ax = plt.subplots(
        figsize=(8.2, 5.6),
        constrained_layout=True,
    )
    style_axis(math_ax, "MATH-500 L1-L3 accuracy, n=100 (%)  →")
    math_ax.set_xlim(40, 54)
    math_ax.set_ylabel("HarmBench attack success rate (%)  ↓")
    math_ax.set_title("Main safety–math trajectories")
    math_ax.scatter(
        math_base_x,
        math_base_y,
        s=105,
        marker="*",
        color="#222222",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    for method in MAIN_ONLY_METHODS:
        method_rows = [
            row
            for row in rows
            if row["method"] == method
            and row["plot_role"] in ("main", "main_patch")
            and row[math_field]
        ]
        xs, ys = zip(
            *(coordinates(row, math_field) for row in method_rows), strict=True
        )
        math_ax.plot(
            [math_base_x, *xs],
            [math_base_y, *ys],
            color=COLORS[method],
            marker=MARKERS[method],
            markevery=range(1, len(xs) + 1),
            markersize=7,
            linewidth=2.3,
            zorder=2,
        )
    math_labels = {
        ("SN-Tune", f"alpha={alpha}"): rf"$\alpha={alpha}$"
        for alpha in (1, 4, 6, 8)
    }
    math_labels.update(
        {
            ("IA3-SFT", f"alpha={alpha}"): rf"$\alpha={alpha}$"
            for alpha in (1, 1.5, 2, 2.5, 3, 3.5)
        }
    )
    math_labels.update(
        {
            ("IA3 guide patch", f"K={k}"): rf"$K={k // 1000}\mathrm{{k}}$"
            for k in (40000, 80000, 160000, 320000)
        }
    )
    math_labels.update(
        {
            ("Grad (on-policy)", f"K={k}, strength=1"): rf"$K={k}$"
            for k in (1000, 2000, 4000)
        }
    )
    math_offsets = {
        ("SN-Tune", "alpha=1"): (7, 7),
        ("SN-Tune", "alpha=4"): (7, 7),
        ("SN-Tune", "alpha=6"): (8, 10),
        ("SN-Tune", "alpha=8"): (-48, -13),
        ("IA3-SFT", "alpha=1"): (7, 7),
        ("IA3-SFT", "alpha=1.5"): (7, 7),
        ("IA3-SFT", "alpha=2"): (7, 7),
        ("IA3-SFT", "alpha=2.5"): (7, -16),
        ("IA3-SFT", "alpha=3"): (-48, -15),
        ("IA3-SFT", "alpha=3.5"): (7, -15),
        ("IA3 guide patch", "K=40000"): (-9, 9),
        ("IA3 guide patch", "K=80000"): (7, 7),
        ("IA3 guide patch", "K=160000"): (-22, -17),
        ("IA3 guide patch", "K=320000"): (7, -15),
        ("Grad (on-policy)", "K=1000, strength=1"): (7, 7),
        ("Grad (on-policy)", "K=2000, strength=1"): (-55, 7),
        ("Grad (on-policy)", "K=4000, strength=1"): (7, 7),
    }
    math_rows = [
        row
        for row in rows
        if row[math_field]
        and row["method"] in MAIN_ONLY_METHODS
        and row["plot_role"] in ("main", "main_patch")
    ]
    annotate_rows(
        math_ax,
        math_rows,
        math_labels,
        math_offsets,
        x_field=math_field,
    )
    math_ax.annotate(
        "Baseline",
        (math_base_x, math_base_y),
        xytext=(7, 7),
        textcoords="offset points",
        color="#222222",
        fontsize=9,
    )
    math_ax.legend(
        handles=main_only_legend_handles[:5],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
    )
    for extension in ("png", "pdf", "svg"):
        math_fig.savefig(
            OUTPUT_DIR / f"math500_harmbench_tradeoff_main_only.{extension}",
            bbox_inches="tight",
        )
    plt.close(math_fig)

    gsm_field = "gsm8k_accuracy_pct"
    gsm_base_x, gsm_base_y = coordinates(baseline, gsm_field)
    gsm_fig, gsm_ax = plt.subplots(
        figsize=(8.2, 5.6),
        constrained_layout=True,
    )
    style_axis(gsm_ax, "GSM8K accuracy (%)  →")
    gsm_ax.set_xlim(65.5, 82)
    gsm_ax.set_ylabel("HarmBench attack success rate (%)  ↓")
    gsm_ax.set_title(
        "Main safety–math trajectories\n"
        "Grad: strength = 1, candidate pool = 20,000"
    )
    gsm_ax.scatter(
        gsm_base_x,
        gsm_base_y,
        s=105,
        marker="*",
        color="#222222",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    for method in METHODS:
        method_rows = [
            row
            for row in rows
            if row["method"] == method
            and row["plot_role"] == "main"
            and row[gsm_field]
        ]
        xs, ys = zip(
            *(coordinates(row, gsm_field) for row in method_rows),
            strict=True,
        )
        gsm_ax.plot(
            [gsm_base_x, *xs],
            [gsm_base_y, *ys],
            color=COLORS[method],
            marker=MARKERS[method],
            markevery=range(1, len(xs) + 1),
            markersize=7,
            linewidth=2.3,
            zorder=2,
        )

    gsm_offsets = {
        ("SN-Tune", "alpha=1"): (8, -16),
        ("SN-Tune", "alpha=4"): (8, 7),
        ("SN-Tune", "alpha=6"): (-48, 8),
        ("SN-Tune", "alpha=8"): (-42, -15),
        ("IA3-SFT", "alpha=1"): (8, -16),
        ("IA3-SFT", "alpha=2"): (-43, 8),
        ("IA3-SFT", "alpha=3"): (8, 5),
        ("IA3-SFT", "alpha=3.5"): (-54, -14),
        ("Grad (off-policy)", "K=1000, strength=1"): (8, -16),
        ("Grad (off-policy)", "K=2000, strength=1"): (-56, -15),
        ("Grad (off-policy)", "K=4000, strength=1"): (8, 8),
        ("Grad (on-policy)", "K=1000, strength=1"): (8, 8),
        ("Grad (on-policy)", "K=2000, strength=1"): (8, -14),
        ("Grad (on-policy)", "K=4000, strength=1"): (8, 8),
    }
    annotate_rows(gsm_ax, main_rows, main_labels, gsm_offsets, gsm_field)
    gsm_ax.annotate(
        "Baseline",
        (gsm_base_x, gsm_base_y),
        xytext=(-62, 8),
        textcoords="offset points",
        color="#222222",
        fontsize=9,
    )
    gsm_ax.legend(
        handles=legend_handles[:5],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
    )
    for extension in ("png", "pdf", "svg"):
        gsm_fig.savefig(
            OUTPUT_DIR / f"gsm8k_harmbench_tradeoff_main_only.{extension}",
            bbox_inches="tight",
        )
    plt.close(gsm_fig)


if __name__ == "__main__":
    main()
