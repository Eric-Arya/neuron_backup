#!/usr/bin/env python3
"""Plot frozen HarmBench/IFEval results for direct and Fisher-guided Grad."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = ROOT / "figures"

DIRECT_COLOR = "#0072B2"
FISHER_COLOR = "#D55E00"


@dataclass(frozen=True)
class Point:
    label: str
    run_dir: str
    family: str
    marker: str
    offset: tuple[int, int]


POINTS = (
    Point("direct 2k, s=1", "grad_firstcue_safe256_tail_positive_k2000_s1_fp32", "direct", "o", (8, 2)),
    Point("direct 4k, s=1", "grad_firstcue_safe256_tail_positive_k4000_s1_fp32", "direct", "o", (8, 17)),
    Point("direct 5k, s=1", "grad_direct_firstcue256_k5000_s1_fp32", "direct", "o", (-80, 8)),
    Point("direct 6k, s=1", "grad_direct_firstcue256_k6000_s1_fp32", "direct", "o", (-78, -18)),
    Point("direct 8k, s=0.75", "grad_direct_firstcue256_k8000_s0p75", "direct_alt", "D", (8, -17)),
    Point("direct 8k, s=0.6", "grad_direct_firstcue256_k8000_s0p6_fp32", "direct_alt", "D", (8, 7)),
    Point("direct 8k, s=1", "grad_direct_firstcue256_k8000_s1_fp32", "direct", "o", (8, 7)),
    Point("Fisher diagonal, cap=1", "grad_fisher_diag_cap1_firstcue256_k8000", "fisher", "s", (8, 8)),
    Point("Fisher select 4k", "grad_fisherselect_sqrt_active4000_s1_firstcue256", "fisher", "^", (8, 8)),
    Point("Fisher select 6k", "grad_fisherselect_fisher_active6000_s1_firstcue256", "fisher", "^", (8, 8)),
    Point("Fisher replace 500/4k", "grad_fisherreplace_sqrt_base4000_replace500_s1_firstcue256", "fisher", "X", (8, 8)),
    Point("bounded Fisher 8k, s=0.6 KL", "grad_boxfisher_actualkl_s0p6_firstcue256_k8000", "fisher", "s", (-135, 12)),
    Point("20% Fisher blend", "grad_blendfisher0p2_actualkl_s0p6_firstcue256_k8000", "fisher", "P", (-92, -18)),
    Point("80% Fisher blend", "grad_blendfisher0p8_actualkl_s0p6_firstcue256_k8000", "fisher", "P", (-90, 8)),
)


def load_point(point: Point) -> dict[str, float | str]:
    path = RESULTS / point.run_dir / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": point.label,
        "family": point.family,
        "marker": point.marker,
        "offset": point.offset,
        "ifeval": float(payload["ifeval"]["strict"]["prompt_accuracy"]),
        "harmbench": float(payload["harmbench"]["attack_success_rate"]),
    }


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [load_point(point) for point in POINTS]
    direct_s1 = [row for row in rows if row["family"] == "direct"]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "svg.hashsalt": "grad-fisher-ifeval-harmbench",
        }
    )
    fig, ax = plt.subplots(figsize=(8.7, 5.8), constrained_layout=True)
    ax.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Connecting only the uniform s=1 sequence makes the direct K sweep explicit.
    ax.plot(
        [float(row["ifeval"]) for row in direct_s1],
        [float(row["harmbench"]) for row in direct_s1],
        color=DIRECT_COLOR,
        linewidth=1.8,
        alpha=0.8,
        zorder=2,
    )

    for row in rows:
        is_fisher = row["family"] == "fisher"
        is_alt = row["family"] == "direct_alt"
        color = FISHER_COLOR if is_fisher else DIRECT_COLOR
        ax.scatter(
            float(row["ifeval"]),
            float(row["harmbench"]),
            s=76 if is_fisher else 66,
            marker=str(row["marker"]),
            facecolor="white" if is_alt else color,
            edgecolor=color,
            linewidth=1.5 if is_alt else 0.8,
            zorder=3,
        )
        ax.annotate(
            str(row["label"]),
            (float(row["ifeval"]), float(row["harmbench"])),
            xytext=row["offset"],
            textcoords="offset points",
            color=color,
            fontsize=8.6,
        )

    ax.set_xlim(52.5, 68.2)
    ax.set_ylim(-0.8, 14.7)
    ax.set_xlabel("IFEval strict prompt accuracy (%)  →")
    ax.set_ylabel("HarmBench attack success rate (%)  ↓")
    ax.set_title("Direct versus Fisher-guided first-cue Grad interventions")
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color=DIRECT_COLOR, marker="o", linewidth=1.8,
                markersize=6.5, label="Direct uniform scaling",
            ),
            Line2D(
                [0], [0], color="none", marker="D", markerfacecolor="white",
                markeredgecolor=DIRECT_COLOR, markersize=6.5,
                label="Direct 8k, alternate strength",
            ),
            Line2D(
                [0], [0], color="none", marker="^", markerfacecolor=FISHER_COLOR,
                markeredgecolor=FISHER_COLOR, markersize=7,
                label="Fisher-guided",
            ),
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.95,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        output = OUTPUT_DIR / f"grad_fisher_ifeval_harmbench.{extension}"
        fig.savefig(output, bbox_inches="tight")
        if extension == "svg":
            normalize_svg(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
