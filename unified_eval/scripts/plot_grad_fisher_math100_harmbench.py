#!/usr/bin/env python3
"""Plot frozen HarmBench/MATH100 results for direct and Fisher Grad only."""

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
    family: str
    marker: str
    offset: tuple[int, int]
    harmbench_run: str
    math_run: str
    harmbench_summary: str | None = None
    harmbench_experiment: str | None = None


POINTS = (
    Point(
        "direct 1k, s=1",
        "direct_s1",
        "o",
        (8, -15),
        "grad_firstcue_safe256_tail_positive_k1000_s1_fp32",
        "math500_l1_l3_n100_grad_firstcue_safe256_tail_positive_k1000_s1_fp32",
    ),
    Point(
        "direct 2k, s=1",
        "direct_s1",
        "o",
        (-88, 8),
        "grad_firstcue_safe256_tail_positive_k2000_s1_fp32",
        "math500_l1_l3_n100_grad_firstcue_safe256_tail_positive_k2000_s1_fp32",
    ),
    Point(
        "direct 4k, s=1",
        "direct_s1",
        "o",
        (8, 8),
        "grad_firstcue_safe256_tail_positive_k4000_s1_fp32",
        "math500_l1_l3_n100_grad_firstcue_safe256_tail_positive_k4000_s1_fp32",
    ),
    Point(
        "direct 4k, s=.75",
        "direct_alt",
        "D",
        (-92, 8),
        "grad_firstcue_safe256_tail_positive_k4000_s0p75_fp32",
        "math500_l1_l3_n100_grad_firstcue_safe256_tail_positive_k4000_s0p75_fp32",
    ),
    Point(
        "direct 8k, s=.6",
        "direct_alt",
        "D",
        (-94, -18),
        "grad_direct_firstcue256_k8000_s0p6_fp32",
        "math100_grad_direct_k8000_s0p6",
    ),
    Point(
        "direct 12k, s=.3",
        "direct_alt",
        "D",
        (-97, 8),
        "grad_direct_firstcue256_k12000_s0p3_frozen_with_bbh",
        "grad_direct_firstcue256_k12000_s0p3_frozen_with_bbh",
    ),
    Point(
        "direct 16k, s=.3",
        "direct_alt",
        "D",
        (-100, -18),
        "grad_direct_firstcue256_k16000_s0p3_frozen_with_bbh",
        "grad_direct_firstcue256_k16000_s0p3_frozen_with_bbh",
    ),
    Point(
        "Fisher 8k, floor=.35",
        "fisher",
        "v",
        (8, 8),
        "grad_floorfisher_f0p35_m0p6_c0p75_d1_firstcue256_k8000",
        "math100_grad_floorfisher_f0p35_m0p6_c0p75_d1_firstcue256_k8000_fp32",
    ),
    Point(
        "Fisher 8k, floor=.4",
        "fisher",
        "v",
        (8, -18),
        "grad_floorfisher_f0p4_m0p6_c0p75_d1_firstcue256_k8000",
        "math100_grad_floorfisher_f0p4_m0p6_c0p75_d1_firstcue256_k8000_fp32",
    ),
    Point(
        "Fisher 12k, c=.22",
        "fisher",
        "*",
        (8, 9),
        "grad_floorfisher_wt2048_k12000_f0_c0p22_cap0p75",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p22_cap0p75",
        "grad_floor_fisher_wikitext2048_gentler12k/frozen_harmbench_c0p22/summary.json",
        "floorfisher_k12000_floor0p0_c0p22_cap0p75_damp1p0",
    ),
    Point(
        "Fisher 12k, c=.24",
        "fisher",
        "*",
        (8, -18),
        "grad_floorfisher_wt2048_k12000_f0_c0p24_cap0p75",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p24_cap0p75",
        "grad_floor_fisher_wikitext2048_gentle12k/frozen_harmbench_c0p24/summary.json",
        "floorfisher_k12000_floor0p0_c0p24_cap0p75_damp1p0",
    ),
    Point(
        "Fisher 12k, c=.48",
        "fisher",
        "*",
        (-105, -18),
        "grad_floorfisher_wt2048_k12000_f0_c0p48",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p48_cap0p75",
        "grad_floor_fisher_wikitext2048_finalists/frozen_harmbench/summary.json",
        "floorfisher_k12000_floor0p0_c0p48_cap0p75_damp1p0",
    ),
    Point(
        "Fisher 12k, cap=.6",
        "fisher",
        "h",
        (-112, 8),
        "grad_floorfisher_wt2048_k12000_f0_c0p56_cap0p6",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p56_cap0p6",
        "grad_floor_fisher_wikitext2048_smaller_cap/frozen_harmbench_cap0p6_f0_c0p56/summary.json",
        "floorfisher_k12000_floor0p0_c0p56_cap0p6_damp1p0",
    ),
    Point(
        "Fisher 16k, c=.22",
        "fisher",
        "*",
        (-108, -18),
        "grad_floorfisher_wt2048_k16000_f0_c0p22_cap0p75_frozen",
        "grad_floorfisher_wt2048_k16000_f0_c0p22_cap0p75_frozen",
    ),
)


def load_point(point: Point) -> dict[str, float | str | tuple[int, int]]:
    harmbench_payload = json.loads(
        (RESULTS / point.harmbench_run / "summary.json").read_text(encoding="utf-8")
    )
    if point.harmbench_summary is None:
        harmbench = float(harmbench_payload["harmbench"]["attack_success_rate"])
    else:
        sweep_payload = json.loads(
            (RESULTS / point.harmbench_summary).read_text(encoding="utf-8")
        )
        match = next(
            row
            for row in sweep_payload["summaries"]
            if row["experiment"] == point.harmbench_experiment
        )
        harmbench = float(match["attack_success_rate"])
    math_payload = json.loads(
        (RESULTS / point.math_run / "summary.json").read_text(encoding="utf-8")
    )
    return {
        "label": point.label,
        "family": point.family,
        "marker": point.marker,
        "offset": point.offset,
        "harmbench": harmbench,
        "math": float(math_payload["math500"]["accuracy"]),
    }


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [load_point(point) for point in POINTS]
    direct_s1 = [row for row in rows if row["family"] == "direct_s1"]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "svg.hashsalt": "grad-fisher-math100-harmbench",
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 6.0), constrained_layout=True)
    ax.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    ax.plot(
        [float(row["math"]) for row in direct_s1],
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
            float(row["math"]),
            float(row["harmbench"]),
            s=78 if is_fisher else 66,
            marker=str(row["marker"]),
            facecolor="white" if is_alt else color,
            edgecolor=color,
            linewidth=1.5 if is_alt else 0.8,
            zorder=3,
        )
        ax.annotate(
            str(row["label"]),
            (float(row["math"]), float(row["harmbench"])),
            xytext=row["offset"],
            textcoords="offset points",
            color=color,
            fontsize=8.5,
        )

    ax.set_xlim(36.0, 50.3)
    ax.set_ylim(-0.3, 24.2)
    ax.set_xlabel("MATH100 accuracy (%)  →")
    ax.set_ylabel("HarmBench attack success rate (%)  ↓")
    ax.set_title("Direct versus Fisher-guided first-cue Grad interventions")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=DIRECT_COLOR,
                marker="o",
                linewidth=1.8,
                markersize=6.5,
                label="Direct uniform scaling, s=1",
            ),
            Line2D(
                [0],
                [0],
                color="none",
                marker="D",
                markerfacecolor="white",
                markeredgecolor=DIRECT_COLOR,
                markersize=6.5,
                label="Direct alternate strength",
            ),
            Line2D(
                [0],
                [0],
                color="none",
                marker="*",
                markerfacecolor=FISHER_COLOR,
                markeredgecolor=FISHER_COLOR,
                markersize=9,
                label="Fisher-guided",
            ),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        output = OUTPUT_DIR / f"grad_fisher_math100_harmbench.{extension}"
        fig.savefig(output, bbox_inches="tight")
        if extension == "svg":
            normalize_svg(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
