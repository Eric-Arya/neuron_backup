#!/usr/bin/env python3
"""Plot the main HarmBench--capability comparisons with direct and Fisher Grad.

The three plots use the same curated Grad subset so that the BBH, MATH100, and
IFEval panels can be compared directly.  Existing method trajectories are read
from the frozen trade-off table; new Grad values are read from their frozen run
artifacts rather than copied into this script.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TRADEOFF_CSV = RESULTS / "ifeval_harmbench_tradeoff.csv"

BASELINE = "Unmodified"
SN_TUNE = "SN-Tune"
IA3_SFT = "IA3-SFT"
IA3_PATCH = "IA3 guide patch"
GRAD_ON_POLICY = "Grad (on-policy)"
DIRECT_GRAD = "Grad (first-cue-256)"
FISHER_GRAD = "Diagonal Fisher Grad (first-cue-256)"

COLORS = {
    SN_TUNE: "#0072B2",
    IA3_SFT: "#CC79A7",
    IA3_PATCH: "#E69F00",
    GRAD_ON_POLICY: "#009E73",
    DIRECT_GRAD: "#56B4E9",
    FISHER_GRAD: "#D55E00",
}
MARKERS = {
    SN_TUNE: "o",
    IA3_SFT: "^",
    IA3_PATCH: "P",
    GRAD_ON_POLICY: "D",
    DIRECT_GRAD: "X",
    FISHER_GRAD: "h",
}

OTHER_ORDER = {
    SN_TUNE: ("alpha=1", "alpha=4", "alpha=6", "alpha=8"),
    IA3_SFT: (
        "alpha=1",
        "alpha=1.5",
        "alpha=2",
        "alpha=2.5",
        "alpha=3",
        "alpha=3.5",
    ),
    IA3_PATCH: ("K=40000", "K=80000", "K=160000", "K=320000"),
    GRAD_ON_POLICY: (
        "K=1000, strength=1",
        "K=2000, strength=1",
        "K=4000, strength=1",
    ),
}

BBH_RUNS = {
    ("Baseline", BASELINE): "bbh_llama3_base_raw_cot_fp32",
    (SN_TUNE, "alpha=1"): "bbh_sn_alpha1_raw_cot_fp32",
    (SN_TUNE, "alpha=4"): "bbh_sn_alpha4_raw_cot_fp32",
    (SN_TUNE, "alpha=6"): "bbh_sn_alpha6_raw_cot_fp32",
    (SN_TUNE, "alpha=8"): "bbh_sn_alpha8_raw_cot_fp32",
    (IA3_SFT, "alpha=1"): "bbh_ia3_sft_snraw_alpha1_raw_cot_fp32",
    (IA3_SFT, "alpha=1.5"): "bbh_ia3_sft_snraw_alpha1p5_raw_cot_fp32",
    (IA3_SFT, "alpha=2"): "bbh_ia3_sft_snraw_alpha2_raw_cot_fp32",
    (IA3_SFT, "alpha=2.5"): "bbh_ia3_sft_snraw_alpha2p5_raw_cot_fp32",
    (IA3_SFT, "alpha=3"): "bbh_ia3_sft_snraw_alpha3_raw_cot_fp32",
    (IA3_SFT, "alpha=3.5"): "bbh_ia3_sft_snraw_alpha3p5_raw_cot_fp32",
    (IA3_PATCH, "K=40000"): "bbh_sft_patch_snraw_alpha3_top40000_raw_cot_bf16",
    (IA3_PATCH, "K=80000"): "bbh_sft_patch_snraw_alpha3_top80000_raw_cot_bf16",
    (GRAD_ON_POLICY, "K=1000, strength=1"): (
        "bbh_grad_onpolicy_expanded_k1000_s1_raw_cot_fp32"
    ),
    (GRAD_ON_POLICY, "K=2000, strength=1"): (
        "bbh_grad_onpolicy_expanded_k2000_s1_raw_cot_fp32"
    ),
    (GRAD_ON_POLICY, "K=4000, strength=1"): (
        "bbh_grad_onpolicy_expanded_k4000_s1_raw_cot_fp32"
    ),
}


@dataclass(frozen=True)
class Point:
    method: str
    setting: str
    harmbench: float
    capability: float


@dataclass(frozen=True)
class DirectSource:
    setting: str
    safety_run: str
    math_run: str
    bbh_run: str


@dataclass(frozen=True)
class FisherSource:
    setting: str
    ifeval_run: str
    math_run: str
    bbh_run: str
    harmbench_summary: str
    harmbench_experiment: str


DIRECT_SOURCES = (
    DirectSource(
        "K=1k, s=1",
        "grad_firstcue_safe256_tail_positive_k1000_s1_fp32",
        "math500_l1_l3_n100_grad_firstcue_safe256_tail_positive_k1000_s1_fp32",
        "bbh_grad_firstcue_safe256_tail_positive_k1000_s1_raw_cot_fp32",
    ),
    DirectSource(
        "K=2k, s=1",
        "grad_firstcue_safe256_tail_positive_k2000_s1_fp32",
        "math500_l1_l3_n100_grad_firstcue_safe256_tail_positive_k2000_s1_fp32",
        "bbh_grad_firstcue_safe256_tail_positive_k2000_s1_raw_cot_fp32",
    ),
    DirectSource(
        "K=4k, s=1",
        "grad_firstcue_safe256_tail_positive_k4000_s1_fp32",
        "math500_l1_l3_n100_grad_firstcue_safe256_tail_positive_k4000_s1_fp32",
        "bbh_grad_firstcue_safe256_tail_positive_k4000_s1_raw_cot_fp32",
    ),
    DirectSource(
        "K=4k, s=.75",
        "grad_firstcue_safe256_tail_positive_k4000_s0p75_fp32",
        "math500_l1_l3_n100_grad_firstcue_safe256_tail_positive_k4000_s0p75_fp32",
        "bbh_grad_firstcue_safe256_tail_positive_k4000_s0p75_raw_cot_fp32",
    ),
)

FISHER_SOURCES = (
    FisherSource(
        "K=12k, c=.22",
        "grad_floorfisher_wt2048_k12000_f0_c0p22_cap0p75",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p22_cap0p75",
        "bbh_grad_floorfisher_wt2048_k12000_f0_c0p22_cap0p75_raw_cot_fp32",
        "grad_floor_fisher_wikitext2048_gentler12k/frozen_harmbench_c0p22/summary.json",
        "floorfisher_k12000_floor0p0_c0p22_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=12k, c=.24",
        "grad_floorfisher_wt2048_k12000_f0_c0p24_cap0p75",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p24_cap0p75",
        "bbh_grad_floorfisher_wt2048_k12000_f0_c0p24_cap0p75_raw_cot_fp32",
        "grad_floor_fisher_wikitext2048_gentle12k/frozen_harmbench_c0p24/summary.json",
        "floorfisher_k12000_floor0p0_c0p24_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=12k, c=.48",
        "grad_floorfisher_wt2048_k12000_f0_c0p48",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p48_cap0p75",
        "bbh_grad_floorfisher_wt2048_k12000_f0_c0p48_cap0p75_raw_cot_fp32",
        "grad_floor_fisher_wikitext2048_finalists/frozen_harmbench/summary.json",
        "floorfisher_k12000_floor0p0_c0p48_cap0p75_damp1p0",
    ),
)

BENCHMARKS = {
    "ifeval": {
        "field": "ifeval_strict_prompt_accuracy_pct",
        "xlabel": "IFEval strict prompt accuracy (%)  →",
        "title": "Safety–instruction-following trade-offs",
        "xlim": (50.0, 75.0),
        "stem": "ifeval_harmbench_tradeoff_main_only",
    },
    "math": {
        "field": "math500_l1_l3_n100_accuracy_pct",
        "xlabel": "MATH-500 L1–L3 accuracy, n=100 (%)  →",
        "title": "Safety–math trade-offs",
        "xlim": (36.0, 54.0),
        "stem": "math500_harmbench_tradeoff_main_only",
    },
    "bbh": {
        "field": None,
        "xlabel": "BBH exact-match accuracy, n=200 (%)  →",
        "title": "Safety–BBH trade-offs",
        "xlim": (38.0, 68.0),
        "stem": "bbh_harmbench_tradeoff_main_only",
    },
}


def read_json(relative: str) -> dict[str, Any]:
    path = RESULTS / relative
    if path.is_dir():
        path /= "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_tradeoff_rows() -> dict[tuple[str, str], dict[str, str]]:
    with TRADEOFF_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["method"], row["setting"])
        if key not in selected or row["plot_role"] in ("main", "main_patch"):
            selected[key] = row
    return selected


def load_bbh(run: str) -> float:
    payload = read_json(run)
    bbh = payload["bbh"]
    if bbh["num_samples"] != 200 or bbh["num_tasks"] != 27:
        raise ValueError(f"Unexpected BBH coverage for {run}")
    return float(bbh["accuracy"])


def load_math(run: str) -> float:
    payload = read_json(run)
    math = payload["math500"]
    if math["num_samples"] != 100:
        raise ValueError(f"Unexpected MATH coverage for {run}")
    return float(math["accuracy"])


def load_direct_points(benchmark: str) -> list[Point]:
    points = []
    for source in DIRECT_SOURCES:
        safety = read_json(source.safety_run)
        harmbench = safety["harmbench"]
        if harmbench["num_samples"] != 200:
            raise ValueError(f"Unexpected HarmBench coverage for {source.safety_run}")
        if benchmark == "ifeval":
            if safety["ifeval"]["num_samples"] != 200:
                raise ValueError(f"Unexpected IFEval coverage for {source.safety_run}")
            capability = float(safety["ifeval"]["strict"]["prompt_accuracy"])
        elif benchmark == "math":
            capability = load_math(source.math_run)
        else:
            capability = load_bbh(source.bbh_run)
        points.append(
            Point(
                DIRECT_GRAD,
                source.setting,
                float(harmbench["attack_success_rate"]),
                capability,
            )
        )
    return points


def load_sweep_harmbench(source: FisherSource) -> float:
    payload = read_json(source.harmbench_summary)
    row = next(
        item
        for item in payload["summaries"]
        if item["experiment"] == source.harmbench_experiment
    )
    if row["num_samples"] != 200:
        raise ValueError(
            f"Unexpected HarmBench coverage for {source.harmbench_experiment}"
        )
    return float(row["attack_success_rate"])


def load_fisher_points(benchmark: str) -> list[Point]:
    points = []
    for source in FISHER_SOURCES:
        if benchmark == "ifeval":
            payload = read_json(source.ifeval_run)
            if payload["ifeval"]["num_samples"] != 200:
                raise ValueError(f"Unexpected IFEval coverage for {source.ifeval_run}")
            capability = float(payload["ifeval"]["strict"]["prompt_accuracy"])
        elif benchmark == "math":
            capability = load_math(source.math_run)
        else:
            capability = load_bbh(source.bbh_run)
        points.append(
            Point(
                FISHER_GRAD,
                source.setting,
                load_sweep_harmbench(source),
                capability,
            )
        )
    return points


def load_other_points(
    benchmark: str, rows: dict[tuple[str, str], dict[str, str]]
) -> tuple[Point, dict[str, list[Point]]]:
    field = BENCHMARKS[benchmark]["field"]
    baseline_row = rows[("Baseline", BASELINE)]
    if benchmark == "bbh":
        baseline_capability = load_bbh(BBH_RUNS[("Baseline", BASELINE)])
    else:
        baseline_capability = float(baseline_row[str(field)])
    baseline = Point(
        BASELINE,
        BASELINE,
        float(baseline_row["harmbench_asr_pct"]),
        baseline_capability,
    )

    groups: dict[str, list[Point]] = {}
    for method, settings in OTHER_ORDER.items():
        method_points = []
        for setting in settings:
            key = (method, setting)
            if benchmark == "bbh" and key not in BBH_RUNS:
                continue
            row = rows[key]
            capability = (
                load_bbh(BBH_RUNS[key])
                if benchmark == "bbh"
                else float(row[str(field)])
            )
            method_points.append(
                Point(method, setting, float(row["harmbench_asr_pct"]), capability)
            )
        groups[method] = method_points
    return baseline, groups


def display_label(point: Point) -> str:
    setting = point.setting
    if setting.startswith("alpha="):
        return rf"$\alpha={setting.removeprefix('alpha=')}$"
    if point.method == IA3_PATCH:
        return rf"$K={int(setting.removeprefix('K=')) // 1000}\mathrm{{k}}$"
    if point.method == FISHER_GRAD:
        return rf"$c={setting.rsplit('=', 1)[1]}$"
    if point.method == DIRECT_GRAD and "s=.75" in setting:
        return r"$K=4\mathrm{k},\ s=.75$"
    k_value = setting.split(",", maxsplit=1)[0].removeprefix("K=")
    return rf"$K={k_value}$"


OFFSETS: dict[str, dict[tuple[str, str], tuple[int, int]]] = {
    "ifeval": {
        (SN_TUNE, "alpha=1"): (-48, -17),
        (SN_TUNE, "alpha=4"): (7, 7),
        (SN_TUNE, "alpha=6"): (7, 7),
        (SN_TUNE, "alpha=8"): (7, 7),
        (IA3_SFT, "alpha=1"): (7, -17),
        (IA3_SFT, "alpha=1.5"): (7, 6),
        (IA3_SFT, "alpha=2"): (7, 6),
        (IA3_SFT, "alpha=2.5"): (-50, -17),
        (IA3_SFT, "alpha=3"): (-48, 6),
        (IA3_SFT, "alpha=3.5"): (7, -16),
        (IA3_PATCH, "K=40000"): (-17, 8),
        (IA3_PATCH, "K=80000"): (7, 6),
        (IA3_PATCH, "K=160000"): (-55, 7),
        (IA3_PATCH, "K=320000"): (8, 10),
        (GRAD_ON_POLICY, "K=1000, strength=1"): (7, -17),
        (GRAD_ON_POLICY, "K=2000, strength=1"): (-10, 14),
        (GRAD_ON_POLICY, "K=4000, strength=1"): (-56, -16),
        (DIRECT_GRAD, "K=1k, s=1"): (-48, 7),
        (DIRECT_GRAD, "K=2k, s=1"): (-58, 7),
        (DIRECT_GRAD, "K=4k, s=1"): (7, -17),
        (DIRECT_GRAD, "K=4k, s=.75"): (8, 15),
        (FISHER_GRAD, "K=12k, c=.22"): (9, -19),
        (FISHER_GRAD, "K=12k, c=.24"): (-48, -19),
        (FISHER_GRAD, "K=12k, c=.48"): (-10, -20),
    },
    "math": {
        (SN_TUNE, "alpha=1"): (7, 7),
        (SN_TUNE, "alpha=4"): (7, 7),
        (SN_TUNE, "alpha=6"): (7, 7),
        (SN_TUNE, "alpha=8"): (-48, -14),
        (IA3_SFT, "alpha=1"): (7, 7),
        (IA3_SFT, "alpha=1.5"): (7, 7),
        (IA3_SFT, "alpha=2"): (7, 7),
        (IA3_SFT, "alpha=2.5"): (7, -16),
        (IA3_SFT, "alpha=3"): (-48, -15),
        (IA3_SFT, "alpha=3.5"): (7, -15),
        (IA3_PATCH, "K=40000"): (-9, 9),
        (IA3_PATCH, "K=80000"): (7, 7),
        (IA3_PATCH, "K=160000"): (-22, -17),
        (IA3_PATCH, "K=320000"): (7, -15),
        (GRAD_ON_POLICY, "K=1000, strength=1"): (7, 7),
        (GRAD_ON_POLICY, "K=2000, strength=1"): (-55, 7),
        (GRAD_ON_POLICY, "K=4000, strength=1"): (7, 7),
        (DIRECT_GRAD, "K=1k, s=1"): (-52, 7),
        (DIRECT_GRAD, "K=2k, s=1"): (7, 7),
        (DIRECT_GRAD, "K=4k, s=1"): (7, -16),
        (DIRECT_GRAD, "K=4k, s=.75"): (-88, 14),
        (FISHER_GRAD, "K=12k, c=.22"): (8, 14),
        (FISHER_GRAD, "K=12k, c=.24"): (7, -17),
        (FISHER_GRAD, "K=12k, c=.48"): (7, 7),
    },
    "bbh": {
        (SN_TUNE, "alpha=1"): (-48, -17),
        (SN_TUNE, "alpha=4"): (-47, 7),
        (SN_TUNE, "alpha=6"): (-47, 7),
        (SN_TUNE, "alpha=8"): (7, 6),
        (IA3_SFT, "alpha=1"): (7, -17),
        (IA3_SFT, "alpha=1.5"): (7, 5),
        (IA3_SFT, "alpha=2"): (-47, 5),
        (IA3_SFT, "alpha=2.5"): (-53, -16),
        (IA3_SFT, "alpha=3"): (7, -15),
        (IA3_SFT, "alpha=3.5"): (7, 5),
        (IA3_PATCH, "K=40000"): (7, 6),
        (IA3_PATCH, "K=80000"): (7, -16),
        (GRAD_ON_POLICY, "K=1000, strength=1"): (-58, 7),
        (GRAD_ON_POLICY, "K=2000, strength=1"): (-58, -16),
        (GRAD_ON_POLICY, "K=4000, strength=1"): (7, -15),
        (DIRECT_GRAD, "K=1k, s=1"): (7, -17),
        (DIRECT_GRAD, "K=2k, s=1"): (-54, 7),
        (DIRECT_GRAD, "K=4k, s=1"): (7, -17),
        (DIRECT_GRAD, "K=4k, s=.75"): (-72, 14),
        (FISHER_GRAD, "K=12k, c=.22"): (8, 13),
        (FISHER_GRAD, "K=12k, c=.24"): (-43, -17),
        (FISHER_GRAD, "K=12k, c=.48"): (7, 7),
    },
}


def annotate(ax: plt.Axes, benchmark: str, point: Point) -> None:
    ax.annotate(
        display_label(point),
        (point.capability, point.harmbench),
        xytext=OFFSETS[benchmark][(point.method, point.setting)],
        textcoords="offset points",
        color=COLORS[point.method],
        fontsize=8.4,
    )


def plot_trajectory(
    ax: plt.Axes,
    benchmark: str,
    baseline: Point,
    points: list[Point],
    *,
    linestyle: str = "-",
    markerfacecolor: str | None = None,
) -> None:
    if not points:
        return
    method = points[0].method
    xs = [baseline.capability, *(point.capability for point in points)]
    ys = [baseline.harmbench, *(point.harmbench for point in points)]
    ax.plot(
        xs,
        ys,
        color=COLORS[method],
        marker=MARKERS[method],
        markevery=range(1, len(xs)),
        markerfacecolor=markerfacecolor,
        markeredgewidth=1.4 if markerfacecolor else 0.8,
        markersize=7.2,
        linewidth=2.15,
        linestyle=linestyle,
        zorder=2,
    )
    for point in points:
        annotate(ax, benchmark, point)


def normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = FIGURES / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        if extension == "svg":
            normalize_svg(path)


def render(benchmark: str, rows: dict[tuple[str, str], dict[str, str]]) -> None:
    config = BENCHMARKS[benchmark]
    baseline, other_groups = load_other_points(benchmark, rows)
    direct = load_direct_points(benchmark)
    fisher = load_fisher_points(benchmark)

    fig, ax = plt.subplots(figsize=(11.4, 6.4), constrained_layout=True)
    ax.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(*config["xlim"])
    ax.set_ylim(-4, 71)
    ax.set_xlabel(str(config["xlabel"]))
    ax.set_ylabel("HarmBench attack success rate (%)  ↓")
    ax.set_title(
        f"{config['title']}\n"
        "Direct: solid s=1 K sweep, dashed K=4k strength branch; "
        "diagonal Fisher: K=12k c sweep"
    )

    ax.scatter(
        baseline.capability,
        baseline.harmbench,
        s=110,
        marker="*",
        color="#222222",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    ax.annotate(
        "Baseline",
        (baseline.capability, baseline.harmbench),
        xytext=(7, 7),
        textcoords="offset points",
        color="#222222",
        fontsize=9,
    )

    for points in other_groups.values():
        plot_trajectory(ax, benchmark, baseline, points)

    direct_s1 = [point for point in direct if point.setting.endswith("s=1")]
    direct_s075 = [point for point in direct if point.setting.endswith("s=.75")]
    plot_trajectory(ax, benchmark, baseline, direct_s1)
    # A fixed-K strength branch, shown hollow and dashed to distinguish it from
    # the direct s=1 K trajectory. It shares the K=4k,s=1 endpoint.
    direct_k4_s1 = next(point for point in direct_s1 if point.setting.startswith("K=4k"))
    branch = [*direct_s075, direct_k4_s1]
    branch_x = [baseline.capability, *(point.capability for point in branch)]
    branch_y = [baseline.harmbench, *(point.harmbench for point in branch)]
    ax.plot(
        branch_x,
        branch_y,
        color=COLORS[DIRECT_GRAD],
        marker=MARKERS[DIRECT_GRAD],
        markevery=[1],
        markerfacecolor="white",
        markeredgewidth=1.5,
        markersize=7.2,
        linewidth=1.7,
        linestyle="--",
        zorder=3,
    )
    annotate(ax, benchmark, direct_s075[0])
    plot_trajectory(ax, benchmark, baseline, fisher)

    legend = [
        Line2D([], [], marker="*", linestyle="none", color="#222222", label=BASELINE),
        *[
            Line2D(
                [],
                [],
                marker=MARKERS[method],
                color=COLORS[method],
                label=("IA3-SFT (SNCorpus raw)" if method == IA3_SFT else method),
            )
            for method in (
                SN_TUNE,
                IA3_SFT,
                IA3_PATCH,
                GRAD_ON_POLICY,
                DIRECT_GRAD,
                FISHER_GRAD,
            )
        ],
    ]
    ax.legend(
        handles=legend,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.205),
        ncol=4,
    )
    save_figure(fig, str(config["stem"]))
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "svg.hashsalt": "main-tradeoffs-with-fisher-v1",
        }
    )
    rows = load_tradeoff_rows()
    for benchmark in ("bbh", "math", "ifeval"):
        render(benchmark, rows)


if __name__ == "__main__":
    main()
