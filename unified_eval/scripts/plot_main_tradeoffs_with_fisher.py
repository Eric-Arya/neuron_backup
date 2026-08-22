#!/usr/bin/env python3
"""Plot the main HarmBench--capability comparisons with direct and Fisher Grad.

All Grad points use the first-cue-256 ranking objective. The three plots use the
same curated Grad subset so that the BBH, MATH100, and IFEval panels can be
compared directly. Existing non-Grad trajectories are read from the frozen
trade-off table; Grad values are read from their frozen run artifacts rather
than copied into this script.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep PDF/SVG metadata deterministic across regenerations.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

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
DIRECT_GRAD = "Direct Grad"
FISHER_GRAD = "Diagonal Fisher Grad"

COLORS = {
    SN_TUNE: "#0072B2",
    IA3_SFT: "#CC79A7",
    IA3_PATCH: "#E69F00",
    DIRECT_GRAD: "#56B4E9",
    FISHER_GRAD: "#D55E00",
}
MARKERS = {
    SN_TUNE: "o",
    IA3_SFT: "o",
    IA3_PATCH: "o",
    DIRECT_GRAD: "o",
    FISHER_GRAD: "o",
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
        "K=1k, c=.64",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k1000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k1000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k1000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/frozen_harmbench/shard0-of-2/summary.json",
        "floorfisher_k1000_floor0p0_c0p64_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=2k, c=.64",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k2000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k2000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k2000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/frozen_harmbench/shard1-of-2/summary.json",
        "floorfisher_k2000_floor0p0_c0p64_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=4k, c=.40",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k4000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k4000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k4000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/frozen_harmbench/shard0-of-2/summary.json",
        "floorfisher_k4000_floor0p0_c0p4_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=6k, c=.52",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k6000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k6000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k6000_cap0p75_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/frozen_harmbench/shard1-of-2/summary.json",
        "floorfisher_k6000_floor0p0_c0p52_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=8k, c=.48",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k8000_c0p48_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k8000_c0p48_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k8000_c0p48_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/frozen_harmbench/shard1-of-2/summary.json",
        "floorfisher_k8000_floor0p0_c0p48_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=12k, c=.22",
        "grad_floorfisher_wt2048_k12000_f0_c0p22_cap0p75",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p22_cap0p75",
        "bbh_grad_floorfisher_wt2048_k12000_f0_c0p22_cap0p75_raw_cot_fp32",
        "grad_fisher_zero_floor_all_k_cap0p75/frozen_harmbench/shard0-of-2/summary.json",
        "floorfisher_k12000_floor0p0_c0p22_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=12k, c=.48",
        "grad_floorfisher_wt2048_k12000_f0_c0p48",
        "math100_grad_floorfisher_wt2048_k12000_f0_c0p48_cap0p75",
        "bbh_grad_floorfisher_wt2048_k12000_f0_c0p48_cap0p75_raw_cot_fp32",
        "grad_floor_fisher_wikitext2048_finalists/frozen_harmbench/summary.json",
        "floorfisher_k12000_floor0p0_c0p48_cap0p75_damp1p0",
    ),
    FisherSource(
        "K=16k, c=.18",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k16000_c0p18_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k16000_c0p18_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/capability/fisher_zero_floor_math100_k16000_c0p18_cap0p75",
        "grad_fisher_zero_floor_all_k_cap0p75/frozen_harmbench/shard1-of-2/summary.json",
        "floorfisher_k16000_floor0p0_c0p18_cap0p75_damp1p0",
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


POINT_IDS = {
    **{(SN_TUNE, f"alpha={value}"): f"SN{index}" for index, value in enumerate((1, 4, 6, 8), 1)},
    **{
        (IA3_SFT, f"alpha={value}"): f"IA{index}"
        for index, value in enumerate((1, 1.5, 2, 2.5, 3, 3.5), 1)
    },
    **{
        (IA3_PATCH, f"K={value}"): f"P{index}"
        for index, value in enumerate((40000, 80000, 160000, 320000), 1)
    },
    (DIRECT_GRAD, "K=1k, s=1"): "D1",
    (DIRECT_GRAD, "K=2k, s=1"): "D2",
    (DIRECT_GRAD, "K=4k, s=1"): "D3",
    (DIRECT_GRAD, "K=4k, s=.75"): "D4",
    (FISHER_GRAD, "K=1k, c=.64"): "F1",
    (FISHER_GRAD, "K=2k, c=.64"): "F2",
    (FISHER_GRAD, "K=4k, c=.40"): "F3",
    (FISHER_GRAD, "K=6k, c=.52"): "F4",
    (FISHER_GRAD, "K=8k, c=.48"): "F5",
    (FISHER_GRAD, "K=12k, c=.22"): "F6",
    (FISHER_GRAD, "K=12k, c=.48"): "F7",
    (FISHER_GRAD, "K=16k, c=.18"): "F8",
}


def point_id(point: Point) -> str:
    return POINT_IDS[(point.method, point.setting)]


def point_key(benchmark: str) -> str:
    patch_values = "40k,80k" if benchmark == "bbh" else "40k,80k,160k,320k"
    return (
        "Point IDs — SN1–4: α=1,4,6,8  |  "
        "IA1–6: α=1,1.5,2,2.5,3,3.5  |  "
        f"P1–{2 if benchmark == 'bbh' else 4}: K={patch_values}\n"
        "D1–4: (K,s)=(1k,1),(2k,1),(4k,1),(4k,.75)\n"
        "F1–8: (K,c)=(1k,.64),(2k,.64),(4k,.40),(6k,.52),(8k,.48),"
        "(12k,.22),(12k,.48),(16k,.18)"
    )


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
        (DIRECT_GRAD, "K=1k, s=1"): (-48, 7),
        (DIRECT_GRAD, "K=2k, s=1"): (-58, 7),
        (DIRECT_GRAD, "K=4k, s=1"): (7, -17),
        (DIRECT_GRAD, "K=4k, s=.75"): (8, 15),
        (FISHER_GRAD, "K=1k, c=.64"): (8, 7),
        (FISHER_GRAD, "K=2k, c=.64"): (8, 7),
        (FISHER_GRAD, "K=4k, c=.40"): (-25, 7),
        (FISHER_GRAD, "K=6k, c=.52"): (-25, -18),
        (FISHER_GRAD, "K=8k, c=.48"): (-25, 7),
        (FISHER_GRAD, "K=12k, c=.22"): (-38, -18),
        (FISHER_GRAD, "K=12k, c=.48"): (8, -18),
        (FISHER_GRAD, "K=16k, c=.18"): (8, -18),
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
        (DIRECT_GRAD, "K=1k, s=1"): (-52, 7),
        (DIRECT_GRAD, "K=2k, s=1"): (7, 7),
        (DIRECT_GRAD, "K=4k, s=1"): (7, -16),
        (DIRECT_GRAD, "K=4k, s=.75"): (-88, 14),
        (FISHER_GRAD, "K=1k, c=.64"): (8, 7),
        (FISHER_GRAD, "K=2k, c=.64"): (8, 7),
        (FISHER_GRAD, "K=4k, c=.40"): (-25, 7),
        (FISHER_GRAD, "K=6k, c=.52"): (-28, -18),
        (FISHER_GRAD, "K=8k, c=.48"): (-25, 7),
        (FISHER_GRAD, "K=12k, c=.22"): (-25, -18),
        (FISHER_GRAD, "K=12k, c=.48"): (8, -18),
        (FISHER_GRAD, "K=16k, c=.18"): (8, 7),
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
        (DIRECT_GRAD, "K=1k, s=1"): (7, -17),
        (DIRECT_GRAD, "K=2k, s=1"): (-54, 7),
        (DIRECT_GRAD, "K=4k, s=1"): (7, -17),
        (DIRECT_GRAD, "K=4k, s=.75"): (-72, 14),
        (FISHER_GRAD, "K=1k, c=.64"): (8, 7),
        (FISHER_GRAD, "K=2k, c=.64"): (-25, 7),
        (FISHER_GRAD, "K=4k, c=.40"): (8, 7),
        (FISHER_GRAD, "K=6k, c=.52"): (-25, -18),
        (FISHER_GRAD, "K=8k, c=.48"): (-25, 7),
        (FISHER_GRAD, "K=12k, c=.22"): (-34, -18),
        (FISHER_GRAD, "K=12k, c=.48"): (8, -18),
        (FISHER_GRAD, "K=16k, c=.18"): (8, -18),
    },
}


def annotate(
    ax: plt.Axes, benchmark: str, point: Point, label: str | None = None
) -> None:
    del benchmark
    hollow = point.setting in {"K=4k, s=.75", "K=12k, c=.48"}
    text_color = (
        COLORS[point.method]
        if hollow
        else ("#5B3A00" if point.method == IA3_PATCH else "white")
    )
    shared = bool(label and "\n" in label)
    ax.scatter(
        [point.capability],
        [point.harmbench],
        s=205 if shared else 155,
        marker="o",
        facecolor="white" if hollow else COLORS[point.method],
        edgecolor=COLORS[point.method] if hollow else "white",
        linewidth=1.5 if hollow else 0.7,
        zorder=5,
    )
    ax.text(
        point.capability,
        point.harmbench,
        label or point_id(point),
        ha="center",
        va="center",
        color=text_color,
        fontsize=4.8 if label and "\n" in label else 6.2,
        fontweight="bold",
        linespacing=0.75,
        zorder=6,
    )


def plot_trajectory(
    ax: plt.Axes,
    benchmark: str,
    baseline: Point,
    points: list[Point],
    *,
    linestyle: str = "-",
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
        linewidth=2.15,
        linestyle=linestyle,
        zorder=2,
    )
    coincident: dict[tuple[float, float], list[Point]] = {}
    for point in points:
        coincident.setdefault((point.capability, point.harmbench), []).append(point)
    for shared in coincident.values():
        annotate(
            ax,
            benchmark,
            shared[0],
            label="\n".join(point_id(point) for point in shared),
        )


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

    fig, ax = plt.subplots(figsize=(11.4, 7.0), constrained_layout=True)
    ax.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(*config["xlim"])
    ax.set_ylim(-4, 71)
    ax.set_xlabel(str(config["xlabel"]))
    ax.set_ylabel("HarmBench attack success rate (%)  ↓")
    ax.set_title(
        f"{config['title']}\n"
        "All Grad results use first-cue-256; direct: solid s=1 K sweep, "
        "dashed fixed-K strength branches; diagonal Fisher: zero-floor K sweep, cap=.75"
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
        linewidth=1.7,
        linestyle="--",
        zorder=3,
    )
    annotate(ax, benchmark, direct_s075[0])
    fisher_c48 = next(point for point in fisher if point.setting == "K=12k, c=.48")
    fisher_c22 = next(point for point in fisher if point.setting == "K=12k, c=.22")
    fisher_k_sweep = [point for point in fisher if point is not fisher_c48]
    plot_trajectory(ax, benchmark, baseline, fisher_k_sweep)
    ax.plot(
        [fisher_c22.capability, fisher_c48.capability],
        [fisher_c22.harmbench, fisher_c48.harmbench],
        color=COLORS[FISHER_GRAD],
        linewidth=1.7,
        linestyle="--",
        zorder=3,
    )
    annotate(ax, benchmark, fisher_c48)

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
                DIRECT_GRAD,
                FISHER_GRAD,
            )
        ],
    ]
    ax.legend(
        handles=legend,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=4,
    )
    ax.text(
        0.5,
        -0.145,
        point_key(benchmark),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.1,
        linespacing=1.45,
        color="#333333",
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
