#!/usr/bin/env python3
"""Build compact Table 1 reproduction artifacts from completed run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "evaluation_outputs"
OUT = ROOT / "reproduction_summary"
VARIANTS = ("origin", "random_rate0p0005", "sn_rate0p0005")
SAFETY_VARIANTS = ("origin", "random", "sn")
LABELS = ("Origin", "Deact-R", "Deact-SN")
PAPER = {
    "HarmBehavior": (30.0, 31.0, 78.0),
    "AdvBehavior": (7.0, 13.0, 96.0),
    "MultiJail-En": (20.0, 21.6, 74.3),
    "MMLU": (65.3, 63.2, 62.7),
    "GSM8K": (75.9, 73.6, 72.4),
}


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--safety-directory",
        default="table1_safety_chat_single_bos",
        help="Directory below evaluation_outputs containing the safety runs.",
    )
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    parser.add_argument("--output-stem", default="table1_llama3")
    parser.add_argument(
        "--safety-variants",
        nargs=3,
        default=SAFETY_VARIANTS,
        metavar=("ORIGIN", "RANDOM", "SN"),
        help="Safety-run subdirectories corresponding to Origin, Deact-R, and Deact-SN.",
    )
    args = parser.parse_args()
    safety_paths = {
        "HarmBehavior": "harm_behavior_first_100",
        "AdvBehavior": "adv_behavior_harmbench_standard_200",
        "MultiJail-En": "multijail_en_315",
    }
    reproduced: dict[str, tuple[float, float, float]] = {}
    sample_counts: dict[str, int] = {}
    for label, directory in safety_paths.items():
        run_directories = [
            ROOT / args.safety_directory / directory / variant for variant in args.safety_variants
        ]
        for run_directory in run_directories:
            metadata = read(run_directory / "run.json")
            actual_prompt_format = metadata["generation"]["prompt_format"]
            if actual_prompt_format != args.prompt_format:
                raise ValueError(
                    f"{run_directory} uses {actual_prompt_format!r}, "
                    f"expected {args.prompt_format!r}"
                )
        summaries = [
            read(run_directory / "summary.json") for run_directory in run_directories
        ]
        reproduced[label] = tuple(float(summary["attack_success_rate"]) for summary in summaries)
        sample_counts[label] = int(summaries[0]["num_samples"])
    for task, label in (("mmlu", "MMLU"), ("gsm8k", "GSM8K")):
        summaries = [
            read(ROOT / "table1_capability" / task / variant / "summary.json")
            for variant in VARIANTS
        ]
        reproduced[label] = tuple(float(summary["accuracy"]) for summary in summaries)
        sample_counts[label] = int(summaries[0]["num_samples"])

    reproduced["Avg. Harmful"] = tuple(
        sum(reproduced[row][column] for row in safety_paths) / len(safety_paths)
        for column in range(3)
    )
    reproduced["Avg. Capability"] = tuple(
        (reproduced["MMLU"][column] + reproduced["GSM8K"][column]) / 2
        for column in range(3)
    )
    paper = dict(PAPER)
    paper["Avg. Harmful"] = (19.0, 21.9, 82.8)
    paper["Avg. Capability"] = (70.6, 68.4, 67.6)
    origin_metadata = read(
        ROOT
        / args.safety_directory
        / safety_paths["HarmBehavior"]
        / args.safety_variants[0]
        / "run.json"
    )
    origin_applied_neurons = int(
        origin_metadata["variant_settings"].get("applied_deactivated_neurons", 0)
    )
    artifact = {
        "model": "Meta-Llama-3-8B-Instruct",
        "deactivation_rate": 0.0005,
        "global_deactivated_neurons": 557,
        "safety_prompt_format": args.prompt_format,
        "seed": 112,
        "sample_counts": sample_counts,
        "variants": LABELS,
        "safety_variant_directories": list(args.safety_variants),
        "origin_applied_deactivated_neurons": origin_applied_neurons,
        "paper": {key: list(value) for key, value in paper.items()},
        "reproduced": {key: list(value) for key, value in reproduced.items()},
        "known_protocol_ambiguity": (
            "The paper does not identify the HarmBehavior row slice; this run uses the first 100 "
            "official AdvBench rows. AdvBehavior uses all 200 HarmBench standard behaviors."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{args.output_stem}.json"
    json_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    rows = (
        "HarmBehavior",
        "AdvBehavior",
        "MultiJail-En",
        "Avg. Harmful",
        "MMLU",
        "GSM8K",
        "Avg. Capability",
    )
    lines = [
        "# Llama-3 Table 1 reproduction",
        "",
        f"Safety evaluations use {args.prompt_format} prompts; deactivation rate is 0.0005 (557 neurons).",
        "",
        "| Dataset | Paper Origin | Paper Deact-R | Paper Deact-SN | Repro Origin | Repro Deact-R | Repro Deact-SN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = paper[row] + reproduced[row]
        lines.append(f"| {row} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "Harmfulness values are ASR (lower is safer); capability values are accuracy (higher is better).",
            "",
        ]
    )
    if args.safety_variants[0] != "origin":
        lines.extend(
            [
                (
                    f"Origin uses `{args.safety_variants[0]}` in the patched environment with "
                    f"{origin_applied_neurons} applied neurons, matching the deactivation variants' "
                    "software stack."
                ),
                "",
            ]
        )
    markdown_path = OUT / f"{args.output_stem}.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
