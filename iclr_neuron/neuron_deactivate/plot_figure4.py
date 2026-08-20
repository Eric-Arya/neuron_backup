#!/usr/bin/env python3
"""Collect the Llama-3 layer-sweep summaries and render the Figure 4 reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(__file__).resolve().parent / "evaluation_outputs" / "figure4_chat_single_bos"
DEFAULT_FRONT = (0, 4, 8, 10, 12, 14, 16, 32)
DEFAULT_BACK = (0, 2, 4, 8, 16, 24, 28, 32)


def load_curve(
    root: Path,
    scope: str,
    cutoffs: tuple[int, ...],
    prompt_format: str,
) -> list[dict[str, object]]:
    curve = []
    for cutoff in cutoffs:
        run_dir = root / f"{scope}{cutoff}"
        summary_path = run_dir / "summary.json"
        metadata_path = run_dir / "run.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual_prompt_format = metadata["generation"]["prompt_format"]
        if actual_prompt_format != prompt_format:
            raise ValueError(
                f"{metadata_path} uses {actual_prompt_format!r}, expected {prompt_format!r}"
            )
        settings = metadata["variant_settings"]
        curve.append(
            {
                "layer_cutoff": cutoff,
                "attack_success_rate": summary["attack_success_rate"],
                "attack_success_count": summary["attack_success_count"],
                "num_samples": summary["num_samples"],
                "applied_deactivated_neurons": settings["applied_deactivated_neurons"],
                "selection_sha256": settings["selected_neurons_sha256"],
            }
        )
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    args = parser.parse_args()
    front = load_curve(args.root, "front", DEFAULT_FRONT, args.prompt_format)
    back = load_curve(args.root, "back", DEFAULT_BACK, args.prompt_format)
    hashes = {str(point["selection_sha256"]) for point in front + back}
    if len(hashes) != 1:
        raise ValueError(f"Figure 4 runs do not share one fixed neuron selection: {hashes}")

    artifact = {
        "model": "Meta-Llama-3-8B-Instruct",
        "prompt_format": args.prompt_format,
        "deactivation_rate": 0.0005,
        "selection_sha256": hashes.pop(),
        "front": front,
        "back": back,
    }
    data_path = args.root / "figure4_llama3.json"
    data_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, title, curve in zip(
        axes,
        ("Deactivate layers before cutoff", "Deactivate layers at/after cutoff"),
        (front, back),
    ):
        axis.plot(
            [point["layer_cutoff"] for point in curve],
            [point["attack_success_rate"] for point in curve],
            marker="o",
            linewidth=2,
            color="#2ca02c",
        )
        axis.set_title(title)
        axis.set_xlabel("Layer cutoff")
        axis.set_xlim(-0.8, 32.8)
        axis.set_ylim(0, 100)
        axis.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Attack Success Rate (%)")
    fig.suptitle(
        f"Figure 4 reproduction: Llama-3-8B-Instruct, {args.prompt_format}, rate 0.0005"
    )
    fig.tight_layout()
    image_path = args.root / "figure4_llama3.png"
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    print(json.dumps({"data": str(data_path), "figure": str(image_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
