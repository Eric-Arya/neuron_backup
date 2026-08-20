#!/usr/bin/env python3
"""Overlay a linear random-neuron estimate on the existing Llama2 safety curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAFETY = ROOT / "results/figure2_left_llama2_safety_neurons/curve.json"
DEFAULT_RANDOM = ROOT / "results/figure2_left_llama2_random_seed42_8000_20000/curve.json"
DEFAULT_OUTPUT = (
    ROOT
    / "results/figure2_left_llama2_safety_neurons/"
    / "figure2_left_llama2_safety_neurons.png"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety-curve", type=Path, default=DEFAULT_SAFETY)
    parser.add_argument("--random-points", type=Path, default=DEFAULT_RANDOM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    safety = json.loads(args.safety_curve.read_text(encoding="utf-8"))["curve"]
    random = json.loads(args.random_points.read_text(encoding="utf-8"))["curve"]
    if [point["top_k"] for point in random] != [8000, 20000]:
        raise ValueError("Expected random-neuron anchors at 8,000 and 20,000")

    safety_x = [point["neurons_percent"] for point in safety]
    safety_y = [point["causal_effect"] for point in safety]
    # Zero patched neurons is exactly Base, so its normalized causal effect is 0.
    # Connecting these three anchors gives a transparent piecewise-linear estimate;
    # no unmeasured model evaluations are implied.
    random_x = [0.0, *[point["neurons_percent"] for point in random]]
    random_y = [0.0, *[point["causal_effect"] for point in random]]

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    axis.plot(
        safety_x,
        safety_y,
        color="#9467bd",
        linewidth=2,
        marker="o",
        markersize=3,
        label="Llama2 Safety",
    )
    axis.plot(
        random_x,
        random_y,
        color="#9467bd",
        linestyle="--",
        linewidth=2,
        marker="s",
        markersize=3,
        label="Llama2 Random (estimated)",
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("Neurons (%)")
    axis.set_ylabel("Causal Effect")
    axis.set_title("Patch Base with DPO")
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(frameon=False)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
