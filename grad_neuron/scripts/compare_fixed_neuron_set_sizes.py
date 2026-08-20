#!/usr/bin/env python3
"""Compare matched K=25 and K=50 fixed-neuron intervention sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_safety_capability_tradeoff import paired_effect


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k25-dir", type=Path,
        default=root / "results/tradeoff/fixed_top25_strength_sweep",
    )
    parser.add_argument(
        "--k50-dir", type=Path,
        default=root / "results/tradeoff/fixed_top50_strength_sweep",
    )
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument(
        "--output", type=Path,
        default=root / (
            "results/tradeoff/fixed_top50_strength_sweep/"
            "k25_vs_k50_paired_comparison.csv"
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safety_condition(directory: Path, strength: float) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(
        (row for row in read_jsonl(directory / "harmbehavior_first100/generations.jsonl")
         if float(row["epsilon"]) == strength),
        key=lambda row: int(row["id"]),
    )
    return (
        np.asarray([bool(row["baseline_jailbroken"]) for row in rows]),
        np.asarray([bool(row["jailbroken"]) for row in rows]),
    )


def capability_condition(directory: Path, strength: float) -> tuple[np.ndarray, np.ndarray]:
    tag = f"strength_{strength:g}".replace(".", "p")
    baseline = sorted(
        read_jsonl(directory / "gsm8k_first100/baseline_responses.jsonl"),
        key=lambda row: int(row["id"]),
    )
    condition = sorted(
        read_jsonl(directory / f"gsm8k_first100/{tag}_responses.jsonl"),
        key=lambda row: int(row["id"]),
    )
    if [row["id"] for row in baseline] != [row["id"] for row in condition]:
        raise ValueError("GSM8K IDs differ between baseline and condition")
    return (
        np.asarray([bool(row["correct"]) for row in baseline]),
        np.asarray([bool(row["correct"]) for row in condition]),
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for benchmark, loader in [
        ("HarmBehavior ASR", safety_condition),
        ("GSM8K accuracy", capability_condition),
    ]:
        for strength in args.strengths:
            baseline, k25 = loader(args.k25_dir, strength)
            baseline50, k50 = loader(args.k50_dir, strength)
            if not np.array_equal(baseline, baseline50):
                raise ValueError(f"{benchmark} baselines differ at strength {strength}")
            for reference_name, reference in [("baseline", baseline), ("K25", k25)]:
                effect = paired_effect(reference, k50, rng, args.bootstrap_samples)
                rows.append({
                    "benchmark": benchmark,
                    "strength": strength,
                    "comparison": f"K50_vs_{reference_name}",
                    "reference_rate_pct": 100 * reference.mean(),
                    "k50_rate_pct": 100 * k50.mean(),
                    **effect,
                })
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
