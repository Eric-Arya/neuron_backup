#!/usr/bin/env python3
"""Reconstruct the exact seeded prompt sample used by neuron detection."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(
            "/workspace/xcy/dataset/projects/iclr_neuron/"
            "safety_neuron/processed/zou_train.txt"
        ),
    )
    parser.add_argument(
        "--training-json",
        type=Path,
        default=Path(
            "/workspace/xcy/dataset/projects/iclr_neuron/"
            "safety_neuron/training/circuit_breakers_train.json"
        ),
    )
    parser.add_argument("--output-csv", type=Path,
                        default=root / "data/processed/zou_detector_sample_seed112_n200.csv")
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_csv}; pass --overwrite")
    lines = [line.strip() for line in args.corpus.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not 0 < args.sample_count <= len(lines):
        raise ValueError("sample-count must be between 1 and corpus size")

    training = json.loads(args.training_json.read_text(encoding="utf-8"))
    training_prompts = [str(row["prompt"]).strip() for row in training]
    if training_prompts != lines:
        raise ValueError("Corpus does not exactly match training JSON prompt order")

    # This is equivalent to random.Random(seed).sample(lines, sample_count), while
    # retaining the original line index for provenance.
    rng = random.Random(args.seed)
    selected_indices = rng.sample(range(len(lines)), args.sample_count)
    expected = random.Random(args.seed).sample(lines, args.sample_count)
    if [lines[index] for index in selected_indices] != expected:
        raise RuntimeError("Index sampling did not reproduce detector text sampling")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("detector_sample_index", "corpus_index", "category", "prompt")
        )
        writer.writeheader()
        for sample_index, corpus_index in enumerate(selected_indices):
            writer.writerow({
                "detector_sample_index": sample_index,
                "corpus_index": corpus_index,
                "category": training[corpus_index]["category"],
                "prompt": lines[corpus_index],
            })
    print(f"saved={args.output_csv} rows={len(selected_indices)} seed={args.seed}")


if __name__ == "__main__":
    main()
