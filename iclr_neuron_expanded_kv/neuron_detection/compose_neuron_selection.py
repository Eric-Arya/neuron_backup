#!/usr/bin/env python3
"""Sample exact per-structure counts into a five-line neuron selection file.

Example:
  python compose_neuron_selection.py detector.txt output.txt \
    --count v=900 --count q=8 --seed 112
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from pathlib import Path


STRUCTURES = ("fwd_up", "fwd_down", "q", "k", "v")


def parse_count(value: str) -> tuple[str, int]:
    try:
        structure, raw_count = value.split("=", 1)
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("count must have the form STRUCTURE=N") from exc
    if structure not in STRUCTURES or count <= 0:
        raise argparse.ArgumentTypeError(
            f"structure must be one of {STRUCTURES} and N must be positive"
        )
    return structure, count


def load_neurons(path: Path) -> list[dict[int, set[int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(STRUCTURES):
        raise ValueError(f"expected five dictionaries in {path}, found {len(lines)}")
    neurons = []
    for structure, line in zip(STRUCTURES, lines):
        value = ast.literal_eval(line)
        if not isinstance(value, dict):
            raise ValueError(f"{structure} is not a dictionary")
        neurons.append({int(layer): set(indices) for layer, indices in value.items()})
    return neurons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", action="append", type=parse_count, required=True)
    parser.add_argument("--seed", type=int, default=112)
    args = parser.parse_args()

    requested = dict(args.count)
    if len(requested) != len(args.count):
        parser.error("pass each structure at most once")
    source = load_neurons(args.input)
    layers = sorted(set().union(*(values.keys() for values in source)))
    selected = [{layer: set() for layer in layers} for _ in STRUCTURES]

    for position, structure in enumerate(STRUCTURES):
        if structure not in requested:
            continue
        pool = [
            (layer, index)
            for layer in sorted(source[position])
            for index in sorted(source[position][layer])
        ]
        count = requested[structure]
        if count > len(pool):
            parser.error(f"requested {count} {structure} neurons from a pool of {len(pool)}")
        # An independent seeded stream keeps a structure's selection fixed when another
        # structure is added or removed from an experimental configuration.
        for layer, index in random.Random(args.seed).sample(pool, count):
            selected[position][layer].add(index)

    payload = "\n".join(repr(values) for values in selected) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    result = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "selected_by_structure": {
            structure: sum(len(indices) for indices in values.values())
            for structure, values in zip(STRUCTURES, selected)
        },
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
