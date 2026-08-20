#!/usr/bin/env python3
"""Keep selected structures from a five-dictionary safety-neuron file."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


STRUCTURES = ("fwd_up", "fwd_down", "q", "k", "v")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--structures", nargs="+", choices=STRUCTURES, required=True)
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(STRUCTURES):
        raise ValueError(f"Expected five dictionaries in {args.input}")
    keep = set(args.structures)
    output = []
    counts = []
    for structure, line in zip(STRUCTURES, lines):
        layers = ast.literal_eval(line)
        normalized = {
            int(layer): sorted(set(indices)) if structure in keep else []
            for layer, indices in sorted(layers.items())
        }
        output.append(normalized)
        counts.append(sum(len(indices) for indices in normalized.values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for layers in output:
            handle.write(repr(layers) + "\n")
    print(f"Wrote {args.output}: counts={counts}, total={sum(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
