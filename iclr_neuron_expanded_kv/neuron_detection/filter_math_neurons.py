#!/usr/bin/env python3
"""Remove neurons consistently active on a tiny math corpus from safety candidates."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


STRUCTURES = ("fwd_up", "fwd_down", "q", "k", "v")


def load(path: Path) -> list[dict[int, set[int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(STRUCTURES):
        raise ValueError(f"Expected five dictionaries in {path}")
    result = []
    for line in lines:
        parsed = ast.literal_eval(line)
        result.append({int(layer): set(indices) for layer, indices in parsed.items()})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harmful", type=Path)
    parser.add_argument("math", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    harmful = load(args.harmful)
    math = load(args.math)
    filtered = []
    for structure, harmful_layers, math_layers in zip(STRUCTURES, harmful, math):
        if set(harmful_layers) != set(math_layers):
            raise ValueError(f"Layer mismatch for {structure}")
        filtered.append(
            {
                layer: harmful_layers[layer] - math_layers[layer]
                for layer in sorted(harmful_layers)
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for layers in filtered:
            serializable = {layer: sorted(indices) for layer, indices in layers.items()}
            handle.write(repr(serializable) + "\n")

    counts = [sum(len(indices) for indices in layers.values()) for layers in filtered]
    print(f"Wrote {args.output}: counts={counts}, total={sum(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
