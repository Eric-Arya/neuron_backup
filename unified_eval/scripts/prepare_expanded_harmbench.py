#!/usr/bin/env python3
"""Convert the frozen unified HarmBench manifest for the expanded-K/V runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/"
    "table1_seed42_n200.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/workspace/xcy/dataset/projects/iclr_neuron_expanded_kv/harmbench/"
    "table1_seed42_n200_prepared.jsonl"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len(source) != 200 or len({row["id"] for row in source}) != 200:
        raise ValueError("Expected the frozen unique 200-row HarmBench manifest")
    converted = [
        {
            "id": index,
            "original_id": row["id"],
            "source_index": row["source_index"],
            "behavior_id": row["behavior_id"],
            "goal": row["prompt"],
            # HarmBench does not provide affirmative targets; retain a nonmatching sentinel.
            "target": "__NO_AFFIRMATIVE_TARGET_IN_FROZEN_HARMBENCH__",
        }
        for index, row in enumerate(source)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
        encoding="utf-8",
    )
    print(f"Wrote {len(converted)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
