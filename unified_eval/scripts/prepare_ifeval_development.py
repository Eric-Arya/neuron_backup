#!/usr/bin/env python3
"""Create an IFEval development subset disjoint from the frozen 200 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/workspace/xcy/dataset/ifeval/instruction_following_eval/data/input_data.jsonl"
)
DEFAULT_FROZEN = Path(
    "/workspace/xcy/dataset/ifeval/subsets/ifeval_seed112_n200_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "/workspace/xcy/dataset/ifeval/subsets/ifeval_development_seed314_n64.jsonl"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--frozen-manifest", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=314)
    args = parser.parse_args()

    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    frozen_keys = {int(key) for key in frozen["selected_keys"]}
    pool = [row for row in read_jsonl(args.source) if int(row["key"]) not in frozen_keys]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    all_types = {item for row in pool for item in row["instruction_id_list"]}
    uncovered = set(all_types)
    selected: list[dict] = []
    remaining = list(pool)
    while uncovered and len(selected) < args.count:
        best_gain = max(
            len(set(row["instruction_id_list"]) & uncovered) for row in remaining
        )
        candidates = [
            row
            for row in remaining
            if len(set(row["instruction_id_list"]) & uncovered) == best_gain
        ]
        chosen = rng.choice(candidates)
        selected.append(chosen)
        remaining.remove(chosen)
        uncovered.difference_update(chosen["instruction_id_list"])
    if uncovered:
        raise RuntimeError(f"Could not cover instruction types: {sorted(uncovered)}")
    selected.extend(remaining[: args.count - len(selected)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(args.output)
    manifest = {
        "benchmark": "IFEval",
        "count": len(selected),
        "seed": args.seed,
        "selection": "coverage-first seeded sample from rows excluded by frozen seed-112 n=200",
        "frozen_manifest": str(args.frozen_manifest.resolve()),
        "selected_keys": [int(row["key"]) for row in selected],
        "instruction_type_count": len(
            {item for row in selected for item in row["instruction_id_list"]}
        ),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
