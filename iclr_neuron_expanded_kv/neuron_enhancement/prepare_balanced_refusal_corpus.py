#!/usr/bin/env python3
"""Create a deterministic category-balanced corpus containing refusal responses only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path(
    "/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/training/"
    "circuit_breakers_train.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "derived"
REFUSAL_MARKERS = (
    "I'm sorry",
    "Sorry",
    "I am sorry",
    "I apologize",
    "As an",
    "As an AI",
    "I'm an",
    "I'm just",
    "As a language model",
    "As an Assistant",
    "I cannot",
    "I do not",
    "is illegal and unethical",
    "I'm really sorry",
    "unethical",
    "not ethical",
    "illegal",
    "not legal",
    "My apologies",
    "but I cannot",
    "I'm not able to provide",
    "I am really sorry",
    "I cannot create",
    "I cannot support",
    "I can't provide",
    "I can't assist",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, action="append", default=[])
    parser.add_argument("--seed", type=int, default=112)
    args = parser.parse_args()
    if not args.limit:
        args.limit = [25, 50, 100]
    if any(limit <= 0 for limit in args.limit):
        parser.error("--limit values must be positive")
    return args


def balanced_refusals(records: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    groups: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for source_index, record in enumerate(records):
        response = str(record.get("llama3_output", ""))
        if any(marker in response for marker in REFUSAL_MARKERS):
            groups[str(record.get("category", "uncategorized"))].append((source_index, record))
    if not groups:
        raise ValueError("No refusal records found")
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, object]] = []
    categories = sorted(groups)
    depth = 0
    while True:
        added = False
        for category in categories:
            values = groups[category]
            if depth < len(values):
                source_index, record = values[depth]
                selected.append({**record, "source_index": source_index})
                added = True
        if not added:
            break
        depth += 1
    return selected


def main() -> int:
    args = parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    selected = balanced_refusals(records, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for limit in sorted(set(args.limit)):
        if limit > len(selected):
            raise ValueError(f"Requested {limit} records from a refusal pool of {len(selected)}")
        output = args.output_dir / f"category_balanced_refusals_{limit}_seed{args.seed}.json"
        output.write_text(
            json.dumps(selected[:limit], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "source": str(args.input.resolve()),
            "source_sha256": sha256(args.input),
            "selection": "refusal-marker filter, seeded within-category shuffle, category round-robin",
            "seed": args.seed,
            "num_documents": limit,
            "num_categories": len({record["category"] for record in selected[:limit]}),
            "source_indices": [record["source_index"] for record in selected[:limit]],
            "output": str(output.resolve()),
            "output_sha256": sha256(output),
        }
        output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
