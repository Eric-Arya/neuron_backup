#!/usr/bin/env python3
"""Prepare the frozen HarmBench prompt files for the NeurIPS Table 1 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/workspace/xcy/dataset/shared/harmbench/raw/behaviors_text_all.csv"
)
DEFAULT_ALL_OUTPUT = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/harmbench/processed/"
    "prompts_all_400.jsonl"
)
DEFAULT_SPLIT_OUTPUT = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/"
    "table1_seed42_n200.jsonl"
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--all-output", type=Path, default=DEFAULT_ALL_OUTPUT)
    parser.add_argument("--split-output", type=Path, default=DEFAULT_SPLIT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()

    with args.source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400:
        raise ValueError(f"Expected 400 HarmBench rows, found {len(rows)} in {args.source}")
    if not 0 < args.sample_size <= len(rows):
        raise ValueError("sample-size must be between 1 and the number of source rows")
    required_columns = {"BehaviorID", "Behavior"}
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        raise ValueError(f"Missing HarmBench columns: {sorted(missing_columns)}")

    records = [
        {
            "dataset": "harmbench",
            "id": f"harmbench_{index}",
            "source_index": index,
            "behavior_id": row["BehaviorID"],
            "prompt": row["Behavior"],
        }
        for index, row in enumerate(rows)
    ]
    selected_indices = random.Random(args.seed).sample(range(len(records)), args.sample_size)
    selected = [records[index] for index in selected_indices]

    write_jsonl(args.all_output, records)
    write_jsonl(args.split_output, selected)
    print(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "source_sha256": sha256_file(args.source),
                "all_output": str(args.all_output.resolve()),
                "all_count": len(records),
                "all_sha256": sha256_file(args.all_output),
                "split_output": str(args.split_output.resolve()),
                "split_count": len(selected),
                "split_sha256": sha256_file(args.split_output),
                "seed": args.seed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
