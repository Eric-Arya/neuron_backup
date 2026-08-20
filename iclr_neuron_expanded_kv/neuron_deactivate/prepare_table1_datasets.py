#!/usr/bin/env python3
"""Prepare the three harmfulness datasets used by the Llama-3 Table 1 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path("/workspace/xcy/dataset")
DEFAULT_SOURCE_DIR = DATASET_ROOT / "table1_sources"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_dataset(
    output_dir: Path,
    name: str,
    records: list[dict[str, object]],
    source: Path,
    selection: str,
) -> dict[str, object]:
    output = output_dir / f"{name}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "name": name,
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "selection": selection,
        "num_samples": len(records),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    advbench_path = DATASET_ROOT / "harmful_behaviors.csv"
    harmbench_path = args.source_dir / "harmbench_behaviors_text_all.csv"
    multijail_path = args.source_dir / "MultiJail.csv"

    advbench = read_csv(advbench_path)
    if len(advbench) < 100 or set(advbench[0]) != {"goal", "target"}:
        raise ValueError(f"Unexpected AdvBench schema or size: {advbench_path}")
    harm_behavior = [
        {
            "id": index,
            "source_index": index,
            "goal": row["goal"],
            "target": row["target"],
        }
        for index, row in enumerate(advbench[:100])
    ]

    harmbench = read_csv(harmbench_path)
    standard = [row for row in harmbench if row["FunctionalCategory"] == "standard"]
    if len(standard) != 200:
        raise ValueError(f"Expected 200 standard HarmBench behaviors, found {len(standard)}")
    adv_behavior = [
        {
            "id": index,
            "source_index": index,
            "source_id": row["BehaviorID"],
            "goal": row["Behavior"],
            "target": "",
        }
        for index, row in enumerate(standard)
    ]

    multijail = read_csv(multijail_path)
    if len(multijail) != 315 or "en" not in multijail[0]:
        raise ValueError(f"Unexpected MultiJail schema or size: {multijail_path}")
    multijail_en = [
        {
            "id": index,
            "source_index": int(row["id"]),
            "source": row["source"],
            "goal": row["en"],
            "target": "",
        }
        for index, row in enumerate(multijail)
    ]

    summaries = [
        write_dataset(
            args.output_dir,
            "harm_behavior_first_100",
            harm_behavior,
            advbench_path,
            "first 100 rows, matching the integer-valued 100-example paper evaluation",
        ),
        write_dataset(
            args.output_dir,
            "adv_behavior_harmbench_standard_200",
            adv_behavior,
            harmbench_path,
            "all 200 rows with FunctionalCategory=standard",
        ),
        write_dataset(
            args.output_dir,
            "multijail_en_315",
            multijail_en,
            multijail_path,
            "English column for all 315 rows",
        ),
    ]
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
