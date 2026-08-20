from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import DatasetDict, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unified_eval.common import atomic_write_json, sha256_file  # noqa: E402


DEFAULT_SOURCE = Path("/workspace/xcy/dataset/math500")
DEFAULT_OUTPUT = DEFAULT_SOURCE / "subsets/math500_l1_l3_seed112_n50"


def proportional_allocation(counts: dict[int, int], total: int) -> dict[int, int]:
    available = sum(counts.values())
    if total <= 0 or total > available:
        raise ValueError(f"Requested {total} rows from an eligible pool of {available}")
    ideals = {level: total * count / available for level, count in counts.items()}
    allocation = {level: math.floor(value) for level, value in ideals.items()}
    remainder = total - sum(allocation.values())
    order = sorted(counts, key=lambda level: (-(ideals[level] % 1), level))
    for level in order[:remainder]:
        allocation[level] += 1
    return allocation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a level-stratified frozen subset of MATH-500."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--base-subset",
        type=Path,
        help="Existing frozen subset whose rows must all remain in the enlarged subset.",
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=112)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing subset: {args.output}")
    levels = sorted(set(args.levels))
    if levels != args.levels or not levels:
        raise ValueError("Levels must be a non-empty sorted list without duplicates")

    source_manifest_path = args.source / "SOURCE.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source = load_from_disk(str(args.source))["test"]
    source_counts = Counter(int(level) for level in source["level"] if level in levels)
    if set(source_counts) != set(levels):
        raise ValueError(f"Requested levels are not all present: {source_counts}")
    allocation = proportional_allocation(dict(source_counts), args.count)

    base_indices_by_level: dict[int, list[int]] = {level: [] for level in levels}
    base_manifest_path = None
    if args.base_subset is not None:
        base_manifest_path = args.base_subset / "SOURCE.json"
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        base_selection = base_manifest.get("selection", {})
        if base_selection.get("seed") != args.seed:
            raise ValueError("Base subset seed does not match the requested seed")
        if base_selection.get("levels") != levels:
            raise ValueError("Base subset levels do not match the requested levels")
        recorded = base_selection.get("selected_source_indices_by_level", {})
        for level in levels:
            base_indices_by_level[level] = [
                int(index) for index in recorded[str(level)]
            ]

    selected_by_level: dict[int, list[int]] = {}
    for level in levels:
        candidates = [
            index for index, value in enumerate(source["level"]) if int(value) == level
        ]
        existing = base_indices_by_level[level]
        if len(existing) > allocation[level] or not set(existing).issubset(candidates):
            raise ValueError(f"Base subset is incompatible with level {level}")
        remaining = sorted(set(candidates) - set(existing))
        additional_count = allocation[level] - len(existing)
        rng = random.Random(
            f"math500-subset:{args.seed}:level:{level}:"
            f"extend:{len(existing)}-to-{allocation[level]}"
        )
        selected_by_level[level] = sorted(
            existing + rng.sample(remaining, additional_count)
        )
    selected_indices = sorted(
        index for indices in selected_by_level.values() for index in indices
    )
    subset = source.select(selected_indices).add_column(
        "source_index", selected_indices
    )
    if len(subset) != args.count or len(set(subset["unique_id"])) != args.count:
        raise AssertionError("Subset count or unique IDs are invalid")
    actual_counts = Counter(int(level) for level in subset["level"])
    if dict(actual_counts) != allocation:
        raise AssertionError(
            f"Subset allocation mismatch: {actual_counts} != {allocation}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"test": subset}).save_to_disk(str(args.output))
    arrow_files = sorted((args.output / "test").glob("*.arrow"))
    if len(arrow_files) != 1:
        raise AssertionError(f"Expected one subset Arrow file, found {arrow_files}")

    manifest = {
        "dataset": {
            "arrow_sha256": sha256_file(arrow_files[0]),
            "config": source_manifest["dataset"]["config"],
            "huggingface_revision": source_manifest["dataset"]["huggingface_revision"],
            "name": source_manifest["dataset"]["name"],
            "rows": args.count,
            "split": "test",
        },
        "selection": {
            "eligible_count": sum(source_counts.values()),
            "levels": levels,
            "method": "seeded within-level sampling with largest-remainder allocation",
            "samples_per_level": {str(level): allocation[level] for level in levels},
            "seed": args.seed,
            "selected_source_indices_by_level": {
                str(level): selected_by_level[level] for level in levels
            },
            "source_count_per_level": {
                str(level): source_counts[level] for level in levels
            },
        },
        "source": {
            "arrow_sha256": source_manifest["dataset"]["arrow_sha256"],
            "manifest_path": str(source_manifest_path.resolve()),
            "manifest_sha256": sha256_file(source_manifest_path),
            "path": str(args.source.resolve()),
        },
        "evaluator": source_manifest["evaluator"],
        "evaluator_dependencies": source_manifest["evaluator_dependencies"],
    }
    if args.base_subset is not None:
        manifest["selection"]["base_subset"] = {
            "manifest_path": str(base_manifest_path.resolve()),
            "manifest_sha256": sha256_file(base_manifest_path),
            "path": str(args.base_subset.resolve()),
            "rows": sum(len(indices) for indices in base_indices_by_level.values()),
        }
    manifest_path = args.output / "SOURCE.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
