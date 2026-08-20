#!/usr/bin/env python3
"""Create a reproducible, subject-balanced MMLU evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import DatasetDict, load_from_disk


DEFAULT_SOURCE = Path("/workspace/xcy/dataset/shared/mmlu/all")
DEFAULT_OUTPUT = Path("/workspace/xcy/dataset/mmlu_balanced_5_per_subject/mmlu/all")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-subject", type=int, default=5)
    parser.add_argument("--seed", type=int, default=112)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples_per_subject <= 0:
        raise SystemExit("--samples-per-subject must be positive")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    source = load_from_disk(str(args.source))
    test = source["test"]
    indices_by_subject: dict[str, list[int]] = defaultdict(list)
    for index, subject in enumerate(test["subject"]):
        indices_by_subject[str(subject)].append(index)

    rng = random.Random(args.seed)
    selected: list[int] = []
    selected_by_subject: dict[str, list[int]] = {}
    for subject in sorted(indices_by_subject):
        candidates = indices_by_subject[subject]
        if len(candidates) < args.samples_per_subject:
            raise ValueError(
                f"Subject {subject} has only {len(candidates)} test rows; "
                f"requested {args.samples_per_subject}"
            )
        chosen = sorted(rng.sample(candidates, args.samples_per_subject))
        selected.extend(chosen)
        selected_by_subject[subject] = chosen

    # Keep all 5 official dev examples per subject as the standard MMLU demonstrations.
    subset = DatasetDict(dev=source["dev"], test=test.select(selected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(args.output))

    source_files = sorted(args.source.glob("**/*.arrow"))
    manifest = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "selection": "Python random.Random(seed).sample independently within each subject",
        "samples_per_subject": args.samples_per_subject,
        "num_subjects": len(selected_by_subject),
        "num_test_samples": len(selected),
        "source_arrow_sha256": {
            str(path.relative_to(args.source)): file_sha256(path) for path in source_files
        },
        "source_test_indices_by_subject": selected_by_subject,
    }
    manifest_path = args.output.parent / "subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "source_test_indices_by_subject": "see manifest"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
