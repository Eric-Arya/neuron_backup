from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def normalized_prompt(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a train- and test-disjoint held-out SN selection manifest."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--harmbench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) <= args.train_count:
        raise ValueError("Source must be a JSON list longer than the training prefix")
    training_prompts = {
        normalized_prompt(str(row["prompt"])) for row in records[: args.train_count]
    }
    test_prompts = {
        normalized_prompt(str(row["prompt"])) for row in read_jsonl(args.harmbench)
    }
    eligible = []
    seen = set(training_prompts)
    excluded_training_overlap = 0
    excluded_test_overlap = 0
    excluded_duplicate = 0
    for source_index, row in enumerate(records[args.train_count :], args.train_count):
        prompt = str(row.get("prompt", "")).strip()
        normalized = normalized_prompt(prompt)
        if not normalized:
            continue
        if normalized in training_prompts:
            excluded_training_overlap += 1
            continue
        if normalized in test_prompts:
            excluded_test_overlap += 1
            continue
        if normalized in seen:
            excluded_duplicate += 1
            continue
        seen.add(normalized)
        eligible.append({**row, "source_index": source_index})
    if len(eligible) < args.count:
        raise ValueError(f"Only {len(eligible)} eligible records for requested {args.count}")

    selection = random.Random(args.seed).sample(eligible, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selection:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "training_prefix_records": args.train_count,
        "candidate_records": len(records) - args.train_count,
        "harmbench": str(args.harmbench.resolve()),
        "harmbench_sha256": sha256(args.harmbench),
        "excluded_training_prompt_overlaps": excluded_training_overlap,
        "excluded_harmbench_prompt_overlaps": excluded_test_overlap,
        "excluded_heldout_duplicates": excluded_duplicate,
        "eligible_unique_records": len(eligible),
        "selection_seed": args.seed,
        "selected_records": len(selection),
        "selected_source_indices": [row["source_index"] for row in selection],
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "prompt_serialization": "raw source prompt plus '.' generation boundary",
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
