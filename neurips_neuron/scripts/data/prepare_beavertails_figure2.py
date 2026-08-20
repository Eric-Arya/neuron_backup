#!/usr/bin/env python3
"""Download and prepare the BeaverTails evaluation prompts used by Figure 2.

The released evaluation samples ``n`` prompts from the final ``3 * n`` rows
with Python's RNG after seeding it.  Defaults below reproduce its seed-42,
200-prompt manifest while retaining a prompt-only file for the complete test
split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Sequence


DEFAULT_URL = (
    "https://huggingface.co/datasets/PKU-Alignment/BeaverTails/resolve/main/"
    "round0/330k/test.jsonl.xz?download=true"
)
DEFAULT_RAW_XZ = Path(
    "/workspace/xcy/dataset/shared/beavertails/raw/round0/330k/test.jsonl.xz"
)
DEFAULT_RAW_JSONL = Path(
    "/workspace/xcy/dataset/shared/beavertails/raw/round0/330k/test.jsonl"
)
DEFAULT_PROCESSED = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/beavertails/processed/"
    "prompts_test_all.jsonl"
)
DEFAULT_SPLIT = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/beavertails/splits/"
    "figure2_seed42_n200.jsonl"
)
DEFAULT_PROVENANCE = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/beavertails/provenance/"
    "figure2_seed42_n200.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, force: bool = False) -> bool:
    """Download atomically, returning whether network transfer occurred."""
    if destination.is_file() and not force:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "neurips-neuron-reproduction/1.0"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        # Opening the stream catches HTML/error pages and truncated downloads.
        with lzma.open(temporary, "rb") as handle:
            handle.read(1)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


def atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def decompress_and_extract(source: Path, raw_output: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Decompress the raw split and return prompt-only records plus label counts."""
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_output.with_suffix(raw_output.suffix + ".tmp")
    records: list[dict[str, Any]] = []
    safe_count = 0
    unsafe_count = 0
    try:
        with lzma.open(source, "rt", encoding="utf-8") as compressed, temporary.open(
            "w", encoding="utf-8"
        ) as raw_handle:
            for source_index, line in enumerate(compressed):
                raw_handle.write(line)
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at compressed row {source_index + 1}") from error
                prompt = example.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Missing prompt at source row {source_index}")
                is_safe = example.get("is_safe")
                if is_safe is True:
                    safe_count += 1
                elif is_safe is False:
                    unsafe_count += 1
                records.append(
                    {
                        "dataset": "beavertails",
                        "id": f"beavertails_{source_index}",
                        "source_index": source_index,
                        "prompt": prompt,
                    }
                )
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        temporary.replace(raw_output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return records, {"safe": safe_count, "unsafe": unsafe_count}


def released_sample(
    records: Sequence[dict[str, Any]], sample_size: int, seed: int, window_multiplier: int
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample-size must be positive")
    if window_multiplier <= 0:
        raise ValueError("window-multiplier must be positive")
    window_size = sample_size * window_multiplier
    if len(records) < window_size:
        raise ValueError(f"Need at least {window_size} rows, found {len(records)}")
    # This is exactly random.sample(prompts[-3*num_samples:], num_samples)
    # from src/eval/arena/run_eval.py after seed_torch(42).
    return random.Random(seed).sample(list(records[-window_size:]), sample_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--raw-xz", type=Path, default=DEFAULT_RAW_XZ)
    parser.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW_JSONL)
    parser.add_argument("--processed-output", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--split-output", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--provenance-output", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-multiplier", type=int, default=3)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    downloaded = download_file(args.url, args.raw_xz, force=args.force_download)
    records, label_counts = decompress_and_extract(args.raw_xz, args.raw_jsonl)
    selected = released_sample(records, args.sample_size, args.seed, args.window_multiplier)
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Generated prompt IDs are not unique")
    if len({record["id"] for record in selected}) != len(selected):
        raise ValueError("Sampled prompt IDs are not unique")

    atomic_write_jsonl(args.processed_output, records)
    atomic_write_jsonl(args.split_output, selected)
    provenance = {
        "dataset": "PKU-Alignment/BeaverTails",
        "source_url": args.url,
        "source_split": "round0/330k/test",
        "downloaded_this_run": downloaded,
        "raw_xz": str(args.raw_xz.resolve()),
        "raw_xz_sha256": sha256_file(args.raw_xz),
        "raw_jsonl": str(args.raw_jsonl.resolve()),
        "raw_jsonl_sha256": sha256_file(args.raw_jsonl),
        "raw_rows": len(records),
        "raw_label_counts": label_counts,
        "processed_output": str(args.processed_output.resolve()),
        "processed_sha256": sha256_file(args.processed_output),
        "processed_rows": len(records),
        "split_output": str(args.split_output.resolve()),
        "split_sha256": sha256_file(args.split_output),
        "split_rows": len(selected),
        "seed": args.seed,
        "candidate_window": {
            "selection": "last rows before random.sample",
            "multiplier": args.window_multiplier,
            "rows": args.sample_size * args.window_multiplier,
            "first_source_index": len(records) - args.sample_size * args.window_multiplier,
            "last_source_index": len(records) - 1,
        },
        "prompt_only": True,
    }
    atomic_write_json(args.provenance_output, provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
