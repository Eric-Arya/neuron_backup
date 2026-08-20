from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unified_eval.common import atomic_write_json, atomic_write_jsonl, sha256_file  # noqa: E402


DEFAULT_ROOT = Path("/workspace/xcy/dataset/big_bench_hard")
DEFAULT_OUTPUT = DEFAULT_ROOT / "subsets/bbh_seed112_n200.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a task-stratified frozen subset from the official BBH release."
    )
    parser.add_argument("--bbh-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=112)
    return parser


def repository_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    args = build_parser().parse_args()
    task_paths = sorted((args.bbh_root / "bbh").glob("*.json"))
    if not task_paths:
        raise FileNotFoundError(f"No BBH task files found under {args.bbh_root / 'bbh'}")
    if args.count < len(task_paths):
        raise ValueError("Count must include at least one example from every BBH task")

    base_count, remainder = divmod(args.count, len(task_paths))
    allocation_rng = random.Random(args.seed)
    extra_tasks = set(allocation_rng.sample([path.stem for path in task_paths], remainder))

    rows = []
    source_files = {}
    for path in task_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = payload.get("examples")
        if not isinstance(examples, list):
            raise ValueError(f"Missing examples list in {path}")
        selected_count = base_count + int(path.stem in extra_tasks)
        if len(examples) < selected_count:
            raise ValueError(f"Task {path.stem} has only {len(examples)} examples")
        task_seed = f"bbh-subset:{args.seed}:{path.stem}"
        indices = sorted(random.Random(task_seed).sample(range(len(examples)), selected_count))
        for source_index in indices:
            example = examples[source_index]
            rows.append(
                {
                    "id": f"{path.stem}:{source_index}",
                    "task": path.stem,
                    "source_index": source_index,
                    "input": example["input"],
                    "target": example["target"],
                }
            )
        source_files[path.name] = {
            "count": len(examples),
            "sha256": sha256_file(path),
        }

    if len(rows) != args.count or len({row["id"] for row in rows}) != args.count:
        raise AssertionError("Subset count or IDs are invalid")
    atomic_write_jsonl(args.output, rows)
    counts = Counter(row["task"] for row in rows)
    manifest = {
        "benchmark": "BIG-Bench Hard",
        "selection": "seeded task-stratified sample over all 27 released JSON task variants",
        "seed": args.seed,
        "count": args.count,
        "task_count": len(task_paths),
        "samples_per_task": dict(sorted(counts.items())),
        "source_root": str(args.bbh_root.resolve()),
        "source_revision": repository_revision(args.bbh_root),
        "source_files": source_files,
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_name(f"{args.output.stem}_manifest.json")
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
