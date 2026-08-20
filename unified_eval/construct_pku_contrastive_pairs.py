"""Construct a clean safety-contrast subset from PKU-SafeRLHF.

Example:
    python construct_pku_contrastive_pairs.py --num-pairs 150 --seed 112 \
        --output /workspace/xcy/dataset/pku_saferlhf/contrastive_pairs_seed112_n150.jsonl
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from unified_eval.common import atomic_write_json, atomic_write_jsonl, sha256_file


DEFAULT_CACHE = Path("/workspace/xcy/dataset/pku_saferlhf/hf_cache")
DEFAULT_OUTPUT = Path(
    "/workspace/xcy/dataset/pku_saferlhf/contrastive_pairs_seed112_n150.jsonl"
)
REFUSAL_TEMPLATE_RE = re.compile(
    r"(?i)\b(?:sorry|apolog(?:ize|ise)|cannot|can't|can’t|won't|won’t|"
    r"unable|not able|do not assist|don't assist|don’t assist|"
    r"will not assist|refuse)\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="PKU-Alignment/PKU-SafeRLHF")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--num-pairs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--min-words", type=int, default=16)
    parser.add_argument("--max-words", type=int, default=256)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    parser.add_argument("--min-unsafe-severity", type=int, default=2)
    parser.add_argument("--allow-inability-templates", action="store_true")
    parser.add_argument("--allow-helpfulness-disagreement", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 100 <= args.num_pairs <= 200:
        raise ValueError("--num-pairs must be between 100 and 200")
    if args.min_words <= 0 or args.max_words < args.min_words:
        raise ValueError("Invalid response word bounds")
    if args.max_length_ratio < 1:
        raise ValueError("--max-length-ratio must be at least 1")
    if args.min_unsafe_severity not in {1, 2, 3}:
        raise ValueError("--min-unsafe-severity must be 1, 2, or 3")
    for path in (args.output, args.output.with_suffix(".summary.json")):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; pass --overwrite")


def load_source(args: argparse.Namespace):
    if args.local_files_only:
        import os

        os.environ["HF_DATASETS_OFFLINE"] = "1"
    return load_dataset(
        args.repo_id,
        name=None if args.config == "default" else args.config,
        split=args.split,
        cache_dir=str(args.cache_dir),
    )


def build_candidates(dataset, args: argparse.Namespace):
    counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for source_index, row in enumerate(dataset):
        counts["source_rows"] += 1
        safe_flags = [bool(row["is_response_0_safe"]), bool(row["is_response_1_safe"])]
        if sum(safe_flags) != 1:
            continue
        counts["exactly_one_safe"] += 1

        safe_id = safe_flags.index(True)
        unsafe_id = 1 - safe_id
        if int(row["safer_response_id"]) != safe_id:
            continue
        counts["safer_id_consistent"] += 1

        safe_response = str(row[f"response_{safe_id}"]).strip()
        unsafe_response = str(row[f"response_{unsafe_id}"]).strip()
        if not safe_response or not unsafe_response:
            continue
        counts["nonempty_responses"] += 1

        if not args.allow_inability_templates and REFUSAL_TEMPLATE_RE.search(safe_response):
            continue
        counts["safe_without_explicit_refusal_template"] += 1

        safe_words = len(safe_response.split())
        unsafe_words = len(unsafe_response.split())
        if not (
            args.min_words <= safe_words <= args.max_words
            and args.min_words <= unsafe_words <= args.max_words
        ):
            continue
        counts["within_word_bounds"] += 1

        length_ratio = max(safe_words, unsafe_words) / min(safe_words, unsafe_words)
        if length_ratio > args.max_length_ratio:
            continue
        counts["within_length_ratio"] += 1

        if (
            not args.allow_helpfulness_disagreement
            and int(row["better_response_id"]) != safe_id
        ):
            continue
        counts["safe_also_more_helpful"] += 1

        unsafe_severity = int(row[f"response_{unsafe_id}_severity_level"])
        if unsafe_severity < args.min_unsafe_severity:
            continue
        counts["unsafe_meets_severity"] += 1

        harm_flags = dict(row[f"response_{unsafe_id}_harm_category"])
        harm_categories = sorted(key for key, value in harm_flags.items() if value)
        if not harm_categories:
            continue
        counts["unsafe_has_harm_category"] += 1

        candidates.append(
            {
                "pair_id": f"pku_{args.config}_{args.split}_{source_index}",
                "source_index": source_index,
                "prompt": str(row["prompt"]).strip(),
                "safe_response": safe_response,
                "unsafe_response": unsafe_response,
                "safe_original_response_id": safe_id,
                "unsafe_original_response_id": unsafe_id,
                "safe_response_source": row[f"response_{safe_id}_source"],
                "unsafe_response_source": row[f"response_{unsafe_id}_source"],
                "safe_response_sha256": row[f"response_{safe_id}_sha256"],
                "unsafe_response_sha256": row[f"response_{unsafe_id}_sha256"],
                "safe_word_count": safe_words,
                "unsafe_word_count": unsafe_words,
                "response_length_ratio": length_ratio,
                "unsafe_severity_level": unsafe_severity,
                "unsafe_harm_categories": harm_categories,
                "prompt_source": row["prompt_source"],
                "safe_also_more_helpful": int(row["better_response_id"]) == safe_id,
                "label_checks": {
                    "exactly_one_safe": True,
                    "safer_response_id_consistent": True,
                    "safe_label": True,
                    "unsafe_label": False,
                },
            }
        )
    return candidates, counts


def balanced_sample(
    candidates: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} candidates remain; requested {count}")

    category_frequency: Counter[str] = Counter()
    for row in candidates:
        category_frequency.update(row["unsafe_harm_categories"])
    for row in candidates:
        row["selection_primary_category"] = min(
            row["unsafe_harm_categories"],
            key=lambda category: (category_frequency[category], category),
        )

    strata: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        strata[(row["selection_primary_category"], row["unsafe_severity_level"])].append(row)
    rng = random.Random(seed)
    for rows in strata.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    selected_prompts: set[str] = set()
    stratum_keys = sorted(strata)
    while len(selected) < count:
        progressed = False
        for key in stratum_keys:
            while strata[key]:
                candidate = strata[key].pop()
                normalized_prompt = " ".join(candidate["prompt"].casefold().split())
                if normalized_prompt in selected_prompts:
                    continue
                selected.append(candidate)
                selected_prompts.add(normalized_prompt)
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError("Balanced sampler exhausted all strata")
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset = load_source(args)
    candidates, filter_counts = build_candidates(dataset, args)
    selected = balanced_sample(candidates, args.num_pairs, args.seed)
    atomic_write_jsonl(args.output, selected)

    category_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    severity_counts: Counter[int] = Counter()
    prompt_sources: Counter[str] = Counter()
    response_sources: Counter[str] = Counter()
    for row in selected:
        category_counts.update(row["unsafe_harm_categories"])
        primary_counts[row["selection_primary_category"]] += 1
        severity_counts[row["unsafe_severity_level"]] += 1
        prompt_sources[str(row["prompt_source"])] += 1
        response_sources[str(row["safe_response_source"])] += 1
        response_sources[str(row["unsafe_response_source"])] += 1

    summary = {
        "source": {
            "repo_id": args.repo_id,
            "config": args.config,
            "split": args.split,
            "dataset_fingerprint": dataset._fingerprint,
            "license": "CC-BY-NC-4.0",
            "cache_dir": str(args.cache_dir.resolve()),
        },
        "selection": {
            "seed": args.seed,
            "requested_pairs": args.num_pairs,
            "selected_pairs": len(selected),
            "method": "round-robin over primary harm category and unsafe severity",
            "filters": {
                "exactly_one_safe": True,
                "safer_response_id_must_agree": True,
                "exclude_explicit_refusal_templates": not args.allow_inability_templates,
                "safe_must_also_be_more_helpful": not args.allow_helpfulness_disagreement,
                "min_words": args.min_words,
                "max_words": args.max_words,
                "max_length_ratio": args.max_length_ratio,
                "min_unsafe_severity": args.min_unsafe_severity,
                "harmbench_deduplication": False,
            },
            "filter_counts": dict(filter_counts),
        },
        "distribution": {
            "unsafe_severity": dict(sorted(severity_counts.items())),
            "primary_harm_category": dict(sorted(primary_counts.items())),
            "all_unsafe_harm_categories": dict(sorted(category_counts.items())),
            "prompt_source": dict(sorted(prompt_sources.items())),
            "response_source": dict(sorted(response_sources.items())),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
        },
    }
    atomic_write_json(args.output.with_suffix(".summary.json"), summary)
    print(f"selected {len(selected)} from {len(candidates)} eligible pairs")
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
