"""Sample repeated Grad completions and score them with Beaver cost.

Example:
    python sample_grad_beaver.py --num-prompts 5 --samples-per-prompt 6 \
        --temperature 1.0 --top-p 0.95 --output results/grad_beaver_samples.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from transformers import set_seed

from unified_eval.common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    score_asr,
    sha256_file,
)
from unified_eval.methods import DEFAULT_GRAD_RANKING, DEFAULT_LLAMA3, GradMethod, _trim_generated


DEFAULT_MANIFEST = Path("results/grad_harmbench_development/tuning_manifest.jsonl")
DEFAULT_TEST = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/"
    "table1_seed42_n200.jsonl"
)
DEFAULT_COST_MODEL = Path("/workspace/xcy/models/beaver-7b-v1.0-cost")
DEFAULT_NEURIPS_REPO = Path("/workspace/xcy/safety_repro/neurips_neuron")
DEFAULT_OUTPUT = Path("results/grad_beaver_samples.jsonl")
WORD_RE = re.compile(r"\w+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--frozen-test-manifest", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--samples-per-prompt", type=int, default=6)
    parser.add_argument("--prompt-seed", type=int, default=112)
    parser.add_argument("--sampling-seed", type=int, default=112)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_LLAMA3)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_GRAD_RANKING)
    parser.add_argument("--grad-top-k", type=int, default=25)
    parser.add_argument("--grad-strength", type=float, default=1.0)
    parser.add_argument("--grad-scope", choices=("last", "all"), default="last")
    parser.add_argument(
        "--grad-direction", choices=("signed", "positive-only"), default="positive-only"
    )
    parser.add_argument("--generation-device", default="cuda:0")
    parser.add_argument("--cost-model", type=Path, default=DEFAULT_COST_MODEL)
    parser.add_argument("--cost-device", default="cuda:1")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--cost-batch-size", type=int, default=16)
    parser.add_argument("--neurips-repo", type=Path, default=DEFAULT_NEURIPS_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if not 4 <= args.samples_per_prompt <= 8:
        raise ValueError("--samples-per-prompt must be between 4 and 8")
    if args.max_new_tokens <= 0 or args.temperature <= 0:
        raise ValueError("max tokens and temperature must be positive")
    if not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("top-p must be in (0, 1] and top-k must be non-negative")
    if args.cost_batch_size <= 0:
        raise ValueError("--cost-batch-size must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite")


def choose_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = read_jsonl(args.manifest)
    frozen_test = read_jsonl(args.frozen_test_manifest)
    test_ids = {str(row["id"]) for row in frozen_test}
    test_prompts = {str(row["prompt"]) for row in frozen_test}
    overlap = [
        row for row in candidates
        if str(row["id"]) in test_ids or str(row["prompt"]) in test_prompts
    ]
    if overlap:
        raise ValueError(f"Candidate manifest overlaps frozen test set: {len(overlap)} rows")
    if args.num_prompts > len(candidates):
        raise ValueError("Requested more prompts than the candidate manifest contains")
    return random.Random(args.prompt_seed).sample(candidates, args.num_prompts)


def sample_responses(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    method = GradMethod(
        model_path=args.model,
        ranking_path=args.ranking,
        top_k=args.grad_top_k,
        strength=args.grad_strength,
        scope=args.grad_scope,
        direction=args.grad_direction,
        max_batch_size=len(prompts),
        device=args.generation_device,
        dtype_name=args.dtype,
    )
    records: list[dict[str, Any]] = []
    try:
        prompt_texts = [str(row["prompt"]) for row in prompts]
        for sample_index in range(args.samples_per_prompt):
            seed = args.sampling_seed + sample_index
            set_seed(seed)
            encoded = method._encode(prompt_texts)
            input_width = encoded.input_ids.shape[1]
            with torch.inference_mode():
                output = method.model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    use_cache=True,
                    pad_token_id=method.tokenizer.pad_token_id,
                )
            eos = method.model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            for prompt_index, (source, generated) in enumerate(zip(prompts, output)):
                prompt_ids = encoded.input_ids[prompt_index][
                    encoded.attention_mask[prompt_index].bool()
                ].tolist()
                generated_ids = _trim_generated(
                    generated[input_width:].tolist(), eos_ids, method.tokenizer.pad_token_id
                )
                response = method.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                score_text = method.tokenizer.decode(
                    prompt_ids + generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                jailbroken, matched_refusal_prefixes = score_asr(response)
                records.append(
                    {
                        **source,
                        "prompt_index": prompt_index,
                        "sample_index": sample_index,
                        "sampling_seed": seed,
                        "response": response,
                        "score_text": score_text,
                        "generated_token_count": len(generated_ids),
                        "jailbroken": jailbroken,
                        "matched_refusal_prefixes": matched_refusal_prefixes,
                    }
                )
            print(f"generated sample round {sample_index + 1}/{args.samples_per_prompt}", flush=True)
    finally:
        method.close()
    return records


def score_with_beaver(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, float]:
    repo = args.neurips_repo.resolve()
    for path in (repo, repo / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from eval import table1_harmbench as scorer

    model, tokenizer = scorer.load_score_model(args.cost_model, args.cost_device, args.dtype)
    try:
        direction = scorer.validate_cost_direction(model, tokenizer)
        for start in range(0, len(records), args.cost_batch_size):
            batch = records[start : start + args.cost_batch_size]
            costs = scorer.score_texts(model, tokenizer, [row["score_text"] for row in batch])
            for row, cost in zip(batch, costs):
                row["beaver_cost"] = cost
                row["reward"] = -cost
            print(f"scored {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    finally:
        model = tokenizer = None
        torch.cuda.empty_cache()
    return direction


def normalized_words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def diversity_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for prompt_index in sorted({int(row["prompt_index"]) for row in records}):
        rows = [row for row in records if int(row["prompt_index"]) == prompt_index]
        normalized = [" ".join(normalized_words(str(row["response"]))) for row in rows]
        token_sets = [set(value.split()) for value in normalized]
        jaccard_similarities = []
        sequence_similarities = []
        for (left_text, left), (right_text, right) in combinations(zip(normalized, token_sets), 2):
            union = left | right
            jaccard_similarities.append(len(left & right) / len(union) if union else 1.0)
            sequence_similarities.append(difflib.SequenceMatcher(None, left_text, right_text).ratio())
        costs = [float(row["beaver_cost"]) for row in rows]
        summaries.append(
            {
                "prompt_index": prompt_index,
                "id": rows[0]["id"],
                "behavior_id": rows[0].get("behavior_id"),
                "prompt": rows[0]["prompt"],
                "num_samples": len(rows),
                "unique_normalized_responses": len(set(normalized)),
                "exact_unique_ratio": len(set(normalized)) / len(rows),
                "mean_pairwise_token_jaccard_similarity": statistics.fmean(jaccard_similarities),
                "mean_pairwise_sequence_similarity": statistics.fmean(sequence_similarities),
                "beaver_cost_mean": statistics.fmean(costs),
                "beaver_cost_population_std": statistics.pstdev(costs),
                "beaver_cost_min": min(costs),
                "beaver_cost_max": max(costs),
                "beaver_cost_range": max(costs) - min(costs),
                "refusal_substring_jailbroken_count": sum(
                    bool(row["jailbroken"]) for row in rows
                ),
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    validate_args(args)
    prompts = choose_prompts(args)
    records = sample_responses(args, prompts)
    direction = score_with_beaver(args, records)
    atomic_write_jsonl(args.output, records)
    summaries = diversity_summary(records)
    metadata = {
        "output": str(args.output.resolve()),
        "num_prompts": args.num_prompts,
        "samples_per_prompt": args.samples_per_prompt,
        "prompt_selection": "uniform without replacement from manifest",
        "prompt_seed": args.prompt_seed,
        "sampling_seed_start": args.sampling_seed,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "grad": {
            "model": str(args.model.resolve()),
            "ranking": str(args.ranking.resolve()),
            "top_k": args.grad_top_k,
            "strength": args.grad_strength,
            "scope": args.grad_scope,
        },
        "beaver": {
            "model": str(args.cost_model.resolve()),
            "reward_definition": "reward = -beaver_cost",
            "lower_cost_is_safer": True,
            "direction_check": direction,
        },
        "data": {
            "candidate_manifest": str(args.manifest.resolve()),
            "candidate_manifest_sha256": sha256_file(args.manifest),
            "frozen_test_manifest": str(args.frozen_test_manifest.resolve()),
            "frozen_test_manifest_sha256": sha256_file(args.frozen_test_manifest),
            "verified_id_and_prompt_overlap": 0,
        },
        "per_prompt": summaries,
        "overall": {
            "mean_exact_unique_ratio": statistics.fmean(row["exact_unique_ratio"] for row in summaries),
            "mean_pairwise_token_jaccard_similarity": statistics.fmean(
                row["mean_pairwise_token_jaccard_similarity"] for row in summaries
            ),
            "mean_pairwise_sequence_similarity": statistics.fmean(
                row["mean_pairwise_sequence_similarity"] for row in summaries
            ),
            "mean_within_prompt_beaver_cost_std": statistics.fmean(
                row["beaver_cost_population_std"] for row in summaries
            ),
            "mean_within_prompt_beaver_cost_range": statistics.fmean(
                row["beaver_cost_range"] for row in summaries
            ),
        },
    }
    if not all(math.isfinite(float(value)) for value in metadata["overall"].values()):
        raise ValueError("Non-finite summary metric")
    summary_path = args.output.with_suffix(".summary.json")
    atomic_write_json(summary_path, metadata)
    print(json.dumps(metadata["overall"], indent=2), flush=True)
    print(f"wrote {args.output} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
