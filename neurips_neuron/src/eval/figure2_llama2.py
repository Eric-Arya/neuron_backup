#!/usr/bin/env python3
"""Reproduce Figure 2 left for Llama2 safety neurons only."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

import torch

from eval.table1_harmbench import (
    DEFAULT_COST_MODEL,
    DEFAULT_DPO_ADAPTER,
    DEFAULT_MODEL,
    DEFAULT_RANKING,
    DEFAULT_SFT_ADAPTER,
    atomic_write_json,
    freeze_manifest,
    generate_missing_condition,
    group_neurons,
    load_hooked_model,
    load_ranked_neurons,
    load_score_model,
    load_shards,
    missing_records,
    score_missing_condition,
    sha256_file,
    validate_cost_direction,
)


LOGGER = logging.getLogger("figure2_llama2")
DEFAULT_DATASET = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/beavertails/splits/"
    "figure2_seed42_n200.jsonl"
)
DEFAULT_OUTPUT = Path("results/figure2_left_llama2_safety_neurons")
DEFAULT_TOP_K = [
    0, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 3000, 4000,
    5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000,
    18000, 20000,
]


def patched_name(top_k: int) -> str:
    return f"patched_top{top_k:06d}"


def clear_models() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def ensure_config(args: argparse.Namespace) -> None:
    config = {
        "figure": "Figure 2 left: Patch Base with DPO",
        "model": "Llama2-7b",
        "neuron_type": f"{args.selection} neurons",
        "selection": args.selection,
        "random_seed": args.random_seed if args.selection == "random" else None,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "ranking": str(args.ranking.resolve()),
        "ranking_sha256": sha256_file(args.ranking),
        "base_model": str(args.model.resolve()),
        "sft_adapter": str(args.sft_adapter.resolve()),
        "dpo_adapter": str(args.dpo_adapter.resolve()),
        "cost_model": str(args.cost_model.resolve()),
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "generation_batch_size": args.generation_batch_size,
        "score_batch_size": args.score_batch_size,
        "decoding": "greedy",
        "prompt_format": "tulu",
    }
    path = args.output_dir / "run_config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        semantic_keys = {
            "dataset_sha256", "ranking_sha256", "base_model", "sft_adapter",
            "dpo_adapter", "cost_model", "top_k", "max_new_tokens",
            "selection", "random_seed",
        }
        # Runs created before selection was configurable are safety-neuron runs.
        previous.setdefault("selection", "safety")
        previous.setdefault("random_seed", None)
        if any(previous.get(key) != config.get(key) for key in semantic_keys):
            raise ValueError(f"Existing run differs in {path}; use a new output directory")
    else:
        atomic_write_json(path, config)


def generate(args: argparse.Namespace, manifest: Sequence[dict[str, Any]]) -> int:
    conditions = ["base", "dpo", *[patched_name(k) for k in args.top_k]]
    need = {
        condition: bool(
            missing_records(manifest, load_shards(args.output_dir / "generations" / condition))
        )
        for condition in conditions
    }
    if not any(need.values()):
        LOGGER.info("All generations are complete")
        _, ranking_count = load_ranked_neurons(args.ranking, max(args.top_k))
        return ranking_count

    base_model = guide_model = None
    try:
        base_model, tokenizer = load_hooked_model(
            args.model, args.tokenizer, [], args.base_device, args.dtype
        )
        guide_model, guide_tokenizer = load_hooked_model(
            args.model,
            args.tokenizer,
            [args.sft_adapter, args.dpo_adapter],
            args.guide_device,
            args.dtype,
        )
        if tokenizer.get_vocab() != guide_tokenizer.get_vocab():
            raise ValueError("Base and DPO tokenizers differ")
        ranking, ranking_count = load_ranked_neurons(args.ranking, max(args.top_k), base_model)
        if args.selection == "random":
            all_neurons, ranking_count = load_ranked_neurons(
                args.ranking, ranking_count, base_model
            )
            generator = torch.Generator(device="cpu").manual_seed(args.random_seed)
            ordering = torch.randperm(ranking_count, generator=generator)
            ranking = all_neurons[ordering[: max(args.top_k)]]
        if need["base"]:
            generate_missing_condition(
                "base", manifest, args.output_dir, args.generation_batch_size,
                args.max_new_tokens, base_model, tokenizer, None, {},
            )
        if need["dpo"]:
            generate_missing_condition(
                "dpo", manifest, args.output_dir, args.generation_batch_size,
                args.max_new_tokens, guide_model, tokenizer, None, {},
            )
        for top_k in args.top_k:
            condition = patched_name(top_k)
            if not need[condition]:
                continue
            selected = ranking[:top_k]
            generate_missing_condition(
                condition,
                manifest,
                args.output_dir,
                args.generation_batch_size,
                args.max_new_tokens,
                base_model,
                tokenizer,
                guide_model,
                group_neurons(selected),
            )
    finally:
        base_model = None
        guide_model = None
        clear_models()
    return ranking_count


def score(args: argparse.Namespace, manifest: Sequence[dict[str, Any]]) -> None:
    conditions = ["base", "dpo", *[patched_name(k) for k in args.top_k]]
    model = None
    try:
        model, tokenizer = load_score_model(args.cost_model, args.cost_device, args.dtype)
        validation = validate_cost_direction(model, tokenizer)
        atomic_write_json(args.output_dir / "cost_model_validation.json", validation)
        for condition in conditions:
            score_missing_condition(
                condition, manifest, args.output_dir, args.score_batch_size, model, tokenizer
            )
    finally:
        model = None
        clear_models()


def mean_cost(args: argparse.Namespace, manifest: Sequence[dict[str, Any]], condition: str) -> float:
    rows = load_shards(args.output_dir / "costs" / condition)
    missing = missing_records(manifest, rows)
    if missing:
        raise ValueError(f"{condition} is missing {len(missing)} scores")
    values = [float(rows[record["id"]]["cost"]) for record in manifest]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{condition} contains non-finite costs")
    return statistics.fmean(values)


def aggregate(args: argparse.Namespace, manifest: Sequence[dict[str, Any]], ranking_count: int):
    base_cost = mean_cost(args, manifest, "base")
    dpo_cost = mean_cost(args, manifest, "dpo")
    denominator = dpo_cost - base_cost
    if abs(denominator) < 1e-12:
        raise ZeroDivisionError("Base and DPO costs are equal")
    points = []
    for top_k in args.top_k:
        patched_cost = mean_cost(args, manifest, patched_name(top_k))
        points.append(
            {
                "top_k": top_k,
                "neurons_percent": 100.0 * top_k / ranking_count,
                "mean_cost": patched_cost,
                "causal_effect": (patched_cost - base_cost) / denominator,
                "causal_effect_percent": 100.0 * (patched_cost - base_cost) / denominator,
            }
        )
    result = {
        "figure": "Figure 2 left: Patch Base with DPO",
        "model": "Llama2-7b",
        "dataset": "BeaverTails",
        "neuron_type": f"{args.selection} neurons",
        "random_seed": args.random_seed if args.selection == "random" else None,
        "prompt_count": len(manifest),
        "ranked_neuron_count": ranking_count,
        "base_mean_cost": base_cost,
        "dpo_mean_cost": dpo_cost,
        "curve": points,
    }
    atomic_write_json(args.output_dir / "curve.json", result)
    csv_path = args.output_dir / "curve.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    temporary.replace(csv_path)
    return result


def plot_curve(args: argparse.Namespace, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    x = [row["neurons_percent"] for row in result["curve"]]
    y = [row["causal_effect"] for row in result["curve"]]
    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    linestyle = "--" if args.selection == "random" else "-"
    axis.plot(
        x, y, color="#9467bd", linestyle=linestyle, linewidth=2,
        marker="o", markersize=3, label=f"Llama2 {args.selection}",
    )
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xlabel("Neurons (%)")
    axis.set_ylabel("Causal Effect")
    axis.set_title("Patch Base with DPO")
    axis.grid(True, linestyle=":", alpha=0.6)
    axis.legend(frameon=False)
    figure.tight_layout()
    stem = f"figure2_left_llama2_{args.selection}_neurons"
    figure.savefig(args.output_dir / f"{stem}.png", dpi=200)
    figure.savefig(args.output_dir / f"{stem}.pdf")
    plt.close(figure)


def write_checksums(args: argparse.Namespace) -> None:
    paths = sorted(
        path for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "checksums.json" and "/logs/" not in str(path)
    )
    checksums = {str(path.relative_to(args.output_dir)): sha256_file(path) for path in paths}
    atomic_write_json(args.output_dir / "checksums.json", checksums)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--tokenizer", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--sft-adapter", type=Path, default=DEFAULT_SFT_ADAPTER)
    result.add_argument("--dpo-adapter", type=Path, default=DEFAULT_DPO_ADAPTER)
    result.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    result.add_argument("--cost-model", type=Path, default=DEFAULT_COST_MODEL)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--top-k", type=int, nargs="+", default=DEFAULT_TOP_K)
    result.add_argument("--selection", choices=("safety", "random"), default="safety")
    result.add_argument("--random-seed", type=int, default=42)
    result.add_argument("--expected-prompts", type=int, default=200)
    result.add_argument("--max-new-tokens", type=int, default=128)
    result.add_argument("--generation-batch-size", type=int, default=16)
    result.add_argument("--score-batch-size", type=int, default=16)
    result.add_argument("--base-device", default="cuda:0")
    result.add_argument("--guide-device", default="cuda:1")
    result.add_argument("--cost-device", default="cuda:0")
    result.add_argument("--dtype", default="bfloat16")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.top_k != sorted(set(args.top_k)) or args.top_k[0] < 0:
        raise ValueError("top-k values must be unique, non-negative, and ascending")
    manifest = freeze_manifest(
        args.dataset.resolve(), args.output_dir / "prompt_manifest.jsonl", args.expected_prompts
    )
    ensure_config(args)
    ranking_count = generate(args, manifest)
    score(args, manifest)
    result = aggregate(args, manifest, ranking_count)
    plot_curve(args, result)
    write_checksums(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
