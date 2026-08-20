#!/usr/bin/env python3
"""Extract per-example GradAct signals for the Llama-3 "I cannot" refusal pivot."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-column", default="goal")
    parser.add_argument("--response-column", default="model_response")
    parser.add_argument("--response-prefix", default="I cannot")
    parser.add_argument("--assistant-prefill", default="I")
    parser.add_argument("--target-text", default=" cannot")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="0 means use all matching examples")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--top-per-layer", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
        )
    return rank, local_rank, world_size


def attach_alpha_hooks(model: torch.nn.Module) -> tuple[list[torch.Tensor], list[object]]:
    """Insert alpha immediately before each MLP down projection."""
    alphas: list[torch.Tensor] = []
    handles: list[object] = []
    for layer in model.model.layers:
        width = layer.mlp.down_proj.in_features
        alpha = torch.ones(width, device=model.device, dtype=torch.float32, requires_grad=True)

        def scale_input(_module, inputs, current_alpha=alpha):
            activation = inputs[0]
            scaled = activation * current_alpha.to(dtype=activation.dtype)
            return (scaled, *inputs[1:])

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(scale_input))
        alphas.append(alpha)
    return alphas, handles


def build_summary(
    gradients: torch.Tensor,
    top_k: int,
    top_per_layer: int,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    values = gradients.float()
    mean_g = values.mean(dim=0)
    std_g = values.std(dim=0, unbiased=False)
    mean_abs_g = values.abs().mean(dim=0)
    positive_fraction = (values > 0).float().mean(dim=0)
    sign_consistency = torch.maximum(positive_fraction, 1.0 - positive_fraction)
    snr = mean_g.abs() / (std_g + 1e-12)
    summary = {
        "mean_g": mean_g,
        "abs_mean_g": mean_g.abs(),
        "mean_abs_g": mean_abs_g,
        "std_g": std_g,
        "positive_fraction": positive_fraction,
        "sign_consistency": sign_consistency,
        "snr": snr,
    }

    n_layers, width = mean_g.shape
    selected: set[int] = set()
    flat_strength = mean_g.abs().flatten()
    overall_k = min(top_k, flat_strength.numel())
    selected.update(torch.topk(flat_strength, overall_k).indices.tolist())
    layer_k = min(top_per_layer, width)
    for layer_idx in range(n_layers):
        neurons = torch.topk(mean_g[layer_idx].abs(), layer_k).indices.tolist()
        selected.update(layer_idx * width + neuron for neuron in neurons)

    rows = []
    for flat_idx in selected:
        layer_idx, neuron_idx = divmod(flat_idx, width)
        rows.append(
            {
                "layer": layer_idx,
                "neuron": neuron_idx,
                "direction": (
                    "supports_cannot"
                    if mean_g[layer_idx, neuron_idx].item() > 0
                    else "suppresses_cannot"
                ),
                "mean_g": mean_g[layer_idx, neuron_idx].item(),
                "abs_mean_g": mean_g[layer_idx, neuron_idx].abs().item(),
                "mean_abs_g": mean_abs_g[layer_idx, neuron_idx].item(),
                "std_g": std_g[layer_idx, neuron_idx].item(),
                "positive_fraction": positive_fraction[layer_idx, neuron_idx].item(),
                "sign_consistency": sign_consistency[layer_idx, neuron_idx].item(),
                "snr": snr[layer_idx, neuron_idx].item(),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        ["abs_mean_g", "sign_consistency"], ascending=[False, False]
    )
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return summary, ranking


def split_half_stability(gradients: torch.Tensor, seed: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(gradients.shape[0], generator=generator)
    midpoint = gradients.shape[0] // 2
    first = gradients[permutation[:midpoint]].float().mean(dim=0).flatten()
    second = gradients[permutation[midpoint:]].float().mean(dim=0).flatten()
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    correlation = (first_centered @ second_centered) / (
        torch.linalg.vector_norm(first_centered)
        * torch.linalg.vector_norm(second_centered)
    )
    overlaps = {}
    for k in (10, 25, 50, 100, 500, 1000, 2000):
        effective_k = min(k, first.numel())
        first_top = set(torch.topk(first.abs(), effective_k).indices.tolist())
        second_top = set(torch.topk(second.abs(), effective_k).indices.tolist())
        overlap = len(first_top.intersection(second_top))
        overlaps[str(k)] = {
            "count": overlap,
            "fraction": overlap / effective_k,
        }
    return {
        "seed": seed,
        "first_half_examples": midpoint,
        "second_half_examples": gradients.shape[0] - midpoint,
        "pearson_all_neurons": correlation.item(),
        "top_abs_mean_overlap": overlaps,
    }


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    output_dir = Path(args.output_dir)
    final_gradient_path = output_dir / "per_example_g.pt"

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if final_gradient_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists: {final_gradient_path}; pass --overwrite to replace it"
            )
    if world_size > 1:
        dist.barrier()

    frame = pd.read_csv(args.input_csv)
    required = {args.prompt_column, args.response_column, "source_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing input columns: {sorted(missing)}")
    mask = frame[args.response_column].fillna("").astype(str).str.startswith(args.response_prefix)
    selected = frame.loc[mask].copy()
    if args.max_examples > 0:
        selected = selected.iloc[: args.max_examples].copy()
    selected = selected.reset_index(drop=True)
    local_examples = selected.iloc[rank::world_size]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    target_ids = tokenizer.encode(args.target_text, add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(
            f"target-text must be exactly one token; got {target_ids} for {args.target_text!r}"
        )
    target_id = target_ids[0]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    alphas, hook_handles = attach_alpha_hooks(model)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    local_gradients: list[torch.Tensor] = []
    local_scores: list[float] = []
    local_indices: list[int] = []
    started = time.perf_counter()

    for completed, (_, row) in enumerate(local_examples.iterrows(), start=1):
        messages = [{"role": "user", "content": str(row[args.prompt_column])}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + args.assistant_prefill
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        for alpha in alphas:
            alpha.grad = None

        outputs = model(**inputs, use_cache=False)
        score = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)[target_id]
        score.backward()
        gradient = torch.stack([alpha.grad.detach().cpu() for alpha in alphas])
        local_gradients.append(gradient.to(torch.float16))
        local_scores.append(float(score.detach().cpu()))
        local_indices.append(int(row["source_index"]))

        if completed % 10 == 0 or completed == len(local_examples):
            elapsed = time.perf_counter() - started
            print(
                f"rank={rank} examples={completed}/{len(local_examples)} "
                f"elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    for handle in hook_handles:
        handle.remove()
    local_tensor = torch.stack(local_gradients) if local_gradients else torch.empty(0)
    part_path = output_dir / f"per_example_g.rank{rank}.pt"
    torch.save(
        {
            "source_indices": torch.tensor(local_indices, dtype=torch.int64),
            "log_prob_cannot": torch.tensor(local_scores, dtype=torch.float32),
            "g": local_tensor,
        },
        part_path,
    )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        parts = [
            torch.load(output_dir / f"per_example_g.rank{part_rank}.pt", weights_only=True)
            for part_rank in range(world_size)
        ]
        indices = torch.cat([part["source_indices"] for part in parts])
        scores = torch.cat([part["log_prob_cannot"] for part in parts])
        gradients = torch.cat([part["g"] for part in parts])
        order = torch.argsort(indices)
        merged = {
            "source_indices": indices[order],
            "log_prob_cannot": scores[order],
            "g": gradients[order],
        }
        torch.save(merged, final_gradient_path)
        for part_rank in range(world_size):
            (output_dir / f"per_example_g.rank{part_rank}.pt").unlink()

        summary, ranking = build_summary(
            merged["g"], top_k=args.top_k, top_per_layer=args.top_per_layer
        )
        stability = split_half_stability(merged["g"], seed=args.seed)
        torch.save(summary, output_dir / "g_summary.pt")
        ranking.to_csv(output_dir / "top_neurons.csv", index=False)
        ranking.loc[ranking["mean_g"] > 0].head(100).to_csv(
            output_dir / "top_refusal_supporting.csv", index=False
        )
        ranking.loc[ranking["mean_g"] < 0].head(100).to_csv(
            output_dir / "top_refusal_suppressing.csv", index=False
        )
        selected.to_csv(output_dir / "examples.csv", index=False)
        metadata = {
            "model": args.model,
            "input_csv": args.input_csv,
            "num_examples": len(selected),
            "selection": f"{args.response_column}.startswith({args.response_prefix!r})",
            "score": (
                f"log p({args.target_text!r} | chat(goal) + assistant prefill "
                f"{args.assistant_prefill!r})"
            ),
            "target_token_id": target_id,
            "target_token_decoded": tokenizer.decode([target_id]),
            "gradient_shape": list(merged["g"].shape),
            "gradient_storage_dtype": str(merged["g"].dtype),
            "ranking_metric": "abs(mean(g over examples))",
            "alpha_scope": "one alpha per layer/neuron, shared across all token positions",
            "world_size": world_size,
            "seed": args.seed,
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        with (output_dir / "split_half_stability.json").open("w", encoding="utf-8") as handle:
            json.dump(stability, handle, indent=2)
        print(
            f"saved={output_dir} examples={len(selected)} shape={list(merged['g'].shape)}",
            flush=True,
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
