#!/usr/bin/env python3
"""Extract standard joint-path Integrated Gradients for the I-cannot score."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_i_cannot_gradients import build_summary, split_half_stability


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct"))
    parser.add_argument(
        "--input-csv", type=Path,
        default=root / "data/processed/llama3_8b_instruct_first200.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "results/gradients/integrated_gradients",
    )
    parser.add_argument("--prompt-column", default="goal")
    parser.add_argument("--response-column", default="model_response")
    parser.add_argument("--response-prefix", default="I cannot")
    parser.add_argument("--assistant-prefill", default="I")
    parser.add_argument("--target-text", default=" cannot")
    parser.add_argument("--baseline-alpha", type=float, default=0.9)
    parser.add_argument("--integration-steps", type=int, default=16)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="0 means all matching examples")
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


def quadrature(steps: int) -> tuple[list[float], list[float]]:
    if steps <= 0:
        raise ValueError("integration-steps must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    # Map Gauss-Legendre nodes and weights from [-1, 1] to [0, 1].
    return ((nodes + 1.0) / 2.0).tolist(), (weights / 2.0).tolist()


def attach_alpha_hooks(model) -> tuple[list[torch.Tensor], list[object]]:
    alphas: list[torch.Tensor] = []
    handles: list[object] = []
    for layer in model.model.layers:
        width = layer.mlp.down_proj.in_features
        alpha = torch.ones(width, device=model.device, dtype=torch.float32, requires_grad=True)

        def scale_input(_module, inputs, current_alpha=alpha):
            activation = inputs[0]
            return (activation * current_alpha.to(dtype=activation.dtype), *inputs[1:])

        alphas.append(alpha)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(scale_input))
    return alphas, handles


def set_alpha(alphas: list[torch.Tensor], value: float) -> None:
    with torch.no_grad():
        for alpha in alphas:
            alpha.fill_(value)


def score_at(model, inputs, alphas, alpha_value: float, target_id: int) -> float:
    set_alpha(alphas, alpha_value)
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
        score = torch.log_softmax(logits, dim=-1)[target_id]
    return float(score.cpu())


def main() -> None:
    args = parse_args()
    if not 0 <= args.baseline_alpha < 1:
        raise ValueError("baseline-alpha must satisfy 0 <= baseline < 1")
    rank, local_rank, world_size = setup_distributed()
    output_dir = args.output_dir.resolve()
    final_path = output_dir / "per_example_ig.pt"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if final_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {final_path}; pass --overwrite to replace it")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    target_ids = tokenizer.encode(args.target_text, add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(f"Target must be one token, got {target_ids}")
    target_id = target_ids[0]
    model_dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=model_dtype,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    alphas, handles = attach_alpha_hooks(model)
    nodes, weights = quadrature(args.integration_steps)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    local_ig: list[torch.Tensor] = []
    source_indices: list[int] = []
    baseline_scores: list[float] = []
    actual_scores: list[float] = []
    completeness_deltas: list[float] = []
    started = time.perf_counter()

    for completed, (_, row) in enumerate(local_examples.iterrows(), start=1):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row[args.prompt_column])}],
            tokenize=False,
            add_generation_prompt=True,
        ) + args.assistant_prefill
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        score_baseline = score_at(
            model, inputs, alphas, args.baseline_alpha, target_id
        )
        score_actual = score_at(model, inputs, alphas, 1.0, target_id)
        integrated_gradient = torch.zeros(
            (len(alphas), alphas[0].numel()), device=model.device, dtype=torch.float32
        )

        for node, weight in zip(nodes, weights):
            path_alpha = args.baseline_alpha + node * (1.0 - args.baseline_alpha)
            set_alpha(alphas, path_alpha)
            outputs = model(**inputs, use_cache=False)
            score = torch.log_softmax(outputs.logits[0, -1].float(), dim=-1)[target_id]
            gradients = torch.autograd.grad(score, alphas)
            integrated_gradient.add_(torch.stack(gradients), alpha=weight)

        integrated_gradient.mul_(1.0 - args.baseline_alpha)
        completeness_delta = float(
            integrated_gradient.sum().detach().cpu()
            - (score_actual - score_baseline)
        )
        local_ig.append(integrated_gradient.detach().cpu().to(torch.float16))
        source_indices.append(int(row["source_index"]))
        baseline_scores.append(score_baseline)
        actual_scores.append(score_actual)
        completeness_deltas.append(completeness_delta)

        if completed % 5 == 0 or completed == len(local_examples):
            print(
                f"rank={rank} examples={completed}/{len(local_examples)} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )

    set_alpha(alphas, 1.0)
    for handle in handles:
        handle.remove()
    part = {
        "source_indices": torch.tensor(source_indices, dtype=torch.int64),
        "baseline_log_prob": torch.tensor(baseline_scores, dtype=torch.float32),
        "actual_log_prob": torch.tensor(actual_scores, dtype=torch.float32),
        "completeness_delta": torch.tensor(completeness_deltas, dtype=torch.float32),
        "ig": torch.stack(local_ig) if local_ig else torch.empty(0),
    }
    part_path = output_dir / f"per_example_ig.rank{rank}.pt"
    torch.save(part, part_path)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        parts = [
            torch.load(output_dir / f"per_example_ig.rank{part_rank}.pt", weights_only=True)
            for part_rank in range(world_size)
        ]
        merged = {
            key: torch.cat([part[key] for part in parts])
            for key in ("source_indices", "baseline_log_prob", "actual_log_prob",
                        "completeness_delta", "ig")
        }
        order = torch.argsort(merged["source_indices"])
        merged = {key: value[order] for key, value in merged.items()}
        torch.save(merged, final_path)
        for part_rank in range(world_size):
            (output_dir / f"per_example_ig.rank{part_rank}.pt").unlink()

        raw_summary, ranking = build_summary(
            merged["ig"], top_k=args.top_k, top_per_layer=args.top_per_layer
        )
        summary = {
            key.replace("_g", "_ig"): value for key, value in raw_summary.items()
        }
        ranking = ranking.rename(
            columns={
                "mean_g": "mean_ig",
                "abs_mean_g": "abs_mean_ig",
                "mean_abs_g": "mean_abs_ig",
                "std_g": "std_ig",
            }
        )
        torch.save(summary, output_dir / "ig_summary.pt")
        ranking.to_csv(output_dir / "top_neurons.csv", index=False)
        ranking.loc[ranking["mean_ig"] > 0].head(100).to_csv(
            output_dir / "top_refusal_supporting.csv", index=False
        )
        ranking.loc[ranking["mean_ig"] < 0].head(100).to_csv(
            output_dir / "top_refusal_suppressing.csv", index=False
        )
        selected.to_csv(output_dir / "examples.csv", index=False)
        stability = split_half_stability(merged["ig"], seed=args.seed)
        (output_dir / "split_half_stability.json").write_text(
            json.dumps(stability, indent=2) + "\n", encoding="utf-8"
        )

        score_change = merged["actual_log_prob"] - merged["baseline_log_prob"]
        abs_delta = merged["completeness_delta"].abs()
        relative = abs_delta / score_change.abs().clamp_min(1e-6)
        metadata = {
            "method": "standard Integrated Gradients; all alpha coordinates move jointly",
            "model": str(args.model.resolve()),
            "input_csv": str(args.input_csv.resolve()),
            "num_examples": len(selected),
            "selection": f"{args.response_column}.startswith({args.response_prefix!r})",
            "score": (
                f"log p({args.target_text!r} | chat(goal) + assistant prefill "
                f"{args.assistant_prefill!r})"
            ),
            "baseline_alpha": args.baseline_alpha,
            "actual_alpha": 1.0,
            "path": "alpha(t) = baseline_alpha + t * (1 - baseline_alpha), for every neuron",
            "quadrature": "Gauss-Legendre on [0,1]",
            "integration_steps": args.integration_steps,
            "target_token_id": target_id,
            "ig_shape": list(merged["ig"].shape),
            "ig_storage_dtype": str(merged["ig"].dtype),
            "model_compute_dtype": args.dtype,
            "mean_baseline_probability": merged["baseline_log_prob"].exp().mean().item(),
            "mean_actual_probability": merged["actual_log_prob"].exp().mean().item(),
            "mean_score_change": score_change.mean().item(),
            "completeness": {
                "mean_absolute_delta": abs_delta.mean().item(),
                "median_absolute_delta": abs_delta.median().item(),
                "max_absolute_delta": abs_delta.max().item(),
                "median_relative_error": relative.median().item(),
                "p95_relative_error": torch.quantile(relative, 0.95).item(),
            },
            "ranking_metric": "abs(mean(IG over examples))",
            "world_size": world_size,
            "seed": args.seed,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
