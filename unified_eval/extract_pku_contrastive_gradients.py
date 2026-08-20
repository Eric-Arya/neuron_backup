"""Extract MLP-neuron gradients from PKU safe/unsafe response pairs.

Example:
    python extract_pku_contrastive_gradients.py --limit 150 \
        --output-dir results/pku_contrastive_gradients
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unified_eval.common import atomic_write_json, read_jsonl, sha256_file
from unified_eval.methods import DEFAULT_LLAMA3


DEFAULT_PAIRS = Path(
    "/workspace/xcy/dataset/pku_saferlhf/contrastive_pairs_seed112_n150.jsonl"
)
DEFAULT_OUTPUT = Path("results/pku_contrastive_gradients")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_LLAMA3)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--response-prefix", default=" ")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--stability-top-k", type=int, default=100)
    parser.add_argument("--stability-seed", type=int, default=112)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def attach_completion_hooks(model):
    alphas: list[torch.Tensor] = []
    handles = []
    state = {"start_position": 0}
    for layer in model.model.layers:
        width = layer.mlp.down_proj.in_features
        alpha = torch.ones(
            width,
            device=layer.mlp.down_proj.weight.device,
            dtype=torch.float32,
            requires_grad=True,
        )

        def scale_completion(_module, inputs, current_alpha=alpha):
            activation = inputs[0]
            start = state["start_position"]
            before = activation[:, :start]
            completion = activation[:, start:] * current_alpha.to(activation.dtype)
            return (torch.cat((before, completion), dim=1), *inputs[1:])

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(scale_completion))
        alphas.append(alpha)
    return alphas, handles, state


def completion_score_and_gradient(
    model,
    tokenizer,
    alphas: list[torch.Tensor],
    hook_state: dict[str, int],
    prompt: str,
    response: str,
    response_prefix: str,
) -> tuple[float, torch.Tensor, int]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    response_ids = tokenizer.encode(response_prefix + response, add_special_tokens=False)
    if not response_ids:
        raise ValueError("Response tokenized to zero tokens")
    maximum = int(getattr(model.config, "max_position_embeddings", 8192))
    if len(prompt_ids) + len(response_ids) - 1 > maximum:
        raise ValueError(
            f"Pair exceeds context: prompt={len(prompt_ids)} response={len(response_ids)}"
        )
    input_ids = torch.tensor(
        [prompt_ids + response_ids[:-1]], dtype=torch.long, device=model.device
    )
    hook_state["start_position"] = len(prompt_ids) - 1
    for alpha in alphas:
        alpha.grad = None

    logits = model(input_ids=input_ids, use_cache=False).logits[0]
    positions = torch.arange(
        len(prompt_ids) - 1,
        len(prompt_ids) - 1 + len(response_ids),
        device=model.device,
    )
    targets = torch.tensor(response_ids, dtype=torch.long, device=model.device)
    token_logps = torch.log_softmax(logits[positions].float(), dim=-1)[
        torch.arange(len(response_ids), device=model.device), targets
    ]
    score = token_logps.mean()
    score.backward()
    gradient = torch.stack([alpha.grad.detach().cpu() for alpha in alphas])
    return float(score.detach().cpu()), gradient, len(response_ids)


def ranking_summary(gradients: torch.Tensor, top_k: int):
    values = gradients.float()
    mean_g = values.mean(0)
    std_g = values.std(0, unbiased=False)
    mean_abs_g = values.abs().mean(0)
    positive_fraction = (values > 0).float().mean(0)
    sign_consistency = torch.maximum(positive_fraction, 1 - positive_fraction)
    signal_to_noise = mean_g.abs() / (std_g + 1e-12)
    summary = {
        "mean_g": mean_g,
        "abs_mean_g": mean_g.abs(),
        "mean_abs_g": mean_abs_g,
        "std_g": std_g,
        "positive_fraction": positive_fraction,
        "sign_consistency": sign_consistency,
        "signal_to_noise": signal_to_noise,
    }
    width = mean_g.shape[1]
    indices = torch.topk(mean_g.abs().flatten(), min(top_k, mean_g.numel())).indices
    rows: list[dict[str, Any]] = []
    for rank, flat_index in enumerate(indices.tolist(), 1):
        layer, neuron = divmod(flat_index, width)
        value = float(mean_g[layer, neuron])
        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "neuron": neuron,
                "direction": (
                    "supports_safe_relative_to_unsafe"
                    if value > 0
                    else "supports_unsafe_relative_to_safe"
                ),
                "mean_g": value,
                "abs_mean_g": abs(value),
                "mean_abs_g": float(mean_abs_g[layer, neuron]),
                "std_g": float(std_g[layer, neuron]),
                "positive_fraction": float(positive_fraction[layer, neuron]),
                "sign_consistency": float(sign_consistency[layer, neuron]),
                "signal_to_noise": float(signal_to_noise[layer, neuron]),
            }
        )
    return summary, pd.DataFrame(rows)


def split_half_stability(gradients: torch.Tensor, seed: int, top_k: int):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(gradients.shape[0], generator=generator)
    midpoint = gradients.shape[0] // 2
    first = gradients[order[:midpoint]].float().mean(0).flatten()
    second = gradients[order[midpoint:]].float().mean(0).flatten()
    first_indices = torch.topk(first.abs(), top_k).indices
    second_indices = torch.topk(second.abs(), top_k).indices
    first_set = set(first_indices.tolist())
    second_set = set(second_indices.tolist())
    overlap = first_set & second_set
    sign_agreement = (
        sum(bool(torch.sign(first[index]) == torch.sign(second[index])) for index in overlap)
        / len(overlap)
        if overlap
        else 0.0
    )
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    correlation = float(
        (centered_first * centered_second).sum()
        / (
            torch.linalg.vector_norm(centered_first)
            * torch.linalg.vector_norm(centered_second)
            + 1e-12
        )
    )
    return {
        "seed": seed,
        "first_half_examples": midpoint,
        "second_half_examples": gradients.shape[0] - midpoint,
        "top_k": top_k,
        "top_k_overlap_count": len(overlap),
        "top_k_jaccard": len(overlap) / len(first_set | second_set),
        "overlap_sign_agreement": sign_agreement,
        "full_vector_pearson": correlation,
    }


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.top_k <= 0 or args.stability_top_k <= 0:
        raise ValueError("Limits and top-k values must be positive")
    rows = read_jsonl(args.pairs)[args.offset : args.offset + args.limit]
    if len(rows) != args.limit:
        raise ValueError("Requested pair slice is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "per_pair_g.pt"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; pass --overwrite")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        device_map={"": args.device},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    alphas, handles, hook_state = attach_completion_hooks(model)

    gradients = []
    safe_scores = []
    unsafe_scores = []
    safe_tokens = []
    unsafe_tokens = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    try:
        for index, row in enumerate(rows, 1):
            safe_score, safe_gradient, safe_count = completion_score_and_gradient(
                model,
                tokenizer,
                alphas,
                hook_state,
                str(row["prompt"]),
                str(row["safe_response"]),
                args.response_prefix,
            )
            unsafe_score, unsafe_gradient, unsafe_count = completion_score_and_gradient(
                model,
                tokenizer,
                alphas,
                hook_state,
                str(row["prompt"]),
                str(row["unsafe_response"]),
                args.response_prefix,
            )
            gradients.append((safe_gradient - unsafe_gradient).to(torch.float16))
            safe_scores.append(safe_score)
            unsafe_scores.append(unsafe_score)
            safe_tokens.append(safe_count)
            unsafe_tokens.append(unsafe_count)
            if index % 5 == 0 or index == len(rows):
                print(
                    f"pairs={index}/{len(rows)} "
                    f"elapsed_seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    tensor = torch.stack(gradients)
    elapsed = time.perf_counter() - started
    torch.save(
        {
            "pair_ids": [row["pair_id"] for row in rows],
            "source_indices": [int(row["source_index"]) for row in rows],
            "safe_mean_log_prob": torch.tensor(safe_scores),
            "unsafe_mean_log_prob": torch.tensor(unsafe_scores),
            "safe_token_count": torch.tensor(safe_tokens),
            "unsafe_token_count": torch.tensor(unsafe_tokens),
            "g": tensor,
        },
        output_path,
    )
    summary, ranking = ranking_summary(tensor, args.top_k)
    torch.save(summary, args.output_dir / "g_summary.pt")
    ranking.to_csv(args.output_dir / "top_neurons.csv", index=False)
    stable_ranking = ranking.sort_values(
        ["signal_to_noise", "abs_mean_g"], ascending=False
    ).reset_index(drop=True)
    stable_ranking["rank"] = stable_ranking.index + 1
    stable_ranking.to_csv(args.output_dir / "top_neurons_stable.csv", index=False)
    stability_k = min(args.stability_top_k, tensor.shape[1] * tensor.shape[2])
    stability = split_half_stability(tensor, args.stability_seed, stability_k)
    metadata = {
        "model": str(args.model.resolve()),
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "offset": args.offset,
        "limit": args.limit,
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "response_prefix": args.response_prefix,
        "objective": (
            "mean completion-token log P(safe|prompt) minus mean completion-token "
            "log P(unsafe|prompt)"
        ),
        "alpha_scope": (
            "final prompt position and all teacher-forced completion prediction positions"
        ),
        "gradient_shape": list(tensor.shape),
        "gradient_storage_dtype": str(tensor.dtype),
        "mean_safe_log_prob": sum(safe_scores) / len(safe_scores),
        "mean_unsafe_log_prob": sum(unsafe_scores) / len(unsafe_scores),
        "mean_contrast": sum(
            safe - unsafe for safe, unsafe in zip(safe_scores, unsafe_scores)
        )
        / len(safe_scores),
        "elapsed_seconds": elapsed,
        "pairs_per_second": len(rows) / elapsed,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(model.device) / 2**30
            if torch.cuda.is_available()
            else None
        ),
        "split_half_stability": stability,
    }
    if not all(
        math.isfinite(value)
        for value in (
            metadata["mean_safe_log_prob"],
            metadata["mean_unsafe_log_prob"],
            metadata["mean_contrast"],
        )
    ):
        raise ValueError("Non-finite score summary")
    atomic_write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
