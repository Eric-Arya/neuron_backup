#!/usr/bin/env python3
"""Rank MLP neurons by the gradient of a raw-prompt refusal phrase.

The intervention variable alpha scales one SwiGLU intermediate dimension before
``down_proj``.  It is applied only at the final prompt position and at the
teacher-forced refusal positions, matching the inference-time controller.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct"))
    parser.add_argument(
        "--prepared", type=Path,
        default=Path("/workspace/xcy/safety_repro/iclr_neuron/neuron_deactivate/"
                     "evaluation_data/harm_behavior_first_100.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path,
                        default=root / "results/gradients/raw_refusal_train40")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--refusal-phrase", default="I cannot")
    parser.add_argument(
        "--alpha-scope", choices=("tail", "global"), default="tail"
    )
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def attach_alpha_hooks(model, scope: str):
    alphas: list[torch.Tensor] = []
    handles = []
    state = {"start_position": 0}
    for layer in model.model.layers:
        width = layer.mlp.down_proj.in_features
        alpha = torch.ones(width, device=model.device, dtype=torch.float32,
                           requires_grad=True)

        def scale_tail(_module, inputs, current_alpha=alpha):
            activation = inputs[0]
            if scope == "global":
                return (
                    activation * current_alpha.to(activation.dtype),
                    *inputs[1:],
                )
            # Examples are processed one at a time. The forward input begins at
            # the raw prompt and ends with teacher-forced refusal-prefix tokens.
            start = state["start_position"]
            before = activation[:, :start]
            tail = activation[:, start:] * current_alpha.to(activation.dtype)
            return (torch.cat((before, tail), dim=1), *inputs[1:])

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(scale_tail))
        alphas.append(alpha)
    return alphas, handles, state


def build_summary(gradients: torch.Tensor, top_k: int):
    values = gradients.float()
    mean_g = values.mean(0)
    std_g = values.std(0, unbiased=False)
    mean_abs_g = values.abs().mean(0)
    positive_fraction = (values > 0).float().mean(0)
    sign_consistency = torch.maximum(positive_fraction, 1 - positive_fraction)
    summary = {
        "mean_g": mean_g,
        "abs_mean_g": mean_g.abs(),
        "mean_abs_g": mean_abs_g,
        "std_g": std_g,
        "positive_fraction": positive_fraction,
        "sign_consistency": sign_consistency,
    }
    width = mean_g.shape[1]
    indices = torch.topk(mean_g.abs().flatten(), min(top_k, mean_g.numel())).indices
    rows = []
    for rank, flat_index in enumerate(indices.tolist(), start=1):
        layer, neuron = divmod(flat_index, width)
        value = mean_g[layer, neuron].item()
        rows.append({
            "rank": rank,
            "layer": layer,
            "neuron": neuron,
            "direction": "supports_refusal" if value > 0 else "suppresses_refusal",
            "mean_g": value,
            "abs_mean_g": abs(value),
            "mean_abs_g": mean_abs_g[layer, neuron].item(),
            "std_g": std_g[layer, neuron].item(),
            "positive_fraction": positive_fraction[layer, neuron].item(),
            "sign_consistency": sign_consistency[layer, neuron].item(),
        })
    return summary, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "per_example_g.pt"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; pass --overwrite")

    all_rows = read_jsonl(args.prepared)
    rows = all_rows[args.offset:args.offset + args.limit]
    if len(rows) != args.limit:
        raise ValueError("Requested slice exceeds prepared data")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    target_ids = tokenizer.encode(args.refusal_phrase, add_special_tokens=False)
    if not target_ids:
        raise ValueError("Refusal phrase tokenized to zero tokens")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, device_map={"": 0},
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    alphas, handles, hook_state = attach_alpha_hooks(model, args.alpha_scope)

    gradients = []
    scores = []
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        prompt_ids = tokenizer.encode(str(row["goal"]), add_special_tokens=True)
        input_ids = torch.tensor(
            [prompt_ids + target_ids[:-1]], dtype=torch.long, device=model.device
        )
        # Hook only the final prompt position and subsequent target positions.
        hook_state["start_position"] = len(prompt_ids) - 1
        for alpha in alphas:
            alpha.grad = None

        logits = model(input_ids=input_ids, use_cache=False).logits[0]
        positions = torch.arange(
            len(prompt_ids) - 1, len(prompt_ids) - 1 + len(target_ids), device=model.device
        )
        targets = torch.tensor(target_ids, device=model.device)
        token_scores = torch.log_softmax(logits[positions].float(), dim=-1)[
            torch.arange(len(target_ids), device=model.device), targets
        ]
        score = token_scores.mean()
        score.backward()
        gradients.append(torch.stack([alpha.grad.detach().cpu() for alpha in alphas]).half())
        scores.append(score.detach().cpu())
        if index % 5 == 0 or index == len(rows):
            print(f"examples={index}/{len(rows)} elapsed_seconds={time.monotonic()-started:.1f}",
                  flush=True)

    for handle in handles:
        handle.remove()
    tensor = torch.stack(gradients)
    payload = {
        "ids": torch.tensor([int(row["id"]) for row in rows]),
        "mean_log_prob": torch.stack(scores).float(),
        "g": tensor,
    }
    torch.save(payload, output_path)
    summary, ranking = build_summary(tensor, args.top_k)
    torch.save(summary, args.output_dir / "g_summary.pt")
    ranking.to_csv(args.output_dir / "top_neurons.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model": str(args.model.resolve()),
        "prepared": str(args.prepared.resolve()),
        "offset": args.offset,
        "limit": args.limit,
        "prompt_format": "raw",
        "refusal_phrase": args.refusal_phrase,
        "target_token_ids": target_ids,
        "target_tokens": [tokenizer.decode([token]) for token in target_ids],
        "score": "mean teacher-forced log probability of refusal phrase",
        "alpha_scope": args.alpha_scope,
        "gradient_shape": list(tensor.shape),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
