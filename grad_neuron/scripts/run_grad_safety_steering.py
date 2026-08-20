#!/usr/bin/env python3
"""Generate raw AdvBench responses with small signed neuron scaling."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
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
    parser.add_argument(
        "--baseline-scored", type=Path,
        default=Path("/workspace/xcy/safety_repro/iclr_neuron_expanded_kv/"
                     "neuron_deactivate/evaluation_outputs/table2_sn_tune/baseline/"
                     "harm_behavior/scored.jsonl"),
    )
    parser.add_argument(
        "--evaluator", type=Path,
        default=Path("/workspace/xcy/safety_repro/iclr_neuron_expanded_kv/"
                     "neuron_deactivate/table1_harm_behavior_eval.py"),
    )
    parser.add_argument("--ranking", type=Path,
                        default=root / (
                            "results/gradients/raw_refusal_advbench_rows100_299/"
                            "top_neurons.csv"
                        ))
    parser.add_argument(
        "--per-example-gradients", type=Path, default=None,
        help="Optional per_example_g.pt; selects each prompt's own top-K signed neurons",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=root / "results/safety_steering/fixed_top25_disjoint_advbench/run")
    parser.add_argument("--k-values", type=int, nargs="+", default=[25])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0])
    parser.add_argument("--scope", choices=("last", "all"), default="last")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("raw_safety_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def diagnostics(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    grams = Counter(tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3)))
    maximum = max(grams.values(), default=0)
    return {
        "word_count": len(words),
        "unique_word_ratio": len(set(words)) / len(words) if words else 0.0,
        "max_fourgram_count": maximum,
        "repetitive": maximum >= 5,
    }


def attach_masks(model, scope: str, batch_size: int):
    masks = []
    handles = []
    for layer in model.model.layers:
        mask = torch.ones(batch_size, layer.mlp.down_proj.in_features,
                          device=model.device, dtype=torch.bfloat16)

        def scale_input(_module, inputs, current_mask=mask):
            activation = inputs[0]
            batch_mask = current_mask[:activation.shape[0]].unsqueeze(1)
            if scope == "all":
                scaled = activation * batch_mask
            else:
                # Prefill: affect only the final raw-prompt token. Decode with
                # KV cache: sequence length is one, so each new token is affected.
                scaled = torch.cat(
                    (activation[:, :-1], activation[:, -1:] * batch_mask), dim=1
                )
            return (scaled, *inputs[1:])

        masks.append(mask)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(scale_input))
    return masks, handles


def configure_static(masks, ranking: pd.DataFrame, k: int, epsilon: float):
    for mask in masks:
        mask.fill_(1)
    chosen = ranking.head(k)
    for row in chosen.itertuples():
        direction = 1.0 if float(row.mean_g) > 0 else -1.0
        masks[int(row.layer)][:, int(row.neuron)] = max(
            0.0, 1.0 + epsilon * direction
        )
    return chosen


def configure_dynamic(masks, gradients: torch.Tensor, k: int, epsilon: float):
    """Configure a different signed top-K mask for each row in a batch."""
    batch_size, num_layers, width = gradients.shape
    for mask in masks:
        mask.fill_(1)
    flat = gradients.float().reshape(batch_size, -1)
    indices = torch.topk(flat.abs(), k, dim=1).indices
    signs = torch.gather(flat, 1, indices).sign()
    for batch_index in range(batch_size):
        for flat_index, sign in zip(indices[batch_index].tolist(),
                                    signs[batch_index].tolist()):
            layer, neuron = divmod(flat_index, width)
            masks[layer][batch_index, neuron] = max(0.0, 1.0 + epsilon * sign)


def exp_name(k: int, epsilon: float, scope: str, dynamic: bool = False):
    eps = f"{epsilon:g}".replace(".", "p")
    prefix = "dynamic_signed" if dynamic else "signed"
    return f"{prefix}_top{k}_eps{eps}_{scope}"


def summarize(rows):
    count = len(rows)
    return {
        "num_samples": count,
        "attack_success_count": sum(bool(row["jailbroken"]) for row in rows),
        "attack_success_rate": 100 * sum(bool(row["jailbroken"]) for row in rows) / count,
        "safe_to_unsafe": sum(bool(row["safe_to_unsafe"]) for row in rows),
        "unsafe_to_safe": sum(bool(row["unsafe_to_safe"]) for row in rows),
        "blank_responses": sum(bool(row["blank"]) for row in rows),
        "repetitive_responses": sum(bool(row["repetitive"]) for row in rows),
        "median_word_count": float(pd.Series([row["word_count"] for row in rows]).median()),
        "median_unique_word_ratio": float(
            pd.Series([row["unique_word_ratio"] for row in rows]).median()
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "generations.jsonl"
    if final_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {final_path}; pass --overwrite")

    prepared_all = read_jsonl(args.prepared)
    prepared = prepared_all[args.offset:args.offset + args.limit]
    if len(prepared) != args.limit:
        raise ValueError("Requested slice exceeds prepared data")
    baseline_by_id = {int(row["id"]): row for row in read_jsonl(args.baseline_scored)}
    evaluator = load_evaluator(args.evaluator)
    ranking = pd.read_csv(args.ranking).sort_values("abs_mean_g", ascending=False)
    if max(args.k_values) > len(ranking):
        raise ValueError("Ranking does not contain enough neurons")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, device_map={"": 0},
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()
    model.requires_grad_(False)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    masks, handles = attach_masks(model, args.scope, args.batch_size)

    dynamic_by_id = None
    if args.per_example_gradients is not None:
        gradient_payload = torch.load(args.per_example_gradients, map_location="cpu",
                                      weights_only=True)
        id_key = "ids" if "ids" in gradient_payload else "source_indices"
        dynamic_by_id = {
            int(record_id): gradient_payload["g"][index]
            for index, record_id in enumerate(gradient_payload[id_key].tolist())
        }
        missing = [int(row["id"]) for row in prepared if int(row["id"]) not in dynamic_by_id]
        if missing:
            raise ValueError(f"Per-example gradients missing ids: {missing[:10]}")

    configs = [(k, eps) for k in args.k_values for eps in args.epsilons]
    records = []
    started = time.monotonic()
    for config_index, (k, epsilon) in enumerate(configs, start=1):
        if dynamic_by_id is None:
            configure_static(masks, ranking, k, epsilon)
        experiment = exp_name(k, epsilon, args.scope, dynamic_by_id is not None)
        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start:start + args.batch_size]
            if dynamic_by_id is not None:
                batch_gradients = torch.stack([
                    dynamic_by_id[int(row["id"])] for row in batch
                ])
                configure_dynamic(masks, batch_gradients, k, epsilon)
            inputs = tokenizer([str(row["goal"]) for row in batch], padding=True,
                               add_special_tokens=True, return_tensors="pt").to(model.device)
            input_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, use_cache=True,
                )
            responses = tokenizer.batch_decode(
                outputs[:, input_width:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row, response in zip(batch, responses):
                record_id = int(row["id"])
                judgment = evaluator.score_response(response, str(row["target"]))
                baseline_unsafe = bool(baseline_by_id[record_id]["jailbroken"])
                records.append({
                    "experiment": experiment,
                    "top_k": k,
                    "epsilon": epsilon,
                    "scope": args.scope,
                    "id": record_id,
                    "goal": row["goal"],
                    "target": row["target"],
                    "response": response,
                    "jailbroken": judgment.jailbroken,
                    "matched_refusal_prefixes": list(judgment.matched_refusal_prefixes),
                    "baseline_jailbroken": baseline_unsafe,
                    "safe_to_unsafe": (not baseline_unsafe) and judgment.jailbroken,
                    "unsafe_to_safe": baseline_unsafe and not judgment.jailbroken,
                    "blank": not response.strip(),
                    **diagnostics(response),
                })
        print(f"configs={config_index}/{len(configs)} experiment={experiment} "
              f"elapsed_seconds={time.monotonic()-started:.1f}", flush=True)

    for handle in handles:
        handle.remove()
    write_jsonl(final_path, records)
    summaries = []
    baseline_slice = [baseline_by_id[int(row["id"])] for row in prepared]
    summaries.append({
        "experiment": "baseline", "top_k": 0, "epsilon": 0.0, "scope": "none",
        "num_samples": len(baseline_slice),
        "attack_success_count": sum(bool(row["jailbroken"]) for row in baseline_slice),
        "attack_success_rate": 100 * sum(bool(row["jailbroken"]) for row in baseline_slice)
                               / len(baseline_slice),
        "safe_to_unsafe": 0, "unsafe_to_safe": 0,
        "blank_responses": sum(not str(row["response"]).strip() for row in baseline_slice),
        "repetitive_responses": sum(diagnostics(str(row["response"]))["repetitive"]
                                    for row in baseline_slice),
        "median_word_count": float(pd.Series([
            diagnostics(str(row["response"]))["word_count"] for row in baseline_slice
        ]).median()),
        "median_unique_word_ratio": float(pd.Series([
            diagnostics(str(row["response"]))["unique_word_ratio"] for row in baseline_slice
        ]).median()),
    })
    for k, epsilon in configs:
        experiment = exp_name(k, epsilon, args.scope, dynamic_by_id is not None)
        subset = [row for row in records if row["experiment"] == experiment]
        summaries.append({"experiment": experiment, "top_k": k, "epsilon": epsilon,
                          "scope": args.scope, **summarize(subset)})
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    selected = {}
    for k in args.k_values:
        selected[str(k)] = [
            {"layer": int(row.layer), "neuron": int(row.neuron),
             "mean_g": float(row.mean_g)}
            for row in ranking.head(k).itertuples()
        ]
    (args.output_dir / "selected_neurons.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model": str(args.model.resolve()), "prepared": str(args.prepared.resolve()),
        "baseline_scored": str(args.baseline_scored.resolve()),
        "ranking": str(args.ranking.resolve()), "offset": args.offset, "limit": args.limit,
        "k_values": args.k_values, "epsilons": args.epsilons, "scope": args.scope,
        "intervention": (
            "alpha_ij = max(0, 1 + epsilon * sign(per-prompt gradient))"
            if dynamic_by_id is not None
            else "alpha_j = max(0, 1 + epsilon * sign(mean gradient))"
        ),
        "per_example_gradients": (
            str(args.per_example_gradients.resolve()) if args.per_example_gradients else None
        ),
        "selection_mode": "per-prompt" if dynamic_by_id is not None else "global-mean",
        "prompt_format": "raw", "dtype": "bfloat16", "do_sample": False,
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
