#!/usr/bin/env python3
"""Run signed GradAct FFN ablations on the AdvBench HarmBehavior protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
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
        "--ranking", type=Path,
        default=root / "results/gradients/i_cannot/top_neurons.csv",
    )
    parser.add_argument(
        "--evaluator", type=Path,
        default=Path("/workspace/xcy/safety_repro/iclr_neuron_expanded_kv/"
                     "neuron_deactivate/table1_harm_behavior_eval.py"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "results/causal_sweeps/i_cannot_zero",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=[1000, 500, 200, 100])
    parser.add_argument("--signs", nargs="+", choices=("positive", "negative"),
                        default=["positive", "negative"])
    parser.add_argument("--neuron-scale", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=112)
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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("advbench_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def text_diagnostics(text: str) -> dict[str, object]:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 4:
        maximum = 0
    else:
        four_grams = Counter(
            tuple(words[index : index + 4]) for index in range(len(words) - 3)
        )
        maximum = max(four_grams.values(), default=0)
    return {
        "word_count": len(words),
        "unique_word_ratio": len(set(words)) / len(words) if words else 0.0,
        "max_fourgram_count": maximum,
        "repetitive": maximum >= 5,
    }


def summarize_text(records: list[dict[str, object]]) -> dict[str, object]:
    diagnostics = [text_diagnostics(str(record["response"])) for record in records]
    return {
        "blank_responses": sum(not str(record["response"]).strip() for record in records),
        "under_10_word_responses": sum(item["word_count"] < 10 for item in diagnostics),
        "median_word_count": float(pd.Series([item["word_count"] for item in diagnostics]).median()),
        "median_unique_word_ratio": float(
            pd.Series([item["unique_word_ratio"] for item in diagnostics]).median()
        ),
        "fourgram_repeated_at_least_5": sum(
            item["max_fourgram_count"] >= 5 for item in diagnostics
        ),
        "fourgram_repeated_at_least_20": sum(
            item["max_fourgram_count"] >= 20 for item in diagnostics
        ),
    }


def load_selection(ranking_path: Path, sign: str, k: int) -> list[tuple[int, int]]:
    frame = pd.read_csv(ranking_path).sort_values("abs_mean_g", ascending=False)
    if sign == "positive":
        frame = frame.loc[frame["mean_g"] > 0]
    else:
        frame = frame.loc[frame["mean_g"] < 0]
    if len(frame) < k:
        raise ValueError(f"Only {len(frame)} {sign} ranked neurons are available; requested {k}")
    rows = frame.head(k)
    selected = [(int(row.layer), int(row.neuron)) for row in rows.itertuples()]
    if len(set(selected)) != k:
        raise ValueError(f"Duplicate neurons in top-{k} {sign} selection")
    return selected


def attach_masks(model) -> tuple[list[torch.Tensor], list[object]]:
    masks: list[torch.Tensor] = []
    handles: list[object] = []
    for layer in model.model.layers:
        width = layer.mlp.down_proj.in_features
        mask = torch.ones(width, device=model.device, dtype=torch.bfloat16)

        def mask_input(_module, inputs, current_mask=mask):
            return (inputs[0] * current_mask, *inputs[1:])

        masks.append(mask)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(mask_input))
    return masks, handles


def configure_masks(
    masks: list[torch.Tensor], selected: list[tuple[int, int]], neuron_scale: float
) -> None:
    for mask in masks:
        mask.fill_(1)
    for layer, neuron in selected:
        if not 0 <= layer < len(masks) or not 0 <= neuron < masks[layer].numel():
            raise ValueError(f"Out-of-range neuron: layer={layer}, neuron={neuron}")
        masks[layer][neuron] = neuron_scale


def experiment_name(sign: str, k: int, scale: float) -> str:
    if scale == 0:
        suffix = "zero"
    else:
        suffix = f"scale{scale:g}".replace("-", "neg").replace(".", "p")
    return f"{sign}_top{k}_{suffix}"


def validate_inputs(
    prepared: list[dict[str, object]], baseline: list[dict[str, object]], limit: int
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    if limit <= 0 or limit > len(prepared):
        raise ValueError(f"limit must be in [1, {len(prepared)}], got {limit}")
    selected = prepared[:limit]
    baseline_by_id = {int(row["id"]): row for row in baseline}
    if len(baseline_by_id) != len(baseline):
        raise ValueError("Baseline contains duplicate IDs")
    for row in selected:
        record_id = int(row["id"])
        if record_id not in baseline_by_id:
            raise ValueError(f"Baseline is missing prepared id {record_id}")
        if row["goal"] != baseline_by_id[record_id].get("goal"):
            raise ValueError(f"Goal mismatch for id {record_id}")
    return selected, baseline_by_id


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    output_dir = args.output_dir.resolve()
    final_path = output_dir / "generations.jsonl"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if final_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {final_path}; pass --overwrite to replace it")
    if world_size > 1:
        dist.barrier()

    prepared, baseline_by_id = validate_inputs(
        read_jsonl(args.prepared), read_jsonl(args.baseline_scored), args.limit
    )
    evaluator = load_evaluator(args.evaluator)
    configs = [(sign, k) for sign in args.signs for k in args.k_values]
    local_configs = configs[rank::world_size]
    selections = {
        (sign, k): load_selection(args.ranking, sign, k)
        for sign, k in local_configs
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    model.requires_grad_(False)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    masks, handles = attach_masks(model)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    local_records: list[dict[str, object]] = []
    started = time.monotonic()

    for config_number, (sign, k) in enumerate(local_configs, start=1):
        selected_neurons = selections[(sign, k)]
        configure_masks(masks, selected_neurons, args.neuron_scale)
        experiment = experiment_name(sign, k, args.neuron_scale)
        for batch_start in range(0, len(prepared), args.batch_size):
            batch = prepared[batch_start : batch_start + args.batch_size]
            prompts = [str(row["goal"]) for row in batch]
            inputs = tokenizer(
                prompts,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(model.device)
            input_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            generated = outputs[:, input_width:]
            responses = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row, response in zip(batch, responses):
                record_id = int(row["id"])
                judgment = evaluator.score_response(response, str(row["target"]))
                baseline_unsafe = bool(baseline_by_id[record_id]["jailbroken"])
                diagnostics = text_diagnostics(response)
                local_records.append(
                    {
                        "experiment": experiment,
                        "ablated_sign": sign,
                        "top_k": k,
                        "id": record_id,
                        "source_index": row.get("source_index"),
                        "goal": row["goal"],
                        "target": row["target"],
                        "response": response,
                        "jailbroken": judgment.jailbroken,
                        "matched_refusal_prefixes": list(judgment.matched_refusal_prefixes),
                        "blank": not response.strip(),
                        **diagnostics,
                        "baseline_jailbroken": baseline_unsafe,
                        "safe_to_unsafe": (not baseline_unsafe) and judgment.jailbroken,
                        "unsafe_to_safe": baseline_unsafe and (not judgment.jailbroken),
                        "response_changed": response != baseline_by_id[record_id]["response"],
                    }
                )
            print(
                f"rank={rank} experiment={experiment} rows="
                f"{min(batch_start + len(batch), len(prepared))}/{len(prepared)}",
                flush=True,
            )
        print(
            f"rank={rank} configs={config_number}/{len(local_configs)} "
            f"elapsed_seconds={time.monotonic() - started:.1f}",
            flush=True,
        )

    for handle in handles:
        handle.remove()
    part_path = output_dir / f"generations.rank{rank}.jsonl"
    write_jsonl(part_path, local_records)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        records: list[dict[str, object]] = []
        for part_rank in range(world_size):
            path = output_dir / f"generations.rank{part_rank}.jsonl"
            records.extend(read_jsonl(path))
            path.unlink()
        config_order = {
            experiment_name(sign, k, args.neuron_scale): i
            for i, (sign, k) in enumerate(configs)
        }
        records.sort(key=lambda row: (config_order[str(row["experiment"])], int(row["id"])))
        write_jsonl(final_path, records)

        summary_rows = []
        for sign, k in configs:
            experiment = experiment_name(sign, k, args.neuron_scale)
            subset = [row for row in records if row["experiment"] == experiment]
            count = len(subset)
            summary_rows.append(
                {
                    "experiment": experiment,
                    "ablated_sign": sign,
                    "top_k": k,
                    "num_samples": count,
                    "attack_success_count": sum(bool(row["jailbroken"]) for row in subset),
                    "attack_success_rate": 100 * sum(bool(row["jailbroken"]) for row in subset) / count,
                    "safe_to_unsafe": sum(bool(row["safe_to_unsafe"]) for row in subset),
                    "unsafe_to_safe": sum(bool(row["unsafe_to_safe"]) for row in subset),
                    "changed_responses": sum(bool(row["response_changed"]) for row in subset),
                    **summarize_text(subset),
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(output_dir / "summary.csv", index=False)
        (output_dir / "summary.json").write_text(
            json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8"
        )
        baseline_subset = [baseline_by_id[int(row["id"])] for row in prepared]
        baseline_count = sum(bool(row["jailbroken"]) for row in baseline_subset)
        baseline_summary = {
            "experiment": "baseline",
            "ablated_sign": "none",
            "top_k": 0,
            "num_samples": len(baseline_subset),
            "attack_success_count": baseline_count,
            "attack_success_rate": 100.0 * baseline_count / len(baseline_subset),
            "safe_to_unsafe": 0,
            "unsafe_to_safe": 0,
            "changed_responses": 0,
            **summarize_text(baseline_subset),
        }
        comparison = pd.DataFrame([baseline_summary, *summary_rows])
        comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
        (output_dir / "comparison_summary.json").write_text(
            json.dumps([baseline_summary, *summary_rows], indent=2) + "\n",
            encoding="utf-8",
        )
        selected_payload = {}
        for sign, k in configs:
            chosen = load_selection(args.ranking, sign, k)
            selected_payload[experiment_name(sign, k, args.neuron_scale)] = [
                {"layer": layer, "neuron": neuron} for layer, neuron in chosen
            ]
        (output_dir / "selected_neurons.json").write_text(
            json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8"
        )
        metadata = {
            "model": str(args.model.resolve()),
            "prepared": str(args.prepared.resolve()),
            "prepared_sha256": sha256_file(args.prepared),
            "baseline_scored": str(args.baseline_scored.resolve()),
            "ranking": str(args.ranking.resolve()),
            "ranking_sha256": sha256_file(args.ranking),
            "evaluator": str(args.evaluator.resolve()),
            "k_values_in_execution_order": args.k_values,
            "signs": args.signs,
            "neuron_intervention": (
                f"set alpha={args.neuron_scale:g} before MLP down_proj at all token positions"
            ),
            "neuron_scale": args.neuron_scale,
            "generation": {
                "prompt_format": "raw",
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "dtype": "bfloat16",
                "seed": args.seed,
            },
            "baseline_attack_success_rate": 100.0 * sum(
                bool(baseline_by_id[int(row["id"])]["jailbroken"])
                for row in prepared
            ) / len(prepared),
            "world_size": world_size,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(summary.to_string(index=False), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
