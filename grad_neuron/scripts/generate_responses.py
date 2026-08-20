#!/usr/bin/env python3
"""Generate chat-model responses for a CSV of prompts, optionally with torchrun."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--prompt-column", default="goal")
    parser.add_argument("--response-column", default="model_response")
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def make_chat(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    output_path = Path(args.output_csv)

    if rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output_path}; pass --overwrite to replace it")
    if world_size > 1:
        dist.barrier()

    input_path = Path(args.input_csv)
    if input_path.suffix == ".jsonl":
        source = pd.read_json(input_path, lines=True)
    else:
        source = pd.read_csv(input_path)
    if args.prompt_column not in source.columns:
        raise KeyError(f"Missing prompt column {args.prompt_column!r}; found {list(source.columns)}")
    stop = min(args.start_index + args.num_examples, len(source))
    selected = source.iloc[args.start_index:stop].copy()
    if "source_index" not in selected.columns:
        selected.insert(0, "source_index", selected.index)
    local_rows = selected.iloc[rank::world_size]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
    )
    model.eval()
    if args.temperature == 0:
        # Some checkpoints persist sampling-only values in generation_config.
        # Clear them so deterministic runs do not emit misleading warnings.
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    generated: list[dict[str, object]] = []
    started = time.perf_counter()

    for offset in range(0, len(local_rows), args.batch_size):
        batch = local_rows.iloc[offset : offset + args.batch_size]
        if args.prompt_format == "chat":
            chats = [make_chat(str(prompt), args.system_prompt)
                     for prompt in batch[args.prompt_column]]
            texts = [
                tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                for chat in chats
            ]
        else:
            if args.system_prompt:
                raise ValueError("--system-prompt is only valid with --prompt-format chat")
            texts = [str(prompt) for prompt in batch[args.prompt_column]]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        do_sample = args.temperature > 0
        generation_args: dict[str, object] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_args.update(temperature=args.temperature, top_p=args.top_p)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_args)
        new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for (_, row), response, token_ids in zip(batch.iterrows(), responses, new_ids):
            record = row.to_dict()
            record[args.response_column] = response.strip()
            record["response_token_count"] = int(
                (token_ids != tokenizer.pad_token_id).sum().item()
            )
            generated.append(record)

        print(
            f"rank={rank} generated={min(offset + len(batch), len(local_rows))}/{len(local_rows)}",
            flush=True,
        )

    part_path = output_path.with_suffix(output_path.suffix + f".rank{rank}.jsonl")
    with part_path.open("w", encoding="utf-8") as handle:
        for record in generated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        records: list[dict[str, object]] = []
        for part_rank in range(world_size):
            path = output_path.with_suffix(output_path.suffix + f".rank{part_rank}.jsonl")
            with path.open(encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle)
            path.unlink()
        records.sort(key=lambda row: int(row["source_index"]))
        if output_path.suffix == ".jsonl":
            with output_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            fieldnames = list(records[0]) if records else [
                "source_index", *source.columns, args.response_column
            ]
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
        elapsed = time.perf_counter() - started
        print(f"saved={output_path} rows={len(records)} elapsed_seconds={elapsed:.1f}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
