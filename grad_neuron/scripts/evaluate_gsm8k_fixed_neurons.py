#!/usr/bin/env python3
"""Evaluate the fixed FFN-neuron controller on the ICLR GSM8K protocol."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_grad_safety_steering import attach_masks, configure_static


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("/workspace/xcy/dataset/shared/gsm8k/main"))
    parser.add_argument(
        "--ranking", type=Path,
        default=root / (
            "results/gradients/raw_refusal_advbench_rows100_299/top_neurons.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "results/capability/gsm8k_fixed_top25_strong_first100",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--epsilons", type=float, nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_number(value: str) -> str:
    value = value.replace(",", "").strip().rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return format(number, ".12g")


def extract_gsm_answer(text: str) -> Optional[str]:
    text = text.split("\nQuestion:", 1)[0]
    hashes = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if hashes:
        return normalize_number(hashes.group(1))
    values = NUMBER_RE.findall(text)
    return normalize_number(values[-1]) if values else None


def build_prompts(dataset, tokenizer) -> list[str]:
    prompts = []
    for record in dataset["test"]:
        content = (
            f"Question: {record['question']}\n"
            "Answer: Let's think step by step."
        )
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        ))
    return prompts


def text_diagnostics(text: str) -> dict[str, object]:
    words = re.findall(r"\b\w+\b", text.lower())
    grams = Counter(tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3)))
    maximum = max(grams.values(), default=0)
    return {
        "blank": not text.strip(),
        "word_count": len(words),
        "unique_word_ratio": len(set(words)) / len(words) if words else 0.0,
        "max_fourgram_count": maximum,
        "repetitive": maximum >= 5,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_condition(
    name: str, model, tokenizer, prompts: list[str], test, args: argparse.Namespace
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    started = time.monotonic()
    for start in range(0, args.limit, args.batch_size):
        indices = list(range(start, min(start + args.batch_size, args.limit)))
        inputs = tokenizer(
            [prompts[index] for index in indices],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )[:, input_width:]
        responses = tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for index, response, token_ids in zip(indices, responses, generated):
            prediction = extract_gsm_answer(response)
            answer = extract_gsm_answer(str(test[index]["answer"]))
            rows.append({
                "condition": name,
                "id": index,
                "question": test[index]["question"],
                "response": response,
                "prediction": prediction,
                "answer": answer,
                "correct": prediction is not None and prediction == answer,
                "response_token_count": int(
                    (token_ids != tokenizer.pad_token_id).sum().item()
                ),
                **text_diagnostics(response),
            })
        print(f"{name} saved {len(rows)}/{args.limit} "
              f"({time.monotonic()-started:.1f}s)", flush=True)

    summary = {
        "condition": name,
        "num_samples": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": 100 * sum(bool(row["correct"]) for row in rows) / len(rows),
        "extraction_failures": sum(row["prediction"] is None for row in rows),
        "blank_responses": sum(bool(row["blank"]) for row in rows),
        "repetitive_responses": sum(bool(row["repetitive"]) for row in rows),
        "median_response_tokens": float(
            pd.Series([row["response_token_count"] for row in rows]).median()
        ),
        "median_word_count": float(pd.Series([row["word_count"] for row in rows]).median()),
        "median_unique_word_ratio": float(
            pd.Series([row["unique_word_ratio"] for row in rows]).median()
        ),
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {summary_path}; pass --overwrite")
    if args.limit <= 0 or args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("limit, batch-size, and max-new-tokens must be positive")

    dataset = load_from_disk(str(args.dataset))
    if args.limit > len(dataset["test"]):
        raise ValueError("limit exceeds GSM8K test split")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = build_prompts(dataset, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, device_map={"": 0},
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()
    model.requires_grad_(False)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    masks, handles = attach_masks(model, "last", args.batch_size)
    ranking = pd.read_csv(args.ranking).sort_values("abs_mean_g", ascending=False)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    for mask in masks:
        mask.fill_(1)
    baseline_rows, baseline_summary = generate_condition(
        "baseline", model, tokenizer, prompts, dataset["test"], args
    )

    condition_rows = {"baseline": baseline_rows}
    summaries = [baseline_summary]
    for epsilon in args.epsilons:
        configure_static(masks, ranking, args.top_k, epsilon)
        tag = f"strength_{epsilon:g}".replace(".", "p")
        rows, summary = generate_condition(
            tag, model, tokenizer, prompts, dataset["test"], args
        )
        condition_rows[tag] = rows
        summaries.append(summary)
    for handle in handles:
        handle.remove()

    baseline_by_id = {int(row["id"]): row for row in baseline_rows}
    transitions = {}
    for tag, rows in condition_rows.items():
        write_jsonl(args.output_dir / f"{tag}_responses.jsonl", rows)
        if tag == "baseline":
            continue
        transitions[tag] = {
            "incorrect_to_correct": sum(
                not baseline_by_id[int(row["id"])]["correct"] and row["correct"]
                for row in rows
            ),
            "correct_to_incorrect": sum(
                baseline_by_id[int(row["id"])]["correct"] and not row["correct"]
                for row in rows
            ),
        }
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    payload = {
        "protocol": {
            "dataset": str(args.dataset.resolve()),
            "test_rows": f"first {args.limit}",
            "prompt_format": "Llama-3 chat template",
            "num_fewshot": 0,
            "prompt_suffix": "Answer: Let's think step by step.",
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "dtype": "bfloat16",
            "batch_size": args.batch_size,
            "answer_extraction": "ICLR reproduction: #### number else final number",
        },
        "controller": {
            "ranking": str(args.ranking.resolve()),
            "top_k": args.top_k,
            "strengths": args.epsilons,
            "positive_scale": "1 + strength",
            "negative_scale": "max(0, 1 - strength)",
            "scope": "final prefill position and cached decoding positions",
            "prompt_specific": False,
        },
        "results": summaries,
        "transitions_vs_baseline": transitions,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "selected_neurons.json").write_text(json.dumps({
        "neurons": [
            {"layer": int(row.layer), "neuron": int(row.neuron),
             "mean_g": float(row.mean_g),
             "sign": 1 if float(row.mean_g) > 0 else -1}
            for row in ranking.head(args.top_k).itertuples()
        ]
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
