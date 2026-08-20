#!/usr/bin/env python3
"""Evaluate the Llama-3 Table 1 MMLU and GSM8K capability rows."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
import transformers
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationMixin

from run_table1_deactivated import DEFAULT_NEURONS, STRUCTURES, build_deactivation_factory


EXPECTED_ENV = Path("/workspace/xcy/miniconda3/envs/iclr_neuron_deactivation")
DEFAULT_MODEL = Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct")
DEFAULT_DATA_ROOT = Path("/workspace/xcy/dataset/shared")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "evaluation_outputs" / "table1_capability"
CHOICES = ("A", "B", "C", "D")


def verify_environment() -> None:
    transformers_path = Path(transformers.__file__).resolve()
    if Path(sys.prefix).resolve() != EXPECTED_ENV or EXPECTED_ENV not in transformers_path.parents:
        raise RuntimeError(f"Run with {EXPECTED_ENV}/bin/python; current Python is {sys.executable}")
    if "activate_keys_fwd_up_set" not in inspect.signature(GenerationMixin.generate).parameters:
        raise RuntimeError("The deactivation Transformers overlay is not installed")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def no_op_factory(config) -> tuple[dict[str, object], dict[str, object]]:
    layers = int(config.num_hidden_layers)
    empty = [{layer: () for layer in range(layers)} for _ in STRUCTURES]
    kwargs = {
        "activate_keys_fwd_up_set": empty[0],
        "activate_keys_fwd_down_set": empty[1],
        "activate_keys_q_set": empty[2],
        "activate_keys_k_set": empty[3],
        "activate_keys_v_set": empty[4],
        "under_layer": layers,
        "gen_layer": 0,
        "atten_number": 0,
        "ffn_number": 0,
        "whether_under": True,
        "whether_reason": False,
        "whether_gen": False,
        "whether_under_fwd": True,
        "whether_reason_fwd": False,
        "whether_gen_fwd": False,
    }
    return kwargs, {"deactivation": False, "deactivation_mode": "origin"}


def variant_factory(args: argparse.Namespace, config):
    if args.variant == "origin":
        return no_op_factory(config)
    selection_args = Namespace(
        deact_rate=args.deact_rate,
        deact_mode=args.variant,
        seed=args.seed,
        neurons=args.neurons if args.variant == "sn" else None,
        layer_scope="all",
        layer_cutoff=None,
    )
    return build_deactivation_factory(selection_args)(config)


def append_jsonl(handle, record: dict[str, object]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def load_completed(path: Path, fingerprint: str) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    completed: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id")
            if not isinstance(record_id, int) or record_id in completed:
                raise ValueError(f"Invalid or duplicate id at {path}:{line_number}")
            if record.get("run_fingerprint") != fingerprint:
                raise ValueError(f"Fingerprint mismatch at {path}:{line_number}")
            completed[record_id] = record
    return completed


def format_mmlu_question(record: dict[str, object], include_answer: bool) -> str:
    lines = [str(record["question"])]
    lines.extend(f"{letter}. {choice}" for letter, choice in zip(CHOICES, record["choices"]))
    suffix = CHOICES[int(record["answer"])] if include_answer else ""
    lines.append(f"Answer: {suffix}")
    return "\n".join(lines)


def mmlu_prompts(dataset) -> list[str]:
    dev_by_subject: dict[str, list[dict[str, object]]] = {}
    for record in dataset["dev"]:
        dev_by_subject.setdefault(record["subject"], []).append(record)
    prompts: list[str] = []
    for record in dataset["test"]:
        subject = record["subject"]
        header = (
            "The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n"
        )
        demonstrations = "\n\n".join(
            format_mmlu_question(example, True) for example in dev_by_subject[subject][:5]
        )
        prompts.append(header + demonstrations + "\n\n" + format_mmlu_question(record, False))
    return prompts


def evaluate_mmlu(
    args: argparse.Namespace,
    model,
    tokenizer,
    generation_kwargs: dict[str, object],
    fingerprint: str,
    responses_path: Path,
) -> dict[str, object]:
    dataset = load_from_disk(str(args.data_root / "mmlu" / "all"))
    test = dataset["test"]
    prompts = mmlu_prompts(dataset)
    completed = load_completed(responses_path, fingerprint)
    sample_count = min(len(test), args.limit) if args.limit is not None else len(test)
    expected_ids = set(range(sample_count))
    if set(completed) - expected_ids:
        raise ValueError("MMLU response file contains out-of-range IDs")

    option_ids = []
    for choice in CHOICES:
        encoded = tokenizer.encode(" " + choice, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"MMLU option {choice!r} is not one token: {encoded}")
        option_ids.append(encoded[0])

    pending = [index for index in range(sample_count) if index not in completed]
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with responses_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start : start + args.batch_size]
            inputs = tokenizer(
                [prompts[index] for index in indices],
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(model.get_input_embeddings().weight.device)
            with torch.inference_mode():
                logits = model(**inputs, **generation_kwargs).logits[:, -1, option_ids]
            predictions = logits.argmax(dim=-1).tolist()
            for index, prediction in zip(indices, predictions):
                record = test[index]
                output = {
                    "id": index,
                    "subject": record["subject"],
                    "prediction": int(prediction),
                    "prediction_letter": CHOICES[prediction],
                    "answer": int(record["answer"]),
                    "correct": int(prediction) == int(record["answer"]),
                    "run_fingerprint": fingerprint,
                }
                append_jsonl(handle, output)
                completed[index] = output
            print(
                f"MMLU saved {len(completed)}/{sample_count} ({time.monotonic()-started:.1f}s)",
                flush=True,
            )

    by_subject: dict[str, list[bool]] = {}
    for record in completed.values():
        by_subject.setdefault(str(record["subject"]), []).append(bool(record["correct"]))
    correct = sum(bool(record["correct"]) for record in completed.values())
    return {
        "benchmark": "MMLU",
        "num_fewshot": 5,
        "num_samples": len(completed),
        "correct": correct,
        "accuracy": 100.0 * correct / len(completed),
        "accuracy_by_subject": {
            subject: 100.0 * sum(values) / len(values)
            for subject, values in sorted(by_subject.items())
        },
    }


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


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


def gsm8k_prompts(dataset) -> list[str]:
    demonstrations = "\n\n".join(
        f"Question: {record['question']}\nAnswer: {record['answer']}"
        for record in dataset["train"].select(range(5))
    )
    return [
        demonstrations + f"\n\nQuestion: {record['question']}\nAnswer:"
        for record in dataset["test"]
    ]


def evaluate_gsm8k(
    args: argparse.Namespace,
    model,
    tokenizer,
    generation_kwargs: dict[str, object],
    fingerprint: str,
    responses_path: Path,
) -> dict[str, object]:
    dataset = load_from_disk(str(args.data_root / "gsm8k" / "main"))
    test = dataset["test"]
    prompts = gsm8k_prompts(dataset)
    completed = load_completed(responses_path, fingerprint)
    sample_count = min(len(test), args.limit) if args.limit is not None else len(test)
    expected_ids = set(range(sample_count))
    if set(completed) - expected_ids:
        raise ValueError("GSM8K response file contains out-of-range IDs")
    pending = [index for index in range(sample_count) if index not in completed]

    generate_kwargs = {
        **generation_kwargs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with responses_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start : start + args.batch_size]
            inputs = tokenizer(
                [prompts[index] for index in indices],
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(model.get_input_embeddings().weight.device)
            input_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generated = model.generate(**inputs, **generate_kwargs)[:, input_width:]
            texts = tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for index, response in zip(indices, texts):
                prediction = extract_gsm_answer(response)
                answer = extract_gsm_answer(str(test[index]["answer"]))
                output = {
                    "id": index,
                    "response": response,
                    "prediction": prediction,
                    "answer": answer,
                    "correct": prediction is not None and prediction == answer,
                    "run_fingerprint": fingerprint,
                }
                append_jsonl(handle, output)
                completed[index] = output
            print(
                f"GSM8K saved {len(completed)}/{sample_count} ({time.monotonic()-started:.1f}s)",
                flush=True,
            )

    correct = sum(bool(record["correct"]) for record in completed.values())
    extraction_failures = sum(record["prediction"] is None for record in completed.values())
    return {
        "benchmark": "GSM8K",
        "num_fewshot": 5,
        "num_samples": len(completed),
        "correct": correct,
        "accuracy": 100.0 * correct / len(completed),
        "extraction_failures": extraction_failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("mmlu", "gsm8k"))
    parser.add_argument("variant", choices=("origin", "random", "sn"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--neurons", type=Path, default=DEFAULT_NEURONS)
    parser.add_argument("--deact-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N test rows (for smoke tests).")
    return parser


def main() -> int:
    verify_environment()
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("batch-size, max-new-tokens, and limit must be positive")
    args.model = args.model.resolve()
    args.data_root = args.data_root.resolve()
    args.neurons = args.neurons.resolve()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    generation_kwargs, variant_metadata = variant_factory(args, config)
    run_config = {
        "task": args.task,
        "variant": args.variant,
        "model": str(args.model),
        "model_config_sha256": file_sha256(args.model / "config.json"),
        "data_root": str(args.data_root),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens if args.task == "gsm8k" else None,
        "limit": args.limit,
        "prompt_format": "raw capability benchmark prompt",
        "variant_settings": variant_metadata,
    }
    fingerprint = json_hash(run_config)
    rate_tag = format(args.deact_rate, ".12g").replace(".", "p")
    variant_tag = args.variant if args.variant == "origin" else f"{args.variant}_rate{rate_tag}"
    run_dir = args.output_root / args.task / variant_tag
    responses_path = run_dir / "responses.jsonl"
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "run.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **run_config,
        "run_fingerprint": fingerprint,
        "python": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "responses_path": str(responses_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("run_fingerprint") != fingerprint:
            raise SystemExit(f"Existing run has a different fingerprint: {metadata_path}")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Loading {args.variant} model for {args.task}: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.task == "mmlu":
        summary = evaluate_mmlu(
            args, model, tokenizer, generation_kwargs, fingerprint, responses_path
        )
    else:
        summary = evaluate_gsm8k(
            args, model, tokenizer, generation_kwargs, fingerprint, responses_path
        )
    summary.update(variant=args.variant, run_fingerprint=fingerprint)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
