"""Shared generation pipeline for the Table 1 baseline and deactivation runners."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

try:
    from . import table1_harm_behavior_eval as evaluator
except ImportError:
    import table1_harm_behavior_eval as evaluator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct")
DEFAULT_PREPARED = SCRIPT_DIR / "evaluation_data" / "harm_behavior_first_100.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "evaluation_outputs"


def add_common_arguments(parser: argparse.ArgumentParser, output_stem: str) -> None:
    run_dir = DEFAULT_OUTPUT_DIR / output_stem
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--prepared",
        "--prompt-jsonl",
        "--prompts",
        dest="prepared",
        type=Path,
        default=DEFAULT_PREPARED,
        help=(
            "Prepared evaluation JSONL path. Each record must contain integer 'id' "
            "and string 'goal' and 'target' fields."
        ),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=run_dir / "responses.jsonl",
        help="Resumable JSONL containing generation text and per-response judgments.",
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=run_dir / "scored.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=run_dir / "summary.json",
    )
    parser.add_argument(
        "--run-metadata",
        type=Path,
        default=run_dir / "run.json",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--prompt-format",
        choices=("chat", "raw"),
        default="chat",
        help="Use the model chat template or tokenize each goal verbatim (default: chat).",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Greedy decoding is the reproducible default.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a matching partial response JSONL (default: true).",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Process at most this many pending samples, leaving a resumable partial run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, tokenizer, configuration, and variant settings without loading weights.",
    )


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _append_jsonl(handle, record: dict[str, object]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _load_prepared(path: Path) -> list[dict[str, object]]:
    records = evaluator.read_jsonl(path)
    indexed = evaluator.index_unique(records, path)
    ordered = [indexed[record_id] for record_id in sorted(indexed)]
    for record in ordered:
        if not isinstance(record.get("goal"), str) or not isinstance(record.get("target"), str):
            raise ValueError(f"Prepared record {record.get('id')} needs string goal and target")
    return ordered


def _load_existing_responses(
    path: Path, expected_ids: set[int], fingerprint: str, resume: bool
) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    if not resume:
        raise ValueError(f"Response file already exists and --no-resume was used: {path}")
    if path.stat().st_size == 0:
        return {}
    records = evaluator.read_jsonl(path)
    indexed = evaluator.index_unique(records, path)
    extras = sorted(set(indexed) - expected_ids)
    if extras:
        raise ValueError(f"Existing response file contains unexpected IDs: {extras[:10]}")
    for record_id, record in indexed.items():
        if record.get("run_fingerprint") != fingerprint:
            raise ValueError(
                f"Existing response id {record_id} belongs to a different run configuration"
            )
        if not isinstance(record.get("response"), str):
            raise ValueError(f"Existing response id {record_id} is not a string")
    return indexed


def _generated_token_count(token_ids, eos_token_ids: set[int]) -> int:
    for index, token_id in enumerate(token_ids.tolist(), start=1):
        if token_id in eos_token_ids:
            return index
    return len(token_ids)


def _build_prompts(
    tokenizer, records: Sequence[dict[str, object]], prompt_format: str
) -> list[str]:
    if prompt_format == "raw":
        return [str(record["goal"]) for record in records]
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["goal"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for record in records
    ]


def run_generation(
    args: argparse.Namespace,
    variant: str,
    variant_factory: Callable[[object], tuple[dict[str, object], dict[str, object]]],
) -> int:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch size and max new tokens must be positive")
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError("stop-after must be positive")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("temperature must be positive when sampling")

    model_path = args.model.resolve()
    prepared_path = args.prepared.resolve()
    if not model_path.is_dir():
        raise ValueError(f"Model directory does not exist: {model_path}")

    prepared = _load_prepared(prepared_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = _build_prompts(tokenizer, prepared, args.prompt_format)

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    variant_generate_kwargs, variant_metadata = variant_factory(config)
    generation_settings = {
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "prompt_format": args.prompt_format,
        "padding_side": tokenizer.padding_side,
        "pad_token_id": tokenizer.pad_token_id,
        "prompt_construction": (
            "tokenizer.apply_chat_template(user goal, add_generation_prompt=True)"
            if args.prompt_format == "chat"
            else "goal string verbatim"
        ),
        "add_special_tokens": args.prompt_format == "raw",
        "input_schema": "prepared_advbench",
    }
    fingerprint_input = {
        "variant": variant,
        "model_path": str(model_path),
        "model_config_sha256": evaluator.sha256_file(model_path / "config.json"),
        "prepared_sha256": evaluator.sha256_file(prepared_path),
        "generation": generation_settings,
        "variant_settings": variant_metadata,
    }
    fingerprint = _json_hash(fingerprint_input)
    run_metadata = {
        **fingerprint_input,
        "run_fingerprint": fingerprint,
        "python": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "transformers_path": transformers.__file__,
        "num_samples": len(prepared),
        "responses_path": str(args.responses.resolve()),
        "scored_path": str(args.scored.resolve()),
        "summary_path": str(args.summary.resolve()),
    }

    if args.dry_run:
        print(json.dumps(run_metadata, indent=2))
        return 0

    if args.run_metadata.exists():
        old_metadata = json.loads(args.run_metadata.read_text(encoding="utf-8"))
        if old_metadata.get("run_fingerprint") != fingerprint:
            raise ValueError(f"Run metadata has a different configuration: {args.run_metadata}")
    else:
        _write_json(args.run_metadata, run_metadata)

    expected_ids = {int(record["id"]) for record in prepared}
    completed = _load_existing_responses(args.responses, expected_ids, fingerprint, args.resume)
    if set(completed) == expected_ids:
        print(f"All {len(prepared)} responses already exist; rescoring without loading model weights.")
        score_args = argparse.Namespace(
            prepared=args.prepared,
            responses=args.responses,
            output=args.scored,
            summary=args.summary,
        )
        evaluator.score(score_args)
        return 0
    pending_pairs = [
        (record, prompt)
        for record, prompt in zip(prepared, prompts)
        if int(record["id"]) not in completed
    ]
    if args.stop_after is not None:
        pending_pairs = pending_pairs[: args.stop_after]

    dtype = getattr(torch, args.dtype)
    print(
        f"Loading {variant} model with global batch size {args.batch_size}: {model_path}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=args.device_map,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    if not args.do_sample:
        # The local Llama-3 generation_config.json contains sampling defaults.
        # Clear them for an explicit greedy run and avoid misleading warnings.
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    input_device = model.get_input_embeddings().weight.device
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    generation_kwargs: dict[str, object] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        **variant_generate_kwargs,
    }
    if args.do_sample:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_token_ids = {eos_ids}
    else:
        eos_token_ids = set(eos_ids or [])

    args.responses.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    processed_now = 0
    with args.responses.open("a", encoding="utf-8") as response_handle:
        for batch_start in range(0, len(pending_pairs), args.batch_size):
            batch_pairs = pending_pairs[batch_start : batch_start + args.batch_size]
            batch_records = [pair[0] for pair in batch_pairs]
            batch_prompts = [pair[1] for pair in batch_pairs]
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=False,
                # The rendered Llama-3 chat template already begins with BOS.
                add_special_tokens=args.prompt_format == "raw",
                return_tensors="pt",
            ).to(input_device)
            input_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(**inputs, **generation_kwargs)
            generated = outputs[:, input_width:]
            response_texts = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for index, (record, prompt, response) in enumerate(
                zip(batch_records, batch_prompts, response_texts)
            ):
                judgment = evaluator.score_response(response, str(record["target"]))
                output_record = {
                    "id": record["id"],
                    "source_index": record.get("source_index"),
                    "goal": record["goal"],
                    "target": record["target"],
                    "prompt": prompt,
                    "response": response,
                    "jailbroken": judgment.jailbroken,
                    "target_exact_match": judgment.target_exact_match,
                    "matched_refusal_prefixes": list(judgment.matched_refusal_prefixes),
                    "prompt_token_count": int(inputs["attention_mask"][index].sum().item()),
                    "generated_token_count": _generated_token_count(
                        generated[index], eos_token_ids
                    ),
                    "variant": variant,
                    "run_fingerprint": fingerprint,
                }
                _append_jsonl(response_handle, output_record)
                completed[int(record["id"])] = output_record
                processed_now += 1

            elapsed = time.monotonic() - started
            print(
                f"Saved {len(completed)}/{len(prepared)} responses "
                f"({processed_now} this invocation, {elapsed:.1f}s)",
                flush=True,
            )

    if set(completed) != expected_ids:
        print(
            f"Partial run saved at {args.responses}: {len(completed)}/{len(prepared)} complete. "
            "Re-run the same command to resume.",
            flush=True,
        )
        return 0

    score_args = argparse.Namespace(
        prepared=args.prepared,
        responses=args.responses,
        output=args.scored,
        summary=args.summary,
    )
    evaluator.score(score_args)
    return 0
