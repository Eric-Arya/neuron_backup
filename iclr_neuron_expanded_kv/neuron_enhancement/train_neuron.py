#!/usr/bin/env python3
"""Train Llama 3 by updating only detected safety-neuron parameter slices."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from trl import SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct")
DEFAULT_DATASET = Path(
    "/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/training/"
    "circuit_breakers_train.json"
)
DEFAULT_NEURONS = (
    PROJECT_ROOT
    / "neuron_detection/output_neurons/"
    "Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_"
    "kvexpanded_raw_vpool_200.txt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "neuron_enhancement/outputs/llama3_sn_tune_raw_50"
STRUCTURES = ("fwd_up", "fwd_down", "q", "k", "v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--neurons", type=Path, default=DEFAULT_NEURONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=Path("/workspace/xcy/dataset/.cache/sn_tune"))
    parser.add_argument("--data-limit", type=int, default=50)
    parser.add_argument("--dataset-start", type=int, default=0)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="llama3_output")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--neuron-cap", type=int, default=100)
    parser.add_argument(
        "--selection-method",
        choices=("set-order", "file-order"),
        default="set-order",
        help=(
            "Choose capped neurons using legacy Python-set order or preserve ranked list order "
            "from the detector file."
        ),
    )
    parser.add_argument(
        "--kv-index-space", choices=("expanded", "physical"), default="expanded"
    )
    parser.add_argument(
        "--kv-map",
        choices=("head-aware", "paper-code-divide"),
        default="head-aware",
        help="Map expanded K/V coordinates to physical projection rows.",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prepare-for-kbit-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply PEFT's preparation helper. On this non-quantized checkpoint it reproduces "
            "the original code's conversion of all BF16 parameters to FP32."
        ),
    )
    parser.add_argument(
        "--sparse-fp32-deltas",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only selected slices as FP32 deltas, then merge into an FP32 checkpoint.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--optim", default="paged_adamw_32bit")
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.data_limit <= 0 or args.dataset_start < 0:
        parser.error("--data-limit must be positive and --dataset-start must be nonnegative")
    if args.neuron_cap <= 0 or args.max_seq_length <= 0:
        parser.error("--neuron-cap and --max-seq-length must be positive")
    return args


class SparseNeuronLinear(nn.Module):
    """Frozen linear layer plus trainable FP32 row or column deltas."""

    def __init__(self, base: nn.Linear, indices: set[int], axis: str):
        super().__init__()
        if axis not in {"row", "column"}:
            raise ValueError(f"Unsupported sparse axis: {axis}")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.axis = axis
        ordered = torch.tensor(sorted(indices), dtype=torch.long, device=base.weight.device)
        self.register_buffer("indices", ordered, persistent=False)
        shape = (
            (len(indices), base.in_features)
            if axis == "row"
            else (base.out_features, len(indices))
        )
        self.delta = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32, device=base.weight.device),
            requires_grad=bool(indices),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if self.axis == "row":
            delta_output = F.linear(inputs, self.delta.to(inputs.dtype))
            return output.index_add(-1, self.indices, delta_output)
        selected_inputs = inputs.index_select(-1, self.indices)
        return output + F.linear(selected_inputs, self.delta.to(inputs.dtype))

    def merge(self) -> nn.Linear:
        weight = self.base.weight.detach().float()
        delta = self.delta.detach().float()
        if self.axis == "row":
            weight.index_add_(0, self.indices, delta)
        else:
            weight.index_add_(1, self.indices, delta)
        self.base.weight = nn.Parameter(weight, requires_grad=False)
        if self.base.bias is not None:
            self.base.bias = nn.Parameter(self.base.bias.detach().float(), requires_grad=False)
        return self.base


def install_sparse_deltas(model, selected: list[dict[int, list[int]]]) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    total = 0
    for layer_index, layer in enumerate(model.model.layers):
        specs = (
            (layer.mlp, "up_proj", selected[0], "row"),
            (layer.mlp, "down_proj", selected[1], "column"),
            (layer.self_attn, "q_proj", selected[2], "row"),
            (layer.self_attn, "k_proj", selected[3], "row"),
            (layer.self_attn, "v_proj", selected[4], "row"),
        )
        for parent, attribute, values, axis in specs:
            indices = set(values[str(layer_index)])
            wrapped = SparseNeuronLinear(getattr(parent, attribute), indices, axis)
            setattr(parent, attribute, wrapped)
            total += wrapped.delta.numel()
    return total


def merge_sparse_deltas(model) -> None:
    for layer in model.model.layers:
        for parent, attribute in (
            (layer.mlp, "up_proj"),
            (layer.mlp, "down_proj"),
            (layer.self_attn, "q_proj"),
            (layer.self_attn, "k_proj"),
            (layer.self_attn, "v_proj"),
        ):
            wrapped = getattr(parent, attribute)
            if not isinstance(wrapped, SparseNeuronLinear):
                raise TypeError(f"Expected SparseNeuronLinear at {attribute}")
            setattr(parent, attribute, wrapped.merge())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expanded_to_physical(index: int, head_dim: int, kv_repeat: int, mode: str) -> int:
    if mode == "paper-code-divide":
        return index // kv_repeat
    expanded_head, coordinate = divmod(index, head_dim)
    return (expanded_head // kv_repeat) * head_dim + coordinate


def select_indices(indices, cap: int, method: str) -> set[int]:
    if method == "file-order":
        if not isinstance(indices, list):
            raise ValueError("file-order selection requires ordered lists in the neuron file")
        return set(indices[:cap])
    return set(itertools.islice(set(indices), cap))


def load_neurons(path: Path, config, args: argparse.Namespace):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(STRUCTURES):
        raise ValueError(f"Expected five neuron dictionaries in {path}, found {len(lines)}")
    neurons = []
    for name, line in zip(STRUCTURES, lines):
        parsed = ast.literal_eval(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} entry is not a dictionary")
        normalized = {
            int(layer): (
                list(dict.fromkeys(int(index) for index in values))
                if args.selection_method == "file-order"
                else set(int(index) for index in values)
            )
            for layer, values in parsed.items()
        }
        neurons.append(normalized)

    expected_layers = set(range(int(config.num_hidden_layers)))
    if any(set(values) != expected_layers for values in neurons):
        raise ValueError("Every neuron structure must contain exactly the model's layers")

    q_dim = int(config.hidden_size)
    head_dim = q_dim // int(config.num_attention_heads)
    physical_kv_dim = int(config.num_key_value_heads) * head_dim
    kv_repeat = int(config.num_attention_heads) // int(config.num_key_value_heads)
    ffn_dim = int(config.intermediate_size)
    limits = (ffn_dim, ffn_dim, q_dim, q_dim if args.kv_index_space == "expanded" else physical_kv_dim,
              q_dim if args.kv_index_space == "expanded" else physical_kv_dim)
    for name, values, limit in zip(STRUCTURES, neurons, limits):
        for layer, indices in values.items():
            if any(index < 0 or index >= limit for index in indices):
                raise ValueError(f"{name} layer {layer} contains an index outside [0, {limit})")

    if args.kv_index_space == "expanded":
        for position in (3, 4):
            neurons[position] = {
                layer: (
                    list(
                        dict.fromkeys(
                            expanded_to_physical(index, head_dim, kv_repeat, args.kv_map)
                            for index in indices
                        )
                    )
                    if args.selection_method == "file-order"
                    else {
                        expanded_to_physical(index, head_dim, kv_repeat, args.kv_map)
                        for index in indices
                    }
                )
                for layer, indices in neurons[position].items()
            }

    # Legacy mode reproduces original-paper-code. Ranked detector files use explicit list order.
    selected = [
        {
            layer: select_indices(indices, args.neuron_cap, args.selection_method)
            for layer, indices in values.items()
        }
        for values in neurons
    ]
    manifest_selection = [
        {str(layer): sorted(indices) for layer, indices in sorted(values.items())}
        for values in selected
    ]
    selection_hash = hashlib.sha256(
        json.dumps(manifest_selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return neurons, manifest_selection, selection_hash


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if args.prepare_for_kbit_training and args.sparse_fp32_deltas:
        raise ValueError("Use either full FP32 preparation or sparse FP32 deltas, not both")
    if args.prepare_for_kbit_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    mapped_neurons, selected, selection_hash = load_neurons(args.neurons, model.config, args)
    sparse_trainable_parameters = None
    if args.sparse_fp32_deltas:
        sparse_trainable_parameters = install_sparse_deltas(model, selected)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    selected_counts = {
        name: sum(len(indices) for indices in values.values())
        for name, values in zip(STRUCTURES, selected)
    }

    dataset = load_dataset(
        "json", data_files=str(args.dataset), split="train", cache_dir=str(args.cache_dir)
    )
    stop = min(len(dataset), args.dataset_start + args.data_limit)
    if args.dataset_start >= stop:
        raise ValueError("Dataset slice is empty")
    dataset = dataset.select(range(args.dataset_start, stop))
    missing = {args.prompt_field, args.response_field} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing fields: {sorted(missing)}")

    def formatting_prompts_func(batch):
        texts = []
        for prompt, response in zip(batch[args.prompt_field], batch[args.response_field]):
            if args.prompt_format == "chat":
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": str(prompt)},
                        {"role": "assistant", "content": str(response)},
                    ],
                    tokenize=False,
                )
            else:
                text = f"{prompt}. {response}"
            texts.append(text)
        return texts

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        save_strategy="no",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=False,
    )
    training_args.activate_neuron = mapped_neurons
    training_args.neuron_cap = args.neuron_cap
    training_args.kv_indices_are_physical = True
    training_args.sparse_delta_mode = args.sparse_fp32_deltas

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    manifest = {
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_start": args.dataset_start,
        "num_documents": len(dataset),
        "prompt_field": args.prompt_field,
        "response_field": args.response_field,
        "prompt_format": args.prompt_format,
        "neuron_file": str(args.neurons.resolve()),
        "neuron_file_sha256": sha256_file(args.neurons),
        "neuron_cap_per_layer_structure": args.neuron_cap,
        "selection_method": args.selection_method,
        "kv_index_space": args.kv_index_space,
        "kv_map": args.kv_map,
        "selection_counts": selected_counts,
        "selection_sha256": selection_hash,
        "selection": selected,
        "world_size": world_size,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.per_device_batch_size
        * args.gradient_accumulation_steps
        * world_size,
        "gradient_checkpointing": args.gradient_checkpointing,
        "prepare_for_kbit_training": args.prepare_for_kbit_training,
        "sparse_fp32_deltas": args.sparse_fp32_deltas,
        "sparse_trainable_parameters": sparse_trainable_parameters,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "optim": args.optim,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "benchmark": args.benchmark,
    }
    if training_args.should_save:
        (args.output_dir / "run_config.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_prompts_func,
        args=training_args,
        packing=False,
    )
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start

    if training_args.should_save:
        metrics = dict(result.metrics)
        metrics["wall_time_seconds"] = elapsed
        metrics["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        (args.output_dir / "train_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        if args.save_model and not args.benchmark:
            if args.sparse_fp32_deltas:
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                merge_sparse_deltas(unwrapped)
                unwrapped.float()
                unwrapped.config.torch_dtype = torch.float32
            trainer.save_model(str(args.output_dir))
            tokenizer.save_pretrained(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
