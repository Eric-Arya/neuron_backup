#!/usr/bin/env python3
"""Amplify a merged SN-Tune checkpoint's delta from its base model."""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE = Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"--output-dir must be absent or empty: {args.output_dir}")
    return args


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {name: single.name for name in handle.keys()}


def scaled_tensor(base: torch.Tensor, tuned: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return base + alpha * (tuned - base) in the tuned tensor's dtype."""
    base = base.to(dtype=tuned.dtype)
    return torch.lerp(base, tuned, alpha)


def main() -> int:
    args = parse_args()
    base_dir = args.base_model.resolve()
    checkpoint_dir = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_map = load_weight_map(base_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    state = model.state_dict()
    if set(state) != set(base_map):
        missing = sorted(set(base_map) - set(state))
        extra = sorted(set(state) - set(base_map))
        raise ValueError(f"Tensor mismatch: missing={missing}, extra={extra}")

    with ExitStack() as stack, torch.no_grad():
        base_files = {
            filename: stack.enter_context(
                safe_open(base_dir / filename, framework="pt", device="cpu")
            )
            for filename in set(base_map.values())
        }
        for name, tensor in state.items():
            base = base_files[base_map[name]].get_tensor(name)
            tensor.copy_(scaled_tensor(base, tensor, args.alpha))

    model.config.use_cache = True
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    tokenizer.save_pretrained(output_dir)

    source_run_config = checkpoint_dir / "run_config.json"
    if source_run_config.exists():
        run_config = json.loads(source_run_config.read_text(encoding="utf-8"))
        run_config["delta_scaling"] = {
            "alpha": args.alpha,
            "base_model": str(base_dir),
            "source_checkpoint": str(checkpoint_dir),
            "formula": "base + alpha * (source_checkpoint - base)",
        }
        (output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
        )
    source_metrics = checkpoint_dir / "train_metrics.json"
    if source_metrics.exists():
        shutil.copy2(source_metrics, output_dir / "source_train_metrics.json")

    result = {
        "alpha": args.alpha,
        "base_model": str(base_dir),
        "source_checkpoint": str(checkpoint_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "delta_scale_config.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
