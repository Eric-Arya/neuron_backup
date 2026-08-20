#!/usr/bin/env python3
"""Validate that an SN-Tune checkpoint changes only its selected parameter slices."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_BASE = Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to CHECKPOINT/weight_diff_validation.json.",
    )
    return parser.parse_args()


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    single_file = model_dir / "model.safetensors"
    if not single_file.exists():
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")
    with safe_open(single_file, framework="pt", device="cpu") as handle:
        return {name: single_file.name for name in handle.keys()}


def selected_axis(name: str, selection: list[dict[str, list[int]]]):
    parts = name.split(".")
    if len(parts) < 5 or parts[:2] != ["model", "layers"] or parts[-1] != "weight":
        return None
    try:
        layer = parts[2]
    except (IndexError, ValueError):
        return None
    suffix = ".".join(parts[3:])
    specs = {
        "mlp.up_proj.weight": (0, 0),
        "mlp.down_proj.weight": (1, 1),
        "self_attn.q_proj.weight": (2, 0),
        "self_attn.k_proj.weight": (3, 0),
        "self_attn.v_proj.weight": (4, 0),
    }
    if suffix not in specs:
        return None
    structure, axis = specs[suffix]
    return selection[structure][layer], axis


def main() -> int:
    args = parse_args()
    base_dir = args.base_model.resolve()
    checkpoint_dir = args.checkpoint.resolve()
    output = args.output or checkpoint_dir / "weight_diff_validation.json"
    config = json.loads((checkpoint_dir / "run_config.json").read_text(encoding="utf-8"))
    selection = config["selection"]
    base_map = load_weight_map(base_dir)
    checkpoint_map = load_weight_map(checkpoint_dir)
    if set(base_map) != set(checkpoint_map):
        missing = sorted(set(base_map) - set(checkpoint_map))
        extra = sorted(set(checkpoint_map) - set(base_map))
        raise ValueError(f"Checkpoint tensor mismatch: missing={missing}, extra={extra}")

    changed_tensor_count = 0
    changed_element_count = 0
    max_abs_change = 0.0
    unexpected: list[dict[str, object]] = []
    with ExitStack() as stack:
        base_files = {
            filename: stack.enter_context(
                safe_open(base_dir / filename, framework="pt", device="cpu")
            )
            for filename in set(base_map.values())
        }
        checkpoint_files = {
            filename: stack.enter_context(
                safe_open(checkpoint_dir / filename, framework="pt", device="cpu")
            )
            for filename in set(checkpoint_map.values())
        }
        for name in sorted(base_map):
            base = base_files[base_map[name]].get_tensor(name).float()
            tuned = checkpoint_files[checkpoint_map[name]].get_tensor(name).float()
            difference = tuned.sub_(base)
            changed = int(torch.count_nonzero(difference).item())
            if not changed:
                continue
            changed_tensor_count += 1
            changed_element_count += changed
            max_abs_change = max(max_abs_change, float(difference.abs().max().item()))
            allowed = selected_axis(name, selection)
            allowed_changed = 0
            if allowed is not None:
                indices, axis = allowed
                index = torch.tensor(indices, dtype=torch.long)
                allowed_changed = int(
                    torch.count_nonzero(difference.index_select(axis, index)).item()
                )
            if changed != allowed_changed:
                unexpected.append(
                    {
                        "tensor": name,
                        "changed_elements": changed,
                        "allowed_changed_elements": allowed_changed,
                        "unexpected_changed_elements": changed - allowed_changed,
                    }
                )

    result = {
        "changed_tensor_count": changed_tensor_count,
        "changed_element_count": changed_element_count,
        "max_abs_change": max_abs_change,
        "unexpected_tensor_count": len(unexpected),
        "unexpected": unexpected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
