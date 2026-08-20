#!/usr/bin/env python3
"""Build a merged SN-Tune checkpoint containing a filtered weight delta."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE = Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct")
SPECS = (
    ("mlp.up_proj.weight", 0),
    ("mlp.down_proj.weight", 1),
    ("self_attn.q_proj.weight", 0),
    ("self_attn.k_proj.weight", 0),
    ("self_attn.v_proj.weight", 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--mode", choices=("sign-flip", "top-abs"), required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        help=(
            "Number of coordinates retained by top-abs. By default this matches the "
            "number of sign-flipping coordinates at the requested alpha."
        ),
    )
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()
    if not torch.isfinite(torch.tensor(args.alpha)) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.mode == "sign-flip" and args.top_k is not None:
        parser.error("--top-k only applies to --mode top-abs")
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


def load_selected_deltas(base_dir: Path, checkpoint_dir: Path):
    run_config = json.loads(
        (checkpoint_dir / "run_config.json").read_text(encoding="utf-8")
    )
    selection = run_config["selection"]
    base_map = load_weight_map(base_dir)
    checkpoint_map = load_weight_map(checkpoint_dir)
    if set(base_map) != set(checkpoint_map):
        raise ValueError("Base and SN-Tune checkpoint tensor names differ")

    slices = []
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
        for structure, (suffix, axis) in enumerate(SPECS):
            for layer in range(len(selection[structure])):
                indices = selection[structure][str(layer)]
                if not indices:
                    continue
                name = f"model.layers.{layer}.{suffix}"
                index = torch.tensor(indices, dtype=torch.long)
                base = torch.index_select(
                    base_files[base_map[name]].get_tensor(name), axis, index
                ).float()
                tuned = torch.index_select(
                    checkpoint_files[checkpoint_map[name]].get_tensor(name), axis, index
                ).float()
                slices.append(
                    {
                        "name": name,
                        "axis": axis,
                        "indices": index,
                        "base": base,
                        "delta": tuned - base,
                    }
                )
    return run_config, slices


def build_masks(slices, mode: str, alpha: float, top_k: int | None):
    flat_deltas = torch.cat([item["delta"].reshape(-1) for item in slices])
    flat_bases = torch.cat([item["base"].reshape(-1) for item in slices])
    sign_flips = torch.signbit(flat_bases + alpha * flat_deltas) != torch.signbit(
        flat_bases
    )
    sign_flip_count = int(sign_flips.sum().item())
    keep_count = top_k if top_k is not None else sign_flip_count
    if keep_count > flat_deltas.numel():
        raise ValueError(f"Requested top-k {keep_count} exceeds delta size")

    if mode == "sign-flip":
        flat_mask = sign_flips
    else:
        flat_mask = torch.zeros(flat_deltas.numel(), dtype=torch.bool)
        top_indices = torch.topk(
            flat_deltas.abs(), keep_count, largest=True, sorted=False
        ).indices
        flat_mask[top_indices] = True

    masks = []
    offset = 0
    digest = hashlib.sha256()
    for item in slices:
        count = item["delta"].numel()
        mask = flat_mask[offset : offset + count].reshape(item["delta"].shape)
        masks.append(mask)
        digest.update(mask.numpy().tobytes())
        offset += count
    if offset != flat_mask.numel():
        raise RuntimeError("Mask partition did not consume every delta coordinate")
    return masks, sign_flip_count, int(flat_mask.sum().item()), digest.hexdigest()


def main() -> int:
    args = parse_args()
    base_dir = args.base_model.resolve()
    checkpoint_dir = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    source_config, slices = load_selected_deltas(base_dir, checkpoint_dir)
    masks, sign_flip_count, kept_count, mask_hash = build_masks(
        slices, args.mode, args.alpha, args.top_k
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    state = model.state_dict()
    with torch.no_grad():
        for item, mask in zip(slices, masks):
            final = item["base"] + args.alpha * item["delta"] * mask
            state[item["name"]].index_copy_(
                item["axis"], item["indices"], final
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.config.torch_dtype = torch.float32
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    tokenizer.save_pretrained(output_dir)

    ablation = {
        "mode": args.mode,
        "alpha": args.alpha,
        "base_model": str(base_dir),
        "source_checkpoint": str(checkpoint_dir),
        "source_selected_parameter_count": sum(
            item["delta"].numel() for item in slices
        ),
        "sign_flip_count_at_alpha": sign_flip_count,
        "kept_parameter_count": kept_count,
        "matched_parameter_budget": kept_count == sign_flip_count,
        "mask_sha256": mask_hash,
        "formula": "base + alpha * source_delta * keep_mask",
    }
    output_config = dict(source_config)
    output_config["delta_ablation"] = ablation
    (output_dir / "run_config.json").write_text(
        json.dumps(output_config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "delta_scale_config.json").write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "base_model": str(base_dir),
                "source_checkpoint": str(checkpoint_dir),
                "output_dir": str(output_dir),
                "ablation_mode": args.mode,
                "kept_parameter_count": kept_count,
                "mask_sha256": mask_hash,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "ablation_config.json").write_text(
        json.dumps(ablation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(ablation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
