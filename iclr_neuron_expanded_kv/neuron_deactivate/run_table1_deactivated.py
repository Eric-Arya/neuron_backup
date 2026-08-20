#!/usr/bin/env python3
"""Run Llama-3 with the custom safety-neuron deactivation overlay and score outputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import transformers
from transformers.generation.utils import GenerationMixin

try:
    from .table1_generation_common import (
        DEFAULT_OUTPUT_DIR,
        SCRIPT_DIR,
        add_common_arguments,
        run_generation,
    )
except ImportError:
    from table1_generation_common import (
        DEFAULT_OUTPUT_DIR,
        SCRIPT_DIR,
        add_common_arguments,
        run_generation,
    )


EXPECTED_ENV = Path("/workspace/xcy/miniconda3/envs/safety-neuron-expanded-kv")
DEFAULT_NEURONS = (
    SCRIPT_DIR.parent
    / "neuron_detection"
    / "output_neurons"
    / "Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vonly_200.txt"
)
STRUCTURES = ("fwd_up", "fwd_down", "q", "k", "v")


def verify_deactivation_transformers() -> None:
    transformers_path = Path(transformers.__file__).resolve()
    if Path(sys.prefix).resolve() != EXPECTED_ENV or EXPECTED_ENV not in transformers_path.parents:
        raise RuntimeError(
            f"Deactivation runner requires {EXPECTED_ENV}; current Python is {sys.executable}"
        )
    if transformers.__version__ != "4.44.2":
        raise RuntimeError(f"Expected Transformers 4.44.2, found {transformers.__version__}")
    generate_parameters = inspect.signature(GenerationMixin.generate).parameters
    if not {"activate_keys_fwd_up_set", "neuron_scale"} <= set(generate_parameters):
        raise RuntimeError("The neuron-scaling Transformers generation overlay is not installed")


def load_neurons(path: Path) -> list[dict[int, tuple[int, ...]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 5:
        raise ValueError(f"Expected five neuron dictionaries in {path}, found {len(lines)} lines")
    result: list[dict[int, tuple[int, ...]]] = []
    for structure, line in zip(STRUCTURES, lines):
        parsed = ast.literal_eval(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Neuron structure {structure} is not a dictionary")
        normalized: dict[int, tuple[int, ...]] = {}
        for layer, indices in parsed.items():
            if not isinstance(layer, int) or not isinstance(indices, (set, list, tuple)):
                raise ValueError(f"Invalid {structure} entry for layer {layer!r}")
            if not all(isinstance(index, int) and index >= 0 for index in indices):
                raise ValueError(f"Invalid neuron index in {structure} layer {layer}")
            normalized[layer] = tuple(sorted(set(indices)))
        result.append(normalized)
    return result


def selected_hash(values: list[dict[int, tuple[int, ...]]]) -> str:
    serializable = [
        {str(layer): list(indices) for layer, indices in sorted(structure.items())}
        for structure in values
    ]
    payload = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_for_rate(dimension: int, rate: float) -> int:
    """Convert a fractional rate to the nearest positive neuron count."""
    return max(1, min(dimension, math.floor(dimension * rate + 0.5)))


def build_deactivation_factory(args: argparse.Namespace):
    def factory(config) -> tuple[dict[str, object], dict[str, object]]:
        neuron_scale = float(getattr(args, "neuron_scale", 0.0))
        if not math.isfinite(neuron_scale):
            raise ValueError("neuron-scale must be finite")
        num_layers = int(config.num_hidden_layers)
        if num_layers != 32:
            raise ValueError(f"This overlay expects 32 Llama layers, found {num_layers}")
        q_dimension = int(config.hidden_size)
        head_dim = q_dimension // int(config.num_attention_heads)
        physical_kv_dimension = int(config.num_key_value_heads) * head_dim
        expanded_kv_dimension = q_dimension
        ffn_dimension = int(config.intermediate_size)
        expected_layers = set(range(num_layers))
        intervention_dimensions = (
            ffn_dimension,
            ffn_dimension,
            q_dimension,
            expanded_kv_dimension,
            expanded_kv_dimension,
        )
        global_dimension = num_layers * sum(intervention_dimensions)
        selected_structures = tuple(getattr(args, "structures", STRUCTURES))
        if not selected_structures or any(name not in STRUCTURES for name in selected_structures):
            raise ValueError(f"structures must be chosen from {STRUCTURES}")
        selected_positions = {STRUCTURES.index(name) for name in selected_structures}
        deact_count = getattr(args, "deact_count", None)
        deact_rate = getattr(args, "deact_rate", None)
        if deact_count is not None and deact_rate is not None:
            raise ValueError("Pass only one of deact-count and deact-rate")
        if deact_count is not None:
            if not 1 <= deact_count <= global_dimension:
                raise ValueError(f"deact-count must be in [1, {global_dimension}]")
            global_count = int(deact_count)
        else:
            if deact_rate is None or not math.isfinite(deact_rate) or not 0 < deact_rate <= 1:
                raise ValueError("Pass a finite deact-rate in (0, 1] or a positive deact-count")
            global_count = count_for_rate(global_dimension, deact_rate)

        def empty_selection() -> list[dict[int, list[int]]]:
            return [
                {layer: [] for layer in range(num_layers)}
                for _ in STRUCTURES
            ]

        neuron_path: Path | None = None
        neuron_file_sha256: str | None = None
        sn_pool_size: int | None = None
        rng = random.Random(args.seed)
        if args.deact_mode == "random":
            selected_lists = empty_selection()
            random_dimension = num_layers * sum(
                dimension
                for position, dimension in enumerate(intervention_dimensions)
                if position in selected_positions
            )
            if global_count > random_dimension:
                raise ValueError(
                    f"deact-count {global_count} exceeds the selected-structure universe "
                    f"of {random_dimension}"
                )
            for flat_index in rng.sample(range(random_dimension), global_count):
                offset = 0
                for position, dimension in enumerate(intervention_dimensions):
                    if position not in selected_positions:
                        continue
                    structure_size = num_layers * dimension
                    if flat_index < offset + structure_size:
                        relative = flat_index - offset
                        layer, index = divmod(relative, dimension)
                        selected_lists[position][layer].append(index)
                        break
                    offset += structure_size
        else:
            neuron_path = args.neurons.resolve()
            neuron_file_sha256 = hashlib.sha256(neuron_path.read_bytes()).hexdigest()
            neurons = load_neurons(neuron_path)
            detector_dimensions = (
                ffn_dimension,
                ffn_dimension,
                q_dimension,
                expanded_kv_dimension,
                expanded_kv_dimension,
            )
            pool: list[tuple[int, int, int]] = []
            for position, (structure, values, detector_dimension) in enumerate(
                zip(STRUCTURES, neurons, detector_dimensions)
            ):
                if set(values) != expected_layers:
                    raise ValueError(
                        f"Neuron structure {structure} does not contain exactly 32 layers"
                    )
                for layer, indices in values.items():
                    if any(index >= detector_dimension for index in indices):
                        raise ValueError(
                            f"{structure} layer {layer} contains an index outside "
                            f"[0, {detector_dimension})"
                        )
                    if position in selected_positions:
                        pool.extend((position, layer, index) for index in indices)
            sn_pool_size = len(pool)
            if sn_pool_size < global_count:
                raise ValueError(
                    f"SN pool has {sn_pool_size} expanded-space candidates, fewer than the "
                    f"requested intervention count of {global_count}"
                )
            selected_lists = empty_selection()
            for position, layer, index in rng.sample(pool, global_count):
                selected_lists[position][layer].append(index)

        selected = [
            {layer: tuple(sorted(indices)) for layer, indices in layers.items()}
            for layers in selected_lists
        ]
        ffn_limit = max(
            len(indices)
            for structure in selected[:2]
            for indices in structure.values()
        )
        attention_limit = max(
            len(indices)
            for structure in selected[2:]
            for indices in structure.values()
        )

        layer_scope = getattr(args, "layer_scope", "all")
        layer_cutoff = getattr(args, "layer_cutoff", None)
        if layer_scope == "all":
            if layer_cutoff is not None:
                raise ValueError("layer-cutoff is only valid with front or back layer scope")
            active_layers = tuple(range(num_layers))
            under_layer = num_layers
            gen_layer = 0
            whether_under = True
            whether_reason = False
            whether_gen = False
        else:
            if layer_cutoff is None or not 0 <= layer_cutoff <= num_layers:
                raise ValueError(f"{layer_scope} layer scope requires --layer-cutoff in [0, {num_layers}]")
            if layer_scope == "front":
                active_layers = tuple(range(layer_cutoff))
                under_layer = layer_cutoff
                gen_layer = 0
                whether_under = True
                whether_reason = False
                whether_gen = False
            else:
                active_layers = tuple(range(layer_cutoff, num_layers))
                under_layer = 0
                gen_layer = num_layers - layer_cutoff
                whether_under = False
                whether_reason = False
                whether_gen = True

        generation_kwargs = {
            "activate_keys_fwd_up_set": selected[0],
            "activate_keys_fwd_down_set": selected[1],
            "activate_keys_q_set": selected[2],
            "activate_keys_k_set": selected[3],
            "activate_keys_v_set": selected[4],
            "neuron_scale": neuron_scale,
            "under_layer": under_layer,
            "gen_layer": gen_layer,
            # These shared limits are at least the largest globally selected per-layer tuple,
            # so the overlay consumes every selected index without imposing per-layer quotas.
            "atten_number": attention_limit,
            "ffn_number": ffn_limit,
            "whether_under": whether_under,
            "whether_reason": whether_reason,
            "whether_gen": whether_gen,
            "whether_under_fwd": whether_under,
            "whether_reason_fwd": False,
            "whether_gen_fwd": whether_gen,
        }
        metadata = {
            "deactivation": True,
            "deactivation_mode": args.deact_mode,
            "neuron_scale": neuron_scale,
            "selection_method": (
                "uniform_without_replacement_from_global_expanded_universe"
                if args.deact_mode == "random"
                else "uniform_without_replacement_from_global_sn_pool"
            ),
            "selection_seed": args.seed,
            "selected_structures": list(selected_structures),
            "neuron_path": str(neuron_path) if neuron_path is not None else None,
            "neuron_file_sha256": neuron_file_sha256,
            "sn_pool_size": sn_pool_size,
            "requested_deact_count": deact_count,
            "requested_deact_rate": deact_rate,
            "global_neuron_dimension": global_dimension,
            "global_deactivated_neurons": global_count,
            "effective_global_deact_rate": global_count / global_dimension,
            "q_dimension": q_dimension,
            "physical_key_value_dimension": physical_kv_dimension,
            "expanded_key_value_dimension": expanded_kv_dimension,
            "ffn_dimension": ffn_dimension,
            "selected_by_structure": {
                structure: sum(len(indices) for indices in layers.values())
                for structure, layers in zip(STRUCTURES, selected)
            },
            "layer_scope": layer_scope,
            "layer_cutoff": layer_cutoff,
            "active_layer_indices": list(active_layers),
            "deactivated_layers": len(active_layers),
            "applied_deactivated_neurons": sum(
                len(layers[layer]) for layers in selected for layer in active_layers
            ),
            "selected_neurons_sha256": selected_hash(selected),
        }
        return generation_kwargs, metadata

    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "deact_sn")
    parser.set_defaults(responses=None, scored=None, summary=None, run_metadata=None)
    parser.add_argument(
        "neuron_txt",
        nargs="?",
        type=Path,
        help="Neuron TXT file containing the five per-layer dictionaries.",
    )
    parser.add_argument(
        "--neurons",
        dest="neurons_option",
        type=Path,
        help="Neuron TXT file (flag form; cannot be combined with neuron_txt).",
    )
    parser.add_argument(
        "--deact-mode",
        "--mode",
        dest="deact_mode",
        type=str.lower,
        choices=("sn", "random"),
        default="sn",
        help="Select detected safety neurons (sn) or random expanded-space neurons (default: sn).",
    )
    count_group = parser.add_mutually_exclusive_group()
    count_group.add_argument(
        "--deact-count",
        type=int,
        help="Exact global number of expanded-space neurons to deactivate (default: 557).",
    )
    count_group.add_argument(
        "--deact-rate",
        type=float,
        help="Fraction of the expanded intervention universe to deactivate.",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        choices=STRUCTURES,
        default=["v"],
        help="Projection structures eligible for selection (default: expanded V only).",
    )
    parser.add_argument(
        "--neuron-scale",
        "--neuron-multiplier",
        "--scale-factor",
        dest="neuron_scale",
        type=float,
        default=0.0,
        help=(
            "Multiply each selected neuron channel by this finite value "
            "(default: 0, reproducing deactivation; examples: 0.5 or 2)."
        ),
    )
    parser.add_argument(
        "--layer-scope",
        choices=("all", "front", "back"),
        default="all",
        help=(
            "Apply the fixed global neuron selection to every layer, layers below the cutoff, "
            "or layers at/above the cutoff (default: all)."
        ),
    )
    parser.add_argument(
        "--layer-cutoff",
        type=int,
        help="Layer boundary for --layer-scope front/back; valid Llama-3 values are 0 through 32.",
    )
    return parser


def main() -> int:
    verify_deactivation_transformers()
    args = build_parser().parse_args()
    if not math.isfinite(args.neuron_scale):
        raise SystemExit("error: --neuron-scale must be finite")
    if args.deact_count is None and args.deact_rate is None:
        args.deact_count = 557
    if args.neuron_txt is not None and args.neurons_option is not None:
        raise SystemExit("error: pass the neuron TXT either positionally or with --neurons, not both")
    supplied_neurons = args.neuron_txt or args.neurons_option
    if args.deact_mode == "random" and supplied_neurons is not None:
        raise SystemExit("error: a neuron file is only valid with --deact-mode sn")
    args.neurons = supplied_neurons or DEFAULT_NEURONS if args.deact_mode == "sn" else None
    safe_stem = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", args.neurons.stem)
        if args.neurons is not None
        else "random"
    )
    if args.deact_count is not None:
        intervention_tag = f"count{args.deact_count}"
    else:
        intervention_tag = "rate" + format(args.deact_rate, ".12g").replace(".", "p")
    mode_tag = f"{safe_stem}_sn" if args.deact_mode == "sn" else "random"
    prepared_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.prepared.stem)
    scope_tag = (
        "all"
        if args.layer_scope == "all"
        else f"{args.layer_scope}{args.layer_cutoff}"
    )
    scale_tag = "" if args.neuron_scale == 0 else (
        "_scale" + format(args.neuron_scale, ".12g").replace("-", "neg").replace(".", "p")
    )
    run_name = (
        f"{mode_tag}_{intervention_tag}{scale_tag}_{scope_tag}_"
        f"{args.prompt_format}_{prepared_tag}"
    )
    run_dir = DEFAULT_OUTPUT_DIR / "deact_sn" / run_name
    args.responses = args.responses or run_dir / "responses.jsonl"
    args.scored = args.scored or run_dir / "scored.jsonl"
    args.summary = args.summary or run_dir / "summary.json"
    args.run_metadata = args.run_metadata or run_dir / "run.json"
    try:
        return run_generation(args, "deact_sn", build_deactivation_factory(args))
    except (OSError, RuntimeError, ValueError) as exc:
        if os.environ.get("NEURON_DEBUG") == "1":
            raise
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
