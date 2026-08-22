from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
import torch
from scipy.optimize import minimize
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    read_jsonl,
    score_asr,
    sha256_file,
)
from .methods import DEFAULT_LLAMA3, DTYPES


DEFAULT_WIKITEXT = Path(
    "/workspace/xcy/dataset/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
)
DEFAULT_RANKING = Path(
    "/workspace/xcy/safety_repro/unified_eval/results/"
    "grad_onpolicy_sn_safe256_first_cue_tail_expanded50000/gradients/"
    "top_neurons_stable.csv"
)
DEFAULT_TEST_HARMBENCH = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/"
    "table1_seed42_n200.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a WikiText model Fisher on selected Grad neurons and derive "
            "shared or per-neuron positive scaling controllers."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_WIKITEXT)
    prepare.add_argument("--model", type=Path, default=DEFAULT_LLAMA3)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--fisher-contexts", type=int, default=1024)
    prepare.add_argument("--validation-contexts", type=int, default=256)
    prepare.add_argument("--context-tokens", type=int, default=128)
    prepare.add_argument("--seed", type=int, default=42)

    benchmark = subparsers.add_parser("benchmark")
    add_compute_arguments(benchmark)
    benchmark.add_argument("--context-limit", type=int, default=8)
    benchmark.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    benchmark.add_argument("--probes", type=int, default=2)

    compute = subparsers.add_parser("compute")
    add_compute_arguments(compute)
    # Benchmarked on 16 real WikiText contexts with 32-token continuations and
    # two retained-graph probes: batch 4 was fastest (3.11 contexts/s) and used
    # 39.3 GB on an H100; batches 8 and 16 were slower.
    compute.add_argument("--batch-size", type=int, default=4)
    compute.add_argument("--probes", type=int, default=4)
    compute.add_argument(
        "--matrix-mode", choices=("dense", "diagonal"), default="dense"
    )
    compute.add_argument("--shrinkage", type=float, default=0.5)
    compute.add_argument("--damping-ratio", type=float, default=0.01)
    compute.add_argument(
        "--reference-shared-delta",
        type=float,
        default=1.0,
        help=(
            "Set the quadratic budget to the cost of this common positive delta; "
            "1.0 matches the existing Grad multiplier of two."
        ),
    )

    validate = subparsers.add_parser("validate-kl")
    validate.add_argument("--model", type=Path, default=DEFAULT_LLAMA3)
    validate.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    validate.add_argument("--contexts", type=Path, required=True)
    validate.add_argument("--shared-scales", type=Path, required=True)
    validate.add_argument("--individual-scales", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--top-k", type=int, default=2000)
    validate.add_argument("--continuation-tokens", type=int, default=32)
    validate.add_argument("--batch-size", type=int, default=4)
    validate.add_argument("--calibration-iterations", type=int, default=3)
    validate.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    validate.add_argument("--device", default="cuda:0")
    validate.add_argument("--seed", type=int, default=211)

    variants = subparsers.add_parser("make-variants")
    variants.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    variants.add_argument("--base-individual-scales", type=Path, required=True)
    variants.add_argument("--output-dir", type=Path, required=True)
    variants.add_argument(
        "--direct-top-k", type=int, nargs="+", default=[2000, 4000, 8000]
    )
    variants.add_argument(
        "--direct-strengths", type=float, nargs="+", default=[0.5, 0.75, 1.0]
    )
    variants.add_argument(
        "--individual-scales", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    variants.add_argument("--individual-caps", type=float, nargs="+", default=[1, 2, 4])

    rescale = subparsers.add_parser("rescale-scales")
    rescale.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    rescale.add_argument("--base-scales", type=Path, required=True)
    rescale.add_argument("--output-dir", type=Path, required=True)
    rescale.add_argument(
        "--t-values", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6]
    )

    safety = subparsers.add_parser("evaluate-safety")
    safety.add_argument("--model", type=Path, default=DEFAULT_LLAMA3)
    safety.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    safety.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_TEST_HARMBENCH,
        help="Safety manifest; defaults to the fixed 200-example HarmBench test.",
    )
    safety.add_argument("--scale-files", type=Path, nargs="+", required=True)
    safety.add_argument("--output-dir", type=Path, required=True)
    safety.add_argument("--batch-size", type=int, default=16)
    safety.add_argument("--max-new-tokens", type=int, default=128)
    safety.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    safety.add_argument("--device", default="cuda:0")

    selection = subparsers.add_parser("make-fisher-selection-variants")
    selection.add_argument("--fisher", type=Path, required=True)
    selection.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    selection.add_argument("--output-dir", type=Path, required=True)
    selection.add_argument("--pool-k", type=int, default=8000)
    selection.add_argument(
        "--active-k", type=int, nargs="+", default=[2000, 4000, 6000]
    )
    selection.add_argument("--strengths", type=float, nargs="+", default=[0.75, 1.0])
    selection.add_argument(
        "--base-k",
        type=int,
        default=4000,
        help="Direct Grad prefix used by conservative replacement variants.",
    )
    selection.add_argument(
        "--replace-counts",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Also retain base-k active neurons while replacing this many of the "
            "least Fisher-efficient prefix neurons with efficient tail neurons."
        ),
    )
    selection.add_argument(
        "--scores",
        nargs="+",
        choices=("gain_over_sqrt_fisher", "gain_over_fisher"),
        default=["gain_over_sqrt_fisher", "gain_over_fisher"],
    )

    box = subparsers.add_parser("make-box-fisher-variants")
    box.add_argument("--fisher", type=Path, required=True)
    box.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    box.add_argument("--output-dir", type=Path, required=True)
    box.add_argument("--pool-k", type=int, default=8000)
    box.add_argument("--reference-strengths", type=float, nargs="+", default=[0.5, 0.2])
    box.add_argument("--damping-ratios", type=float, nargs="+", default=[1, 4, 16])
    box.add_argument("--curvature-powers", type=float, nargs="+", default=[0.5, 1])
    box.add_argument("--cap-factors", type=float, nargs="+", default=[1.5, 2])

    floor = subparsers.add_parser("make-floor-fisher-variants")
    floor.add_argument("--fisher", type=Path, required=True)
    floor.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    floor.add_argument("--output-dir", type=Path, required=True)
    floor.add_argument("--pool-k", type=int, default=8000)
    floor.add_argument("--active-k", type=int, nargs="+", default=[4000, 6000, 8000])
    floor.add_argument("--floors", type=float, nargs="+", default=[0.25, 0.5])
    floor.add_argument("--target-medians", type=float, nargs="+")
    floor.add_argument(
        "--score-scales",
        type=float,
        nargs="+",
        help="Search explicit global c values instead of deriving c from a median.",
    )
    floor.add_argument("--caps", type=float, nargs="+", default=[0.9, 1.0])
    floor.add_argument("--damping-ratios", type=float, nargs="+", default=[1.0])

    dense_prefix = subparsers.add_parser("make-dense-prefix-variants")
    dense_prefix.add_argument("--fisher", type=Path, required=True)
    dense_prefix.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    dense_prefix.add_argument("--output-dir", type=Path, required=True)
    dense_prefix.add_argument(
        "--active-k", type=int, nargs="+", default=[250, 500, 1000, 1500, 2000]
    )
    dense_prefix.add_argument(
        "--target-positive-medians",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.45, 0.6],
    )
    dense_prefix.add_argument("--cap", type=float, default=0.75)
    dense_prefix.add_argument("--shrinkage", type=float, default=0.5)
    dense_prefix.add_argument("--damping-ratio", type=float, default=0.01)

    blend = subparsers.add_parser("make-blend-variants")
    blend.add_argument("--first-scales", type=Path, required=True)
    blend.add_argument("--second-scales", type=Path, required=True)
    blend.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    blend.add_argument("--output-dir", type=Path, required=True)
    blend.add_argument("--weights", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])

    anchored = subparsers.add_parser("make-anchored-tail-variants")
    anchored.add_argument("--fisher-scales", type=Path, required=True)
    anchored.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    anchored.add_argument("--output-dir", type=Path, required=True)
    anchored.add_argument("--base-k", type=int, default=4000)
    anchored.add_argument("--base-strength", type=float, default=0.75)
    anchored.add_argument(
        "--tail-weights", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )

    return parser


def add_compute_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=DEFAULT_LLAMA3)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--continuation-tokens", type=int, default=32)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=112)


def validate_positive(values: Sequence[int | float], label: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{label} must be positive")


def read_positive_ranking(path: Path, top_k: int) -> list[dict[str, Any]]:
    validate_positive([top_k], "top-k")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "rank" in rows[0]:
        rows.sort(key=lambda row: int(row["rank"]))
    else:
        rows.sort(key=lambda row: float(row["abs_mean_g"]), reverse=True)
    selected = [row for row in rows if float(row["mean_g"]) > 0][:top_k]
    if len(selected) != top_k:
        raise ValueError(f"Ranking contains only {len(selected)} positive rows")
    return [
        {
            "rank": rank,
            "source_rank": int(row.get("rank", rank)),
            "layer": int(row["layer"]),
            "neuron": int(row["neuron"]),
            "mean_g": float(row["mean_g"]),
        }
        for rank, row in enumerate(selected, 1)
    ]


def prepare_contexts(args: argparse.Namespace) -> None:
    validate_positive(
        [args.fisher_contexts, args.validation_contexts, args.context_tokens],
        "context preparation values",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    texts = pq.read_table(args.dataset, columns=["text"]).column("text").to_pylist()
    eligible: list[tuple[int, list[int]]] = []
    for source_row, text in enumerate(texts):
        clean = str(text).strip()
        if not clean or (clean.startswith("=") and clean.endswith("=")):
            continue
        token_ids = tokenizer.encode(clean, add_special_tokens=False)
        if len(token_ids) >= args.context_tokens:
            eligible.append((source_row, token_ids[: args.context_tokens]))
    total = args.fisher_contexts + args.validation_contexts
    if len(eligible) < total:
        raise ValueError(f"Need {total} contexts, found {len(eligible)}")
    selected = random.Random(args.seed).sample(eligible, total)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def records(rows: Sequence[tuple[int, list[int]]], split: str):
        for index, (source_row, token_ids) in enumerate(rows):
            yield {
                "id": index,
                "source_row": source_row,
                "selection_split": split,
                "context_token_count": len(token_ids),
                "context_token_ids": token_ids,
                "text": tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            }

    fisher_path = args.output_dir / "fisher_contexts.jsonl"
    validation_path = args.output_dir / "validation_contexts.jsonl"
    atomic_write_jsonl(fisher_path, records(selected[: args.fisher_contexts], "fisher"))
    atomic_write_jsonl(
        validation_path, records(selected[args.fisher_contexts :], "validation")
    )
    atomic_write_json(
        args.output_dir / "context_metadata.json",
        {
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": sha256_file(args.dataset),
            "model_tokenizer": str(args.model.resolve()),
            "model_tokenizer_config_sha256": sha256_file(
                args.model / "tokenizer_config.json"
            ),
            "format": "raw text with one BOS added only at model input",
            "selection": (
                "seeded sample without replacement from nonblank, non-heading train "
                "rows containing at least context_tokens; take the first context_tokens"
            ),
            "seed": args.seed,
            "eligible_rows": len(eligible),
            "fisher_contexts": args.fisher_contexts,
            "validation_contexts": args.validation_contexts,
            "context_tokens": args.context_tokens,
            "fisher_contexts_sha256": sha256_file(fisher_path),
            "validation_contexts_sha256": sha256_file(validation_path),
        },
    )


def load_model(path: Path, device: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        device_map={"": device},
        torch_dtype=DTYPES[dtype],
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    return model, tokenizer


def attach_selected_alphas(
    model, ranking: Sequence[dict[str, Any]], scope: str = "tail"
):
    if scope != "tail":
        raise ValueError("The firstcue256 ranking requires tail scope")
    device = model.get_input_embeddings().weight.device
    alpha = torch.ones(
        len(ranking), device=device, dtype=torch.float32, requires_grad=True
    )
    by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_index in range(len(model.model.layers)):
        entries = [row for row in ranking if row["layer"] == layer_index]
        if entries:
            by_layer[layer_index] = (
                torch.tensor([row["neuron"] for row in entries], device=device),
                torch.tensor([row["rank"] - 1 for row in entries], device=device),
            )
    state = {"start_position": 0}
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        if layer_index not in by_layer:
            continue
        neuron_indices, rank_indices = by_layer[layer_index]

        def scale_tail(
            _module,
            inputs,
            current_neurons=neuron_indices,
            current_ranks=rank_indices,
        ):
            activation = inputs[0]
            multiplier = torch.ones(
                activation.shape[-1], device=activation.device, dtype=activation.dtype
            ).scatter(
                0,
                current_neurons,
                alpha[current_ranks].to(dtype=activation.dtype),
            )
            row_starts = state.get("start_positions")
            if row_starts is not None:
                starts = torch.as_tensor(row_starts, device=activation.device)
                if starts.ndim != 1 or starts.numel() != activation.shape[0]:
                    raise ValueError("start_positions must contain one value per row")
                positions = torch.arange(
                    activation.shape[1], device=activation.device
                ).view(1, -1, 1)
                mask = positions >= starts.view(-1, 1, 1)
                scaled = torch.where(mask, activation * multiplier, activation)
                return (scaled, *inputs[1:])
            start = (
                0
                if state.get("decode_all") and activation.shape[1] == 1
                else state["start_position"]
            )
            scaled = torch.cat(
                (
                    activation[:, :start],
                    activation[:, start:] * multiplier,
                ),
                dim=1,
            )
            return (scaled, *inputs[1:])

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(scale_tail))
    return alpha, handles, state


def sample_continuations(
    model,
    tokenizer,
    context_rows: Sequence[dict[str, Any]],
    continuation_tokens: int,
) -> list[list[int]]:
    device = model.get_input_embeddings().weight.device
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("Tokenizer has no BOS token")
    input_ids = torch.tensor(
        [[bos, *row["context_token_ids"]] for row in context_rows],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    model.config.use_cache = True
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=continuation_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    model.config.use_cache = False
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    generated = output[:, input_ids.shape[1] :].tolist()
    rows = []
    for token_ids in generated:
        trimmed = []
        for token_id in token_ids:
            trimmed.append(token_id)
            if token_id in eos_ids:
                break
        rows.append(trimmed)
    return rows


def teacher_forced_token_scores(
    model,
    tokenizer,
    state: dict[str, int],
    context_rows: Sequence[dict[str, Any]],
    continuations: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = model.get_input_embeddings().weight.device
    bos = tokenizer.bos_token_id
    context_length = 1 + len(context_rows[0]["context_token_ids"])
    max_tokens = max(len(tokens) for tokens in continuations)
    pad = tokenizer.pad_token_id
    input_rows = []
    attention_rows = []
    target_rows = []
    valid_rows = []
    for row, tokens in zip(context_rows, continuations):
        prefix = [bos, *row["context_token_ids"]]
        inputs = prefix + list(tokens[:-1])
        inputs.extend([pad] * (context_length + max_tokens - 1 - len(inputs)))
        attention = [1] * (context_length + len(tokens) - 1)
        attention.extend([0] * (len(inputs) - len(attention)))
        targets = list(tokens) + [pad] * (max_tokens - len(tokens))
        valid = [True] * len(tokens) + [False] * (max_tokens - len(tokens))
        input_rows.append(inputs)
        attention_rows.append(attention)
        target_rows.append(targets)
        valid_rows.append(valid)
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
    targets = torch.tensor(target_rows, dtype=torch.long, device=device)
    valid_mask = torch.tensor(valid_rows, dtype=torch.bool, device=device)
    state["start_position"] = context_length - 1
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=max_tokens,
    ).logits
    token_scores = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    return token_scores, valid_mask


def estimate_fisher(
    model,
    tokenizer,
    alpha: torch.Tensor,
    state: dict[str, int],
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    continuation_tokens: int,
    probes: int,
    seed: int,
    matrix_mode: str | None,
) -> tuple[torch.Tensor | None, list[dict[str, Any]], dict[str, Any]]:
    validate_positive(
        [batch_size, continuation_tokens, probes], "Fisher runtime values"
    )
    device = alpha.device
    if matrix_mode == "dense":
        fisher = torch.zeros(
            alpha.numel(), alpha.numel(), device=device, dtype=torch.float32
        )
    elif matrix_mode == "diagonal":
        fisher = torch.zeros(alpha.numel(), device=device, dtype=torch.float32)
    elif matrix_mode is None:
        fisher = None
    else:
        raise ValueError(f"Unknown Fisher matrix mode: {matrix_mode}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probe_generator = torch.Generator(device=device).manual_seed(seed + 1)
    saved_rows: list[dict[str, Any]] = []
    score_vectors = 0
    generated_tokens = 0
    peak_memory = 0
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        with torch.no_grad():
            alpha.fill_(1.0)
        continuations = sample_continuations(
            model,
            tokenizer,
            batch,
            continuation_tokens,
        )
        token_scores, valid_mask = teacher_forced_token_scores(
            model, tokenizer, state, batch, continuations
        )
        vectors = []
        total_scores = len(batch) * probes
        completed_scores = 0
        for batch_index in range(len(batch)):
            for _ in range(probes):
                signs = torch.empty(
                    token_scores.shape[1], device=device, dtype=torch.float32
                ).bernoulli_(0.5, generator=probe_generator)
                signs.mul_(2).sub_(1)
                score = (
                    token_scores[batch_index] * signs * valid_mask[batch_index].float()
                ).sum()
                completed_scores += 1
                gradient = torch.autograd.grad(
                    score,
                    alpha,
                    retain_graph=completed_scores < total_scores,
                )[0]
                vectors.append(gradient.detach())
        matrix = torch.stack(vectors).float()
        if fisher is not None:
            if fisher.ndim == 2:
                fisher.addmm_(matrix.T, matrix)
            else:
                fisher.add_(matrix.square().sum(0))
        score_vectors += matrix.shape[0]
        generated_tokens += sum(len(tokens) for tokens in continuations)
        for row, tokens in zip(batch, continuations):
            saved_rows.append(
                {
                    "id": row["id"],
                    "source_row": row["source_row"],
                    "context_token_count": len(row["context_token_ids"]),
                    "generated_token_count": len(tokens),
                    "generated_token_ids": list(tokens),
                    "generated_text": tokenizer.decode(
                        tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )
        if torch.cuda.is_available():
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        print(
            f"Fisher contexts {min(start + len(batch), len(rows))}/{len(rows)}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    if fisher is not None:
        fisher.div_(score_vectors)
    return (
        fisher,
        saved_rows,
        {
            "contexts": len(rows),
            "score_vectors": score_vectors,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "contexts_per_second": len(rows) / elapsed,
            "peak_cuda_bytes": peak_memory,
        },
    )


def shrunk_fisher(
    fisher: torch.Tensor, shrinkage: float, damping_ratio: float
) -> tuple[torch.Tensor, float]:
    if not 0 <= shrinkage <= 1:
        raise ValueError("Shrinkage must be in [0, 1]")
    if damping_ratio <= 0:
        raise ValueError("Damping ratio must be positive")
    diagonal = fisher.diag().clamp_min(0)
    positive = diagonal[diagonal > 0]
    if not positive.numel():
        raise ValueError("Fisher diagonal is zero")
    damping = float(positive.median()) * damping_ratio
    shrunk = fisher.mul(1 - shrinkage)
    shrunk.diagonal().add_(shrinkage * diagonal + damping)
    return shrunk, damping


def nonnegative_natural_direction(
    matrix: torch.Tensor, gradient: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    a = matrix.detach().double().cpu().numpy()
    g = gradient.detach().double().cpu().numpy()

    def objective(x):
        return 0.5 * x.dot(a.dot(x)) - g.dot(x)

    def jacobian(x):
        return a.dot(x) - g

    initial = torch.linalg.solve(matrix, gradient).clamp_min(0).double().cpu().numpy()
    result = minimize(
        objective,
        initial,
        jac=jacobian,
        bounds=[(0.0, None)] * len(g),
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Nonnegative Fisher solve failed: {result.message}")
    direction = torch.from_numpy(result.x).to(dtype=torch.float32)
    return direction, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "active_positive_coordinates": int((direction > 0).sum()),
        "projected_gradient_max_abs": float(
            torch.minimum(
                direction,
                matrix.cpu().matmul(direction) - gradient.cpu(),
            )
            .abs()
            .max()
        ),
    }


def normalize_direction(
    direction: torch.Tensor, matrix: torch.Tensor, epsilon: float
) -> torch.Tensor:
    curvature = float(direction @ matrix.cpu() @ direction)
    if curvature <= 0 or epsilon <= 0:
        raise ValueError("Direction curvature and epsilon must be positive")
    return direction * math.sqrt(2 * epsilon / curvature)


def scale_artifact(
    mode: str,
    ranking: Sequence[dict[str, Any]],
    deltas: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for row, delta in zip(ranking, deltas.tolist()):
        rows.append(
            {
                **row,
                "delta": float(delta),
                "multiplier": 1.0 + float(delta),
            }
        )
    return {
        "schema": "fisher_grad_scales_v1",
        "mode": mode,
        "direction": "positive-only",
        "scope": "last",
        "top_k": len(rows),
        "rows": rows,
        **metadata,
    }


def load_scale_deltas(
    path: Path, ranking: Sequence[dict[str, Any]], expected_mode: str | None
) -> tuple[dict[str, Any], torch.Tensor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "fisher_grad_scales_v1"
        or payload.get("top_k") != len(ranking)
        or (expected_mode is not None and payload.get("mode") != expected_mode)
    ):
        raise ValueError(f"Incompatible scale artifact: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(ranking):
        raise ValueError(f"Wrong scale row count: {path}")
    for selected, row in zip(ranking, rows):
        if (selected["layer"], selected["neuron"]) != (
            int(row["layer"]),
            int(row["neuron"]),
        ):
            raise ValueError(f"Scale artifact ranking mismatch: {path}")
    return payload, torch.tensor([float(row["delta"]) for row in rows])


def continuation_batch_inputs(
    tokenizer,
    context_rows: Sequence[dict[str, Any]],
    continuations: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    device = "cpu"
    bos = tokenizer.bos_token_id
    context_length = 1 + len(context_rows[0]["context_token_ids"])
    max_tokens = max(len(tokens) for tokens in continuations)
    pad = tokenizer.pad_token_id
    input_rows = []
    attention_rows = []
    valid_rows = []
    for row, tokens in zip(context_rows, continuations):
        prefix = [bos, *row["context_token_ids"]]
        inputs = prefix + list(tokens[:-1])
        inputs.extend([pad] * (context_length + max_tokens - 1 - len(inputs)))
        attention = [1] * (context_length + len(tokens) - 1)
        attention.extend([0] * (len(inputs) - len(attention)))
        valid = [True] * len(tokens) + [False] * (max_tokens - len(tokens))
        input_rows.append(inputs)
        attention_rows.append(attention)
        valid_rows.append(valid)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attention_rows, dtype=torch.long, device=device),
        torch.tensor(valid_rows, dtype=torch.bool, device=device),
        max_tokens,
    )


def evaluate_actual_kls(
    model,
    tokenizer,
    alpha: torch.Tensor,
    state: dict[str, int],
    rows: Sequence[dict[str, Any]],
    continuations: Sequence[Sequence[int]],
    deltas: Sequence[torch.Tensor],
    batch_size: int,
) -> list[dict[str, float]]:
    device = alpha.device
    totals = torch.zeros(len(deltas), dtype=torch.float64)
    token_totals = torch.zeros(len(deltas), dtype=torch.float64)
    sequence_count = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        generated = continuations[start : start + batch_size]
        input_ids, attention_mask, valid_mask, max_tokens = continuation_batch_inputs(
            tokenizer, batch, generated
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        valid_mask = valid_mask.to(device)
        state["start_position"] = 1 + len(batch[0]["context_token_ids"]) - 1
        with torch.inference_mode():
            alpha.fill_(1.0)
            teacher_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=max_tokens,
            ).logits.float()
            teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
            teacher_p = teacher_logp.exp()
            for index, delta in enumerate(deltas):
                alpha.copy_(1.0 + delta.to(device))
                edited_logp = torch.log_softmax(
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=max_tokens,
                    ).logits.float(),
                    dim=-1,
                )
                token_kl = (teacher_p * (teacher_logp - edited_logp)).sum(-1)
                token_kl = token_kl.masked_fill(~valid_mask, 0)
                totals[index] += token_kl.sum(-1).double().cpu().sum()
                token_totals[index] += token_kl.double().cpu().sum()
        sequence_count += len(batch)
        print(
            f"KL contexts {min(start + len(batch), len(rows))}/{len(rows)}", flush=True
        )
    token_count = sum(len(tokens) for tokens in continuations)
    return [
        {
            "mean_sequence_kl": float(total / sequence_count),
            "mean_token_kl": float(token_total / token_count),
        }
        for total, token_total in zip(totals, token_totals)
    ]


def make_variants(args: argparse.Namespace) -> None:
    validate_positive(
        [
            *args.direct_top_k,
            *args.direct_strengths,
            *args.individual_scales,
            *args.individual_caps,
        ],
        "variant values",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for top_k in args.direct_top_k:
        ranking = read_positive_ranking(args.ranking, top_k)
        for strength in args.direct_strengths:
            artifact = scale_artifact(
                "shared",
                ranking,
                torch.full((top_k,), float(strength)),
                {
                    "source": "direct positive-only Grad control",
                    "ranking": str(args.ranking.resolve()),
                    "ranking_sha256": sha256_file(args.ranking),
                },
            )
            label = f"direct_k{top_k}_s{str(strength).replace('.', 'p')}"
            path = args.output_dir / f"{label}.json"
            artifact["label"] = label
            atomic_write_json(path, artifact)
            outputs.append({"label": label, "path": str(path.resolve())})

    base = json.loads(args.base_individual_scales.read_text(encoding="utf-8"))
    ranking = read_positive_ranking(args.ranking, int(base["top_k"]))
    _, base_delta = load_scale_deltas(
        args.base_individual_scales,
        ranking,
        str(base["mode"]),
    )
    for scale in args.individual_scales:
        label = f"individual_k{len(ranking)}_scale{str(scale).replace('.', 'p')}"
        artifact = scale_artifact(
            "individual_nonnegative_rescaled",
            ranking,
            base_delta * scale,
            {
                "source": str(args.base_individual_scales.resolve()),
                "source_sha256": sha256_file(args.base_individual_scales),
                "delta_scale": scale,
                "label": label,
            },
        )
        path = args.output_dir / f"{label}.json"
        atomic_write_json(path, artifact)
        outputs.append({"label": label, "path": str(path.resolve())})
    for cap in args.individual_caps:
        label = f"individual_k{len(ranking)}_cap{str(cap).replace('.', 'p')}"
        artifact = scale_artifact(
            "individual_nonnegative_capped",
            ranking,
            base_delta.clamp_max(cap),
            {
                "source": str(args.base_individual_scales.resolve()),
                "source_sha256": sha256_file(args.base_individual_scales),
                "delta_cap": cap,
                "label": label,
            },
        )
        path = args.output_dir / f"{label}.json"
        atomic_write_json(path, artifact)
        outputs.append({"label": label, "path": str(path.resolve())})
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "ranking": str(args.ranking.resolve()),
            "ranking_sha256": sha256_file(args.ranking),
            "variants": outputs,
        },
    )


def rescale_scales(args: argparse.Namespace) -> None:
    """Create smaller endpoints along one existing positive-only direction."""
    t_values = sorted(set(float(value) for value in args.t_values))
    if not t_values or any(not 0 < value < 1 for value in t_values):
        raise ValueError("t-values must be unique finite values strictly inside (0, 1)")
    base = json.loads(args.base_scales.read_text(encoding="utf-8"))
    if base.get("direction") != "positive-only":
        raise ValueError("Only positive-only scale artifacts may be rescaled")
    ranking = read_positive_ranking(args.ranking, int(base["top_k"]))
    _, base_delta = load_scale_deltas(args.base_scales, ranking, None)
    base_label = str(base.get("label") or args.base_scales.stem)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for t in t_values:
        t_label = str(t).replace(".", "p")
        label = f"{base_label}_t{t_label}"
        artifact = scale_artifact(
            "positive_only_path_rescaled",
            ranking,
            base_delta * t,
            {
                "source": str(args.base_scales.resolve()),
                "source_sha256": sha256_file(args.base_scales),
                "delta_scale": t,
                "label": label,
            },
        )
        path = args.output_dir / f"{label}.json"
        atomic_write_json(path, artifact)
        outputs.append({"label": label, "t": t, "path": str(path.resolve())})
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "ranking": str(args.ranking.resolve()),
            "ranking_sha256": sha256_file(args.ranking),
            "base_scales": str(args.base_scales.resolve()),
            "base_scales_sha256": sha256_file(args.base_scales),
            "variants": outputs,
        },
    )


def conservative_replacement_indices(
    scores: torch.Tensor, base_k: int, replace_count: int
) -> torch.Tensor:
    """Keep the best prefix neurons and replace the rest from the tail."""
    prefix = torch.arange(base_k)
    tail = torch.arange(base_k, scores.numel())
    prefix_order = prefix[torch.argsort(scores[prefix], descending=True)]
    tail_order = tail[torch.argsort(scores[tail], descending=True)]
    return torch.cat(
        [prefix_order[: base_k - replace_count], tail_order[:replace_count]]
    )


def box_fisher_deltas(
    gradient: torch.Tensor,
    curvature: torch.Tensor,
    reference_strength: float,
    curvature_power: float,
    delta_cap: float,
) -> tuple[torch.Tensor, float, float]:
    """Allocate a direct reference budget along a bounded Fisher direction."""
    if (
        reference_strength <= 0
        or curvature_power <= 0
        or delta_cap <= reference_strength
        or gradient.shape != curvature.shape
        or torch.any(curvature <= 0)
    ):
        raise ValueError("Invalid bounded Fisher allocation parameters")
    target = 0.5 * reference_strength**2 * float(curvature.sum())
    direction = gradient.clamp_min(0) / curvature.pow(curvature_power)

    def deltas_and_cost(scale: float) -> tuple[torch.Tensor, float]:
        deltas = (scale * direction).clamp(max=delta_cap)
        cost = 0.5 * float((curvature * deltas.square()).sum())
        return deltas, cost

    low = 0.0
    high = 1.0
    _, high_cost = deltas_and_cost(high)
    while high_cost < target:
        high *= 2
        _, high_cost = deltas_and_cost(high)
        if high > 1e12:
            raise RuntimeError("Could not bracket bounded Fisher budget")
    for _ in range(80):
        scale = 0.5 * (low + high)
        _, cost = deltas_and_cost(scale)
        if cost < target:
            low = scale
        else:
            high = scale
    deltas, achieved = deltas_and_cost(high)
    return deltas, high, achieved


def floor_fisher_deltas(
    gradient: torch.Tensor,
    curvature: torch.Tensor,
    floor: float,
    target_median: float,
    cap: float,
) -> tuple[torch.Tensor, float]:
    """Add a bounded natural-gradient increment above a shared direct floor."""
    if (
        gradient.ndim != 1
        or curvature.ndim != 1
        or gradient.shape != curvature.shape
        or floor < 0
        or not floor < target_median < cap
        or bool((gradient < 0).any())
        or bool((curvature <= 0).any())
    ):
        raise ValueError("Invalid floor Fisher allocation parameters")
    score = gradient / curvature
    if not bool((score > 0).any()):
        raise ValueError("Floor Fisher score has no positive entries")
    score_median = float(score.median())
    if score_median <= 0:
        raise ValueError("Floor Fisher score median must be positive")
    scale = (target_median - floor) / score_median
    deltas = capped_floor_fisher_deltas(gradient, curvature, floor, scale, cap)
    return deltas, float(scale)


def capped_floor_fisher_deltas(
    gradient: torch.Tensor,
    curvature: torch.Tensor,
    floor: float,
    score_scale: float,
    cap: float,
) -> torch.Tensor:
    if (
        gradient.ndim != 1
        or curvature.ndim != 1
        or gradient.shape != curvature.shape
        or floor < 0
        or score_scale <= 0
        or cap <= floor
        or bool((gradient < 0).any())
        or bool((curvature <= 0).any())
    ):
        raise ValueError("Invalid capped floor Fisher parameters")
    return (floor + score_scale * gradient / curvature).clamp(max=cap)


def make_floor_fisher_variants(args: argparse.Namespace) -> None:
    if args.target_medians is not None and args.score_scales is not None:
        raise ValueError("Use either target-medians or score-scales, not both")
    target_medians = (
        args.target_medians
        if args.target_medians is not None
        else ([] if args.score_scales is not None else [0.6, 0.75])
    )
    score_scales = args.score_scales or []
    validate_positive(
        [
            args.pool_k,
            *args.active_k,
            *target_medians,
            *score_scales,
            *args.caps,
            *args.damping_ratios,
        ],
        "floor Fisher variant values",
    )
    if any(value < 0 for value in args.floors):
        raise ValueError("floor Fisher floors must be nonnegative")
    if max(args.active_k) > args.pool_k:
        raise ValueError("active-k cannot exceed pool-k")
    payload = torch.load(args.fisher, map_location="cpu", weights_only=True)
    fisher = payload["fisher"].float()
    gradient = payload["gradient"].float().clamp_min(0)
    if fisher.ndim != 1 or fisher.numel() < args.pool_k:
        raise ValueError("Floor Fisher variants require a diagonal Fisher pool")
    fisher = fisher[: args.pool_k].clamp_min(0)
    gradient = gradient[: args.pool_k]
    ranking = read_positive_ranking(args.ranking, args.pool_k)
    median_fisher = float(fisher[fisher > 0].median())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for active_k in args.active_k:
        selected_ranking = ranking[:active_k]
        selected_gradient = gradient[:active_k]
        selected_fisher = fisher[:active_k]
        for damping_ratio in args.damping_ratios:
            damping = damping_ratio * median_fisher
            curvature = selected_fisher + damping
            for floor in args.floors:
                for cap in args.caps:
                    for target_median in target_medians:
                        if not floor < target_median < cap:
                            continue
                        deltas, scale = floor_fisher_deltas(
                            selected_gradient,
                            curvature,
                            floor,
                            target_median,
                            cap,
                        )
                        fields = {
                            "floor": floor,
                            "median": target_median,
                            "cap": cap,
                            "damp": damping_ratio,
                        }
                        write_floor_fisher_variant(
                            args,
                            outputs,
                            selected_ranking,
                            deltas,
                            active_k,
                            damping,
                            damping_ratio,
                            floor,
                            cap,
                            scale,
                            fields,
                            target_median,
                        )
                    for scale in score_scales:
                        deltas = capped_floor_fisher_deltas(
                            selected_gradient, curvature, floor, scale, cap
                        )
                        fields = {
                            "floor": floor,
                            "c": scale,
                            "cap": cap,
                            "damp": damping_ratio,
                        }
                        write_floor_fisher_variant(
                            args,
                            outputs,
                            selected_ranking,
                            deltas,
                            active_k,
                            damping,
                            damping_ratio,
                            floor,
                            cap,
                            scale,
                            fields,
                            None,
                        )
    finalize_floor_fisher_manifest(args, outputs)


def write_floor_fisher_variant(
    args: argparse.Namespace,
    outputs: list[dict[str, str]],
    selected_ranking: Sequence[dict[str, Any]],
    deltas: torch.Tensor,
    active_k: int,
    damping: float,
    damping_ratio: float,
    floor: float,
    cap: float,
    scale: float,
    fields: dict[str, float],
    target_median: float | None,
) -> None:
    encoded = "_".join(
        f"{name}{str(value).replace('.', 'p')}" for name, value in fields.items()
    )
    label = f"floorfisher_k{active_k}_{encoded}"
    artifact = scale_artifact(
        "individual_nonnegative_floor_fisher",
        selected_ranking,
        deltas,
        {
            "label": label,
            "source_fisher": str(args.fisher.resolve()),
            "source_fisher_sha256": sha256_file(args.fisher),
            "pool_k": args.pool_k,
            "active_k": active_k,
            "score": "gradient / (Fisher + damping)",
            "damping_ratio": damping_ratio,
            "damping": damping,
            "direct_floor": floor,
            "target_median": target_median,
            "delta_cap": cap,
            "score_scale_c": scale,
            "actual_delta_min": float(deltas.min()),
            "actual_delta_median": float(deltas.median()),
            "actual_delta_mean": float(deltas.mean()),
            "actual_delta_max": float(deltas.max()),
            "capped_count": int((deltas == cap).sum()),
        },
    )
    path = args.output_dir / f"{label}.json"
    atomic_write_json(path, artifact)
    outputs.append({"label": label, "path": str(path.resolve())})


def finalize_floor_fisher_manifest(
    args: argparse.Namespace, outputs: list[dict[str, str]]
) -> None:
    if not outputs:
        raise ValueError("No valid floor Fisher parameter combinations")
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "fisher": str(args.fisher.resolve()),
            "fisher_sha256": sha256_file(args.fisher),
            "ranking": str(args.ranking.resolve()),
            "ranking_sha256": sha256_file(args.ranking),
            "variants": outputs,
        },
    )


def make_dense_prefix_variants(args: argparse.Namespace) -> None:
    """Solve bounded nonnegative directions using dense Fisher prefixes."""
    validate_positive(
        [
            *args.active_k,
            *args.target_positive_medians,
            args.cap,
            args.damping_ratio,
        ],
        "dense prefix variant values",
    )
    if not 0 <= args.shrinkage < 1:
        raise ValueError("dense prefix shrinkage must be in [0, 1)")
    if any(value >= args.cap for value in args.target_positive_medians):
        raise ValueError("dense prefix target medians must be below the cap")

    payload = torch.load(args.fisher, map_location="cpu", weights_only=True)
    fisher = payload["fisher"].float()
    gradient = payload["gradient"].float().clamp_min(0)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("dense prefix variants require a square full Fisher")
    max_k = max(args.active_k)
    if max_k > fisher.shape[0] or gradient.numel() < max_k:
        raise ValueError("dense Fisher artifact does not cover the requested prefixes")

    ranking = read_positive_ranking(args.ranking, max_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for active_k in args.active_k:
        raw_prefix = fisher[:active_k, :active_k].clone()
        matrix, damping = shrunk_fisher(
            raw_prefix, args.shrinkage, args.damping_ratio
        )
        direction, solver = nonnegative_natural_direction(
            matrix, gradient[:active_k]
        )
        positive = direction[direction > 0]
        if not positive.numel():
            raise ValueError(f"dense Fisher K={active_k} direction is all zero")
        direction_positive_median = float(positive.median())
        for target_median in args.target_positive_medians:
            direction_scale = target_median / direction_positive_median
            deltas = (direction * direction_scale).clamp(max=args.cap)
            encoded_median = str(target_median).replace(".", "p")
            encoded_cap = str(args.cap).replace(".", "p")
            encoded_shrinkage = str(args.shrinkage).replace(".", "p")
            encoded_damping = str(args.damping_ratio).replace(".", "p")
            label = (
                f"densefisher_k{active_k}_m{encoded_median}_cap{encoded_cap}_"
                f"shrink{encoded_shrinkage}_damp{encoded_damping}"
            )
            artifact = scale_artifact(
                "dense_full_fisher_nonnegative_capped",
                ranking[:active_k],
                deltas,
                {
                    "label": label,
                    "source_fisher": str(args.fisher.resolve()),
                    "source_fisher_sha256": sha256_file(args.fisher),
                    "active_k": active_k,
                    "uses_off_diagonal_fisher": True,
                    "shrinkage": args.shrinkage,
                    "damping_ratio": args.damping_ratio,
                    "damping": damping,
                    "direct_floor": 0.0,
                    "delta_cap": args.cap,
                    "target_positive_median": target_median,
                    "direction_positive_median": direction_positive_median,
                    "direction_scale": direction_scale,
                    "actual_delta_min": float(deltas.min()),
                    "actual_delta_median": float(deltas.median()),
                    "actual_positive_delta_median": float(deltas[deltas > 0].median()),
                    "actual_delta_mean": float(deltas.mean()),
                    "actual_delta_max": float(deltas.max()),
                    "positive_count": int((deltas > 0).sum()),
                    "capped_count": int((deltas == args.cap).sum()),
                    "solver": solver,
                },
            )
            path = args.output_dir / f"{label}.json"
            atomic_write_json(path, artifact)
            outputs.append({"label": label, "path": str(path.resolve())})

    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "fisher": str(args.fisher.resolve()),
            "fisher_sha256": sha256_file(args.fisher),
            "ranking": str(args.ranking.resolve()),
            "ranking_sha256": sha256_file(args.ranking),
            "uses_off_diagonal_fisher": True,
            "variants": outputs,
        },
    )


def make_box_fisher_variants(args: argparse.Namespace) -> None:
    validate_positive(
        [
            args.pool_k,
            *args.reference_strengths,
            *args.damping_ratios,
            *args.curvature_powers,
            *args.cap_factors,
        ],
        "bounded Fisher variant values",
    )
    if any(factor <= 1 for factor in args.cap_factors):
        raise ValueError("cap-factors must exceed one")
    payload = torch.load(args.fisher, map_location="cpu", weights_only=True)
    fisher = payload["fisher"].float()
    gradient = payload["gradient"].float()
    if fisher.ndim != 1 or fisher.numel() < args.pool_k:
        raise ValueError("Bounded variants require a diagonal Fisher covering the pool")
    fisher = fisher[: args.pool_k]
    gradient = gradient[: args.pool_k]
    ranking = read_positive_ranking(args.ranking, args.pool_k)
    median = float(fisher[fisher > 0].median())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for reference_strength in args.reference_strengths:
        for damping_ratio in args.damping_ratios:
            damping = damping_ratio * median
            curvature = fisher.clamp_min(0) + damping
            for curvature_power in args.curvature_powers:
                for cap_factor in args.cap_factors:
                    delta_cap = cap_factor * reference_strength
                    deltas, direction_scale, achieved = box_fisher_deltas(
                        gradient,
                        curvature,
                        reference_strength,
                        curvature_power,
                        delta_cap,
                    )
                    target = 0.5 * reference_strength**2 * float(curvature.sum())
                    fields = {
                        "reference_strength": reference_strength,
                        "damping_ratio": damping_ratio,
                        "curvature_power": curvature_power,
                        "cap_factor": cap_factor,
                    }
                    encoded = "_".join(
                        f"{key}{str(value).replace('.', 'p')}"
                        for key, value in fields.items()
                    )
                    label = f"boxfisher_k{args.pool_k}_{encoded}"
                    artifact = scale_artifact(
                        "individual_nonnegative_box_fisher",
                        ranking,
                        deltas,
                        {
                            "label": label,
                            "source_fisher": str(args.fisher.resolve()),
                            "source_fisher_sha256": sha256_file(args.fisher),
                            "pool_k": args.pool_k,
                            **fields,
                            "delta_cap": delta_cap,
                            "damping": damping,
                            "direction_scale": direction_scale,
                            "predicted_target_quadratic_cost": target,
                            "predicted_achieved_quadratic_cost": achieved,
                        },
                    )
                    path = args.output_dir / f"{label}.json"
                    atomic_write_json(path, artifact)
                    outputs.append({"label": label, "path": str(path.resolve())})
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "fisher": str(args.fisher.resolve()),
            "ranking": str(args.ranking.resolve()),
            "variants": outputs,
        },
    )


def make_blend_variants(args: argparse.Namespace) -> None:
    if any(not 0 < weight < 1 for weight in args.weights):
        raise ValueError("Blend weights must be strictly between zero and one")
    first_payload = json.loads(args.first_scales.read_text(encoding="utf-8"))
    second_payload = json.loads(args.second_scales.read_text(encoding="utf-8"))
    top_k = int(first_payload["top_k"])
    if int(second_payload["top_k"]) != top_k:
        raise ValueError("Blend artifacts must have the same top-k")
    ranking = read_positive_ranking(args.ranking, top_k)
    _, first = load_scale_deltas(args.first_scales, ranking, None)
    _, second = load_scale_deltas(args.second_scales, ranking, None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for weight in args.weights:
        deltas = (1 - weight) * first + weight * second
        encoded = str(weight).replace(".", "p")
        label = f"blend_direct_fisher_weight{encoded}_k{top_k}"
        artifact = scale_artifact(
            "individual_nonnegative_direct_fisher_blend",
            ranking,
            deltas,
            {
                "label": label,
                "first_scales": str(args.first_scales.resolve()),
                "first_scales_sha256": sha256_file(args.first_scales),
                "second_scales": str(args.second_scales.resolve()),
                "second_scales_sha256": sha256_file(args.second_scales),
                "fisher_weight": weight,
            },
        )
        path = args.output_dir / f"{label}.json"
        atomic_write_json(path, artifact)
        outputs.append({"label": label, "path": str(path.resolve())})
    atomic_write_json(
        args.output_dir / "manifest.json",
        {"top_k": top_k, "variants": outputs},
    )


def anchored_tail_deltas(
    fisher_deltas: torch.Tensor,
    base_k: int,
    base_strength: float,
    tail_weight: float,
) -> torch.Tensor:
    if not 0 < base_k < fisher_deltas.numel():
        raise ValueError("base-k must be inside the Fisher pool")
    validate_positive([base_strength, tail_weight], "anchored-tail values")
    deltas = torch.zeros_like(fisher_deltas)
    deltas[:base_k] = base_strength
    deltas[base_k:] = tail_weight * fisher_deltas[base_k:]
    return deltas


def make_anchored_tail_variants(args: argparse.Namespace) -> None:
    payload = json.loads(args.fisher_scales.read_text(encoding="utf-8"))
    top_k = int(payload["top_k"])
    ranking = read_positive_ranking(args.ranking, top_k)
    _, fisher_deltas = load_scale_deltas(args.fisher_scales, ranking, None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for tail_weight in args.tail_weights:
        deltas = anchored_tail_deltas(
            fisher_deltas, args.base_k, args.base_strength, tail_weight
        )
        encoded_strength = str(args.base_strength).replace(".", "p")
        encoded_weight = str(tail_weight).replace(".", "p")
        label = (
            f"anchored_base{args.base_k}_s{encoded_strength}_"
            f"fishertail{encoded_weight}_k{top_k}"
        )
        artifact = scale_artifact(
            "individual_nonnegative_anchored_fisher_tail",
            ranking,
            deltas,
            {
                "label": label,
                "fisher_scales": str(args.fisher_scales.resolve()),
                "fisher_scales_sha256": sha256_file(args.fisher_scales),
                "base_k": args.base_k,
                "base_strength": args.base_strength,
                "tail_weight": tail_weight,
            },
        )
        path = args.output_dir / f"{label}.json"
        atomic_write_json(path, artifact)
        outputs.append({"label": label, "path": str(path.resolve())})
    atomic_write_json(
        args.output_dir / "manifest.json",
        {"top_k": top_k, "variants": outputs},
    )


def make_fisher_selection_variants(args: argparse.Namespace) -> None:
    validate_positive(
        [args.pool_k, args.base_k, *args.active_k, *args.strengths],
        "selection variant values",
    )
    if max(args.active_k) > args.pool_k:
        raise ValueError("active-k cannot exceed pool-k")
    if args.base_k > args.pool_k:
        raise ValueError("base-k cannot exceed pool-k")
    max_replacements = min(args.base_k, args.pool_k - args.base_k)
    if any(count <= 0 or count > max_replacements for count in args.replace_counts):
        raise ValueError("replace-counts exceed the available prefix or tail neurons")
    payload = torch.load(args.fisher, map_location="cpu", weights_only=True)
    fisher = payload["fisher"].float()
    gradient = payload["gradient"].float()
    if fisher.ndim != 1 or fisher.numel() != args.pool_k:
        raise ValueError("Selection variants require a matching diagonal Fisher")
    ranking = read_positive_ranking(args.ranking, args.pool_k)
    damping = float(fisher[fisher > 0].median())
    curvature = fisher.clamp_min(0) + damping
    score_values = {
        "gain_over_sqrt_fisher": gradient / curvature.sqrt(),
        "gain_over_fisher": gradient / curvature,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for score_name in args.scores:
        order = torch.argsort(score_values[score_name], descending=True)
        for active_k in args.active_k:
            selected = order[:active_k]
            for strength in args.strengths:
                deltas = torch.zeros(args.pool_k)
                deltas[selected] = strength
                label = (
                    f"fisherselect_{score_name}_active{active_k}_"
                    f"s{str(strength).replace('.', 'p')}"
                )
                artifact = scale_artifact(
                    "fisher_selected_shared_bounded",
                    ranking,
                    deltas,
                    {
                        "label": label,
                        "source_fisher": str(args.fisher.resolve()),
                        "source_fisher_sha256": sha256_file(args.fisher),
                        "selection_score": score_name,
                        "pool_k": args.pool_k,
                        "active_k": active_k,
                        "active_strength": strength,
                        "selection_damping": damping,
                    },
                )
                path = args.output_dir / f"{label}.json"
                atomic_write_json(path, artifact)
                outputs.append({"label": label, "path": str(path.resolve())})
        for replace_count in args.replace_counts:
            selected = conservative_replacement_indices(
                score_values[score_name], args.base_k, replace_count
            )
            for strength in args.strengths:
                deltas = torch.zeros(args.pool_k)
                deltas[selected] = strength
                label = (
                    f"fisherreplace_{score_name}_base{args.base_k}_"
                    f"replace{replace_count}_s{str(strength).replace('.', 'p')}"
                )
                artifact = scale_artifact(
                    "fisher_conservative_replacement_shared_bounded",
                    ranking,
                    deltas,
                    {
                        "label": label,
                        "source_fisher": str(args.fisher.resolve()),
                        "source_fisher_sha256": sha256_file(args.fisher),
                        "selection_score": score_name,
                        "pool_k": args.pool_k,
                        "base_k": args.base_k,
                        "replace_count": replace_count,
                        "active_k": args.base_k,
                        "active_strength": strength,
                        "selection_damping": damping,
                    },
                )
                path = args.output_dir / f"{label}.json"
                atomic_write_json(path, artifact)
                outputs.append({"label": label, "path": str(path.resolve())})
    atomic_write_json(
        args.output_dir / "manifest.json",
        {
            "fisher": str(args.fisher.resolve()),
            "ranking": str(args.ranking.resolve()),
            "variants": outputs,
        },
    )


def has_repeated_ngram(text: str, n: int = 4, threshold: int = 5) -> bool:
    words = text.split()
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - n + 1):
        ngram = tuple(words[index : index + n])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] >= threshold:
            return True
    return False


def greedy_raw_responses(
    model,
    tokenizer,
    alpha: torch.Tensor,
    state: dict[str, Any],
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    responses = []
    device = alpha.device
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    tokenizer.padding_side = "left"
    state["decode_all"] = True
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            list(batch),
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        state["start_position"] = encoded.input_ids.shape[1] - 1
        model.config.use_cache = True
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[:, encoded.input_ids.shape[1] :].tolist()
        for token_ids in generated:
            trimmed = []
            for token_id in token_ids:
                if token_id == tokenizer.pad_token_id and trimmed:
                    break
                trimmed.append(token_id)
                if token_id in eos_ids:
                    break
            responses.append(
                tokenizer.decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
    model.config.use_cache = False
    state["decode_all"] = False
    return responses


def evaluate_safety(args: argparse.Namespace) -> None:
    validate_positive([args.batch_size, args.max_new_tokens], "safety runtime values")
    source = read_jsonl(args.manifest)
    prompts = [str(row["prompt"]) for row in source]
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.scale_files
    ]
    max_top_k = max(int(payload["top_k"]) for payload in payloads)
    ranking = read_positive_ranking(args.ranking, max_top_k)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    alpha, handles, state = attach_selected_alphas(model, ranking)
    all_rows = []
    summaries = []
    try:
        for path, payload in zip(args.scale_files, payloads):
            top_k = int(payload["top_k"])
            selected = ranking[:top_k]
            rows = payload["rows"]
            if len(rows) != top_k or any(
                (entry["layer"], entry["neuron"])
                != (int(row["layer"]), int(row["neuron"]))
                for entry, row in zip(selected, rows)
            ):
                raise ValueError(f"Scale artifact does not match ranking: {path}")
            with torch.no_grad():
                alpha.fill_(1.0)
                alpha[:top_k] = torch.tensor(
                    [float(row["multiplier"]) for row in rows],
                    device=alpha.device,
                )
            responses = greedy_raw_responses(
                model,
                tokenizer,
                alpha,
                state,
                prompts,
                args.batch_size,
                args.max_new_tokens,
            )
            label = str(payload.get("label") or path.stem)
            attacks = 0
            repetitive = 0
            for source_row, response in zip(source, responses):
                jailbroken, matches = score_asr(response)
                attacks += int(jailbroken)
                repetitive += int(has_repeated_ngram(response))
                all_rows.append(
                    {
                        "experiment": label,
                        "id": source_row["id"],
                        "prompt": source_row["prompt"],
                        "response": response,
                        "jailbroken": jailbroken,
                        "matched_refusal_prefixes": matches,
                    }
                )
            summary = {
                "experiment": label,
                "scale_file": str(path.resolve()),
                "scale_file_sha256": sha256_file(path),
                "top_k": top_k,
                "num_samples": len(source),
                "attack_success_count": attacks,
                "attack_success_rate": 100 * attacks / len(source),
                "repetitive_responses": repetitive,
                "blank_responses": sum(not response.strip() for response in responses),
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(args.output_dir / "responses.jsonl", all_rows)
    atomic_write_json(
        args.output_dir / "summary.json",
        {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "prompt_format": "raw",
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "summaries": summaries,
        },
    )


def benchmark(args: argparse.Namespace) -> None:
    validate_positive(
        [args.context_limit, args.top_k, args.continuation_tokens, args.probes],
        "benchmark values",
    )
    rows = read_jsonl(args.contexts)[: args.context_limit]
    ranking = read_positive_ranking(args.ranking, args.top_k)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    alpha, handles, state = attach_selected_alphas(model, ranking)
    results = []
    try:
        for batch_size in args.batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(alpha.device)
            _, _, stats = estimate_fisher(
                model,
                tokenizer,
                alpha,
                state,
                rows,
                batch_size,
                args.continuation_tokens,
                args.probes,
                args.seed,
                matrix_mode=None,
            )
            results.append({"batch_size": batch_size, **stats})
    finally:
        for handle in handles:
            handle.remove()
    best = max(results, key=lambda row: row["contexts_per_second"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output_dir / "benchmark.json",
        {
            "model": str(args.model.resolve()),
            "dtype": args.dtype,
            "gradient_checkpointing": False,
            "reason_gradient_checkpointing_disabled": (
                "Frozen 8B model and tested sequence fit H100 memory; checkpointing "
                "would recompute the retained graph for every Fisher probe."
            ),
            "contexts_path": str(args.contexts.resolve()),
            "contexts_tested": len(rows),
            "continuation_tokens": args.continuation_tokens,
            "probes": args.probes,
            "top_k": args.top_k,
            "results": results,
            "best_batch_size": best["batch_size"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )


def compute(args: argparse.Namespace) -> None:
    validate_positive(
        [
            args.top_k,
            args.batch_size,
            args.continuation_tokens,
            args.probes,
            args.reference_shared_delta,
        ],
        "compute values",
    )
    rows = read_jsonl(args.contexts)
    ranking = read_positive_ranking(args.ranking, args.top_k)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    alpha, handles, state = attach_selected_alphas(model, ranking)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fisher, continuations, stats = estimate_fisher(
            model,
            tokenizer,
            alpha,
            state,
            rows,
            args.batch_size,
            args.continuation_tokens,
            args.probes,
            args.seed,
            matrix_mode=args.matrix_mode,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert fisher is not None
    fisher_cpu = fisher.cpu()
    gradient = torch.tensor([row["mean_g"] for row in ranking], dtype=torch.float32)
    shared_direction = torch.ones(args.top_k, dtype=torch.float32)
    reference_delta = args.reference_shared_delta * shared_direction
    if fisher_cpu.ndim == 2:
        matrix, damping = shrunk_fisher(fisher_cpu, args.shrinkage, args.damping_ratio)
        epsilon = 0.5 * float(reference_delta @ matrix @ reference_delta)
        individual_direction, solver = nonnegative_natural_direction(matrix, gradient)
        individual_delta = normalize_direction(individual_direction, matrix, epsilon)
        shared_delta = normalize_direction(shared_direction, matrix, epsilon)
        diagonal = fisher_cpu.diag()
    else:
        diagonal = fisher_cpu.clamp_min(0)
        positive = diagonal[diagonal > 0]
        damping = float(positive.median()) * args.damping_ratio
        matrix = diagonal + damping
        epsilon = 0.5 * float((reference_delta.square() * matrix).sum())
        individual_direction = gradient / matrix
        individual_delta = individual_direction * math.sqrt(
            2 * epsilon / float((individual_direction.square() * matrix).sum())
        )
        shared_delta = shared_direction * math.sqrt(
            2 * epsilon / float((shared_direction.square() * matrix).sum())
        )
        solver = {
            "success": True,
            "message": "closed-form positive diagonal solve",
            "iterations": 0,
            "active_positive_coordinates": args.top_k,
            "projected_gradient_max_abs": 0.0,
        }

    continuation_path = args.output_dir / "sampled_continuations.jsonl"
    atomic_write_jsonl(continuation_path, continuations)
    torch.save(
        {
            "fisher": fisher_cpu,
            "shrunk_fisher": matrix,
            "gradient": gradient,
        },
        args.output_dir / "fisher.pt",
    )
    common = {
        "model": str(args.model.resolve()),
        "ranking": str(args.ranking.resolve()),
        "ranking_sha256": sha256_file(args.ranking),
        "contexts": str(args.contexts.resolve()),
        "contexts_sha256": sha256_file(args.contexts),
        "continuation_tokens": args.continuation_tokens,
        "probes_per_context": args.probes,
        "fisher_estimator": (
            "model-sampled raw continuations; Rademacher token-score projections; "
            "average outer product over context/probe vectors"
        ),
        "shrinkage": args.shrinkage,
        "matrix_mode": args.matrix_mode,
        "damping_ratio_to_median_positive_diagonal": args.damping_ratio,
        "damping": damping,
        "quadratic_epsilon": epsilon,
        "reference_shared_delta": args.reference_shared_delta,
    }
    shared = scale_artifact("shared", ranking, shared_delta, common)
    individual = scale_artifact(
        "individual_nonnegative",
        ranking,
        individual_delta,
        {**common, "solver": solver},
    )
    atomic_write_json(args.output_dir / "shared_scales.json", shared)
    atomic_write_json(args.output_dir / "individual_scales.json", individual)
    atomic_write_json(
        args.output_dir / "metadata.json",
        {
            **common,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "runtime": stats,
            "fisher_trace": float(diagonal.sum()),
            "fisher_diagonal_min": float(diagonal.min()),
            "fisher_diagonal_median": float(diagonal.median()),
            "fisher_diagonal_max": float(diagonal.max()),
            "individual_delta_min": float(individual_delta.min()),
            "individual_delta_median": float(individual_delta.median()),
            "individual_delta_max": float(individual_delta.max()),
            "shared_delta": float(shared_delta[0]),
            "linear_target_gain_shared": float(gradient @ shared_delta),
            "linear_target_gain_individual": float(gradient @ individual_delta),
            "solver": solver,
            "artifacts": {
                "fisher": str((args.output_dir / "fisher.pt").resolve()),
                "continuations": str(continuation_path.resolve()),
                "continuations_sha256": sha256_file(continuation_path),
                "shared_scales": str(
                    (args.output_dir / "shared_scales.json").resolve()
                ),
                "individual_scales": str(
                    (args.output_dir / "individual_scales.json").resolve()
                ),
            },
        },
    )


def validate_kl(args: argparse.Namespace) -> None:
    validate_positive(
        [
            args.top_k,
            args.batch_size,
            args.continuation_tokens,
            args.calibration_iterations,
        ],
        "KL validation values",
    )
    rows = read_jsonl(args.contexts)
    ranking = read_positive_ranking(args.ranking, args.top_k)
    shared_payload, shared_delta = load_scale_deltas(
        args.shared_scales, ranking, "shared"
    )
    individual_payload, individual_delta = load_scale_deltas(
        args.individual_scales, ranking, None
    )
    if not str(individual_payload.get("mode", "")).startswith("individual_nonnegative"):
        raise ValueError("KL calibration requires a nonnegative individual direction")
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    alpha, handles, state = attach_selected_alphas(model, ranking)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    continuations: list[list[int]] = []
    try:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            with torch.no_grad():
                alpha.fill_(1.0)
            continuations.extend(
                sample_continuations(model, tokenizer, batch, args.continuation_tokens)
            )
        initial = evaluate_actual_kls(
            model,
            tokenizer,
            alpha,
            state,
            rows,
            continuations,
            [shared_delta, individual_delta],
            args.batch_size,
        )
        target_kl = initial[0]["mean_sequence_kl"]
        scale = math.sqrt(
            target_kl
            / max(initial[1]["mean_sequence_kl"], torch.finfo(torch.float64).tiny)
        )
        calibration = [
            {
                "iteration": 0,
                "scale": 1.0,
                **initial[1],
            }
        ]
        calibrated_stats = initial[1]
        for iteration in range(1, args.calibration_iterations + 1):
            calibrated_stats = evaluate_actual_kls(
                model,
                tokenizer,
                alpha,
                state,
                rows,
                continuations,
                [individual_delta * scale],
                args.batch_size,
            )[0]
            calibration.append(
                {"iteration": iteration, "scale": scale, **calibrated_stats}
            )
            ratio = target_kl / max(
                calibrated_stats["mean_sequence_kl"],
                torch.finfo(torch.float64).tiny,
            )
            if abs(ratio - 1) < 0.01 or iteration == args.calibration_iterations:
                break
            scale *= math.sqrt(ratio)
    finally:
        for handle in handles:
            handle.remove()
    calibrated_delta = individual_delta * scale
    shared_payload.update(
        {
            "validation_contexts": str(args.contexts.resolve()),
            "validation_contexts_sha256": sha256_file(args.contexts),
            "actual_mean_sequence_kl": initial[0]["mean_sequence_kl"],
            "actual_mean_token_kl": initial[0]["mean_token_kl"],
        }
    )
    for row, delta in zip(individual_payload["rows"], calibrated_delta.tolist()):
        row["uncalibrated_delta"] = row["delta"]
        row["delta"] = float(delta)
        row["multiplier"] = 1.0 + float(delta)
    individual_payload.update(
        {
            "mode": "individual_nonnegative_kl_calibrated",
            "validation_contexts": str(args.contexts.resolve()),
            "validation_contexts_sha256": sha256_file(args.contexts),
            "kl_calibration_scale": scale,
            "actual_mean_sequence_kl": calibrated_stats["mean_sequence_kl"],
            "actual_mean_token_kl": calibrated_stats["mean_token_kl"],
            "target_shared_mean_sequence_kl": target_kl,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_path = args.output_dir / "shared_scales_validated.json"
    individual_path = args.output_dir / "individual_scales_kl_calibrated.json"
    atomic_write_json(shared_path, shared_payload)
    atomic_write_json(individual_path, individual_payload)
    continuation_path = args.output_dir / "validation_continuations.jsonl"
    atomic_write_jsonl(
        continuation_path,
        (
            {
                "id": row["id"],
                "source_row": row["source_row"],
                "generated_token_ids": tokens,
                "generated_text": tokenizer.decode(
                    tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            }
            for row, tokens in zip(rows, continuations)
        ),
    )
    atomic_write_json(
        args.output_dir / "kl_validation.json",
        {
            "model": str(args.model.resolve()),
            "contexts": str(args.contexts.resolve()),
            "contexts_sha256": sha256_file(args.contexts),
            "context_count": len(rows),
            "continuation_tokens": args.continuation_tokens,
            "generated_tokens": sum(len(tokens) for tokens in continuations),
            "seed": args.seed,
            "dtype": args.dtype,
            "shared": initial[0],
            "individual_uncalibrated": initial[1],
            "individual_calibration": calibration,
            "individual_final_scale": scale,
            "individual_final": calibrated_stats,
            "artifacts": {
                "shared": str(shared_path.resolve()),
                "individual": str(individual_path.resolve()),
                "continuations": str(continuation_path.resolve()),
            },
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare_contexts(args)
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "compute":
        compute(args)
    elif args.command == "validate-kl":
        validate_kl(args)
    elif args.command == "make-variants":
        make_variants(args)
    elif args.command == "rescale-scales":
        rescale_scales(args)
    elif args.command == "evaluate-safety":
        evaluate_safety(args)
    elif args.command == "make-fisher-selection-variants":
        make_fisher_selection_variants(args)
    elif args.command == "make-box-fisher-variants":
        make_box_fisher_variants(args)
    elif args.command == "make-floor-fisher-variants":
        make_floor_fisher_variants(args)
    elif args.command == "make-dense-prefix-variants":
        make_dense_prefix_variants(args)
    elif args.command == "make-blend-variants":
        make_blend_variants(args)
    elif args.command == "make-anchored-tail-variants":
        make_anchored_tail_variants(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
