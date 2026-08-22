from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file
from .fisher_grad import (
    DEFAULT_RANKING,
    attach_selected_alphas,
    continuation_batch_inputs,
    load_model,
    load_scale_deltas,
    read_positive_ranking,
    sample_continuations,
    teacher_forced_token_scores,
)
from .methods import DEFAULT_LLAMA3


DEFAULT_TARGET_EXAMPLES = Path(
    "results/grad_onpolicy_sn_safe256_first_cue_tail_expanded50000/"
    "gradients/examples.jsonl"
)
DEFAULT_TARGET_RESPONSES = Path(
    "/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/training/"
    "circuit_breakers_train_llama3_raw_regenerated_bf16_greedy256_safe256/"
    "safe_responses.jsonl"
)
DEFAULT_CONTEXTS = Path(
    "/workspace/xcy/dataset/wikitext/wikitext-2-raw-v1/"
    "firstcue_fisher_seed42_n2048/validation_contexts.jsonl"
)
DEFAULT_FISHER = Path(
    "results/grad_firstcue_fisher_diag_wikitext2048_k16000/fisher.pt"
)
# Updated after benchmarking this exact batched target-span workload on real
# first-cue examples from the 256-example source corpus.
DEFAULT_TARGET_BATCH_SIZE = 8
DEFAULT_GENERAL_BATCH_SIZE = 4
DEFAULT_T_VALUES = [
    0.0,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate finite-radius target-gradient and Fisher approximations "
            "for existing positive-only Grad scale artifacts."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark-target")
    add_common_arguments(benchmark, require_scales=False)
    benchmark.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    benchmark.add_argument("--target-limit", type=int, default=32)

    run = subparsers.add_parser("run")
    add_common_arguments(run, require_scales=True)
    run.add_argument("--fisher", type=Path, default=DEFAULT_FISHER)
    run.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    run.add_argument("--target-limit", type=int)
    run.add_argument("--context-limit", type=int)
    run.add_argument("--target-batch-size", type=int, default=DEFAULT_TARGET_BATCH_SIZE)
    run.add_argument("--general-batch-size", type=int, default=DEFAULT_GENERAL_BATCH_SIZE)
    run.add_argument("--continuation-tokens", type=int, default=32)
    run.add_argument("--probes", type=int, default=4)
    run.add_argument("--t-values", type=float, nargs="+", default=DEFAULT_T_VALUES)
    run.add_argument("--bootstrap-samples", type=int, default=2000)
    run.add_argument("--overwrite", action="store_true")
    return parser


def add_common_arguments(parser: argparse.ArgumentParser, require_scales: bool) -> None:
    parser.add_argument("--model", type=Path, default=DEFAULT_LLAMA3)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--target-examples", type=Path, default=DEFAULT_TARGET_EXAMPLES)
    parser.add_argument("--target-responses", type=Path, default=DEFAULT_TARGET_RESPONSES)
    parser.add_argument("--scale-files", type=Path, nargs="+", required=require_scales)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12000)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=112)


def validate_positive(values: Sequence[int | float], label: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{label} must be positive")


def prepare_target_records(
    tokenizer,
    example_path: Path,
    response_path: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    examples = read_jsonl(example_path)
    responses = {
        int(row["source_index"]): row for row in read_jsonl(response_path)
    }
    if limit is not None:
        validate_positive([limit], "target limit")
        examples = examples[:limit]
    records = []
    for example in examples:
        source_index = int(example["source_index"])
        if source_index not in responses:
            raise ValueError(f"Missing target source row {source_index}")
        source = responses[source_index]
        prompt = str(source["prompt"])
        response = str(source["model_response"])
        response_start = int(example["response_target_char_start"])
        response_end = int(example["response_target_char_end"])
        target_text = str(example["target_text"])
        if response[response_start:response_end] != target_text:
            raise ValueError(f"Target span mismatch for source row {source_index}")
        full_text = prompt + response[:response_end]
        target_start = len(prompt) + response_start
        target_end = len(prompt) + response_end
        encoded = tokenizer(
            full_text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        target_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and end > target_start and start < target_end
        ]
        if not target_indices or target_indices[0] == 0:
            raise ValueError(f"Could not align target for source row {source_index}")
        expected = list(range(target_indices[0], target_indices[-1] + 1))
        if target_indices != expected:
            raise ValueError(f"Non-contiguous target for source row {source_index}")
        records.append(
            {
                "id": str(example["id"]),
                "source_index": source_index,
                "input_ids": list(encoded["input_ids"]),
                "target_indices": target_indices,
                "start_position": target_indices[0] - 1,
                "target_text": target_text,
            }
        )
    return records


def target_batch_tensors(
    records: Sequence[dict[str, Any]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(row["input_ids"]) for row in records)
    input_rows = []
    attention_rows = []
    for row in records:
        token_ids = list(row["input_ids"])
        padding = max_length - len(token_ids)
        input_rows.append(token_ids + [pad_token_id] * padding)
        attention_rows.append([1] * len(token_ids) + [0] * padding)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attention_rows, dtype=torch.long, device=device),
        torch.tensor(
            [int(row["start_position"]) for row in records],
            dtype=torch.long,
            device=device,
        ),
    )


def score_target_records(
    model,
    tokenizer,
    alpha: torch.Tensor,
    state: dict[str, Any],
    records: Sequence[dict[str, Any]],
    delta: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    validate_positive([batch_size], "target batch size")
    if delta.numel() != alpha.numel():
        raise ValueError("Target delta size does not match selected alphas")
    device = alpha.device
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        alpha.copy_(1.0 + delta.to(device))
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            input_ids, attention_mask, starts = target_batch_tensors(
                batch, tokenizer.pad_token_id, device
            )
            state["start_positions"] = starts
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)
                for row_index, row in enumerate(batch):
                    target_indices = torch.tensor(
                        row["target_indices"], dtype=torch.long, device=device
                    )
                    positions = target_indices - 1
                    targets = input_ids[row_index, target_indices]
                    token_scores = log_probs[row_index, positions, targets]
                    scores.append(token_scores.mean().detach().cpu())
    finally:
        state.pop("start_positions", None)
    return torch.stack(scores).double()


def directional_curvatures(
    score_vectors: torch.Tensor, directions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full and diagonal directional curvature per context.

    score_vectors has shape [contexts, probes, coordinates] and directions has
    shape [directions, coordinates]. The output tensors have shape
    [contexts, directions].
    """
    if score_vectors.ndim != 3 or directions.ndim != 2:
        raise ValueError("Directional curvature tensors have invalid ranks")
    if score_vectors.shape[-1] != directions.shape[-1]:
        raise ValueError("Directional curvature coordinate counts do not match")
    projections = torch.einsum("cpk,dk->cpd", score_vectors, directions)
    full = projections.square().mean(dim=1)
    diagonal = torch.einsum(
        "cpk,dk->cpd", score_vectors.square(), directions.square()
    ).mean(dim=1)
    return full, diagonal


def estimate_directional_fisher(
    model,
    tokenizer,
    alpha: torch.Tensor,
    state: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    directions: torch.Tensor,
    batch_size: int,
    continuation_tokens: int,
    probes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    validate_positive(
        [batch_size, continuation_tokens, probes], "directional Fisher values"
    )
    if directions.ndim != 2 or directions.shape[1] != alpha.numel():
        raise ValueError("Directional Fisher directions do not match alpha")
    device = alpha.device
    direction_device = directions.to(device=device, dtype=torch.float32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probe_generator = torch.Generator(device=device).manual_seed(seed + 1)
    full_rows = []
    diagonal_rows = []
    saved_rows: list[dict[str, Any]] = []
    generated_tokens = 0
    started = time.perf_counter()
    state.pop("start_positions", None)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        with torch.no_grad():
            alpha.fill_(1.0)
        continuations = sample_continuations(
            model, tokenizer, batch, continuation_tokens
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
                    token_scores[batch_index]
                    * signs
                    * valid_mask[batch_index].float()
                ).sum()
                completed_scores += 1
                gradient = torch.autograd.grad(
                    score,
                    alpha,
                    retain_graph=completed_scores < total_scores,
                )[0]
                vectors.append(gradient.detach())
        score_vectors = torch.stack(vectors).float().view(
            len(batch), probes, alpha.numel()
        )
        full, diagonal = directional_curvatures(score_vectors, direction_device)
        full_rows.append(full.cpu())
        diagonal_rows.append(diagonal.cpu())
        generated_tokens += sum(len(tokens) for tokens in continuations)
        for row, tokens in zip(batch, continuations):
            saved_rows.append(
                {
                    "id": row["id"],
                    "source_row": row["source_row"],
                    "generated_token_count": len(tokens),
                    "generated_token_ids": list(tokens),
                    "generated_text": tokenizer.decode(
                        tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )
        print(
            f"directional_fisher_contexts={min(start + len(batch), len(rows))}/"
            f"{len(rows)}",
            flush=True,
        )
    return (
        torch.cat(full_rows).double(),
        torch.cat(diagonal_rows).double(),
        saved_rows,
        {
            "contexts": len(rows),
            "probes": probes,
            "score_vectors": len(rows) * probes,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def evaluate_path_kls(
    model,
    tokenizer,
    alpha: torch.Tensor,
    state: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    continuations: Sequence[Sequence[int]],
    deltas: Sequence[torch.Tensor],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-context sequence KL and per-context generated token counts."""
    device = alpha.device
    all_kls = torch.zeros((len(rows), len(deltas)), dtype=torch.float64)
    token_counts = torch.tensor(
        [len(tokens) for tokens in continuations], dtype=torch.float64
    )
    state.pop("start_positions", None)
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
            # The sequence KL is quadratic in small edits, while individual
            # vocabulary terms contain cancelling first-order contributions.
            # Keep model computation in FP32 but perform the full-vocabulary
            # normalization and reduction in FP64 so the smallest path points
            # are not dominated by cancellation error.
            teacher_logp = torch.log_softmax(teacher_logits.double(), dim=-1)
            teacher_p = teacher_logp.exp()
            for delta_index, delta in enumerate(deltas):
                alpha.copy_(1.0 + delta.to(device))
                edited_logp = torch.log_softmax(
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=max_tokens,
                    ).logits.double(),
                    dim=-1,
                )
                token_kl = (teacher_p * (teacher_logp - edited_logp)).sum(-1)
                token_kl = token_kl.masked_fill(~valid_mask, 0)
                sequence_kl = token_kl.sum(-1).double().cpu()
                all_kls[start : start + len(batch), delta_index] = sequence_kl
        print(
            f"actual_kl_contexts={min(start + len(batch), len(rows))}/{len(rows)}",
            flush=True,
        )
    with torch.no_grad():
        alpha.fill_(1.0)
    return all_kls, token_counts


def bootstrap_mean_ci(
    values: torch.Tensor, samples: int, seed: int
) -> tuple[float, float]:
    validate_positive([samples], "bootstrap samples")
    values = values.double().flatten()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(), (samples, values.numel()), generator=generator
    )
    means = values[indices].mean(dim=1)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(bounds[0]), float(bounds[1])


def bootstrap_ratio_ci(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    numerator = numerator.double().flatten()
    denominator = denominator.double().flatten()
    if numerator.numel() != denominator.numel():
        raise ValueError("Paired bootstrap tensors must have equal lengths")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        numerator.numel(), (samples, numerator.numel()), generator=generator
    )
    ratios = numerator[indices].mean(dim=1) / denominator[indices].mean(dim=1)
    bounds = torch.quantile(ratios, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(bounds[0]), float(bounds[1])


def benchmark_target(args: argparse.Namespace) -> None:
    validate_positive([args.top_k, args.target_limit, *args.batch_sizes], "benchmark")
    ranking = read_positive_ranking(args.ranking, args.top_k)
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    records = prepare_target_records(
        tokenizer, args.target_examples, args.target_responses, args.target_limit
    )
    alpha, handles, state = attach_selected_alphas(model, ranking)
    zero = torch.zeros(args.top_k)
    results = []
    try:
        score_target_records(
            model, tokenizer, alpha, state, records[: min(2, len(records))], zero, 1
        )
        for batch_size in args.batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(alpha.device)
                torch.cuda.synchronize(alpha.device)
            started = time.perf_counter()
            status = "ok"
            error = None
            try:
                score_target_records(
                    model, tokenizer, alpha, state, records, zero, batch_size
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(alpha.device)
            except torch.OutOfMemoryError as exc:
                status = "oom"
                error = str(exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elapsed = time.perf_counter() - started
            results.append(
                {
                    "batch_size": batch_size,
                    "status": status,
                    "examples_per_second": (
                        len(records) / elapsed if status == "ok" else None
                    ),
                    "elapsed_seconds": elapsed,
                    "peak_cuda_bytes": (
                        torch.cuda.max_memory_allocated(alpha.device)
                        if torch.cuda.is_available()
                        else 0
                    ),
                    "error": error,
                }
            )
    finally:
        for handle in handles:
            handle.remove()
    viable = [row for row in results if row["status"] == "ok"]
    if not viable:
        raise RuntimeError("Every target-path benchmark batch size failed")
    best = max(viable, key=lambda row: row["examples_per_second"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output_dir / "benchmark.json",
        {
            "model": str(args.model.resolve()),
            "dtype": args.dtype,
            "gradient_checkpointing": False,
            "reason_gradient_checkpointing_disabled": (
                "Inference-only target-span scoring fits without retained gradients."
            ),
            "target_examples": str(args.target_examples.resolve()),
            "target_examples_sha256": sha256_file(args.target_examples),
            "examples_tested": len(records),
            "top_k": args.top_k,
            "results": results,
            "best_batch_size": best["batch_size"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    print(json.dumps({"results": results, "best": best}, indent=2), flush=True)


def validate_t_values(values: Sequence[float]) -> list[float]:
    unique = sorted(set(float(value) for value in values))
    if not unique or unique[0] < 0 or unique[-1] > 1:
        raise ValueError("t values must lie in [0, 1]")
    if 0.0 not in unique or 1.0 not in unique:
        raise ValueError("t values must include both 0 and 1")
    return unique


def run(args: argparse.Namespace) -> None:
    validate_positive(
        [
            args.top_k,
            args.target_batch_size,
            args.general_batch_size,
            args.continuation_tokens,
            args.probes,
            args.bootstrap_samples,
        ],
        "verification runtime values",
    )
    t_values = validate_t_values(args.t_values)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {summary_path}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranking = read_positive_ranking(args.ranking, args.top_k)
    direction_payloads = []
    directions = []
    for scale_path in args.scale_files:
        payload, delta = load_scale_deltas(scale_path, ranking, None)
        direction_payloads.append((scale_path, payload))
        directions.append(delta.float())
    direction_matrix = torch.stack(directions)
    labels = [str(payload["label"]) for _, payload in direction_payloads]
    if len(set(labels)) != len(labels):
        raise ValueError("Scale artifact labels must be unique")

    fisher_payload = torch.load(args.fisher, map_location="cpu", weights_only=True)
    fisher_diagonal = fisher_payload["fisher"].float()[: args.top_k].clamp_min(0)
    if fisher_diagonal.numel() != args.top_k:
        raise ValueError("Fisher artifact does not cover top-k")
    train_diag_curvature = torch.einsum(
        "dk,k->d", direction_matrix.square(), fisher_diagonal
    ).double()

    contexts = read_jsonl(args.contexts)
    if args.context_limit is not None:
        validate_positive([args.context_limit], "context limit")
        contexts = contexts[: args.context_limit]
    model, tokenizer = load_model(args.model, args.device, args.dtype)
    target_records = prepare_target_records(
        tokenizer,
        args.target_examples,
        args.target_responses,
        args.target_limit,
    )
    alpha, handles, state = attach_selected_alphas(model, ranking)
    target_raw: dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    try:
        baseline_target = score_target_records(
            model,
            tokenizer,
            alpha,
            state,
            target_records,
            torch.zeros(args.top_k),
            args.target_batch_size,
        )
        for label, direction in zip(labels, directions):
            values = [baseline_target]
            for t in t_values[1:]:
                print(f"target_path label={label} t={t:g}", flush=True)
                values.append(
                    score_target_records(
                        model,
                        tokenizer,
                        alpha,
                        state,
                        target_records,
                        direction * t,
                        args.target_batch_size,
                    )
                )
            target_raw[label] = torch.stack(values, dim=1)

        full_curvature, validation_diag_curvature, sampled_rows, fisher_stats = (
            estimate_directional_fisher(
                model,
                tokenizer,
                alpha,
                state,
                contexts,
                direction_matrix,
                args.general_batch_size,
                args.continuation_tokens,
                args.probes,
                args.seed,
            )
        )
        continuations = [row["generated_token_ids"] for row in sampled_rows]
        path_deltas = []
        path_keys = []
        for direction_index, (label, direction) in enumerate(zip(labels, directions)):
            for t in t_values[1:]:
                path_keys.append((direction_index, label, t))
                path_deltas.append(direction * t)
        kl_raw, token_counts = evaluate_path_kls(
            model,
            tokenizer,
            alpha,
            state,
            contexts,
            continuations,
            path_deltas,
            args.general_batch_size,
        )
    finally:
        for handle in handles:
            handle.remove()

    target_rows = []
    linear_gains = torch.tensor(
        [sum(float(row["mean_g"]) * float(delta) for row, delta in zip(ranking, d))
         for d in directions],
        dtype=torch.float64,
    )
    for direction_index, label in enumerate(labels):
        path_values = target_raw[label]
        base = path_values[:, 0]
        for t_index, t in enumerate(t_values):
            changes = path_values[:, t_index] - base
            actual = float(changes.mean())
            predicted = float(linear_gains[direction_index] * t)
            ci_low, ci_high = bootstrap_mean_ci(
                changes, args.bootstrap_samples, args.seed + 1000 + t_index
            )
            ratio_ci = None
            if predicted != 0:
                ratio_ci = sorted([ci_low / predicted, ci_high / predicted])
            target_rows.append(
                {
                    "label": label,
                    "t": t,
                    "baseline_mean_log_probability": float(base.mean()),
                    "edited_mean_log_probability": float(path_values[:, t_index].mean()),
                    "actual_change": actual,
                    "actual_change_ci95": [ci_low, ci_high],
                    "linear_predicted_change": predicted,
                    "actual_over_linear": (
                        actual / predicted if predicted != 0 else None
                    ),
                    "actual_over_linear_ci95": ratio_ci,
                    "residual": actual - predicted,
                    "sign_agreement": (
                        actual * predicted > 0 if actual != 0 and predicted != 0 else None
                    ),
                }
            )

    kl_rows = []
    key_to_column = {key: index for index, key in enumerate(path_keys)}
    for direction_index, label in enumerate(labels):
        full_per_context = full_curvature[:, direction_index]
        validation_diag_per_context = validation_diag_curvature[:, direction_index]
        full_mean = float(full_per_context.mean())
        validation_diag_mean = float(validation_diag_per_context.mean())
        train_diag = float(train_diag_curvature[direction_index])
        for t in t_values:
            if t == 0:
                actual_per_context = torch.zeros(len(contexts), dtype=torch.float64)
            else:
                column = key_to_column[(direction_index, label, t)]
                actual_per_context = kl_raw[:, column]
            actual_mean = float(actual_per_context.mean())
            actual_ci = bootstrap_mean_ci(
                actual_per_context,
                args.bootstrap_samples,
                args.seed + 2000 + direction_index * len(t_values) + t_values.index(t),
            )
            full_prediction_per_context = 0.5 * t * t * full_per_context
            validation_diag_prediction_per_context = (
                0.5 * t * t * validation_diag_per_context
            )
            full_prediction = float(full_prediction_per_context.mean())
            validation_diag_prediction = float(
                validation_diag_prediction_per_context.mean()
            )
            train_diag_prediction = 0.5 * t * t * train_diag
            full_ratio_ci = None
            validation_diag_ratio_ci = None
            if t > 0:
                full_ratio_ci = list(
                    bootstrap_ratio_ci(
                        actual_per_context,
                        full_prediction_per_context,
                        args.bootstrap_samples,
                        args.seed + 3000 + direction_index * len(t_values) + t_values.index(t),
                    )
                )
                validation_diag_ratio_ci = list(
                    bootstrap_ratio_ci(
                        actual_per_context,
                        validation_diag_prediction_per_context,
                        args.bootstrap_samples,
                        args.seed + 4000 + direction_index * len(t_values) + t_values.index(t),
                    )
                )
            kl_rows.append(
                {
                    "label": label,
                    "t": t,
                    "actual_mean_sequence_kl": actual_mean,
                    "actual_mean_sequence_kl_ci95": list(actual_ci),
                    "actual_mean_token_kl": float(actual_per_context.sum() / token_counts.sum()),
                    "full_directional_prediction": full_prediction,
                    "validation_diagonal_prediction": validation_diag_prediction,
                    "training_diagonal_prediction": train_diag_prediction,
                    "actual_over_full_directional": (
                        actual_mean / full_prediction if full_prediction > 0 else None
                    ),
                    "actual_over_full_directional_ci95": full_ratio_ci,
                    "actual_over_validation_diagonal": (
                        actual_mean / validation_diag_prediction
                        if validation_diag_prediction > 0
                        else None
                    ),
                    "actual_over_validation_diagonal_ci95": validation_diag_ratio_ci,
                    "actual_over_training_diagonal": (
                        actual_mean / train_diag_prediction
                        if train_diag_prediction > 0
                        else None
                    ),
                }
            )

    direction_rows = []
    for index, ((scale_path, payload), direction) in enumerate(
        zip(direction_payloads, directions)
    ):
        full_mean = float(full_curvature[:, index].mean())
        validation_diag_mean = float(validation_diag_curvature[:, index].mean())
        train_diag = float(train_diag_curvature[index])
        direction_rows.append(
            {
                "label": labels[index],
                "scale_file": str(scale_path.resolve()),
                "scale_file_sha256": sha256_file(scale_path),
                "active_k": int(payload["active_k"]),
                "delta_min": float(direction.min()),
                "delta_median": float(direction.median()),
                "delta_mean": float(direction.mean()),
                "delta_max": float(direction.max()),
                "capped_count": int(payload["capped_count"]),
                "linear_target_gain": float(linear_gains[index]),
                "full_directional_curvature_validation": full_mean,
                "diagonal_curvature_validation": validation_diag_mean,
                "diagonal_curvature_training": train_diag,
                "validation_diagonal_over_full": validation_diag_mean / full_mean,
                "training_diagonal_over_full": train_diag / full_mean,
            }
        )

    atomic_write_jsonl(args.output_dir / "sampled_continuations.jsonl", sampled_rows)
    torch.save(
        {
            "target_values": target_raw,
            "full_curvature_per_context": full_curvature,
            "validation_diagonal_curvature_per_context": validation_diag_curvature,
            "actual_kl_per_context": kl_raw,
            "token_counts": token_counts,
        },
        args.output_dir / "raw_metrics.pt",
    )
    summary = {
        "schema": "grad_approximation_verification_v1",
        "model": str(args.model.resolve()),
        "dtype": args.dtype,
        "direction": "positive-only",
        "scope": "tail/final-position",
        "top_k": args.top_k,
        "t_values": t_values,
        "target": {
            "examples": len(target_records),
            "examples_path": str(args.target_examples.resolve()),
            "examples_sha256": sha256_file(args.target_examples),
            "responses_path": str(args.target_responses.resolve()),
            "responses_sha256": sha256_file(args.target_responses),
            "batch_size": args.target_batch_size,
            "rows": target_rows,
        },
        "general_cost": {
            "contexts": len(contexts),
            "contexts_path": str(args.contexts.resolve()),
            "contexts_sha256": sha256_file(args.contexts),
            "continuation_tokens": args.continuation_tokens,
            "batch_size": args.general_batch_size,
            "model_compute_dtype": args.dtype,
            "kl_normalization_and_reduction_dtype": "float64",
            "probes": args.probes,
            "seed": args.seed,
            "fisher_path": str(args.fisher.resolve()),
            "fisher_sha256": sha256_file(args.fisher),
            "directional_fisher_runtime": fisher_stats,
            "rows": kl_rows,
        },
        "directions": direction_rows,
        "bootstrap_samples": args.bootstrap_samples,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "benchmark-target":
        benchmark_target(args)
    elif args.command == "run":
        run(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
