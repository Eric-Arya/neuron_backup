from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_from_disk

from .common import (
    CHOICES,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    extract_gsm_answer,
    extract_gsm_flexible_answer,
    extract_gsm_strict_answer,
    format_mmlu_question,
    json_hash,
    load_resumable,
    read_jsonl,
    score_asr,
    sha256_file,
)
from .methods import (
    DEFAULT_GRAD_RANKING,
    DEFAULT_LLAMA3,
    DEFAULT_LLAMA3_SFT_ADAPTER,
    DEFAULT_LLAMA3_SFT_PATCH_RANKING,
    DEFAULT_NEURIPS_REPO,
    DEFAULT_SN_ALPHA8,
    DEFAULT_SN_DIRECT_NEURONS,
    build_method,
    clear_memory,
)


DEFAULT_HARMBENCH = Path(
    "/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/"
    "table1_seed42_n200.jsonl"
)
DEFAULT_GSM8K = Path("/workspace/xcy/dataset/shared/gsm8k/main")
DEFAULT_MMLU = Path("/workspace/xcy/dataset/mmlu_balanced_5_per_subject/mmlu/all")
DEFAULT_MMLU_MANIFEST = Path(
    "/workspace/xcy/dataset/mmlu_balanced_5_per_subject/mmlu/subset_manifest.json"
)
DEFAULT_IFEVAL = Path("/workspace/xcy/dataset/ifeval/subsets/ifeval_seed112_n200.jsonl")
DEFAULT_IFEVAL_MANIFEST = Path(
    "/workspace/xcy/dataset/ifeval/subsets/ifeval_seed112_n200_manifest.json"
)
DEFAULT_IFEVAL_SCORER_ROOT = Path("/workspace/xcy/dataset/ifeval")
DEFAULT_BBH = Path(
    "/workspace/xcy/dataset/big_bench_hard/subsets/bbh_seed112_n200.jsonl"
)
DEFAULT_BBH_MANIFEST = Path(
    "/workspace/xcy/dataset/big_bench_hard/subsets/bbh_seed112_n200_manifest.json"
)
DEFAULT_BBH_ROOT = Path("/workspace/xcy/dataset/big_bench_hard")
DEFAULT_BBH_EVALUATOR_ROOT = Path("/workspace/xcy/safety_repro/lm-evaluation-harness")
DEFAULT_MATH500 = Path("/workspace/xcy/dataset/math500")
DEFAULT_MATH500_SOURCE = DEFAULT_MATH500 / "SOURCE.json"
DEFAULT_OUTPUT = Path("/workspace/xcy/safety_repro/unified_eval/results")
EXPECTED_HARMBENCH_SHA256 = (
    "bb5b29ff9db15e420021aee3ad1a07d0ed1ca11a2d8faff024d786168b7be74c"
)
BBH_ANSWER_REGEX = r"(?<=the answer is )(.*)(?=.)"
BBH_STOP_STRINGS = ("</s>", "Q", "\n\n")
MATH500_PROMPT_TEMPLATE = """Solve the following math problem efficiently and clearly. The last line
of your response should be of the following format: 'Therefore, the final
answer is: $\\boxed{ANSWER}$. I hope it is correct' (without quotes)
where ANSWER is just the final number or expression that solves the
problem. Think step by step before answering.

{problem}"""


def configure_float32_execution() -> None:
    """Use full FP32 matmuls when a model is loaded with float32 tensors."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def floating_point_protocol() -> dict[str, Any]:
    return {
        "default_model_dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


LLAMA3_BASE_GSM_SOURCE = Path(
    "/workspace/xcy/safety_repro/iclr_neuron_expanded_kv/neuron_deactivate/"
    "evaluation_outputs/table2_sn_tune/baseline_fp32/capability/gsm8k/"
    "origin_chat_0shot/summary.json"
)
LLAMA3_BASE_MMLU_SOURCE = Path(
    "/workspace/xcy/safety_repro/iclr_neuron_expanded_kv/neuron_deactivate/"
    "evaluation_outputs/sn_tune_table2_mmlu285_multiturn/baseline/mmlu/"
    "origin_chat_5shot/summary.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("validate", "benchmark", "run", "summarize")
    )
    parser.add_argument(
        "--method",
        choices=(
            "llama3_base",
            "llama3_sft",
            "llama3_sft_patch",
            "grad",
            "sn",
            "sn_direct",
            "neurips_direct",
        ),
    )
    parser.add_argument(
        "--run-name",
        help="Output condition name (default: method; useful for NeurIPS top-k sweeps).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=(
            "harmbench",
            "gsm8k",
            "mmlu",
            "ifeval",
            "bbh",
            "math500",
        ),
        default=["harmbench"],
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--harmbench", type=Path, default=DEFAULT_HARMBENCH)
    parser.add_argument("--gsm8k", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--mmlu", type=Path, default=DEFAULT_MMLU)
    parser.add_argument("--mmlu-manifest", type=Path, default=DEFAULT_MMLU_MANIFEST)
    parser.add_argument("--ifeval", type=Path, default=DEFAULT_IFEVAL)
    parser.add_argument("--ifeval-manifest", type=Path, default=DEFAULT_IFEVAL_MANIFEST)
    parser.add_argument(
        "--ifeval-scorer-root", type=Path, default=DEFAULT_IFEVAL_SCORER_ROOT
    )
    parser.add_argument("--bbh", type=Path, default=DEFAULT_BBH)
    parser.add_argument("--bbh-manifest", type=Path, default=DEFAULT_BBH_MANIFEST)
    parser.add_argument("--bbh-root", type=Path, default=DEFAULT_BBH_ROOT)
    parser.add_argument(
        "--bbh-evaluator-root", type=Path, default=DEFAULT_BBH_EVALUATOR_ROOT
    )
    parser.add_argument("--math500", type=Path, default=DEFAULT_MATH500)
    parser.add_argument("--math500-source", type=Path, default=DEFAULT_MATH500_SOURCE)
    parser.add_argument("--llama3-model", type=Path, default=DEFAULT_LLAMA3)
    parser.add_argument(
        "--llama3-base-capability-source",
        choices=("inherited", "fresh"),
        default="fresh",
        help=(
            "Run GSM8K/MMLU with the current unified evaluator (default), or explicitly "
            "reuse the recorded legacy summaries."
        ),
    )
    parser.add_argument(
        "--llama3-sft-adapter", type=Path, default=DEFAULT_LLAMA3_SFT_ADAPTER
    )
    parser.add_argument(
        "--llama3-sft-training-format",
        choices=("raw",),
        default="raw",
        help="Serialization used to train the selected Llama-3 IA3-SFT adapter.",
    )
    parser.add_argument(
        "--llama3-sft-ia3-alpha",
        type=float,
        default=1.0,
        help=(
            "Scale the learned IA3 displacement from identity: "
            "gate = 1 + alpha * (trained_gate - 1)."
        ),
    )
    parser.add_argument(
        "--llama3-sft-patch-ranking",
        type=Path,
        default=DEFAULT_LLAMA3_SFT_PATCH_RANKING,
    )
    parser.add_argument("--llama3-sft-patch-top-k", type=int, default=20_000)
    parser.add_argument(
        "--llama3-base-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float32",
    )
    parser.add_argument(
        "--llama3-sft-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float32",
    )
    parser.add_argument(
        "--llama3-sft-patch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float32",
    )
    parser.add_argument("--sn-model", type=Path, default=DEFAULT_SN_ALPHA8)
    parser.add_argument(
        "--sn-alpha",
        type=float,
        help=(
            "SN-Tune delta multiplier. Inferred from delta_scale_config.json when omitted; "
            "required for an unscaled source checkpoint without that file."
        ),
    )
    parser.add_argument("--grad-ranking", type=Path, default=DEFAULT_GRAD_RANKING)
    parser.add_argument("--grad-top-k", type=int, default=25)
    parser.add_argument("--grad-strength", type=float, default=1.0)
    parser.add_argument(
        "--grad-scale-file",
        type=Path,
        help=(
            "Optional fisher_grad_scales_v1 artifact providing one multiplier per "
            "selected Grad neuron; replaces the uniform --grad-strength multiplier."
        ),
    )
    parser.add_argument("--grad-scope", choices=("last", "all"), default="last")
    parser.add_argument(
        "--grad-direction",
        choices=("signed", "positive-only"),
        default="positive-only",
        help=(
            "positive-only (default) filters out negative-gradient neurons; "
            "signed also weakens negative-gradient neurons"
        ),
    )
    parser.add_argument(
        "--grad-dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    parser.add_argument(
        "--sn-dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    parser.add_argument(
        "--sn-direct-neurons", type=Path, default=DEFAULT_SN_DIRECT_NEURONS
    )
    parser.add_argument("--sn-direct-cap", type=int, default=25)
    parser.add_argument(
        "--sn-direct-strength",
        type=float,
        default=0.5,
        help=(
            "Additive activation strength; selected dimensions use multiplier 1 + strength. "
            "Values in (-1, 0) attenuate the selected dimensions."
        ),
    )
    parser.add_argument(
        "--sn-direct-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float32",
    )
    parser.add_argument("--neurips-repo", type=Path, default=DEFAULT_NEURIPS_REPO)
    parser.add_argument(
        "--neurips-direct-ranking",
        type=Path,
        default=DEFAULT_LLAMA3_SFT_PATCH_RANKING,
        help=(
            "Llama-3 post-MLP ranking used only by neurips_direct. Defaults to "
            "the held-out SNCorpus raw-SFT IA3 ranking."
        ),
    )
    parser.add_argument("--neurips-top-k", type=int, default=20_000)
    parser.add_argument(
        "--neurips-direct-multiplier",
        type=float,
        default=1.0,
        help="Direct multiplier for ranked NeurIPS post-MLP activations.",
    )
    parser.add_argument(
        "--neurips-dtype", choices=("bfloat16", "float16", "float32"), default="float32"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-device", default="cuda:0")
    parser.add_argument("--guide-device", default="cuda:1")
    parser.add_argument("--harmbench-batch-size", type=int)
    parser.add_argument("--gsm8k-batch-size", type=int, default=16)
    parser.add_argument("--mmlu-batch-size", type=int, default=8)
    parser.add_argument("--ifeval-batch-size", type=int)
    parser.add_argument("--bbh-batch-size", type=int)
    parser.add_argument("--math500-batch-size", type=int)
    parser.add_argument("--harmbench-max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--llama3-harm-prompt-format",
        choices=("raw", "chat"),
        default="raw",
        help="Use raw behavior text or the tokenizer-native Llama-3 chat template for HarmBench.",
    )
    parser.add_argument(
        "--gsm8k-max-new-tokens",
        type=int,
        default=None,
        help=(
            "GSM8K generation limit (default: 256)."
        ),
    )
    parser.add_argument("--harmbench-limit", type=int)
    parser.add_argument("--gsm8k-limit", type=int)
    parser.add_argument("--mmlu-limit", type=int)
    parser.add_argument("--ifeval-limit", type=int)
    parser.add_argument("--ifeval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--bbh-limit", type=int)
    parser.add_argument("--bbh-max-new-tokens", type=int, default=1024)
    parser.add_argument("--math500-limit", type=int)
    parser.add_argument("--math500-max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument(
        "--benchmark-batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16]
    )
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--benchmark-task",
        choices=("harmbench", "ifeval", "bbh", "math500"),
        default="harmbench",
    )
    return parser


def require_method(args: argparse.Namespace) -> None:
    if args.command != "summarize" and args.method is None:
        raise SystemExit(f"--method is required for {args.command}")


def neurips_direct_selection_source(ranking: Path) -> str:
    name = ranking.name.lower()
    if "vs_sft" in name:
        return "Llama-3 NeurIPS-style Instruct-vs-SFT safety-neuron ranking"
    return "custom Llama-3 NeurIPS-style safety-neuron ranking"


def resolve_method_defaults(args: argparse.Namespace) -> None:
    """Apply model-family defaults while preserving explicit CLI overrides."""
    if args.gsm8k_max_new_tokens is None and args.method is not None:
        args.gsm8k_max_new_tokens = 256
    if args.math500_batch_size is None:
        # The two-model BF16 IA3 guide patch was benchmarked separately on real
        # MATH-500 prompts; batch 32 is fastest and peaks below 45 GiB on H100.
        args.math500_batch_size = 32 if args.method == "llama3_sft_patch" else 16
    if args.harmbench_batch_size is None:
        args.harmbench_batch_size = (
            32 if args.method in {"sn_direct", "neurips_direct"} else 16
        )
    if args.ifeval_batch_size is None:
        # Real IFEval benchmarks select method-specific defaults. The direct SN
        # overlay is light enough for batch 32; SN checkpoints and the two-model
        # guide patch use 16; Grad and other methods retain the conservative 8.
        args.ifeval_batch_size = (
            32
            if args.method in {"sn_direct", "neurips_direct"}
            else 16
            if args.method in {"sn", "llama3_sft_patch"}
            else 8
        )
    if args.bbh_batch_size is None:
        args.bbh_batch_size = 16 if args.method == "llama3_sft_patch" else 8


def validate_positive(args: argparse.Namespace) -> None:
    numbers = (
        args.grad_top_k,
        args.neurips_top_k,
        args.llama3_sft_patch_top_k,
        args.harmbench_batch_size,
        args.gsm8k_batch_size,
        args.mmlu_batch_size,
        args.ifeval_batch_size,
        args.bbh_batch_size,
        args.math500_batch_size,
        args.harmbench_max_new_tokens,
        args.gsm8k_max_new_tokens,
        args.ifeval_max_new_tokens,
        args.bbh_max_new_tokens,
        args.math500_max_new_tokens,
        args.benchmark_max_new_tokens,
        *args.benchmark_batch_sizes,
    )
    if any(value <= 0 for value in numbers):
        raise ValueError("Counts, batch sizes, and token limits must be positive")
    if not math.isfinite(args.grad_strength) or args.grad_strength < 0:
        raise ValueError("grad-strength must be finite and non-negative")
    if (
        not math.isfinite(args.neurips_direct_multiplier)
        or args.neurips_direct_multiplier <= 0
    ):
        raise ValueError("neurips-direct-multiplier must be finite and positive")
    if args.sn_alpha is not None and (
        not math.isfinite(args.sn_alpha) or args.sn_alpha <= 0
    ):
        raise ValueError("sn-alpha must be finite and positive")
    if not math.isfinite(args.llama3_sft_ia3_alpha) or args.llama3_sft_ia3_alpha < 0:
        raise ValueError("llama3-sft-ia3-alpha must be finite and nonnegative")
    for value in (
        args.harmbench_limit,
        args.gsm8k_limit,
        args.mmlu_limit,
        args.ifeval_limit,
        args.bbh_limit,
        args.math500_limit,
    ):
        if value is not None and value <= 0:
            raise ValueError("Task limits must be positive")


def dataset_arrow_hash(dataset_path: Path, split: str) -> str:
    files = sorted((dataset_path / split).glob("*.arrow"))
    if len(files) != 1:
        raise ValueError(
            f"Expected one Arrow file under {dataset_path / split}, found {files}"
        )
    return sha256_file(files[0])


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    validate_positive(args)
    required = [
        args.harmbench,
        args.gsm8k,
        args.mmlu,
        args.mmlu_manifest,
    ]
    if "ifeval" in args.tasks:
        required.extend(
            (
                args.ifeval,
                args.ifeval_manifest,
                args.ifeval_scorer_root
                / "instruction_following_eval/evaluation_lib.py",
            )
        )
    if "bbh" in args.tasks:
        required.extend(
            (
                args.bbh,
                args.bbh_manifest,
                args.bbh_root / "cot-prompts",
                args.bbh_evaluator_root
                / "lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml",
            )
        )
    if "math500" in args.tasks:
        required.extend((args.math500, args.math500_source))
    if args.method == "llama3_base":
        required.append(args.llama3_model / "config.json")
        if args.llama3_base_capability_source == "inherited":
            required.extend((LLAMA3_BASE_GSM_SOURCE, LLAMA3_BASE_MMLU_SOURCE))
    elif args.method == "llama3_sft":
        required.extend(
            (
                args.llama3_model / "config.json",
                args.llama3_sft_adapter / "adapter_config.json",
                args.llama3_sft_adapter / "adapter_model.safetensors",
            )
        )
    elif args.method == "llama3_sft_patch":
        required.extend(
            (
                args.neurips_repo,
                args.llama3_model / "config.json",
                args.llama3_sft_adapter / "adapter_config.json",
                args.llama3_sft_adapter / "adapter_model.safetensors",
                args.llama3_sft_patch_ranking,
            )
        )
    elif args.method == "grad":
        required.extend((args.llama3_model / "config.json", args.grad_ranking))
        if args.grad_scale_file is not None:
            required.append(args.grad_scale_file)
    elif args.method == "sn":
        required.append(args.sn_model / "config.json")
        if args.sn_alpha is None:
            required.append(args.sn_model / "delta_scale_config.json")
    elif args.method == "sn_direct":
        required.extend((args.llama3_model / "config.json", args.sn_direct_neurons))
    elif args.method == "neurips_direct":
        required.extend(
            (
                args.neurips_repo,
                args.llama3_model / "config.json",
                args.neurips_direct_ranking,
            )
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    harm = read_jsonl(args.harmbench)
    harm_hash = sha256_file(args.harmbench)
    if harm_hash != EXPECTED_HARMBENCH_SHA256:
        raise ValueError(
            f"HarmBench manifest hash mismatch: {harm_hash}; expected {EXPECTED_HARMBENCH_SHA256}"
        )
    if len(harm) != 200 or len({row.get("id") for row in harm}) != 200:
        raise ValueError("HarmBench must be the unique 200-row NeurIPS manifest")
    if any(not isinstance(row.get("prompt"), str) or not row["prompt"] for row in harm):
        raise ValueError("Every HarmBench record needs a prompt")

    gsm = load_from_disk(str(args.gsm8k))
    if len(gsm["test"]) < 100:
        raise ValueError("GSM8K test split has fewer than 100 rows")
    mmlu = load_from_disk(str(args.mmlu))
    if len(mmlu["test"]) != 285 or len(mmlu["dev"]) != 285:
        raise ValueError("MMLU subset must contain 285 dev and 285 test rows")
    subjects = set(mmlu["test"]["subject"])
    if len(subjects) != 57:
        raise ValueError(f"Expected 57 MMLU subjects, found {len(subjects)}")
    manifest = json.loads(args.mmlu_manifest.read_text(encoding="utf-8"))
    if manifest.get("seed") != 112 or manifest.get("samples_per_subject") != 5:
        raise ValueError("MMLU manifest is not the seed-112 five-per-subject subset")

    validation = {
        "method": args.method,
        "harmbench": {
            "path": str(args.harmbench.resolve()),
            "sha256": harm_hash,
            "count": len(harm),
        },
        "gsm8k": {
            "path": str(args.gsm8k.resolve()),
            "test_count": len(gsm["test"]),
            "test_arrow_sha256": dataset_arrow_hash(args.gsm8k, "test"),
            "selected": "first 100 test rows",
        },
        "mmlu": {
            "path": str(args.mmlu.resolve()),
            "test_count": len(mmlu["test"]),
            "dev_count": len(mmlu["dev"]),
            "subjects": len(subjects),
            "manifest_sha256": sha256_file(args.mmlu_manifest),
            "test_arrow_sha256": dataset_arrow_hash(args.mmlu, "test"),
            "dev_arrow_sha256": dataset_arrow_hash(args.mmlu, "dev"),
        },
        "prompt_protocol": {
            "harmbench": (
                "native Llama-3 chat"
                if args.llama3_harm_prompt_format == "chat"
                else "raw behavior text"
            ),
            "gsm8k": "zero-shot chat with step-by-step answer prefix",
            "mmlu": "five-shot multi-turn chat with constrained next-token scoring",
            "ifeval": "task-native single-turn chat",
            "bbh": "official raw three-shot CoT text completion",
            "math500": "publisher zero-shot CoT prompt in task-native chat",
        },
    }
    if "ifeval" in args.tasks:
        ifeval = read_jsonl(args.ifeval)
        ifeval_hash = sha256_file(args.ifeval)
        ifeval_manifest = json.loads(args.ifeval_manifest.read_text(encoding="utf-8"))
        if ifeval_hash != ifeval_manifest.get("output_sha256"):
            raise ValueError("IFEval subset hash does not match its manifest")
        if len(ifeval) != 200 or len({row.get("key") for row in ifeval}) != 200:
            raise ValueError("IFEval subset must contain 200 unique keys")
        if len({row.get("prompt") for row in ifeval}) != 200:
            raise ValueError("IFEval subset prompts must be unique")
        for row in ifeval:
            if not isinstance(row.get("prompt"), str) or not row["prompt"]:
                raise ValueError("Every IFEval record needs a prompt")
            if len(row.get("instruction_id_list", [])) != len(row.get("kwargs", [])):
                raise ValueError(
                    f"IFEval key {row.get('key')} has mismatched instructions"
                )
        validation["ifeval"] = {
            "path": str(args.ifeval.resolve()),
            "sha256": ifeval_hash,
            "count": len(ifeval),
            "manifest_path": str(args.ifeval_manifest.resolve()),
            "manifest_sha256": sha256_file(args.ifeval_manifest),
            "instruction_type_count": len(
                {item for row in ifeval for item in row["instruction_id_list"]}
            ),
            "scorer_root": str(args.ifeval_scorer_root.resolve()),
        }
    if "bbh" in args.tasks:
        bbh = read_jsonl(args.bbh)
        bbh_hash = sha256_file(args.bbh)
        bbh_manifest = json.loads(args.bbh_manifest.read_text(encoding="utf-8"))
        if bbh_hash != bbh_manifest.get("output_sha256"):
            raise ValueError("BBH subset hash does not match its manifest")
        if len(bbh) != 200 or len({row.get("id") for row in bbh}) != 200:
            raise ValueError("BBH subset must contain 200 unique examples")
        tasks = {str(row.get("task")) for row in bbh}
        if len(tasks) != 27:
            raise ValueError(f"Expected 27 BBH task variants, found {len(tasks)}")
        for row in bbh:
            required_fields = ("id", "task", "source_index", "input", "target")
            if any(field not in row for field in required_fields):
                raise ValueError(f"Malformed BBH row: {row.get('id')}")
            source_path = args.bbh_root / "bbh" / f"{row['task']}.json"
            source = json.loads(source_path.read_text(encoding="utf-8"))["examples"]
            source_row = source[int(row["source_index"])]
            if (
                row["input"] != source_row["input"]
                or row["target"] != source_row["target"]
            ):
                raise ValueError(f"BBH source mismatch for {row['id']}")
        prompt_paths = [args.bbh_root / "cot-prompts" / f"{task}.txt" for task in tasks]
        missing_prompts = [str(path) for path in prompt_paths if not path.exists()]
        if missing_prompts:
            raise FileNotFoundError(f"Missing BBH CoT prompts: {missing_prompts}")
        evaluator_template = (
            args.bbh_evaluator_root
            / "lm_eval/tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml"
        )
        evaluator_revision = subprocess.run(
            ["git", "-C", str(args.bbh_evaluator_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        validation["bbh"] = {
            "path": str(args.bbh.resolve()),
            "sha256": bbh_hash,
            "count": len(bbh),
            "task_count": len(tasks),
            "manifest_path": str(args.bbh_manifest.resolve()),
            "manifest_sha256": sha256_file(args.bbh_manifest),
            "source_revision": bbh_manifest["source_revision"],
            "prompt_sha256_by_task": {
                path.stem: sha256_file(path) for path in sorted(prompt_paths)
            },
            "evaluator_root": str(args.bbh_evaluator_root.resolve()),
            "evaluator_revision": evaluator_revision,
            "evaluator_template_sha256": sha256_file(evaluator_template),
        }
    if "math500" in args.tasks:
        math500 = load_from_disk(str(args.math500))
        source = json.loads(args.math500_source.read_text(encoding="utf-8"))
        expected_rows = int(source.get("dataset", {}).get("rows", 0))
        if set(math500) != {"test"} or len(math500["test"]) != expected_rows:
            raise ValueError(
                "MATH-500 row count does not match its source manifest: "
                f"{len(math500['test'])} != {expected_rows}"
            )
        required_fields = {
            "problem",
            "solution",
            "answer",
            "subject",
            "level",
            "unique_id",
        }
        if not required_fields.issubset(math500["test"].column_names):
            raise ValueError("MATH-500 is missing required benchmark fields")
        if len(set(math500["test"]["unique_id"])) != expected_rows:
            raise ValueError("MATH-500 unique_id values must be unique")
        expected_arrow = source.get("dataset", {}).get("arrow_sha256")
        arrow_hash = dataset_arrow_hash(args.math500, "test")
        if expected_arrow != arrow_hash:
            raise ValueError(
                f"MATH-500 Arrow hash {arrow_hash} does not match {expected_arrow}"
            )
        validation["math500"] = {
            "path": str(args.math500.resolve()),
            "count": len(math500["test"]),
            "test_arrow_sha256": arrow_hash,
            "source_manifest": str(args.math500_source.resolve()),
            "source_manifest_sha256": sha256_file(args.math500_source),
            "source_revision": source["dataset"]["huggingface_revision"],
            "selection": source.get("selection"),
            "evaluator": source["evaluator"],
            "evaluator_dependencies": source["evaluator_dependencies"],
        }
    if args.method == "sn":
        scale_path = args.sn_model / "delta_scale_config.json"
        inferred_alpha = None
        if scale_path.exists():
            scale = json.loads(scale_path.read_text())
            inferred_alpha = float(scale.get("alpha", scale.get("scale")))
        if args.sn_alpha is not None and (
            inferred_alpha is not None and args.sn_alpha != inferred_alpha
        ):
            raise ValueError(
                f"Requested SN alpha {args.sn_alpha} disagrees with checkpoint "
                f"alpha {inferred_alpha}"
            )
        alpha = args.sn_alpha if args.sn_alpha is not None else inferred_alpha
        if alpha is None or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Could not determine a finite positive SN alpha")
        validation["sn_alpha"] = alpha
    if args.method == "sn_direct":
        if (
            args.sn_direct_cap <= 0
            or not math.isfinite(args.sn_direct_strength)
            or args.sn_direct_strength <= -1
        ):
            raise ValueError(
                "SN direct requires positive cap and finite strength greater than -1"
            )
    if (
        args.method == "llama3_base"
        and args.llama3_base_capability_source == "inherited"
    ):
        validation["inherited_capability_sources"] = {
            "gsm8k": {
                "path": str(LLAMA3_BASE_GSM_SOURCE),
                "sha256": sha256_file(LLAMA3_BASE_GSM_SOURCE),
            },
            "mmlu": {
                "path": str(LLAMA3_BASE_MMLU_SOURCE),
                "sha256": sha256_file(LLAMA3_BASE_MMLU_SOURCE),
            },
        }
    return validation


def semantic_config(
    args: argparse.Namespace, validation: dict[str, Any]
) -> dict[str, Any]:
    common = {
        "method": args.method,
        "floating_point_protocol": floating_point_protocol(),
        "datasets": {key: validation[key] for key in ("harmbench", "gsm8k", "mmlu")},
        "seed": args.seed,
        "decoding": "greedy",
        "harmbench_max_new_tokens": args.harmbench_max_new_tokens,
        "gsm8k_max_new_tokens": args.gsm8k_max_new_tokens,
        "gsm8k_answer_protocol": (
            "flexible numeric exact match: final ####, final answer phrase, "
            "final equation RHS, then last-number fallback"
        ),
        "gsm8k_hash_delimiter_requested": False,
        "metrics": ["llm-attacks case-sensitive refusal-substring ASR"],
        "cost_scoring": False,
    }
    if "ifeval" in args.tasks:
        common["datasets"]["ifeval"] = validation["ifeval"]
        common["ifeval_max_new_tokens"] = args.ifeval_max_new_tokens
        common["ifeval_prompt_format"] = "task_native_chat"
    if "bbh" in args.tasks:
        common["datasets"]["bbh"] = validation["bbh"]
        common["bbh_max_new_tokens"] = args.bbh_max_new_tokens
        common["bbh_prompt_format"] = "official_3shot_cot_raw_completion"
        common["bbh_answer_protocol"] = (
            "lm-evaluation-harness v4.0 case-sensitive 'the answer is' regex exact match"
        )
        common["bbh_generation_stopping"] = (
            "transformers StopStringCriteria with the harness stop strings"
        )
    if "math500" in args.tasks:
        common["datasets"]["math500"] = validation["math500"]
        common["math500_max_new_tokens"] = args.math500_max_new_tokens
        common["math500_prompt_format"] = (
            "publisher zero-shot CoT instruction in task-native chat"
        )
        common["math500_answer_protocol"] = (
            "math_verify parse/verify over full reference solution and generation"
        )
    if args.llama3_harm_prompt_format == "chat":
        # Keep the historical raw-prompt semantic payload unchanged while
        # fingerprinting native-chat safety runs as a distinct condition.
        common["harmbench_prompt_format"] = "native_llama3_chat"
    if args.method == "llama3_base":
        common["intervention"] = {
            "model": str(args.llama3_model.resolve()),
            "model_config_sha256": sha256_file(args.llama3_model / "config.json"),
            "condition": "unmodified pretrained instruction model",
            "dtype": args.llama3_base_dtype,
        }
        common["capability_source"] = args.llama3_base_capability_source
        if args.llama3_base_capability_source == "inherited":
            common["inherited_capability_sources"] = validation[
                "inherited_capability_sources"
            ]
    elif args.method == "llama3_sft":
        common["intervention"] = {
            "model": str(args.llama3_model.resolve()),
            "model_config_sha256": sha256_file(args.llama3_model / "config.json"),
            "sft_adapter": str(args.llama3_sft_adapter.resolve()),
            "sft_adapter_config_sha256": sha256_file(
                args.llama3_sft_adapter / "adapter_config.json"
            ),
            "sft_adapter_model_sha256": sha256_file(
                args.llama3_sft_adapter / "adapter_model.safetensors"
            ),
            "condition": (
                f"{args.llama3_sft_training_format}-format IA3 SFT on the first 256 "
                "SN-Tune refusal pairs"
            ),
            "training_format": args.llama3_sft_training_format,
            "ia3_displacement_alpha": args.llama3_sft_ia3_alpha,
            "ia3_scaling_formula": "1 + alpha * (trained_gate - 1)",
            "dtype": args.llama3_sft_dtype,
        }
    elif args.method == "llama3_sft_patch":
        common["intervention"] = {
            "model": str(args.llama3_model.resolve()),
            "model_config_sha256": sha256_file(args.llama3_model / "config.json"),
            "sft_adapter": str(args.llama3_sft_adapter.resolve()),
            "sft_adapter_config_sha256": sha256_file(
                args.llama3_sft_adapter / "adapter_config.json"
            ),
            "sft_adapter_model_sha256": sha256_file(
                args.llama3_sft_adapter / "adapter_model.safetensors"
            ),
            "ranking": str(args.llama3_sft_patch_ranking.resolve()),
            "ranking_sha256": sha256_file(args.llama3_sft_patch_ranking),
            "top_k": args.llama3_sft_patch_top_k,
            "patch": "copy IA3-SFT-guide post-MLP activations into Llama-3 Instruct base",
            "ia3_displacement_alpha": args.llama3_sft_ia3_alpha,
            "ia3_scaling_formula": "1 + alpha * (trained_gate - 1)",
            "dtype": args.llama3_sft_patch_dtype,
        }
    elif args.method == "grad":
        intervention = {
            "model": str(args.llama3_model.resolve()),
            "model_config_sha256": sha256_file(args.llama3_model / "config.json"),
            "ranking": str(args.grad_ranking.resolve()),
            "ranking_sha256": sha256_file(args.grad_ranking),
            "top_k": args.grad_top_k,
            "strength": args.grad_strength,
            "positive_multiplier": 1 + args.grad_strength,
            "negative_multiplier": (
                1.0
                if args.grad_direction == "positive-only"
                else max(0.0, 1 - args.grad_strength)
            ),
            "direction": args.grad_direction,
            "scope": args.grad_scope,
            "dtype": args.grad_dtype,
        }
        if args.grad_scale_file is not None:
            scale_payload = json.loads(args.grad_scale_file.read_text(encoding="utf-8"))
            multipliers = [float(row["multiplier"]) for row in scale_payload["rows"]]
            intervention.update(
                {
                    "scale_file": str(args.grad_scale_file.resolve()),
                    "scale_file_sha256": sha256_file(args.grad_scale_file),
                    "scale_schema": scale_payload.get("schema"),
                    "scale_mode": scale_payload.get("mode"),
                    "multiplier_min": min(multipliers),
                    "multiplier_median": statistics.median(multipliers),
                    "multiplier_max": max(multipliers),
                    "strength": None,
                    "positive_multiplier": None,
                }
            )
        common["intervention"] = intervention
    elif args.method == "sn":
        scale_path = args.sn_model / "delta_scale_config.json"
        common["intervention"] = {
            "model": str(args.sn_model.resolve()),
            "model_config_sha256": sha256_file(args.sn_model / "config.json"),
            "delta_scale_config_sha256": (
                sha256_file(scale_path) if scale_path.exists() else None
            ),
            "alpha": validation["sn_alpha"],
            "dtype": args.sn_dtype,
        }
    elif args.method == "sn_direct":
        common["intervention"] = {
            "model": str(args.llama3_model.resolve()),
            "model_config_sha256": sha256_file(args.llama3_model / "config.json"),
            "neuron_file": str(args.sn_direct_neurons.resolve()),
            "neuron_file_sha256": sha256_file(args.sn_direct_neurons),
            "detector_prompt_format": "raw",
            "selection_method": "file-order after head-aware expanded-K/V mapping",
            "cap_per_layer_structure": args.sn_direct_cap,
            "structures": ["fwd_up", "fwd_down", "q", "k", "v"],
            "scope": "all token positions",
            "strength": args.sn_direct_strength,
            "multiplier": 1 + args.sn_direct_strength,
            "condition": "direct activation scaling; no fine-tuning or trained weights",
            "dtype": args.sn_direct_dtype,
        }
    elif args.method == "neurips_direct":
        common["intervention"] = {
            "model": str(args.llama3_model.resolve()),
            "model_config_sha256": sha256_file(args.llama3_model / "config.json"),
            "ranking": str(args.neurips_direct_ranking.resolve()),
            "ranking_sha256": sha256_file(args.neurips_direct_ranking),
            "top_k": args.neurips_top_k,
            "multiplier": args.neurips_direct_multiplier,
            "scope": "all token positions",
            "activation_site": "post-MLP hook_post",
            "selection_source": neurips_direct_selection_source(
                args.neurips_direct_ranking
            ),
            "harmbench_prompt_format": "raw",
            "condition": "direct activation scaling; no guide model or trained weights",
            "dtype": args.neurips_dtype,
        }
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    return common


def ensure_config(args: argparse.Namespace, semantic: dict[str, Any]) -> Path:
    run_dir = args.output_root / str(args.run_name or args.method)
    path = run_dir / "run_config.json"
    config = {
        "semantic": semantic,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "floating_point_protocol": floating_point_protocol(),
            "batch_sizes": {
                "harmbench": args.harmbench_batch_size,
                "gsm8k": args.gsm8k_batch_size,
                "mmlu": args.mmlu_batch_size,
                "bbh": args.bbh_batch_size,
                "math500": args.math500_batch_size,
            },
            "dtypes": {
                "llama3_base": args.llama3_base_dtype,
                "llama3_sft": args.llama3_sft_dtype,
                "llama3_sft_patch": args.llama3_sft_patch_dtype,
                "grad": args.grad_dtype,
                "sn": args.sn_dtype,
                "sn_direct": args.sn_direct_dtype,
                "neurips_direct": args.neurips_dtype,
            },
        },
    }
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("semantic") != semantic:
            raise ValueError(f"Semantic configuration differs from existing {path}")
        if old != config:
            # Runtime settings are intentionally outside the semantic fingerprint;
            # keep the recorded batch sizes/dtypes synchronized with the current run.
            atomic_write_json(path, config)
    else:
        atomic_write_json(path, config)
    return run_dir


def limit_count(total: int, value: int | None, default: int) -> int:
    return min(total, value if value is not None else default)


def render_user(method, content: str) -> str:
    return method.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def render_math500_prompt(method, problem: str) -> str:
    content = MATH500_PROMPT_TEMPLATE.replace("{problem}", problem)
    return render_user(method, content)


def score_math500_response(solution: str, response: str) -> dict[str, Any]:
    from math_verify import parse, verify

    try:
        gold = parse(solution)
        target = parse(response)
        return {
            "correct": bool(verify(gold=gold, target=target)),
            "answer_extracted": bool(target),
            "scoring_error": None,
        }
    except Exception as exc:
        return {
            "correct": False,
            "answer_extracted": False,
            "scoring_error": f"{type(exc).__name__}: {exc}",
        }


def render_harm_prompt(args: argparse.Namespace, method, prompt: str) -> str:
    if getattr(method, "harm_prompt_style", None) == "raw":
        return prompt
    if args.llama3_harm_prompt_format == "chat":
        return render_user(method, prompt)
    return prompt


def load_bbh_cot_prompt(root: Path, task: str) -> str:
    text = (root / "cot-prompts" / f"{task}.txt").read_text(encoding="utf-8")
    if "-----" not in text:
        raise ValueError(f"Malformed official BBH CoT prompt for {task}")
    return text.split("-----", 1)[1].strip()


def render_bbh_prompt(method, root: Path, row: dict[str, Any]) -> str:
    demonstrations = load_bbh_cot_prompt(root, str(row["task"]))
    content = f"{demonstrations}\n\nQ: {row['input']}\nA: Let's think step by step.\n"
    return content


def truncate_bbh_response(response: str) -> tuple[str, str | None]:
    matches = [
        (response.find(stop), stop)
        for stop in BBH_STOP_STRINGS
        if response.find(stop) >= 0
    ]
    if not matches:
        return response, None
    index, stop = min(matches, key=lambda value: value[0])
    return response[:index], stop


def gsm_prompts(method, dataset, count: int) -> list[str]:
    prompts = []
    for record in dataset["test"].select(range(count)):
        content = f"Question: {record['question']}\nAnswer: Let's think step by step."
        prompts.append(render_user(method, content))
    return prompts


def mmlu_prompts(method, dataset, count: int) -> list[str]:
    dev_by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in dataset["dev"]:
        dev_by_subject.setdefault(record["subject"], []).append(record)
    prompts: list[str] = []
    for record in dataset["test"].select(range(count)):
        subject = record["subject"]
        header = (
            "The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n"
        )
        messages: list[dict[str, str]] = []
        for index, example in enumerate(dev_by_subject[subject][:5]):
            question = format_mmlu_question(example, False)
            if index == 0:
                question = header + question
            messages.extend(
                (
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": CHOICES[int(example["answer"])]},
                )
            )
        messages.append(
            {"role": "user", "content": format_mmlu_question(record, False)}
        )
        prompts.append(
            method.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def task_fingerprint(semantic: dict[str, Any], task: str, count: int) -> str:
    return json_hash({"semantic": semantic, "task": task, "count": count})


def run_harmbench(args, method, semantic, run_dir: Path) -> Path:
    source_all = read_jsonl(args.harmbench)
    count = limit_count(len(source_all), args.harmbench_limit, 200)
    source = source_all[:count]
    fingerprint = task_fingerprint(semantic, "harmbench", count)
    task_dir = run_dir / "harmbench"
    response_path = task_dir / "responses.jsonl"
    ids = {row["id"] for row in source}
    completed = load_resumable(response_path, fingerprint, ids)
    pending = [row for row in source if row["id"] not in completed]
    for start in range(0, len(pending), args.harmbench_batch_size):
        batch = pending[start : start + args.harmbench_batch_size]
        prompts = [render_harm_prompt(args, method, row["prompt"]) for row in batch]
        started = time.perf_counter()
        outputs = method.generate(prompts, args.harmbench_max_new_tokens)
        elapsed = time.perf_counter() - started
        rows = [
            {
                **source_row,
                "formatted_prompt": prompt,
                **output,
                "batch_elapsed_seconds": elapsed,
                "run_fingerprint": fingerprint,
            }
            for source_row, prompt, output in zip(batch, prompts, outputs)
        ]
        append_jsonl(response_path, rows)
        completed.update({row["id"]: row for row in rows})
        print(f"{args.method} HarmBench {len(completed)}/{count}", flush=True)
    ordered = [completed[row["id"]] for row in source]
    scored = []
    for row in ordered:
        jailbroken, prefixes = score_asr(str(row["response"]))
        scored.append(
            {**row, "jailbroken": jailbroken, "matched_refusal_prefixes": prefixes}
        )
    atomic_write_jsonl(task_dir / "asr_scored.jsonl", scored)
    attacks = sum(bool(row["jailbroken"]) for row in scored)
    summary = {
        "benchmark": "HarmBench",
        "subset": "NeurIPS paper seed-42 sample of 200/400",
        "num_samples": count,
        "attack_success_count": attacks,
        "attack_success_rate": 100.0 * attacks / count,
        "asr_rule": "case-sensitive llm-attacks refusal-substring absence",
        "blank_responses": sum(not str(row["response"]).strip() for row in scored),
        "run_fingerprint": fingerprint,
    }
    atomic_write_json(task_dir / "summary.json", summary)
    return task_dir


def run_gsm8k(args, method, semantic, run_dir: Path) -> None:
    dataset = load_from_disk(str(args.gsm8k))
    count = limit_count(len(dataset["test"]), args.gsm8k_limit, 100)
    fingerprint = task_fingerprint(semantic, "gsm8k", count)
    task_dir = run_dir / "gsm8k"
    response_path = task_dir / "responses.jsonl"
    ids = set(range(count))
    completed = load_resumable(response_path, fingerprint, ids)
    prompts = gsm_prompts(method, dataset, count)
    pending = [index for index in range(count) if index not in completed]
    for start in range(0, len(pending), args.gsm8k_batch_size):
        indices = pending[start : start + args.gsm8k_batch_size]
        outputs = method.generate(
            [prompts[index] for index in indices], args.gsm8k_max_new_tokens
        )
        rows = []
        for index, output in zip(indices, outputs):
            prediction, prediction_source = extract_gsm_flexible_answer(
                output["response"]
            )
            strict_prediction = extract_gsm_strict_answer(output["response"])
            answer = extract_gsm_answer(str(dataset["test"][index]["answer"]))
            rows.append(
                {
                    "id": index,
                    "question": dataset["test"][index]["question"],
                    **output,
                    "prediction": prediction,
                    "prediction_source": prediction_source,
                    "strict_prediction": strict_prediction,
                    "answer": answer,
                    "correct": prediction is not None and prediction == answer,
                    "format_compliant": None,
                    "run_fingerprint": fingerprint,
                }
            )
        append_jsonl(response_path, rows)
        completed.update({row["id"]: row for row in rows})
        print(f"{args.method} GSM8K {len(completed)}/{count}", flush=True)

    # Always rescore loaded rows so extractor fixes apply without regenerating text.
    for index in range(count):
        row = completed[index]
        prediction, prediction_source = extract_gsm_flexible_answer(row["response"])
        strict_prediction = extract_gsm_strict_answer(row["response"])
        answer = extract_gsm_answer(str(dataset["test"][index]["answer"]))
        completed[index] = {
            **row,
            "prediction": prediction,
            "prediction_source": prediction_source,
            "strict_prediction": strict_prediction,
            "answer": answer,
            "correct": prediction is not None and prediction == answer,
            "format_compliant": None,
        }
    atomic_write_jsonl(response_path, (completed[index] for index in range(count)))

    correct = sum(bool(completed[index]["correct"]) for index in range(count))
    summary = {
        "benchmark": "GSM8K",
        "subset": f"first {count} test rows",
        "num_samples": count,
        "prompt_format": "Llama-3 chat",
        "num_fewshot": 0,
        "max_new_tokens": args.gsm8k_max_new_tokens,
        "answer_extraction": (
            "flexible numeric exact match: final ####, final answer phrase, "
            "final equation RHS, then last-number fallback"
        ),
        "correct": correct,
        "accuracy": 100.0 * correct / count,
        "extraction_failures": sum(
            completed[index]["prediction"] is None for index in range(count)
        ),
        "prediction_source_counts": {
            source: sum(
                completed[index].get("prediction_source") == source
                for index in range(count)
            )
            for source in (
                "hash_delimiter",
                "answer_phrase_equation_rhs",
                "answer_phrase",
                "final_equation_rhs",
                "last_number_fallback",
            )
        },
        "strict_correct": None,
        "strict_accuracy": None,
        "format_compliance_count": None,
        "format_compliance_percent": None,
        "run_fingerprint": fingerprint,
    }
    atomic_write_json(task_dir / "summary.json", summary)


def run_math500(args, method, semantic, run_dir: Path) -> None:
    dataset = load_from_disk(str(args.math500))["test"]
    count = limit_count(len(dataset), args.math500_limit, 500)
    selected = dataset.select(range(count))
    fingerprint = task_fingerprint(semantic, "math500", count)
    task_dir = run_dir / "math500"
    response_path = task_dir / "responses.jsonl"
    ids = set(selected["unique_id"])
    completed = load_resumable(response_path, fingerprint, ids)
    pending = [
        index for index, row in enumerate(selected) if row["unique_id"] not in completed
    ]

    for start in range(0, len(pending), args.math500_batch_size):
        indices = pending[start : start + args.math500_batch_size]
        source_rows = [selected[index] for index in indices]
        prompts = [
            render_math500_prompt(method, str(row["problem"])) for row in source_rows
        ]
        outputs = method.generate(prompts, args.math500_max_new_tokens)
        rows = []
        for source_row, prompt, output in zip(source_rows, prompts, outputs):
            score = score_math500_response(
                str(source_row["solution"]), str(output["response"])
            )
            rows.append(
                {
                    "id": source_row["unique_id"],
                    "problem": source_row["problem"],
                    "solution": source_row["solution"],
                    "answer": source_row["answer"],
                    "subject": source_row["subject"],
                    "level": source_row["level"],
                    "formatted_prompt": prompt,
                    **output,
                    **score,
                    "run_fingerprint": fingerprint,
                }
            )
        append_jsonl(response_path, rows)
        completed.update({row["id"]: row for row in rows})
        print(f"{args.method} MATH-500 {len(completed)}/{count}", flush=True)

    ordered = []
    for source_row in selected:
        row = completed[source_row["unique_id"]]
        score = score_math500_response(
            str(source_row["solution"]), str(row["response"])
        )
        ordered.append({**row, **score})
    atomic_write_jsonl(response_path, ordered)

    by_subject: dict[str, list[bool]] = {}
    by_level: dict[str, list[bool]] = {}
    for row in ordered:
        by_subject.setdefault(str(row["subject"]), []).append(bool(row["correct"]))
        by_level.setdefault(str(row["level"]), []).append(bool(row["correct"]))
    subject_accuracy = {
        subject: 100.0 * sum(values) / len(values)
        for subject, values in sorted(by_subject.items())
    }
    level_accuracy = {
        level: 100.0 * sum(values) / len(values)
        for level, values in sorted(by_level.items(), key=lambda item: int(item[0]))
    }
    correct = sum(bool(row["correct"]) for row in ordered)
    summary = {
        "benchmark": "MATH-500",
        "subset": f"first {count} of {len(dataset)} locally selected test rows",
        "num_samples": count,
        "num_fewshot": 0,
        "prompt_format": "publisher zero-shot CoT instruction in task-native chat",
        "decoding": "greedy",
        "max_new_tokens": args.math500_max_new_tokens,
        "evaluator": "math_verify parse/verify over full solution and generation",
        "metric": "deterministic symbolic equivalence",
        "uses_llm_judge": False,
        "correct": correct,
        "accuracy": 100.0 * correct / count,
        "subject_macro_accuracy": statistics.fmean(subject_accuracy.values()),
        "accuracy_by_subject": subject_accuracy,
        "accuracy_by_level": level_accuracy,
        "answer_extraction_failures": sum(
            not bool(row["answer_extracted"]) for row in ordered
        ),
        "scoring_errors": sum(row["scoring_error"] is not None for row in ordered),
        "run_fingerprint": fingerprint,
        **_generation_diagnostics(ordered, args.math500_max_new_tokens),
    }
    atomic_write_json(task_dir / "summary.json", summary)


def run_mmlu(args, method, semantic, run_dir: Path) -> None:
    dataset = load_from_disk(str(args.mmlu))
    count = limit_count(len(dataset["test"]), args.mmlu_limit, 285)
    fingerprint = task_fingerprint(semantic, "mmlu", count)
    task_dir = run_dir / "mmlu"
    response_path = task_dir / "responses.jsonl"
    ids = set(range(count))
    completed = load_resumable(response_path, fingerprint, ids)
    prompts = mmlu_prompts(method, dataset, count)
    option_ids = method.option_token_ids()
    pending = [index for index in range(count) if index not in completed]
    for start in range(0, len(pending), args.mmlu_batch_size):
        indices = pending[start : start + args.mmlu_batch_size]
        logits = method.option_logits([prompts[index] for index in indices], option_ids)
        predictions = logits.argmax(dim=-1).tolist()
        rows = []
        for index, prediction, scores in zip(indices, predictions, logits.tolist()):
            record = dataset["test"][index]
            rows.append(
                {
                    "id": index,
                    "subject": record["subject"],
                    "prediction": int(prediction),
                    "prediction_letter": CHOICES[prediction],
                    "answer": int(record["answer"]),
                    "correct": int(prediction) == int(record["answer"]),
                    "option_logits": dict(zip(CHOICES, scores)),
                    "run_fingerprint": fingerprint,
                }
            )
        append_jsonl(response_path, rows)
        completed.update({row["id"]: row for row in rows})
        print(f"{args.method} MMLU {len(completed)}/{count}", flush=True)
    correct = sum(bool(completed[index]["correct"]) for index in range(count))
    by_subject: dict[str, list[bool]] = {}
    for index in range(count):
        row = completed[index]
        by_subject.setdefault(row["subject"], []).append(bool(row["correct"]))
    subject_accuracy = {
        subject: 100.0 * sum(values) / len(values)
        for subject, values in sorted(by_subject.items())
    }
    summary = {
        "benchmark": "MMLU",
        "subset": "seed-112, 5 test questions per 57 subjects",
        "num_samples": count,
        "num_fewshot": 5,
        "prompt_format": "multi-turn Llama-3 chat",
        "correct": correct,
        "accuracy": 100.0 * correct / count,
        "subject_macro_accuracy": sum(subject_accuracy.values())
        / len(subject_accuracy),
        "accuracy_by_subject": subject_accuracy,
        "run_fingerprint": fingerprint,
    }
    atomic_write_json(task_dir / "summary.json", summary)


def _import_ifeval_scorer(root: Path):
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    # Install the official scorer dependencies globally in the Python environment
    # used to launch this runner before starting an IFEval run:
    #   python -m pip install absl-py nltk langdetect immutabledict
    # Installing them first prevents a completed generation job from failing only
    # when it reaches the final scoring stage.
    try:
        import nltk
        from langdetect import DetectorFactory
        from instruction_following_eval import evaluation_lib
    except ImportError as exc:
        raise ImportError(
            "IFEval scorer dependencies are missing. Install them globally in the "
            "Python environment used to launch this runner before running IFEval: "
            "python -m pip install absl-py nltk langdetect immutabledict. "
            f"The scorer source is under {root / 'instruction_following_eval'}."
        ) from exc
    nltk_data = str((root / "nltk_data").resolve())
    if nltk_data not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data)
    # The official checkers call langdetect for language/case constraints. Its
    # default factory is stochastic, so pin it to make repeated scoring stable.
    DetectorFactory.seed = 0
    return evaluation_lib


def _generation_diagnostics(
    rows: Sequence[dict[str, Any]], max_new_tokens: int
) -> dict[str, Any]:
    repetitive = 0
    unique_ratios = []
    for row in rows:
        words = re.findall(r"\b\w+\b", str(row["response"]).lower())
        grams: dict[tuple[str, ...], int] = {}
        for index in range(max(0, len(words) - 3)):
            gram = tuple(words[index : index + 4])
            grams[gram] = grams.get(gram, 0) + 1
        repetitive += max(grams.values(), default=0) >= 5
        unique_ratios.append(len(set(words)) / len(words) if words else 0.0)
    return {
        "blank_responses": sum(not str(row["response"]).strip() for row in rows),
        "repetitive_responses": repetitive,
        "repetition_rule": "a four-word sequence occurs at least five times",
        "median_unique_word_ratio": statistics.median(unique_ratios),
        "mean_generated_tokens": statistics.fmean(
            float(row["generated_token_count"]) for row in rows
        ),
        "responses_at_token_limit": sum(
            int(row["generated_token_count"]) >= max_new_tokens for row in rows
        ),
    }


def run_bbh(args, method, semantic, run_dir: Path) -> None:
    source_all = read_jsonl(args.bbh)
    count = limit_count(len(source_all), args.bbh_limit, 200)
    source = source_all[:count]
    fingerprint = task_fingerprint(semantic, "bbh", count)
    task_dir = run_dir / "bbh"
    response_path = task_dir / "responses.jsonl"
    ids = {row["id"] for row in source}
    completed = load_resumable(response_path, fingerprint, ids)
    pending = [row for row in source if row["id"] not in completed]
    for start in range(0, len(pending), args.bbh_batch_size):
        batch = pending[start : start + args.bbh_batch_size]
        prompts = [render_bbh_prompt(method, args.bbh_root, row) for row in batch]
        started = time.perf_counter()
        outputs = method.generate(
            prompts,
            args.bbh_max_new_tokens,
            stop_strings=BBH_STOP_STRINGS,
        )
        elapsed = time.perf_counter() - started
        rows = []
        for source_row, prompt, output in zip(batch, prompts, outputs):
            harness_response, stop_string = truncate_bbh_response(output["response"])
            rows.append(
                {
                    **source_row,
                    "formatted_prompt": prompt,
                    **output,
                    "harness_response": harness_response,
                    "matched_stop_string": stop_string,
                    "batch_elapsed_seconds": elapsed,
                    "run_fingerprint": fingerprint,
                }
            )
        append_jsonl(response_path, rows)
        completed.update({row["id"]: row for row in rows})
        print(f"{args.method} BBH {len(completed)}/{count}", flush=True)

    # Use the installed lm-evaluation-harness filter verbatim, then exact string equality.
    evaluator_text = str(args.bbh_evaluator_root.resolve())
    if evaluator_text not in sys.path:
        sys.path.insert(0, evaluator_text)
    from lm_eval.filters.extraction import RegexFilter

    ordered = [completed[row["id"]] for row in source]
    for row in ordered:
        harness_response, stop_string = truncate_bbh_response(str(row["response"]))
        row["harness_response"] = harness_response
        row["matched_stop_string"] = stop_string
    predictions = RegexFilter(regex_pattern=BBH_ANSWER_REGEX).apply(
        [[row["harness_response"]] for row in ordered], source
    )
    for row, prediction_set in zip(ordered, predictions):
        prediction = prediction_set[0]
        row["prediction"] = prediction
        row["correct"] = prediction == row["target"]
    atomic_write_jsonl(response_path, ordered)

    by_task: dict[str, list[bool]] = {}
    for row in ordered:
        by_task.setdefault(str(row["task"]), []).append(bool(row["correct"]))
    task_accuracy = {
        task: 100.0 * sum(values) / len(values)
        for task, values in sorted(by_task.items())
    }
    correct = sum(bool(row["correct"]) for row in ordered)
    summary = {
        "benchmark": "BIG-Bench Hard",
        "subset": f"seed-112 task-stratified subset; first {count} of 200 rows",
        "num_samples": count,
        "num_tasks": len(by_task),
        "num_fewshot": 3,
        "prompt_format": "official raw three-shot CoT text completion",
        "decoding": "greedy",
        "max_new_tokens": args.bbh_max_new_tokens,
        "until": list(BBH_STOP_STRINGS),
        "answer_extraction": BBH_ANSWER_REGEX,
        "metric": "case-sensitive exact match",
        "evaluator": "EleutherAI lm-evaluation-harness BBH CoT few-shot v4.0",
        "evaluator_revision": semantic["datasets"]["bbh"]["evaluator_revision"],
        "correct": correct,
        "accuracy": 100.0 * correct / count,
        "task_macro_accuracy": statistics.fmean(task_accuracy.values()),
        "accuracy_by_task": task_accuracy,
        "extraction_failures": sum(row["prediction"] == "[invalid]" for row in ordered),
        "run_fingerprint": fingerprint,
        **_generation_diagnostics(ordered, args.bbh_max_new_tokens),
    }
    atomic_write_json(task_dir / "summary.json", summary)


def _ifeval_score_summary(outputs: Sequence[Any]) -> dict[str, Any]:
    prompt_correct = sum(bool(output.follow_all_instructions) for output in outputs)
    instruction_total = sum(len(output.follow_instruction_list) for output in outputs)
    instruction_correct = sum(sum(output.follow_instruction_list) for output in outputs)
    per_type: dict[str, list[bool]] = {}
    for output in outputs:
        for instruction_id, followed in zip(
            output.instruction_id_list, output.follow_instruction_list
        ):
            per_type.setdefault(instruction_id, []).append(bool(followed))
    return {
        "prompt_correct": prompt_correct,
        "prompt_total": len(outputs),
        "prompt_accuracy": 100.0 * prompt_correct / len(outputs),
        "instruction_correct": instruction_correct,
        "instruction_total": instruction_total,
        "instruction_accuracy": 100.0 * instruction_correct / instruction_total,
        "instruction_accuracy_by_type": {
            instruction_id: 100.0 * sum(values) / len(values)
            for instruction_id, values in sorted(per_type.items())
        },
    }


def run_ifeval(args, method, semantic, run_dir: Path) -> None:
    source_all = read_jsonl(args.ifeval)
    count = limit_count(len(source_all), args.ifeval_limit, 200)
    source = source_all[:count]
    fingerprint = task_fingerprint(semantic, "ifeval", count)
    task_dir = run_dir / "ifeval"
    response_path = task_dir / "responses.jsonl"
    ids = {row["key"] for row in source}
    completed = load_resumable(response_path, fingerprint, ids)
    pending = [row for row in source if row["key"] not in completed]
    for start in range(0, len(pending), args.ifeval_batch_size):
        batch = pending[start : start + args.ifeval_batch_size]
        prompts = [render_user(method, str(row["prompt"])) for row in batch]
        started = time.perf_counter()
        outputs = method.generate(prompts, args.ifeval_max_new_tokens)
        elapsed = time.perf_counter() - started
        rows = [
            {
                **source_row,
                "id": source_row["key"],
                "formatted_prompt": prompt,
                **output,
                "batch_elapsed_seconds": elapsed,
                "run_fingerprint": fingerprint,
            }
            for source_row, prompt, output in zip(batch, prompts, outputs)
        ]
        append_jsonl(response_path, rows)
        completed.update({row["id"]: row for row in rows})
        print(f"{args.method} IFEval {len(completed)}/{count}", flush=True)

    ordered = [completed[row["key"]] for row in source]
    atomic_write_jsonl(response_path, ordered)
    scorer = _import_ifeval_scorer(args.ifeval_scorer_root)
    prompt_to_response = {row["prompt"]: row["response"] for row in ordered}
    inputs = [
        scorer.InputExample(
            key=row["key"],
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            kwargs=row["kwargs"],
        )
        for row in source
    ]
    # Upstream instruction construction contains random fallbacks for malformed
    # kwargs (one released row asks the letter checker to count "!"). Reset the
    # seed per mode so strict and loose scoring are repeatable and comparable.
    random.seed(args.seed)
    strict_outputs = [
        scorer.test_instruction_following_strict(row, prompt_to_response)
        for row in inputs
    ]
    random.seed(args.seed)
    loose_outputs = [
        scorer.test_instruction_following_loose(row, prompt_to_response)
        for row in inputs
    ]
    for label, outputs in (("strict", strict_outputs), ("loose", loose_outputs)):
        atomic_write_jsonl(
            task_dir / f"scored_{label}.jsonl",
            (
                {
                    "id": source_row["key"],
                    "instruction_id_list": output.instruction_id_list,
                    "follow_all_instructions": output.follow_all_instructions,
                    "follow_instruction_list": output.follow_instruction_list,
                    "run_fingerprint": fingerprint,
                }
                for source_row, output in zip(source, outputs)
            ),
        )
    summary = {
        "benchmark": "IFEval",
        "subset": f"seed-112 coverage-first subset; first {count} of 200 rows",
        "num_samples": count,
        "num_instruction_constraints": sum(
            len(row["instruction_id_list"]) for row in source
        ),
        "prompt_format": "task-native single-turn chat",
        "max_new_tokens": args.ifeval_max_new_tokens,
        "official_scorer": str(args.ifeval_scorer_root.resolve()),
        "langdetect_seed": 0,
        "scorer_random_seed": args.seed,
        "strict": _ifeval_score_summary(strict_outputs),
        "loose": _ifeval_score_summary(loose_outputs),
        "run_fingerprint": fingerprint,
        **_generation_diagnostics(ordered, args.ifeval_max_new_tokens),
    }
    atomic_write_json(task_dir / "summary.json", summary)


def inherit_llama3_base_capability(run_dir: Path) -> None:
    """Attach immutable source-experiment summaries without regenerating capability data."""
    sources = {
        "gsm8k": LLAMA3_BASE_GSM_SOURCE,
        "mmlu": LLAMA3_BASE_MMLU_SOURCE,
    }
    for task, source in sources.items():
        summary = json.loads(source.read_text(encoding="utf-8"))
        summary.update(
            {
                "inherited": True,
                "provenance": "existing source reproduction; not regenerated by unified_eval",
                "source_path": str(source),
                "source_sha256": sha256_file(source),
            }
        )
        if task == "gsm8k":
            summary.update(
                {
                    "subset": "first 100 test rows",
                    "max_new_tokens": 256,
                    "answer_extraction": "source experiment: #### number, else final number",
                }
            )
        else:
            summary.update(
                {
                    "subset": "seed-112, 5 test questions per 57 subjects",
                    "subject_macro_accuracy": summary["accuracy"],
                }
            )
        atomic_write_json(run_dir / task / "summary.json", summary)


def collect_method_summary(method: str, run_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {"method": method}
    for task in (
        "harmbench",
        "gsm8k",
        "mmlu",
        "ifeval",
        "bbh",
        "math500",
    ):
        path = run_dir / task / "summary.json"
        if path.exists():
            task_summary = json.loads(path.read_text(encoding="utf-8"))
            if task == "harmbench":
                responses = read_jsonl(run_dir / task / "responses.jsonl")
                repetitive = 0
                unique_ratios = []
                for row in responses:
                    words = re.findall(r"\b\w+\b", str(row["response"]).lower())
                    grams: dict[tuple[str, ...], int] = {}
                    for index in range(max(0, len(words) - 3)):
                        gram = tuple(words[index : index + 4])
                        grams[gram] = grams.get(gram, 0) + 1
                    repetitive += max(grams.values(), default=0) >= 5
                    unique_ratios.append(len(set(words)) / len(words) if words else 0.0)
                task_summary.update(
                    {
                        "repetitive_responses": repetitive,
                        "repetition_rule": "a four-word sequence occurs at least five times",
                        "median_unique_word_ratio": statistics.median(unique_ratios),
                        "mean_generated_tokens": statistics.fmean(
                            float(row["generated_token_count"]) for row in responses
                        ),
                    }
                )
                atomic_write_json(path, task_summary)
            output[task] = task_summary
    atomic_write_json(run_dir / "summary.json", output)
    return output


def write_result_checksums(directory: Path) -> None:
    checksums = {}
    for path in sorted(directory.rglob("*")):
        if (
            not path.is_file()
            or path.name.endswith(".log")
            or path.name == "checksums.json"
        ):
            continue
        checksums[str(path.relative_to(directory))] = sha256_file(path)
    atomic_write_json(directory / "checksums.json", checksums)


def benchmark(args, semantic: dict[str, Any], run_dir: Path) -> None:
    source_path = {
        "harmbench": args.harmbench,
        "ifeval": args.ifeval,
        "bbh": args.bbh,
        "math500": args.math500,
    }[args.benchmark_task]
    source = (
        load_from_disk(str(source_path))["test"]
        if args.benchmark_task == "math500"
        else read_jsonl(source_path)
    )
    candidates = sorted(set(args.benchmark_batch_sizes))
    method = build_method(args, max(candidates))
    results = []
    try:
        for batch_size in candidates:
            if args.benchmark_task == "harmbench":
                prompts = [
                    render_harm_prompt(args, method, row["prompt"])
                    for row in source[:batch_size]
                ]
            elif args.benchmark_task == "ifeval":
                prompts = [
                    render_user(method, row["prompt"]) for row in source[:batch_size]
                ]
            elif args.benchmark_task == "math500":
                prompts = [
                    render_math500_prompt(method, row["problem"])
                    for row in source.select(range(batch_size))
                ]
            else:
                prompts = [
                    render_bbh_prompt(method, args.bbh_root, row)
                    for row in source[:batch_size]
                ]
            for device_index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device_index)
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = method.generate(
                prompts,
                args.benchmark_max_new_tokens,
                stop_strings=(
                    BBH_STOP_STRINGS if args.benchmark_task == "bbh" else None
                ),
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            token_count = sum(int(row["generated_token_count"]) for row in outputs)
            results.append(
                {
                    "batch_size": batch_size,
                    "elapsed_seconds": elapsed,
                    "examples_per_second": batch_size / elapsed,
                    "tokens_per_second": token_count / elapsed,
                    "peak_gpu_memory_gib": [
                        torch.cuda.max_memory_allocated(index) / 2**30
                        for index in range(torch.cuda.device_count())
                    ],
                }
            )
            print(
                f"benchmark {args.method} batch={batch_size} elapsed={elapsed:.3f}s",
                flush=True,
            )
    finally:
        method.close()
    best = max(results, key=lambda row: row["examples_per_second"])
    report = {
        "method": args.method,
        "dataset": f"first real {args.benchmark_task} prompts",
        "benchmark_task": args.benchmark_task,
        "max_new_tokens": args.benchmark_max_new_tokens,
        "results": results,
        "recommended_batch_size": best["batch_size"],
        "semantic_config_hash": json_hash(semantic),
    }
    atomic_write_json(run_dir / "benchmark.json", report)


def summarize_all(output_root: Path) -> None:
    rows = []
    payload: dict[str, Any] = {}
    preferred = [
        "llama3_base",
        "llama3_sft",
        "llama3_sft_patch_20k_chat",
        "grad",
        "sn",
        "sn_direct",
        "neurips_direct",
    ]
    available = [
        path.name for path in output_root.iterdir() if (path / "summary.json").is_file()
    ]
    methods = [name for name in preferred if name in available]
    methods.extend(sorted(set(available) - set(methods)))
    for method in methods:
        run_dir = output_root / method
        path = run_dir / "summary.json"
        if not path.exists():
            continue
        summary = collect_method_summary(method, run_dir)
        write_result_checksums(run_dir)
        payload[method] = summary
        rows.append(
            {
                "method": method,
                "model_family": "Llama-3",
                "harmbench_n": summary.get("harmbench", {}).get("num_samples"),
                "harmbench_asr_percent": summary.get("harmbench", {}).get(
                    "attack_success_rate"
                ),
                "gsm8k_n": summary.get("gsm8k", {}).get("num_samples"),
                "gsm8k_accuracy_percent": summary.get("gsm8k", {}).get("accuracy"),
                "mmlu_n": summary.get("mmlu", {}).get("num_samples"),
                "mmlu_accuracy_percent": summary.get("mmlu", {}).get("accuracy"),
                "ifeval_n": summary.get("ifeval", {}).get("num_samples"),
                "ifeval_strict_prompt_accuracy_percent": summary.get("ifeval", {})
                .get("strict", {})
                .get("prompt_accuracy"),
                "ifeval_loose_prompt_accuracy_percent": summary.get("ifeval", {})
                .get("loose", {})
                .get("prompt_accuracy"),
                "bbh_n": summary.get("bbh", {}).get("num_samples"),
                "bbh_accuracy_percent": summary.get("bbh", {}).get("accuracy"),
                "bbh_task_macro_accuracy_percent": summary.get("bbh", {}).get(
                    "task_macro_accuracy"
                ),
                "math500_n": summary.get("math500", {}).get("num_samples"),
                "math500_accuracy_percent": summary.get("math500", {}).get("accuracy"),
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "unified_summary.json", payload)
    with (output_root / "unified_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(rows[0]) if rows else ["method"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    top_checksums = {
        "unified_summary.json": sha256_file(output_root / "unified_summary.json"),
        "unified_summary.csv": sha256_file(output_root / "unified_summary.csv"),
    }
    atomic_write_json(output_root / "checksums.json", top_checksums)
    print(json.dumps(rows, indent=2), flush=True)


def run(args: argparse.Namespace) -> None:
    require_method(args)
    if args.command == "summarize":
        summarize_all(args.output_root)
        return
    configure_float32_execution()
    resolve_method_defaults(args)
    validation = validate_inputs(args)
    semantic = semantic_config(args, validation)
    run_dir = ensure_config(args, semantic)
    atomic_write_json(run_dir / "validation.json", validation)
    if args.command == "validate":
        print(json.dumps(validation, indent=2), flush=True)
        return
    if args.command == "benchmark":
        benchmark(args, semantic, run_dir)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    max_batch = max(
        args.harmbench_batch_size,
        args.gsm8k_batch_size,
        args.mmlu_batch_size,
        args.ifeval_batch_size,
        args.bbh_batch_size,
        args.math500_batch_size,
    )
    inherited_only = (
        args.method == "llama3_base"
        and args.llama3_base_capability_source == "inherited"
        and set(args.tasks) <= {"gsm8k", "mmlu"}
    )
    needs_method = not inherited_only
    method = build_method(args, max_batch) if needs_method else None
    try:
        if "harmbench" in args.tasks:
            run_harmbench(args, method, semantic, run_dir)
        if "gsm8k" in args.tasks and (
            args.method != "llama3_base"
            or args.llama3_base_capability_source == "fresh"
        ):
            run_gsm8k(args, method, semantic, run_dir)
        if "mmlu" in args.tasks and (
            args.method != "llama3_base"
            or args.llama3_base_capability_source == "fresh"
        ):
            run_mmlu(args, method, semantic, run_dir)
        if "ifeval" in args.tasks:
            run_ifeval(args, method, semantic, run_dir)
        if "bbh" in args.tasks:
            run_bbh(args, method, semantic, run_dir)
        if "math500" in args.tasks:
            run_math500(args, method, semantic, run_dir)
    finally:
        if method is not None:
            method.close()
        method = None
        gc.collect()
        clear_memory()
    if (
        args.method == "llama3_base"
        and args.llama3_base_capability_source == "inherited"
    ):
        inherit_llama3_base_capability(run_dir)
    summary = collect_method_summary(str(args.run_name or args.method), run_dir)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
