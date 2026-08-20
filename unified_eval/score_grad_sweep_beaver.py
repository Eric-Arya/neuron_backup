"""Score a Grad development sweep with Beaver cost.

Example:
    python score_grad_sweep_beaver.py \
        --sweep-dir results/grad_harmbench_development/pku_contrastive_tuning
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

from unified_eval.common import atomic_write_json, atomic_write_jsonl, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path,
        default=Path("results/grad_harmbench_development/baseline/scored.jsonl"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("results/grad_harmbench_development/tuning_manifest.jsonl"),
    )
    parser.add_argument(
        "--cost-model", type=Path,
        default=Path("/workspace/xcy/models/beaver-7b-v1.0-cost"),
    )
    parser.add_argument(
        "--neurips-repo", type=Path,
        default=Path("/workspace/xcy/safety_repro/neurips_neuron"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.sweep_dir / "beaver_costs.jsonl"
    summary_path = args.sweep_dir / "beaver_summary.json"
    if not args.overwrite:
        for path in (output, summary_path):
            if path.exists():
                raise FileExistsError(f"Output exists: {path}; pass --overwrite")
    tuning_ids = {str(row["id"]) for row in read_jsonl(args.manifest)}
    rows = [
        {**row, "experiment": "baseline"}
        for row in read_jsonl(args.baseline)
        if str(row["id"]) in tuning_ids
    ]
    rows.extend(read_jsonl(args.sweep_dir / "generations.jsonl"))
    if not rows or args.batch_size <= 0:
        raise ValueError("No rows to score or invalid batch size")

    repo = args.neurips_repo.resolve()
    for path in (repo, repo / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from eval import table1_harmbench as scorer

    model, tokenizer = scorer.load_score_model(args.cost_model, args.device, args.dtype)
    try:
        direction = scorer.validate_cost_direction(model, tokenizer)
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts = [str(row["prompt"]) + str(row["response"]) for row in batch]
            costs = scorer.score_texts(model, tokenizer, texts)
            for row, cost in zip(batch, costs):
                row["beaver_cost"] = cost
            print(f"scored={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    finally:
        model = tokenizer = None
        torch.cuda.empty_cache()
    atomic_write_jsonl(output, rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment"])].append(float(row["beaver_cost"]))
    summary = []
    for experiment, costs in grouped.items():
        summary.append(
            {
                "experiment": experiment,
                "count": len(costs),
                "mean_beaver_cost": statistics.fmean(costs),
                "population_std": statistics.pstdev(costs),
                "min": min(costs),
                "max": max(costs),
            }
        )
    summary.sort(key=lambda row: row["mean_beaver_cost"])
    atomic_write_json(
        summary_path,
        {"lower_cost_is_safer": True, "direction_check": direction, "results": summary},
    )
    print(f"wrote {output} and {summary_path}")


if __name__ == "__main__":
    main()
