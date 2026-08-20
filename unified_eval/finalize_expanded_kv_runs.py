#!/usr/bin/env python3
"""Score and normalize the corrected expanded-K/V sweep artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from datasets import load_from_disk

from unified_eval.common import (
    atomic_write_json,
    atomic_write_jsonl,
    extract_gsm_answer,
    extract_gsm_flexible_answer,
    finite_summary,
    read_jsonl,
    score_asr,
)
from unified_eval.methods import _import_neurips_eval, clear_memory
from unified_eval.runner import DEFAULT_COST_MODEL, DEFAULT_NEURIPS_REPO, write_result_checksums


RUNS = {
    "sn_expanded_raw_overlay_baseline": "1",
    "sn_expanded_raw_rate0p0002_m1p1": "1p1",
    "sn_expanded_raw_rate0p0002_m1p25": "1p25",
    "sn_expanded_raw_rate0p0002_m1p5": "1p5",
}
GSM8K = Path("/workspace/xcy/dataset/shared/gsm8k/main")


def repetitive(response: str) -> bool:
    words = re.findall(r"\b\w+\b", response.lower())
    counts: dict[tuple[str, ...], int] = {}
    for index in range(max(0, len(words) - 3)):
        gram = tuple(words[index : index + 4])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values(), default=0) >= 5


def find_one(root: Path, pattern: str) -> Path:
    paths = list(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} under {root}, found {paths}")
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    prepared = read_jsonl(
        Path("/workspace/xcy/dataset/projects/iclr_neuron_expanded_kv/harmbench/")
        / "table1_seed42_n200_prepared.jsonl"
    )
    prepared_by_id = {row["id"]: row for row in prepared}
    cost_inputs: dict[str, list[dict[str, object]]] = {}
    for name in RUNS:
        run_dir = args.results_root / name
        overlay = read_jsonl(run_dir / "harmbench/overlay_responses.jsonl")
        if len(overlay) != 200:
            raise ValueError(f"{name} does not contain 200 HarmBench responses")
        normalized = []
        for row in overlay:
            source = prepared_by_id[row["id"]]
            jailbroken, prefixes = score_asr(str(row["response"]))
            normalized.append(
                {
                    **row,
                    "id": source["original_id"],
                    "behavior_id": source["behavior_id"],
                    "formatted_prompt": row["prompt"],
                    "score_text": str(row["prompt"]) + str(row["response"]),
                    "jailbroken": jailbroken,
                    "matched_refusal_prefixes": prefixes,
                }
            )
        atomic_write_jsonl(run_dir / "harmbench/responses.jsonl", normalized)
        atomic_write_jsonl(run_dir / "harmbench/asr_scored.jsonl", normalized)
        cost_inputs[name] = normalized

    module = _import_neurips_eval(DEFAULT_NEURIPS_REPO)
    model, tokenizer = module.load_score_model(DEFAULT_COST_MODEL, args.device, "bfloat16")
    direction = module.validate_cost_direction(model, tokenizer)
    try:
        for name, responses in cost_inputs.items():
            run_dir = args.results_root / name
            cost_path = run_dir / "harmbench/costs.jsonl"
            if cost_path.exists() and len(read_jsonl(cost_path)) == 200:
                continue
            values = []
            for start in range(0, len(responses), args.batch_size):
                batch = responses[start : start + args.batch_size]
                values.extend(
                    module.score_texts(
                        model, tokenizer, [str(row["score_text"]) for row in batch]
                    )
                )
                print(f"{name} Beaver cost {len(values)}/200", flush=True)
            atomic_write_jsonl(
                cost_path,
                (
                    {"id": row["id"], "cost": value}
                    for row, value in zip(responses, values)
                ),
            )
            atomic_write_json(run_dir / "harmbench/cost_direction.json", direction)
    finally:
        model = tokenizer = None
        clear_memory()

    gsm_dataset = load_from_disk(str(GSM8K))
    for name, scale_tag in RUNS.items():
        run_dir = args.results_root / name
        responses = read_jsonl(run_dir / "harmbench/responses.jsonl")
        costs = read_jsonl(run_dir / "harmbench/costs.jsonl")
        cost_by_id = {row["id"]: float(row["cost"]) for row in costs}
        harm_summary = {
            "benchmark": "HarmBench",
            "num_samples": len(responses),
            "prompt_format": "raw behavior text",
            "attack_success_count": sum(bool(row["jailbroken"]) for row in responses),
            "attack_success_rate": 100 * sum(bool(row["jailbroken"]) for row in responses) / len(responses),
            "mean_beaver_cost": statistics.fmean(cost_by_id[row["id"]] for row in responses),
            "beaver_cost": finite_summary([cost_by_id[row["id"]] for row in responses]),
            "lower_cost_is_safer": True,
            "blank_responses": sum(not str(row["response"]).strip() for row in responses),
            "repetitive_responses": sum(repetitive(str(row["response"])) for row in responses),
            "repetition_rule": "a four-word sequence occurs at least five times",
            "mean_generated_tokens": statistics.fmean(float(row["generated_token_count"]) for row in responses),
        }
        atomic_write_json(run_dir / "harmbench/summary.json", harm_summary)

        gsm_source = find_one(
            run_dir / "capability/gsm8k",
            f"sn_rate0p0002_scale{scale_tag}_chat_0shot/responses.jsonl",
        )
        gsm_rows = read_jsonl(gsm_source)
        rescored = []
        for row in gsm_rows:
            prediction, source = extract_gsm_flexible_answer(str(row["response"]))
            answer = extract_gsm_answer(str(gsm_dataset["test"][row["id"]]["answer"]))
            rescored.append(
                {
                    **row,
                    "prediction": prediction,
                    "prediction_source": source,
                    "answer": answer,
                    "correct": prediction is not None and prediction == answer,
                }
            )
        atomic_write_jsonl(run_dir / "gsm8k/responses.jsonl", rescored)
        gsm_summary = {
            "benchmark": "GSM8K",
            "num_samples": len(rescored),
            "prompt_format": "Llama-3 chat",
            "num_fewshot": 0,
            "max_new_tokens": 256,
            "correct": sum(bool(row["correct"]) for row in rescored),
            "accuracy": 100 * sum(bool(row["correct"]) for row in rescored) / len(rescored),
            "extraction_failures": sum(row["prediction"] is None for row in rescored),
            "answer_extraction": "unified flexible numeric exact match",
        }
        atomic_write_json(run_dir / "gsm8k/summary.json", gsm_summary)

        mmlu_source = find_one(
            run_dir / "capability/mmlu",
            f"sn_rate0p0002_scale{scale_tag}_chat_5shot/summary.json",
        )
        mmlu_summary = json.loads(mmlu_source.read_text(encoding="utf-8"))
        atomic_write_json(run_dir / "mmlu/summary.json", mmlu_summary)
        mmlu_responses = find_one(
            run_dir / "capability/mmlu",
            f"sn_rate0p0002_scale{scale_tag}_chat_5shot/responses.jsonl",
        )
        atomic_write_jsonl(run_dir / "mmlu/responses.jsonl", read_jsonl(mmlu_responses))

        run_metadata = json.loads(
            (run_dir / "harmbench/overlay_run.json").read_text(encoding="utf-8")
        )
        aggregate = {
            "method": name,
            "expanded_kv": True,
            "intervention": run_metadata["variant_settings"],
            "harmbench": harm_summary,
            "gsm8k": gsm_summary,
            "mmlu": mmlu_summary,
        }
        atomic_write_json(run_dir / "summary.json", aggregate)
        write_result_checksums(run_dir)
        print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
