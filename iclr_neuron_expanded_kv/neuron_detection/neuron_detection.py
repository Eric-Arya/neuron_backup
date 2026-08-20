import argparse
import os
from dataclasses import field, dataclass
from typing import Optional, Any
import transformers
from rouge_score import rouge_scorer
import random
from itertools import groupby
import pdb
import re
import sys
from tqdm import tqdm
from typing import List
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import csv

random.seed(112)


model_name = "/workspace/xcy/models/Meta-Llama-3-8B-Instruct"
model_tag = "Meta-Llama-3-8B-Instruct"
log_responses = os.environ.get("NEURON_LOG_RESPONSES", "0") == "1"


def parse_emit_config(value):
    try:
        attn, ffn = (int(part) for part in value.split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected ATTN:FFN, for example 50:100") from exc
    if attn <= 0 or ffn <= 0:
        raise argparse.ArgumentTypeError("ATTN and FFN must be positive")
    return attn, ffn


def aggregate_prefix(collection, limit, aggregation):
    result = {}
    for layer in collection[0]:
        if all(layer in sample for sample in collection):
            prefixes = [list(map(int, sample[layer][:limit])) for sample in collection]
            sets = [set(prefix) for prefix in prefixes]
            if aggregation == "intersection":
                result[layer] = set.intersection(*sets)
            elif aggregation == "union":
                result[layer] = set.union(*sets)
            else:
                common = set.intersection(*sets)
                rank_sums = {
                    index: sum(prefix.index(index) for prefix in prefixes)
                    for index in common
                }
                # Lower mean rank is more important. The index is a deterministic tie-breaker.
                result[layer] = sorted(common, key=lambda index: (rank_sums[index], index))
    return result


def build_prompt_inputs(tokenizer, prompt, prompt_format):
    if prompt_format == "chat":
        if tokenizer.chat_template is None:
            raise ValueError("The tokenizer does not define a chat template")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # The rendered Llama 3 template already contains its BOS token.
        return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    if prompt_format == "raw":
        return tokenizer(prompt, return_tensors="pt")
    raise ValueError(f"Unsupported prompt format: {prompt_format}")


def Prompting(model, tokenizer, prompt, candidate_premature_layers, prompt_format):
    input_device = model.get_input_embeddings().weight.device
    inputs = build_prompt_inputs(tokenizer, prompt, prompt_format).to(input_device)
    hidden_states, outputs, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        candidate_premature_layers=candidate_premature_layers,
    )
    hidden_embed = {}
    # pdb.set_trace()
    for i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = tokenizer.decode(hidden_states[early_exit_layer][0])
        # knowledge_neurons_word[early_exit_layer] = tokenizer.decode(knowledge_neurons[early_exit_layer][0])
        # hidden_info[early_exit_layer] = tokenizer.decode(torch.tensor(hidden_values[early_exit_layer]).to("cuda"))
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys


def main(argv):
    parser = argparse.ArgumentParser(description="Detect Llama safety neurons.")
    parser.add_argument("corpus", help="Corpus filename stem under ./corpus_all")
    parser.add_argument("sample_count", type=int, help="Number of corpus prompts to sample")
    parser.add_argument("sample_seed", type=int, nargs="?", default=112)
    parser.add_argument("run_tag", nargs="?")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        help="Explicit corpus TXT path (datasets should normally live under /workspace/xcy/dataset).",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("chat", "raw"),
        default="chat",
        help="Use the model chat template by default; raw is retained for comparisons.",
    )
    parser.add_argument(
        "--top-attn",
        type=int,
        default=1000,
        help="Per-layer Q/K/V candidates retained per prompt (default: 1000).",
    )
    parser.add_argument(
        "--top-ffn",
        type=int,
        default=2000,
        help="Per-layer FFN candidates retained per prompt (default: 2000).",
    )
    parser.add_argument(
        "--emit-config",
        action="append",
        type=parse_emit_config,
        help=(
            "Write an additional prefix intersection as ATTN:FFN. May be repeated; "
            "values cannot exceed --top-attn/--top-ffn."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=("intersection", "union", "ranked-intersection"),
        default="intersection",
        help=(
            "Aggregate per-prompt candidate prefixes by intersection (default), union, or "
            "intersection ordered by mean per-prompt importance rank."
        ),
    )
    args = parser.parse_args(argv)
    if args.sample_count <= 0 or args.top_attn <= 0 or args.top_ffn <= 0:
        parser.error("sample_count, --top-attn, and --top-ffn must be positive")
    emit_configs = args.emit_config or []
    emit_configs.append((args.top_attn, args.top_ffn))
    emit_configs = list(dict.fromkeys(emit_configs))
    if any(attn > args.top_attn or ffn > args.top_ffn for attn, ffn in emit_configs):
        parser.error("--emit-config values cannot exceed the detector top limits")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.config.safety_neuron_top_attn = args.top_attn
    model.config.safety_neuron_top_ffn = args.top_ffn

    lines = []
    file_path = args.corpus_path or Path("/workspace/xcy/dataset/safety_neuron") / f"{args.corpus}.txt"
    if not file_path.exists():
        file_path = Path("./corpus_all") / f"{args.corpus}.txt"
    with file_path.open("r") as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    sample_count = min(args.sample_count, len(lines))
    lines = random.Random(args.sample_seed).sample(lines, sample_count)


    candidate_premature_layers = []
    for i in range(32):
        candidate_premature_layers.append(i)


    activate_keys_set_fwd_up = []
    activate_keys_set_fwd_down = []
    activate_keys_set_q = []
    activate_keys_set_k = []
    activate_keys_set_v = []

    count = 0

    for sample_index, prompt in enumerate(tqdm(lines), start=1):
        try:
            hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, _, _ = Prompting(
                model,
                tokenizer,
                prompt,
                candidate_premature_layers,
                args.prompt_format,
            )
            if log_responses:
                print(f"\n--- sample {sample_index}/{len(lines)} ---")
                print(f"PROMPT: {prompt}")
                print(f"RESPONSE: {answer}")
            activate_keys_set_fwd_up.append(activate_keys_fwd_up)
            activate_keys_set_fwd_down.append(activate_keys_fwd_down)
            activate_keys_set_q.append(activate_keys_q)
            activate_keys_set_k.append(activate_keys_k)
            activate_keys_set_v.append(activate_keys_v)
        except Exception as e:
            count += 1
            # Handle the OutOfMemoryError here
            print(count)
            print(e)

        if sample_index % 20 == 0 or sample_index == len(lines):
            print(
                f"Processed {sample_index}/{len(lines)} prompts "
                f"(failures: {count})",
                flush=True,
            )


    if not activate_keys_set_fwd_up:
        raise RuntimeError("All samples failed; no neuron activations were collected.")

    os.makedirs("./output_neurons", exist_ok=True)
    result_tag = f"{args.run_tag}_{sample_count-count}" if args.run_tag else str(sample_count-count)
    collections = (
        activate_keys_set_fwd_up,
        activate_keys_set_fwd_down,
        activate_keys_set_q,
        activate_keys_set_k,
        activate_keys_set_v,
    )
    for emit_attn, emit_ffn in emit_configs:
        limits = (emit_ffn, emit_ffn, emit_attn, emit_attn, emit_attn)
        common_structures = [
            aggregate_prefix(collection, limit, args.aggregation)
            for collection, limit in zip(collections, limits)
        ]
        aggregation_tag = {
            "intersection": "",
            "union": "union_",
            "ranked-intersection": "ranked_",
        }[args.aggregation]
        file_path = (
            "./output_neurons/"
            + model_tag
            + "_"
            + args.corpus
            + f"_attn{emit_attn}_ffn{emit_ffn}_kvexpanded_{aggregation_tag}"
            + args.prompt_format
            + "_"
            + result_tag
            + ".txt"
        )
        with open(file_path, "w") as file:
            for structure in common_structures:
                file.write(str(structure) + "\n")
        counts = [sum(len(indices) for indices in structure.values()) for structure in common_structures]
        print(f"Wrote {file_path}: counts={counts}, total={sum(counts)}", flush=True)




if __name__ == "__main__":
    main(sys.argv[1:])
