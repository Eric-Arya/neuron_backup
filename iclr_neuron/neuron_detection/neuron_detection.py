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
logging.basicConfig(level=logging.INFO)
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import csv
from pathlib import Path

random.seed(112)


model_name = "/workspace/xcy/models/Meta-Llama-3-8B-Instruct"
model_tag = "Meta-Llama-3-8B-Instruct"
log_responses = os.environ.get("NEURON_LOG_RESPONSES", "0") == "1"
default_corpus_dir = Path(
    "/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/processed"
)


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
    parser.add_argument("corpus", help="Corpus filename stem under --corpus-dir")
    parser.add_argument("sample_count", type=int, help="Number of corpus prompts to sample")
    parser.add_argument("sample_seed", type=int, nargs="?", default=112)
    parser.add_argument("run_tag", nargs="?")
    parser.add_argument("--corpus-dir", type=Path, default=default_corpus_dir)
    parser.add_argument(
        "--prompt-format",
        choices=("chat", "raw"),
        default="chat",
        help="Use the model chat template by default; raw is retained for comparisons.",
    )
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    lines = []
    file_path = args.corpus_dir / f"{args.corpus}.txt"
    with open(file_path, 'r') as file:
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

    # Initialize dictionary for common elements
    common_elements_dict_fwd_up = {}
    common_elements_dict_fwd_down = {}
    common_elements_dict_q = {}
    common_elements_dict_k = {}
    common_elements_dict_v = {}


    # Iterate through the keys of the first dictionary
    for key in activate_keys_set_fwd_up[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_up):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_up]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_up[key] = common_elements
    # print(common_elements_dict_fwd_up)


    for key in activate_keys_set_fwd_down[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_down):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_down]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_down[key] = common_elements
    # print(common_elements_dict_fwd_down)


    for key in activate_keys_set_q[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_q):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_q]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_q[key] = common_elements
    # print(common_elements_dict_q)


    for key in activate_keys_set_k[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_k):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_k]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_k[key] = common_elements
    # print(common_elements_dict_k)


    for key in activate_keys_set_v[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_v):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_v]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_v[key] = common_elements
    # print(common_elements_dict_v)



    os.makedirs("./output_neurons", exist_ok=True)
    result_tag = f"{args.run_tag}_{sample_count-count}" if args.run_tag else str(sample_count-count)
    file_path = (
        "./output_neurons/"
        + model_tag
        + "_"
        + args.corpus
        + "_attn100_ffn200_kvnative_"
        + args.prompt_format
        + "_"
        + result_tag
        + ".txt"
    )

    with open(file_path, 'w') as file:
        file.write(str(common_elements_dict_fwd_up) + '\n')
        file.write(str(common_elements_dict_fwd_down) + '\n')
        file.write(str(common_elements_dict_q) + '\n')
        file.write(str(common_elements_dict_k) + '\n')
        file.write(str(common_elements_dict_v) + '\n')




if __name__ == "__main__":
    main(sys.argv[1:])
