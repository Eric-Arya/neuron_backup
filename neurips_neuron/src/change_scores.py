import os
import argparse
import json
import re

import torch
import datasets
from transformers import AutoTokenizer

from src.utils import seed_torch
from src.activation_processor import ActivationContrasting
from eval.templates import create_prompt_with_tulu_chat_format


def extract_prompt(example):
    """Read released prompt-only data or raw Anthropic HH-RLHF JSONL."""
    if "prompt" in example:
        return example["prompt"]
    if "chosen" in example:
        match = re.search(r"Human:\s*(.*?)\s*Assistant:", example["chosen"], re.DOTALL)
        if match:
            return match.group(1)
    raise ValueError("Dataset rows must contain `prompt` or Anthropic HH-RLHF `chosen` text.")


def format_prompts(eval_dataset, args):
    tokenizer = None
    if args.chat_format == "native":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        if tokenizer.chat_template is None:
            raise ValueError("Native chat formatting requires a tokenizer chat template")

    prompts = []
    for example in eval_dataset:
        prompt = extract_prompt(example).strip()
        if args.chat_format == "raw":
            prompts.append(prompt + args.generation_startswith)
            continue
        messages = [{"role": "user", "content": prompt}]
        if args.chat_format == "native":
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
        prompts.append(prompt + args.generation_startswith)
    return prompts


def main(args):
    
    seed_torch(42)
    eval_dataset = datasets.load_dataset('json', data_files=args.dataset)["train"]
    if args.num_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.num_samples, len(eval_dataset))))
        
    prompts = format_prompts(eval_dataset, args)

    with open(os.path.join(args.model_name_or_path, 'config.json')) as fin:
        model_config = json.load(fin)
    last_layer = model_config['num_hidden_layers'] - 1
    names_filter = lambda name: (
        name.endswith('hook_post')
        and (args.include_last_layer or f'.layers.{last_layer}.' not in name)
    )
    ac = ActivationContrasting(
        args.model_name_or_path,
        args.first_peft_path,
        args.second_peft_path,
        batchsize=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        first_ia3_alpha=args.first_ia3_alpha,
        second_ia3_alpha=args.second_ia3_alpha,
        device_map='balanced_low_0'
    )
    change_scores, neuron_ranks, first_mean, first_std, second_mean, second_std = ac.compute_change_scores(prompts, names_filter, args.token_type)
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    torch.save((change_scores.cpu(), neuron_ranks.cpu(), first_mean.cpu(), first_std.cpu(), second_mean.cpu(), second_std.cpu()), args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute change scores via generation-time activation contrasting.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens in generation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--output_file",
        type=str, 
        default="../data/default.pt"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--first_peft_path", 
        nargs='*',
        default=None, 
        help="The folder contains peft checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--second_peft_path", 
        nargs='*',
        default=None, 
        help="The folder contains peft checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--token_type", 
        type=str, 
        default='completion', 
        choices=['prompt', 'prompt_last', 'completion'],
        help="Compute change scores from which token position."
    )
    parser.add_argument(
        "--generation_startswith", 
        type=str, 
        default='', 
        help="Generation start with given prefix."
    )
    parser.add_argument(
        "--chat_format",
        choices=["raw", "tulu", "native"],
        default="tulu",
        help="Serialize ranking prompts as raw text, legacy Tulu, or tokenizer-native chat.",
    )
    parser.add_argument(
        "--first_ia3_alpha",
        type=float,
        default=1.0,
        help="Scale first-model IA3 displacement from identity.",
    )
    parser.add_argument(
        "--second_ia3_alpha",
        type=float,
        default=1.0,
        help="Scale second-model IA3 displacement from identity.",
    )
    parser.add_argument(
        "--include_last_layer",
        action="store_true",
        help="Include the final MLP layer. By default it is excluded to match the paper/released code."
    )

    args = parser.parse_args()
    main(args)
