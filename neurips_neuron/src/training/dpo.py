#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional, List
import datasets
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from trl import DPOConfig, DPOTrainer
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    IA3Config,
    PromptTuningConfig,
    PromptTuningInit,
)


logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    use_lora: Optional[bool] = False
    lora_rank: Optional[int] = field(default=64, metadata={"help": "the lora rank for LoRA traing"})
    lora_alpha: Optional[float] = field(default=64, metadata={"help": "the lora alpha for LoRA traing"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout for LoRA traing"})
    lora_module: Optional[List[str]] = field(default_factory=lambda : ['k_proj', 'q_proj'], metadata={"help": "the target lora modules for LoRA traing"})
    """LoRAConfig"""
    use_ia3: Optional[bool] = False
    ia3_module: Optional[List[str]] = field(default_factory=lambda : ['down_proj'], metadata={"help": "The IA3 modules to use"})
    feedforward_modules: Optional[List[str]] = field(default_factory=lambda : ['down_proj'], metadata={"help": "The IA3 modules that is feedforward"})
    """IA3Config"""
    use_prompt_tuning: Optional[bool] = False
    num_virtual_tokens: Optional[int] = 16
    prompt_tuning_init_text: Optional[str] = "You are a helpful and harmless AI assistant."
    """PromptTuningConfig"""
    peft_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The peft checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    save_final_model: bool = field(
        default=True,
        metadata={"help": "Save the final adapter/model. Disable for throughput-only benchmarks."},
    )
    chat_format: str = field(
        default="tulu",
        metadata={
            "help": (
                "Preference-data serialization. Use 'tulu' for the released Llama-2 "
                "reproduction or 'native' for the tokenizer's own chat template."
            )
        },
    )

    def __post_init__(self):
        if self.chat_format not in {"tulu", "native"}:
            raise ValueError("chat_format must be 'tulu' or 'native'")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a json/jsonl file)."})
    test_file: Optional[str] = field(default=None, metadata={"help": "The input test data file (a json/jsonl file)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError("Need either a dataset name or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."

def convert_to_messages(text):
    messages = []
    for pair in text.split('Human:')[1:]:
        temp = pair.split('Assistant:')
        try:
            user, assist = temp[0], temp[1]
        except:
            user, assist = temp[0], ' '
        messages.append({
            "role": "user",
            "content": user
        })
        messages.append({
            "role": "assistant",
            "content": assist
        })
    return messages

def create_prompt_with_tulu_chat_format(messages, bos="<s>", eos="</s>", add_bos=False):
    formatted_text = ""
    for message in messages:
        if message["role"] == "system":
            formatted_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            formatted_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            formatted_text += "<|assistant|>\n" + message["content"].strip() + "\n"
        else:
            raise ValueError(
                "Tulu chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    formatted_text = bos + formatted_text if add_bos else formatted_text
    return formatted_text

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "<|assistant|>\n"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


ANTHROPIC_ROLE_RE = re.compile(r"(?:\A|\n\n)(Human|Assistant):[ \t]?")


def parse_anthropic_dialogue(text):
    """Parse Anthropic's double-newline role delimiters into chat messages."""
    matches = list(ANTHROPIC_ROLE_RE.finditer(text))
    messages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        messages.append(
            {
                "role": "user" if match.group(1) == "Human" else "assistant",
                "content": text[match.end():end].strip(),
            }
        )
    return messages


def native_hh_parts(sample):
    """Find the last shared assistant boundary, including malformed HH continuations."""
    chosen_text = sample["chosen"]
    rejected_text = sample["rejected"]
    common_length = 0
    for chosen_char, rejected_char in zip(chosen_text, rejected_text):
        if chosen_char != rejected_char:
            break
        common_length += 1
    assistant_boundaries = [
        match
        for match in ANTHROPIC_ROLE_RE.finditer(chosen_text)
        if match.group(1) == "Assistant" and match.end() <= common_length
    ]
    if not assistant_boundaries:
        raise ValueError("HH pair has no shared assistant response boundary")
    boundary = assistant_boundaries[-1]
    prompt_messages = parse_anthropic_dialogue(chosen_text[: boundary.start()])
    chosen_completion = parse_anthropic_dialogue(
        "\n\nAssistant: " + chosen_text[boundary.end():]
    )
    rejected_completion = parse_anthropic_dialogue(
        "\n\nAssistant: " + rejected_text[boundary.end():]
    )
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        raise ValueError("HH native prompt must end with a user turn")
    if not chosen_completion or chosen_completion[0]["role"] != "assistant":
        raise ValueError("HH chosen continuation must start with an assistant turn")
    if not rejected_completion or rejected_completion[0]["role"] != "assistant":
        raise ValueError("HH rejected continuation must start with an assistant turn")
    return prompt_messages, chosen_completion, rejected_completion


def format_hh_preference(sample, tokenizer, chat_format: str):
    """Convert one Anthropic HH row without mixing model-family templates."""
    if chat_format == "tulu":
        chosen_messages = convert_to_messages(sample["chosen"])
        rejected_messages = convert_to_messages(sample["rejected"])
        if not chosen_messages or chosen_messages[-1]["role"] != "assistant":
            raise ValueError("HH chosen dialogue must end with an assistant response")
        if not rejected_messages or rejected_messages[-1]["role"] != "assistant":
            raise ValueError("HH rejected dialogue must end with an assistant response")
        formatted_chosen = create_prompt_with_tulu_chat_format(chosen_messages)
        formatted_rejected = create_prompt_with_tulu_chat_format(rejected_messages)
        prompt = extract_anthropic_prompt(formatted_chosen)
        return {
            "prompt": prompt,
            "chosen": formatted_chosen[len(prompt):],
            "rejected": formatted_rejected[len(prompt):],
        }

    if chat_format != "native":
        raise ValueError(f"Unsupported chat format: {chat_format}")
    if tokenizer is None or tokenizer.chat_template is None:
        raise ValueError("Native chat formatting requires a tokenizer with chat_template")

    prompt_messages, chosen_completion, rejected_completion = native_hh_parts(sample)
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    formatted_chosen = tokenizer.apply_chat_template(
        prompt_messages + chosen_completion,
        tokenize=False,
    )
    formatted_rejected = tokenizer.apply_chat_template(
        prompt_messages + rejected_completion,
        tokenize=False,
    )
    if not formatted_chosen.startswith(prompt) or not formatted_rejected.startswith(prompt):
        raise ValueError("Tokenizer chat template does not preserve the assistant boundary")
    chosen = formatted_chosen[len(prompt):]
    rejected = formatted_rejected[len(prompt):]
    eos = tokenizer.eos_token or ""
    if eos and chosen.endswith(eos):
        chosen = chosen[: -len(eos)]
    if eos and rejected.endswith(eos):
        rejected = rejected[: -len(eos)]
    # DPOTrainer appends tokenizer.eos_token_id to each completion after
    # tokenization. For Llama 3 Instruct this is <|eot_id|>, so keeping the
    # strings as response text avoids accidentally training on two EOT tokens.
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def get_hh(
    data_files: str,
    sanity_check: bool = False,
    cache_dir: str = None,
    tokenizer=None,
    chat_format: str = "tulu",
    max_samples: int = None,
):
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset('json', data_files=data_files, cache_dir=cache_dir)['train']
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    return dataset.map(
        format_hh_preference,
        fn_kwargs={"tokenizer": tokenizer, "chat_format": chat_format},
    )

def get_reward_bench(data_files: str, sanity_check: bool = False, cache_dir: str = None):
    dataset = load_dataset('json', data_files=data_files, cache_dir=cache_dir)['train']
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def format_sample(sample):
        prompt = "<|user|>\n" + sample['prompt'].strip() + "\n" + "<|assistant|>\n"
        return {
            "prompt": prompt,
            "chosen": sample['chosen'].strip(),
            "rejected": sample['rejected'].strip(),
        }

    return dataset.map(format_sample)    

def get_SHP(data_files: str, sanity_check: bool = False, cache_dir: str = None):
    dataset = load_dataset('json', data_files=data_files, cache_dir=cache_dir)['train']
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def format_sample(sample):
        prompt = "<|user|>\n" + sample['history'].strip() + "\n" + "<|assistant|>\n"
        return {
            "prompt": prompt,
            "chosen": sample['human_ref_A'].strip() if sample['labels'] == 0 else sample['human_ref_B'].strip(),
            "rejected": sample['human_ref_A'].strip() if sample['labels'] == 1 else sample['human_ref_B'].strip()
        }

    return dataset.map(format_sample)   

def get_H4_stack_exchange(data_files: str, sanity_check: bool = False, cache_dir: str = None):
    dataset = load_dataset('json', data_files=data_files, cache_dir=cache_dir)['train']
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def format_sample(sample):
        prompt = "<|user|>\n" + sample['question'].strip() + "\n" + "<|assistant|>\n"
        return {
            "prompt": prompt,
            "chosen": sample['chosen'].strip(),
            "rejected": sample['rejected'].strip(),
        }

    return dataset.map(format_sample)     

def get_IEFeedback(data_files: str, sanity_check: bool = False, cache_dir: str = None):
    dataset = load_dataset('json', data_files=data_files, cache_dir=cache_dir)['train']
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def format_sample(sample):
        prompt = "<|user|>\n" + sample['messages'][0]['content'].strip() + "\n" + "<|assistant|>\n"
        return {
            "prompt": prompt,
            "chosen": sample['chosen'][-1]['content'].strip(),
            "rejected": sample['rejected'][-1]['content'].strip(),
        }

    return dataset.map(format_sample)   


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DPOConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in Transformers v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this finetuning script."
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this finetuning script."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and normalize the preference dataset to prompt/chosen/rejected.
    if data_args.train_file and 'hh_rlhf' in data_args.train_file:
        train_dataset = get_hh(
            data_args.train_file,
            cache_dir=model_args.cache_dir,
            tokenizer=tokenizer,
            chat_format=model_args.chat_format,
            max_samples=data_args.max_train_samples,
        )
    elif data_args.train_file:
        dataset_name = os.path.basename(os.path.dirname(data_args.train_file))
        loader = globals().get(f'get_{dataset_name}')
        if loader is None:
            train_dataset = load_dataset(
                'json', data_files=data_args.train_file, cache_dir=model_args.cache_dir
            )['train']
            required_columns = {'prompt', 'chosen', 'rejected'}
            if not required_columns.issubset(train_dataset.column_names):
                raise ValueError(
                    f"Unknown preference dataset format in {data_args.train_file}. "
                    "Provide prompt/chosen/rejected columns or add a dataset formatter."
                )
        else:
            train_dataset = loader(data_args.train_file, cache_dir=model_args.cache_dir)
    else:
        train_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            split='train',
            streaming=data_args.streaming,
        )

    if data_args.max_train_samples is not None:
        if data_args.streaming:
            train_dataset = train_dataset.take(data_args.max_train_samples)
        else:
            train_dataset = train_dataset.select(
                range(min(len(train_dataset), data_args.max_train_samples))
            )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        attention_kwargs = (
            {"attn_implementation": "flash_attention_2"}
            if model_args.use_flash_attn else {}
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            **attention_kwargs,
        )
        if model_args.peft_name_or_path:
            requested_adapters = sum(
                bool(value)
                for value in (
                    model_args.use_lora,
                    model_args.use_ia3,
                    model_args.use_prompt_tuning,
                )
            )
            if requested_adapters != 1:
                raise ValueError(
                    "When peft_name_or_path supplies the frozen SFT adapter, request "
                    "exactly one fresh DPO adapter with use_lora, use_ia3, or "
                    "use_prompt_tuning."
                )
            logger.info(
                "Loading frozen SFT adapter %s before initializing a fresh DPO adapter",
                model_args.peft_name_or_path,
            )
            model = PeftModel.from_pretrained(
                model=model,
                model_id=model_args.peft_name_or_path,
                is_trainable=False,
            )

        if model_args.use_lora:
            logger.info("Training with LORA...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=model_args.lora_rank, 
                lora_alpha=model_args.lora_alpha, 
                lora_dropout=model_args.lora_dropout,
                target_modules=model_args.lora_module
            )
        elif model_args.use_ia3:
            logger.info("Training with IA3...")
            peft_config = IA3Config(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                target_modules=model_args.ia3_module,
                feedforward_modules=model_args.feedforward_modules
            )
        elif model_args.use_prompt_tuning:
            logger.info("Training with PROMPT_TUNING...")
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                inference_mode=False, 
                num_virtual_tokens=model_args.num_virtual_tokens,
                prompt_tuning_init_text=model_args.prompt_tuning_init_text, 
                tokenizer_name_or_path=model_args.model_name_or_path,
            )   
        else:
            logger.info("Full fintuning...")
            peft_config = None 

        if peft_config is None:
            ref_model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                **attention_kwargs,
            )
            ref_model.config.use_cache = False
        else:
            # With PEFT, DPOTrainer merges the frozen SFT adapter into the
            # base, initializes the fresh adapter above, and obtains reference
            # logits by temporarily disabling only that fresh DPO adapter.
            ref_model = None
            
    else:
        logger.warning("No pretrained model_name_or_path is given. Training new model from scratch.")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        
    # set use_cache to False to enable activation checkpointing 
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if torch.cuda.is_available():
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3
            logger.warning("Peak CUDA memory allocated on this rank: %.2f GiB", peak_gib)
        if model_args.save_final_model:
            trainer.save_model()  # Saves the tokenizer too for easy upload.

if __name__ == "__main__":
    main()
