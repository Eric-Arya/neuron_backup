# SafetyNeuron
Data and code for the paper: Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons (NeurIPS 2025)


## Installation

``` bash
git clone https://github.com/THU-KEG/SafetyNeuron.git
cd SafetyNeuron
```

## Data Preparation
The datasets used in our experiments can be downloaded with scripts in `scripts/data`. You can also add your datasets by converting the format to a unified format. Details can be found in `scripts/data/reformat_datasets.py`.


## Model Alignment

Our code to fine-tune model is stored in the `src/training` folder. You can also use the following scripts to obtain the models we used in paper.

``` bash
bash scripts/training/finetune_lora_with_accelerate.sh
bash scripts/training/dpo.sh
```

### Llama-3-8B-Instruct DPO

The Llama 3 launcher treats the instruction checkpoint as the already-SFT
reference, trains a fresh IA3 adapter on HH-RLHF harmless preferences, and uses
the tokenizer's native Llama 3 chat template. It defaults to two GPUs and keeps
the paper's effective batch size of 120:

```bash
bash scripts/training/dpo_llama3_8b_instruct.sh
```

Benchmark a minimal set of longest real examples before changing memory-related
settings. For example:

```bash
BASE_MODEL=/workspace/xcy/models/Meta-Llama-3-8B-Instruct \
TOKENIZER_PATH=/workspace/xcy/models/Meta-Llama-3-8B-Instruct \
SFT_ADAPTER= CHAT_FORMAT=native PER_DEVICE_TRAIN_BATCH_SIZE=3 \
GRADIENT_CHECKPOINTING=0 bash scripts/training/benchmarking/dpo_throughput.sh
```

## Finding Safety Neurons

### HookedModel Class

Our `HookedPreTrainedModel` in `src/models/HookedModelBase.py` inherits both from `TransformerLens` and huggingface `transformers`, supporting methods such as `model.run_with_cache()`, `model.generate()`. You can also add your models following the implementation in `src/models/HookedLlama.py`

### Compute Change Scores

Our implementation of *Generation-Time Activation Contrasting* is in `src/activation_processor.py`. You can use the following script to compute and save the *change scores* and *neuron ranks*.

``` bash
bash scripts/safety_neuron/get_change_scores.sh
```
The meaning of important arguments
- `--first_peft_path`: The dir containing peft checkpoint. If not provided, we will use the base model.
- `--second_peft_path`: The same as before, and we use generation based on the second model.
- `--token_type`: Which token position to compare neuron activation. Support **full prompt**, **last token of prompt**, **completion**.

### Dynamic Activation Patching

We implement *Dynamic Activation Patching* by overwriting the `generate()` method of `transformers` models in `src/models/HookedModelBase.py` (currently we only implement the greedy seach decoding, the other sampling strategies are similar).

You can perform *Dynamic Activation Patching* by adding 3 extra arguments to `model.generate()`
- `--guided_model`: The model whose activations are used for patching.
- `--index`: The neurons we want to patch, obtained in previous step.
- `--hook_fn`: The hook function actually performs patching.

### Table 1: Llama2 Base on HarmBench

The current-Transformers, resumable Table 1 reproduction has a dedicated
two-GPU pipeline. It freezes the seed-42 prompt manifest, benchmarks real
HarmBench examples, smoke-tests zero-neuron parity and top-20,000 divergence,
then generates and Beaver-scores Base, full DPO, and dynamically patched Base.

```bash
bash scripts/eval/table1_llama_base_harmbench_pipeline.sh
```

To run one phase or override defaults, use environment variables or append
Python flags. For example:

```bash
COMMAND=smoke TOP_K=20000 SMOKE_MAX_NEW_TOKENS=8 \
  bash scripts/eval/table1_llama_base_harmbench.sh
```

Outputs default to `results/table1_llama_base_harmbench/`; generation and cost
files are atomic per-batch shards and can be resumed with `COMMAND=run`.

### Figure 2 left: Llama2 safety neurons

Prepare the prompt-only BeaverTails test split and freeze the exact 200-prompt,
seed-42 sample used by the released evaluation code:

```bash
python scripts/data/prepare_beavertails_figure2.py
```

Then reproduce only the Llama2 safety-neuron curve for patching Base with DPO
on two GPUs. The script is resumable and accepts model, adapter, neuron-ranking,
batch-size, token-length, and top-k overrides:

```bash
bash scripts/eval/figure2_left_llama2_safety_neurons.sh
```

Outputs default to `results/figure2_left_llama2_safety_neurons/`, including the
frozen manifest, per-batch generations and Beaver costs, JSON/CSV curve data,
PNG/PDF plots, run configuration, and SHA-256 checksums.


### Evaluation

Code of this part is stored in the `src/eval` folder. 

You can use the following script to evaluate the results of dynamic activation patching

```bash
bash scripts/eval/arena.sh
```
Here are some important arguments in the script
- `--guided_generation`: If not specified, only evaluate as usual LLMs.
- `--cost_model_name_or_path`: The model used to evaluate the safety of responses.
- `--topk_ablate`: The number of neurons we want to intervene. 
- `--red_peft_path`: The model being patched. If not provided, we will use the base model.
- `--blue_peft_path`: The model used for patching. 
- `--index_path`: Safety neurons index.

## LLM Safeguard

Use `src/neuron_activation.py` to create the training datasets. Use `src/predict_before_gen.py` to evalute the performance of trained safeguard.

## Other Experiment

- `src/ppl.py`: Compute the perplexity after dynamic activation patching.
- `src/neuron2word.py`: Project the neuron weights to vocabulary space.
- `src/training_free_neurons.py`: . 
- `eval/*/run_eval.py`: Evaluate the general capabilities of patched model.





# Cite
If you find our code useful, we will sincerely appreciate it and encourage you to cite the following article:

```bibtex
@inproceedings{
chen2025towards,
title={Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons},
author={Jianhui Chen and Xiaozhi Wang and Zijun Yao and Yushi Bai and Lei Hou and Juanzi Li},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=AAXMcAyNF6}
}
 ```
