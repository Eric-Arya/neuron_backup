# [ICLR 2025] Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron

This repository contains code for the paper "[Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron](https://openreview.net/pdf?id=yR47RmND1m)". 

<img src="./figures/safety.png" alt="./" style="zoom:63%;" />

## Dataset layout

The dataset root is `/workspace/xcy/dataset`. Reusable source datasets are under
`shared/`, artifacts prepared specifically for this reproduction are under
`projects/iclr_neuron/`, and Hugging Face cache files are under
`_cache/huggingface/`. Table 1 scripts use these locations by default while
retaining command-line path overrides.

## Neuron Detection (PLND) 

The codebase is totally the same as [How do Large Language Models Handle Multilingualism?](https://arxiv.org/abs/2402.18815)  We provide codes for detecting neurons in Llama, Mistral and Gemma.

### Installation

The package can be installed by running the following command at the root of this repository: 

```shell
conda create -n iclr_neuron_detection python=3.9
conda activate iclr_neuron_detection
pip install -r requirement.txt
```

### Running

Detect corpus is harmful behavior dataset of [llm-attack](https://github.com/llm-attacks/llm-attacks/tree/main/data), we need to  **change transformers package**. When detecting, we need to define the language and number of documents used to detect. Detected neurons will be stored in folder `./output_neurons`.

```sh
cd /neuron_detection
python neuron_detection.py zou_train 200 112 native --prompt-format chat
```

The default corpus directory is
`/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/processed`; override it
with `--corpus-dir` when testing another corpus location.

`--prompt-format chat` is the default for instruction-tuned models and applies the
tokenizer's chat template with an assistant generation prompt. Use
`--prompt-format raw` only for an explicit raw-prompt comparison. The prompt
format is included in the output filename so chat and raw neuron sets cannot
overwrite each other.

### Parameters

**Number of Top-k neurons in each layer**

```python
top_number_attn = 100
top_number_ffn = 200
```

## Neuron Deactivation

We provide codes for detecting neurons in Llama, Mistral and Gemma.

### Installation

The package can be installed by running the following command at the root of this repository: 

```shell
conda create -n iclr_neuron_deactivation python=3.9
conda activate iclr_neuron_deactivation
pip install -r requirement.txt
```

### Running

We need to  **change transformers package**. 

```sh
cd /neuron_deactivate
python test_mistral_gsm.py {language} {understanding layer} {generation layer} {attn deact_number} {ffn deact_number} {whether under_attn} {whether reason_attn} {whether gen_attn} {whether under_ffn} {whether reason_ffn} {whether gen_ffn}
```

## Neuron Specific Enhancement

The checked-in enhancement code targets the 32-layer Llama 3 8B architecture,
including its 4:1 grouped-query attention ratio.

### Installation

For Llama 3, use a dedicated environment. The enhancement overlay is based on
Transformers 4.38.2 and must not be installed into the detection or deactivation
environment.

```shell
conda create -n iclr_neuron_enhancement python=3.9
conda activate iclr_neuron_enhancement
conda install pytorch==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
python -m pip install \
  "transformers==4.38.2" \
  "tokenizers==0.15.2" \
  "accelerate==0.27.2" \
  "datasets==2.15.0" \
  "peft==0.9.0" \
  "trl==0.7.11" \
  "bitsandbytes==0.43.0"

TRANSFORMERS_DIR=$(python -c 'import os, transformers; print(os.path.dirname(transformers.__file__))')
cp neuron_enhancement/transformers/trainer.py "$TRANSFORMERS_DIR/trainer.py"
```

The repository-level `requirement.txt` is not used for this phase: it omits the
enhancement training dependencies and includes unrelated evaluation packages.
See `enhancement_phase_LLAMA3_CONDA_SETUP.md` for the verified local setup.

### Running

After installing the Transformers overlay as shown above, replace the checked-in
placeholder paths in `train_neuron.py` with the neuron file, dataset, cache, and
output paths for the run.

```sh
cd neuron_enhancement
python train_neuron.py
```

### Parameters

Note that `attn_k` and `attn_v` needs to be  divided by `kv_repeat`. `index_keys` requires fitting to model you want to train and number of understanding layer and generation layer needs to be changed correspondingly.

```python
index_keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]         

index_keys_under = [i for i in range(8)]
index_keys_gen = [31-i for i in range(4)]

attn_k = {key: {num//4 for num in value} for key, value in attn_k.items()}
attn_v = {key: {num//4 for num in value} for key, value in attn_v.items()}
```

## Citation

If you found this repository useful, please consider

```latex
@inproceedings{
zhao2025understanding,
title={Understanding and Enhancing Safety Mechanisms of {LLM}s via Safety-Specific Neuron},
author={Yiran Zhao and Wenxuan Zhang and Yuxi Xie and Anirudh Goyal and Kenji Kawaguchi and Michael Shieh},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=yR47RmND1m}
}
```
