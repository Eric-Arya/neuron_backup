# Llama 3 Neuron Enhancement Conda Setup

This note records the Llama 3-only neuron-specific enhancement setup. The
detection and deactivation environments were not modified.

## Environment

The dedicated environment is installed at:

```text
/workspace/xcy/miniconda3/envs/iclr_neuron_enhancement
```

Activate it with:

```bash
source /workspace/xcy/miniconda3/bin/activate
conda activate iclr_neuron_enhancement
```

It uses:

- Python 3.9
- PyTorch 2.5.1 with CUDA 12.1 support
- `transformers==4.38.2`
- `tokenizers==0.15.2`
- `accelerate==0.27.2`
- `datasets==2.15.0`
- `peft==0.9.0`
- `trl==0.7.11`
- `bitsandbytes==0.43.0`

The environment was cloned from the working CUDA-enabled deactivation
environment to reuse PyTorch, then the packages above were independently
pinned. The original detection and deactivation package versions remain
unchanged.

## Transformers overlay

`neuron_enhancement/transformers/trainer.py` matches the upstream Transformers
4.38.2 trainer except for the neuron-gradient mask. It is installed at:

```text
/workspace/xcy/miniconda3/envs/iclr_neuron_enhancement/lib/python3.9/site-packages/transformers/trainer.py
```

No detection or deactivation overlay is installed in this environment.

The training script attaches `activate_neuron` to the `TrainingArguments`
instance after construction. This is required because `activate_neuron` is
enhancement state consumed by the custom trainer, not an upstream
`TrainingArguments` constructor field.

The mask tensors are created on the same device as each parameter gradient, so
the enhancement loop works when Llama 3 is on a CUDA device.

## Verification

The setup passed these checks on an NVIDIA H100:

- PyTorch detects CUDA and all pinned libraries import together.
- The legacy `SFTTrainer` interface used by `train_neuron.py` imports.
- A paged AdamW 32-bit optimizer step succeeds on CUDA.
- The local 8,030,261,248-parameter Llama 3 Instruct checkpoint loads fully in
  BF16 on CUDA with network access disabled.
- A one-step training test succeeds with a tiny 32-layer Llama model using
  Llama 3-style grouped-query attention (`num_attention_heads=4`,
  `num_key_value_heads=1`).
- Selected Q, up-projection, and down-projection neuron dimensions change,
  while masked dimensions and the non-layer LM head remain unchanged.

`train_neuron.py` uses the local Llama 3 Instruct checkpoint at
`/workspace/xcy/models/Meta-Llama-3-8B-Instruct`. A full run still requires a
training JSON dataset with `original_question` and `response` fields, a
detected-neuron file, and concrete output/cache paths in place of the remaining
checked-in placeholders.
