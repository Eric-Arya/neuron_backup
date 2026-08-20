# Chat-format `I cannot` Grad experiment

Updated: 2026-08-18

## Outcome

The experiment extracted an MLP-neuron gradient signal for the exact assistant prefix
`I cannot` from the SN-Tune corpus using the native Llama-3 chat template, enhanced the selected
neurons at all token positions, and evaluated the fixed controller against an identical
strength-zero chat control.

| Condition | HarmBench ASR ↓ | Beaver cost ↓ | GSM8K-100 ↑ | MMLU-285 ↑ |
|---|---:|---:|---:|---:|
| Chat strength-0 control | 29.0% (58/200) | -5.014 | 77% (77/100) | 69.82% (199/285) |
| **Chat `I cannot` Grad, top-25, strength 0.5, all positions** | **28.0% (56/200)** | **-5.150** | **77% (77/100)** | **69.47% (198/285)** |

The neuron edit gives a small aggregate safety improvement: ASR decreases by one percentage point
and mean Beaver cost decreases by 0.136. GSM8K is unchanged and MMLU loses one correct answer. The main ASR
reduction relative to the historical raw-prompt Llama-3 baseline comes from native chat formatting,
not from the neuron edit.

## Gradient and selection protocol

- Model: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct`, BF16.
- Source: records 0--255 of `circuit_breakers_train.json`.
- Used records: 235 whose `llama3_output` starts exactly with `I cannot`; no normalized exact prompt
  matches with the frozen HarmBench test manifest were found.
- Input format: native tokenizer chat template with one user message and an assistant generation
  header.
- Objective: mean log probability of the exact two-token target `I cannot` (token IDs 40, 4250).
- Gradient variables: one multiplicative scalar for each of the 14,336 post-activation MLP
  dimensions in each of 32 layers.
- Ranking: stability score among the 2,000 largest absolute mean gradients over all 235 examples.
- Development split: 47 HarmBench complement prompts, disjoint from the frozen 200-prompt test.
- Minimal tuning grid: top-k 10/25/50, strength 0.5/0.75/1.0, all-token scope only. Selection
  penalized safe-to-unsafe regressions and required no more than a five-point GSM8K-20 drop.
- Selected controller: top 25, strength 0.5, applied at all token positions. The multiplier is
  `1 + 0.5 * sign(mean_gradient)` for each selected neuron.

On the off-test tuning split, the chat baseline scored 34.0% ASR (16/47). The selected controller
scored 29.8% (14/47), with two unsafe-to-safe changes, zero safe-to-unsafe changes, and unchanged
GSM8K-20 accuracy of 65%.

## Benchmark and runtime

All GPU work used physical GPU 1. A 32-example real-data generation benchmark selected batch 32:

| Batch | Examples/s | Peak allocated GPU memory |
|---:|---:|---:|
| 8 | 19.7 | 15.06 GiB |
| 16 | 51.8 | 15.14 GiB |
| 32 | 95.4 | 15.28 GiB |

The full 235-example gradient extraction took 13.4 seconds after model loading. Frozen evaluation
used greedy decoding, 128 HarmBench tokens, 256 GSM8K tokens, the seed-112 MMLU-285 subset, and the
Beaver cost scorer.

## Artifacts

- Gradient ranking and metadata: `results/grad_i_cannot_sn_corpus_chat/corpus_gradients/`
- Tuning sweep: `results/grad_i_cannot_sn_corpus_chat/tuning_all_chat/`
- Selected frozen evaluation: `results/grad_i_cannot_chat_all_k25_s0p5/`
- Strength-zero chat control: `results/grad_chat_strength0_control/`
- Matched capability control: `results/llama3_base_chat_capability_unified_b16_m8/`
- Matched chat Grad capability: `results/grad_i_cannot_chat_capability_unified_b16_m8/`

The extractor and development evaluator support `--prompt-format chat` in
`unified_eval/grad_development.py`. Frozen HarmBench evaluation uses
`--llama3-harm-prompt-format chat` in `unified_eval.runner`.
