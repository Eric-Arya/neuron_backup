# Llama-3-8B-Instruct HH-DPO experiment

Date: 2026-08-18

## What was done

- Added a native Llama-3 preference formatter to the NeurIPS training code while preserving its Tulu/Llama-2 path.
- Added flexible multi-GPU DPO launchers with overridable environment variables, defaults, and examples.
- Benchmarked training on the longest examples sampled from the real HH harmless dataset before starting the full run.
- Trained an IA3 adapter on all 42,537 HH harmless preference pairs for three epochs on two GPUs.
- Added `llama3_dpo` as a first-class unified-evaluation method, benchmarked inference batch size, and ran the frozen HarmBench, GSM8K, and MMLU suites.
- Validated the adapter tensors, output row counts, evaluation checksums, shell syntax, Python compilation, and automated tests.

## Chat and preference format

The run uses the tokenizer's native Llama-3 chat template via `apply_chat_template`, not the older Tulu delimiters. Anthropic HH dialogues are parsed into role/content messages. The shared dialogue prefix becomes the DPO prompt, ending at the assistant generation header, and the chosen/rejected assistant continuations are supplied separately. The formatter removes the template's final EOT from each completion because TRL appends the EOS/EOT during DPO tokenization; the resulting training sequence has one BOS and one final EOT.

## Training benchmark

The benchmark used long examples selected from the real HH file with the production sequence limits. Throughput is aggregate across two GPUs.

| Per-device batch | Gradient checkpointing | Result | Examples/s | Peak allocated/GPU |
|---:|:---:|:---|---:|---:|
| 2 | no | pass | 5.084 | 44.85–45.77 GiB |
| 3 | no | pass, selected | 6.156 | 58.93–59.67 GiB |
| 4 | no | OOM | — | approximately 79 GiB |
| 4 | yes | pass | 5.436 | 32.33–32.77 GiB |

Batch 3 without checkpointing was selected because it was the fastest passing configuration. With two GPUs and 20 accumulation steps, the effective batch size was 120.

## Full training

| Setting | Value |
|---|---|
| Reference/SFT model | `/workspace/xcy/models/Meta-Llama-3-8B-Instruct` |
| Dataset | `/workspace/xcy/dataset/shared/hh_rlhf/harmless_base/train.jsonl` |
| Examples | 42,537 |
| Adapter | IA3 on each MLP `down_proj` |
| Trainable parameters | 458,752 (32 tensors of shape 1 x 14,336) |
| DPO beta | 0.1 |
| Learning rate / schedule | 1e-3 / cosine |
| Epochs / optimizer steps | 3 / 1,065 |
| Max total / prompt length | 4,096 / 2,048 |
| Precision / attention | bfloat16 / Flash Attention 2 |
| Per-device / effective batch | 3 / 120 |
| Gradient checkpointing | off |
| Runtime | 1:37:08 |
| Mean training loss | 0.5470904 |
| Training throughput | 21.895 examples/s |
| Peak allocated/GPU | 59.13 and 59.38 GiB |

The final adapter contains 458,752 finite parameters. Its model-file SHA-256 is `6f768885ee94fd174598ee733ce6c2a28239b99042bef48cdd3d4c457f147fad`.

## Unified evaluation

Inference benchmarking on the first real HarmBench prompts selected batch 16. At eight generated tokens it reached 90.36 examples/s and 15.12 GiB peak allocation on one GPU. The final run used greedy decoding and the same frozen inputs and scoring code as the existing unified runs:

- HarmBench: seed-42 frozen sample of 200, 128 generated tokens, refusal-substring ASR, and Beaver-7B cost.
- GSM8K: first 100 test rows, zero-shot native Llama-3 chat, 256 generated tokens, flexible exact numeric match.
- MMLU: five seed-112 questions from each of 57 subjects (285 total), five-shot multi-turn native Llama-3 chat.

| Condition | HarmBench ASR (lower) | Mean Beaver cost (lower) | GSM8K (higher) | MMLU (higher) |
|---|---:|---:|---:|---:|
| Llama-3-8B-Instruct base | 65.50% | 0.594 | 63.00% | 67.37% |
| **Llama-3-8B-Instruct + HH-DPO IA3** | **60.00%** | **-0.634** | **76.00%** | **68.42%** |
| Llama-2 base | 99.00% | 7.850 | 24.00% | 39.30% |
| NeurIPS Llama-2 SFT+DPO | 50.00% | -3.348 | 11.00% | 35.79% |

Relative to its Llama-3 base, HH-DPO reduced HarmBench ASR by 5.5 percentage points and mean Beaver cost by 1.228, while GSM8K increased by 13 points and MMLU by 1.05 points. These are measured outcomes on the frozen subsets, not confidence-adjusted population estimates. Direct cross-family capability comparisons should account for the model-specific Llama-2 paper prompting versus native Llama-3 prompting.

The DPO HarmBench generations included 33/200 responses flagged by the evaluator's repetition heuristic, compared with 26/200 for the Llama-3 base. This should be considered alongside the improved safety scores.

### Native-chat HarmBench follow-up

Wrapping the same 200 HarmBench behaviors in the native Llama-3 user/assistant chat template changed
the standalone DPO result from 60.0% to 35.0% ASR and from -0.634 to -5.035 mean Beaver cost.
Repetitive responses fell from 33 to 9 and mean generated length from 127.6 to 53.5 tokens. Capability
tasks were not rerun because they already use native Llama-3 chat formatting.

## Artifacts and verification

- Final adapter: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH`
- Training metrics: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH/train_results.json`
- Training log: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH/training.log`
- Evaluation results: `/workspace/xcy/safety_repro/unified_eval/results/llama3_dpo`
- Native-chat safety result: `/workspace/xcy/safety_repro/unified_eval/results/llama3_dpo_chat_harmbench`
- Cross-method summary: `/workspace/xcy/safety_repro/unified_eval/results/unified_summary.json`
- Training benchmark artifacts: `/workspace/xcy/tmp/llama3_dpo_benchmark`

All 13 evaluation artifact checksums passed. The DPO formatter tests passed 3/3, and the unified-evaluator tests passed 18/18.
