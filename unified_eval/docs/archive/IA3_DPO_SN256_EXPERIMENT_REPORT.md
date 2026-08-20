# IA3-DPO on the first 256 SN-corpus pairs

Date: 2026-08-19

## Setup

The first 256 rows of `circuit_breakers_train.json` were converted to raw DPO triples as follows:

- `prompt = prompt + ". "`
- `chosen = llama3_output` (safe/refusal response)
- `rejected = output` (unsafe/compliant response)

The resulting dataset has 256 complete, non-identical preference pairs and SHA-256
`58b5e33a37f5a84dd294c0a74928bdb21c10d8c92fabb2394b0804496fed3f5c`:

`/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/training/dpo/circuit_breakers_train_first256_llama3_raw_snformat_dpo.jsonl`

A row-wise check confirmed that its prompts and chosen responses are exactly the same 256 raw
prompt/completion examples used by IA3-SFT. The base and reference model was
`/workspace/xcy/models/Meta-Llama-3-8B-Instruct`; no SFT adapter was preloaded.

## Real-data training benchmark

All conditions used the 120 longest real SN preference pairs, BF16, two H100 80GB GPUs, max
length 4096, and effective batch size 120.

| Per-device batch | Accumulation | Checkpointing | Step time | Peak allocated | Result |
|---:|---:|:---:|---:|---:|:---|
| 4 | 15 | no | **7.80 s** | 58.98 GiB | **selected** |
| 6 | 10 | no | -- | about 79 GiB | OOM |
| 10 | 6 | no | -- | 76.52 GiB before another 4.61 GiB request | OOM |
| 10 | 6 | yes | 9.44 s | 47.87 GiB | stable but slower |

## Training

Both runs used IA3 on `down_proj`, learning rate `1e-3`, AdamW, cosine scheduling, weight decay
`0.1`, effective batch 120, and DPO beta `0.1`.

The initial paper-config run used three epochs, producing nine optimizer steps in 45.10 seconds.
Mean train loss was 0.2996, and the final logged preference margin was 1.7375. Its adapter had a
mean absolute IA3-gate displacement from identity of 0.003348. After its evaluation was frozen and
checksummed, the E3 model directory was permanently removed as requested.

The replacement run used 20 epochs, matching the IA3-SFT epoch count. It produced 60 optimizer
steps in 296.41 seconds, with mean train loss 0.0638 and a final logged preference margin of 5.2765.
Preference accuracy was 100% from the second epoch onward. Checkpoints were saved at steps
15, 30, 45, and 60, corresponding to epochs 5, 10, 15, and 20.

The active E20 adapter contains 32 finite tensors and 458,752 parameters. Its mean absolute IA3-gate
displacement from identity is 0.010800.

- Adapter: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-SNRawDot256-E20`
- Adapter weights SHA-256: `45f53f67df557111e18e00ec0322e1c101ef5a0c3463414fe5ea2ce0feb9d7c6`
- Adapter config SHA-256: `b36fdecdff627d3bb81609e3c1f6d3c6e86af4c7654286d68a7834962db7cb3f`

## Evaluation results

Evaluation used FP32 with TF32 disabled. HarmBench used the frozen seed-42 200-example subset,
raw prompts, greedy decoding, and 128 new tokens. IFEval used the frozen seed-112 200-example
subset, task-native Llama-3 chat, greedy decoding, 1,024 new tokens, and the official deterministic
strict/loose scorers.

| Condition | HarmBench ASR lower is better | Mean Beaver cost lower is better | Strict IFEval prompt | Strict instruction | Loose prompt | Loose instruction |
|:---|---:|---:|---:|---:|---:|---:|
| Llama-3 baseline | 65.5% (131/200) | 0.594 | 69.5% (139/200) | 77.24% (241/312) | 77.0% (154/200) | 84.29% (263/312) |
| SN256 IA3-DPO, 3 epochs | 63.5% (127/200) | 0.596 | 69.0% (138/200) | 76.92% (240/312) | 76.5% (153/200) | 83.97% (262/312) |
| **SN256 IA3-DPO, 20 epochs** | **63.5% (127/200)** | **0.035** | **65.5% (131/200)** | **75.32% (235/312)** | **73.5% (147/200)** | **82.05% (256/312)** |

Against baseline, E20 HarmBench had 12 unsafe-to-safe changes and 8 safe-to-unsafe changes
(paired exact McNemar `p=0.503`). Relative to E3, E20 had 7 safety gains and 7 losses, leaving ASR
unchanged. Beaver cost nevertheless improved by 0.561 from E3 and by 0.558 from baseline, so the
continuous safety scorer detects an improvement that the refusal-substring ASR does not.

Against baseline, E20 strict IFEval had 5 gains and 13 losses (`p=0.096`); loose IFEval had 5 gains
and 12 losses (`p=0.143`). Relative to E3, strict prompt accuracy fell 3.5 points and loose prompt
accuracy fell 3.0 points. E20 produced 31 repetitive HarmBench responses and 17 repetitive IFEval
responses, compared with 25 and 20 for E3 and 26 and 21 for baseline.

Longer DPO training therefore moves the IA3 adapter farther from identity and improves mean Beaver
cost, but it does not improve the binary HarmBench ASR and it reduces general instruction-following
accuracy. This supplies a modest safety-capability trade-off signal, although the paired changes are
not significant at the 0.05 level on these 200-example subsets.

Evaluation artifacts are in:

`results/archive/sncorpus_dpo/llama3_dpo_ia3_snrawdot256_e20_raw_harmbench_ifeval_fp32`

The frozen historical E3 evaluation remains at:

`results/archive/sncorpus_dpo/llama3_dpo_ia3_snrawdot256_e3_raw_harmbench_ifeval_fp32`

The adapter path and hashes in `run_config.json`, together with the training provenance above,
identify this run as SN-corpus DPO.
