# Unified safety-neuron evaluation

This directory evaluates reproduced baselines and interventions on frozen, shared subsets:

| Method | Evaluated model/intervention |
|---|---|
| `llama3_base` | Unmodified Llama-3-8B-Instruct baseline |
| `llama3_dpo` | Llama-3-8B-Instruct with the HH-harmless DPO IA3 adapter |
| `llama3_dpo_patch` | Llama-3 base dynamically patched from the HH-DPO guide at ranked post-MLP neurons |
| `llama3_sft` | Llama-3-8B-Instruct with the native-chat SN-corpus IA3-SFT adapter |
| `llama3_sft_patch` | Llama-3 base dynamically patched from the IA3-SFT guide at ranked post-MLP neurons |
| `llama2_base` | Unmodified Llama-2-7B base model, without SFT/DPO adapters or patching |
| `grad` | Llama-3-8B-Instruct with the fixed top-25 signed gradient controller, strength 1 |
| `sn` | Llama-3-8B-Instruct SN-Tune merged checkpoint at delta scale `alpha=8` |
| `sn_direct` | Unmodified Llama-3 with raw-detected SN-Tune activation dimensions directly strengthened, without fine-tuning |
| `neurips` | Llama-2-7B Base dynamically patched from the DPO guide at the top 20,000 safety neurons |
| `neurips_direct` | Unmodified Llama-2-7B Base with the NeurIPS-ranked post-MLP activation dimensions directly multiplied |
| `neurips_dpo` | NeurIPS Llama-2 SFT+DPO guide without dynamic neuron patching |

HarmBench uses the exact NeurIPS seed-42 200/400 manifest and 128-token greedy decoding. It reports both the case-sensitive `llm-attacks` refusal-substring ASR and mean `beaver-7b-v1.0-cost`. Capability evaluation is unified across model families by default: GSM8K uses the first 100 test rows, zero-shot chat prompting, greedy decoding, a 256-token limit, and flexible deterministic numeric exact match (final `####`, explicit final-answer phrase, final equation RHS, then last-number fallback); MMLU uses the seed-112 balanced 285-example subset, five-shot multi-turn chat prompting, and constrained next-token A/B/C/D scoring. Llama-3 uses its native chat template and Llama-2 uses Tulu serialization. The released Llama-2 paper protocol remains available explicitly with `--neurips-capability-protocol paper`.

Safety prompting defaults to each reproduced convention: raw prompts for the Llama-3 methods and
Tulu prompts for NeurIPS. Pass `--llama3-harm-prompt-format chat` to render Llama-3 HarmBench with
the same native chat template used by its capability tasks. Beaver cost is calculated from the full
decoded prompt plus completion, as in the NeurIPS implementation. Lower cost is safer.

The optional `beavertails` task uses the frozen paper-protocol seed-42 sample of 200 prompts from
the final 600 rows of `round0/330k/test`. It reports both `beaver-7b-v1.0-cost` (lower is safer) and
`beaver-7b-v1.0-reward` (higher is more helpful). Llama-3 can be compared under raw or native-chat
serialization with `--llama3-beavertails-prompt-format`.

## Commands

Validate paths and frozen manifests:

```bash
python -m unified_eval.runner validate --method grad
python -m unified_eval.runner validate --method llama3_dpo
python -m unified_eval.runner validate --method sn
python -m unified_eval.runner validate --method neurips
```

Benchmark batch sizes on a minimal real HarmBench slice before a full run:

```bash
COMMAND=benchmark METHODS="grad sn neurips" bash run_unified_eval.sh
```

IFEval uses the official 541-prompt Google Research dataset and its deterministic strict and loose
constraint checkers (not an LLM judge). The frozen seed-112 subset contains 200 prompts, covers all
25 instruction types, and is stored at
`/workspace/xcy/dataset/ifeval/subsets/ifeval_seed112_n200.jsonl`. Run it with task-native chat
formatting and the benchmark-selected batch size of 16:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method sn --run-name ifeval_sn_alpha4_fp32 \
  --sn-model /workspace/xcy/safety_repro/iclr_neuron_expanded_kv/neuron_enhancement/outputs/sn_delta_scale/exact100_200_cap25_docs256_ep20_alpha4_mmlu_regenerated \
  --tasks ifeval --ifeval-batch-size 16 --ifeval-max-new-tokens 1024 \
  --sn-dtype float32 --device cuda:0
```

The matched baseline and SN-Tune alpha-1/4/8 results are reported in
[`RESULTS_I_CARE_ABOUT.md`](RESULTS_I_CARE_ABOUT.md).

Run all methods. The launcher evaluates the two independent Llama-3 methods concurrently on GPUs 0 and 1, then gives both GPUs to NeurIPS dynamic patching:

```bash
COMMAND=run METHODS="grad sn neurips" bash run_unified_eval.sh
```

Evaluate the standalone Llama-3 DPO adapter on the same frozen suite:

```bash
COMMAND=run METHODS="llama3_dpo" bash run_unified_eval.sh
```

Run only standalone-DPO HarmBench with the native Llama-3 chat rendering used by GSM8K:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method llama3_dpo --run-name llama3_dpo_chat_harmbench \
  --tasks harmbench --llama3-harm-prompt-format chat
```

Run the Llama-3 baseline BeaverTails safety/helpfulness evaluation on physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method llama3_base --run-name llama3_base_beavertails_chat \
  --tasks beavertails --llama3-beavertails-prompt-format chat \
  --beavertails-batch-size 32 --cost-batch-size 32 --reward-batch-size 32 \
  --device cuda:0 --cost-device cuda:0 --reward-device cuda:0
```

Evaluate the DPO-guide activation patch at top-20k and top-8k. The ranking contrasts native-chat
completion activations of Llama-3 Instruct and its HH-DPO adapter on 200 held-out HH examples:

```bash
COMMAND=run TOP_K_VALUES="20000 8000" bash run_llama3_dpo_patch_sweep.sh
```

Generate the test-disjoint native-chat IA3-SFT ranking, then run its fixed top-20k patch on all
three benchmarks with chat formatting:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash \
  /workspace/xcy/safety_repro/neurips_neuron/scripts/safety_neuron/get_change_scores_llama3_sft.sh

CUDA_VISIBLE_DEVICES=0,1 python -m unified_eval.runner run \
  --method llama3_sft_patch --run-name llama3_sft_patch_20k_chat \
  --llama3-harm-prompt-format chat --base-device cuda:0 --guide-device cuda:1 \
  --harmbench-batch-size 16 --gsm8k-batch-size 16 --mmlu-batch-size 8
```

The completed training and evaluation results are documented in
[`LLAMA3_DPO_EXPERIMENT_REPORT.md`](LLAMA3_DPO_EXPERIMENT_REPORT.md).

All hyperparameters and paths have defaults and can be overridden either through the shell variables shown at the top of `run_unified_eval.sh` or direct Python flags. For example:

All unified-evaluator model, intervention, cost-model, and reward-model dtype flags default to
FP32. TF32 matmuls are disabled and `float32_matmul_precision` is set to `highest`. Every
`run_config.json` records this under both `semantic.floating_point_protocol` (so it participates in
the run fingerprint) and `runtime.floating_point_protocol`, along with the resolved per-model
dtypes. Pass an explicit dtype flag only for a separately named reduced-precision experiment.

Grad activation editing defaults to `--grad-direction positive-only`: K is counted after filtering
to positive-gradient neurons, selected activations are multiplied by `1 + strength`, and
negative-gradient neurons are left unchanged. Use `--grad-direction signed` explicitly to restore
the bidirectional controller.

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run --method grad \
  --grad-top-k 50 --grad-strength 0.75 --harmbench-batch-size 8
```

Directly scale the ranked raw SN-Tune cap-25 mask at every token position. Here
`strength=s` means multiplier `1+s`, so attenuation is available with `-1 < s < 0`; safety
prompting remains raw by default. The benchmarked defaults are batch 32 for HarmBench and IFEval:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run --method sn_direct \
  --run-name sn_direct_raw_cap25_m0p95 --tasks harmbench ifeval \
  --sn-direct-strength -0.05 --llama3-harm-prompt-format raw
```

Directly scale the NeurIPS top-20k post-MLP activation pool at all token positions. The multiplier
is specified directly; raw HarmBench prompting is enforced for this diagnostic. The benchmarked
defaults are batch 32 for HarmBench and batch 16 for full-length IFEval:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run --method neurips_direct \
  --run-name neurips_direct_top20k_m1p2 --tasks harmbench ifeval \
  --neurips-direct-multiplier 1.2
```

Outputs are resumable and configuration-fingerprinted under `results/<method>/`. The cross-method files are `results/unified_summary.json` and `results/unified_summary.csv`. The already completed and checksum-verified NeurIPS HarmBench patched condition is reused by default when its exact 200-row/128-token/top-20,000 configuration is requested; pass `--no-reuse-neurips-harm-cache` to regenerate it.

## Leakage-free Grad development

Develop a HarmBench-matched Grad controller on the complement of the frozen test
manifest, using only physical GPU 0:

```bash
COMMAND=all GPU=0 bash run_grad_harmbench_development.sh
```

The split builder removes exact prompt duplicates of frozen-test behaviors before
creating the 150-example gradient-selection and 47-example tuning splits. The
pipeline benchmarks generation first, extracts full-refusal contrastive gradients,
and evaluates the fixed controller grid on tuning HarmBench and GSM8K examples.

The optional direct-refusal cascade is a separately named two-pass inference
policy, not the original fixed Grad controller. It keeps a baseline completion
when that completion contains a direct refusal and otherwise uses the Grad
completion. For example:

```bash
python -m unified_eval.grad_development cascade \
  --cascade-name grad_k50_direct_refusal_test \
  --manifest /workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/table1_seed42_n200.jsonl \
  --baseline-scored results/llama3_base/harmbench/asr_scored.jsonl \
  --controller-scored results/grad_advbench_devtuned_k50/harmbench/asr_scored.jsonl \
  --baseline-costs results/llama3_base/harmbench/costs.jsonl \
  --controller-costs results/grad_advbench_devtuned_k50/harmbench/costs.jsonl
```

For `llama3_base`, HarmBench, GSM8K-100, and MMLU-285 are generated by the unified evaluator by default. The checksum-recorded historical capability summaries remain available explicitly with `--llama3-base-capability-source inherited`.
