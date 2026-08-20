# IA3-SFT alpha-3 held-out safety-neuron patching

## Setup

- Base: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct`
- Guide adapter: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20`
- IA3 scaling: `gate = 1 + 3 * (trained_gate - 1)`
- Training rows: SN corpus rows 0--255
- Selection: 200 seed-42 records sampled from rows 256 onward after training-prefix, frozen
  HarmBench, and duplicate-prompt exclusions
- Selection format: raw prompt plus the SN-training `.` boundary
- Ranking positions: 50,964 base-generated completion tokens
- Eligible neurons: 444,416 post-MLP dimensions in layers 0--30
- Ranking SHA-256: `3185b320c8681c11e8dabefb798d712d9716933e879b6ee03f847cc54d06474a`
- Evaluation device: physical GPU 1 only
- Evaluation dtype: BF16 for the base, guide, and Beaver cost model
- Safety protocol: raw HarmBench; BeaverTails was not run
- Degeneration protocol: all 200 rows of the frozen seed-112 coverage-first IFEval subset,
  containing 312 instruction constraints, task-native single-turn chat, greedy decoding, and
  1,024 maximum new tokens
- IFEval inference batch size: 16, selected by the real-prompt benchmark below

## Safety results

| Condition | Patched share | HarmBench ASR lower is safer | Beaver cost lower is safer | Repetitive | Mean generated tokens |
|---|---:|---:|---:|---:|---:|
| Matched raw BF16 base | 0% | 65.5% (131/200) | 0.594 | 26/200 | 127.24 |
| Alpha-3 patch, top-8k | 1.8% | 32.5% (65/200) | -5.039 | 6/200 | 68.12 |
| Alpha-3 patch, top-20k | 4.5% | 24.0% (48/200) | -5.371 | 7/200 | 59.64 |
| Alpha-3 patch, top-40k | 9.0% | 22.0% (44/200) | -5.610 | 5/200 | 56.81 |
| Alpha-3 patch, top-80k | 18.0% | 15.5% (31/200) | -6.115 | 4/200 | 48.99 |
| Alpha-3 patch, top-160k | 36.0% | 12.0% (24/200) | -6.171 | 3/200 | 44.96 |
| Alpha-3 patch, top-240k | 54.0% | 10.0% (20/200) | -6.445 | 2/200 | 42.21 |
| Alpha-3 patch, top-320k | 72.0% | 9.0% (18/200) | -6.356 | 2/200 | 41.70 |
| Alpha-3 patch, all 444,416 | 100% | 9.0% (18/200) | -6.401 | 2/200 | 43.73 |

ASR improves monotonically and then saturates: 320k and the full patch both score 9.0%. Beaver
cost is not monotonic; its lowest value is -6.445 at 240k, while 320k and the full patch score
-6.356 and -6.401. Relative to the matched base, 320k lowers ASR by 56.5 percentage points and
cost by 6.950. Patching the remaining 124,416 dimensions produces no additional ASR reduction.

## Full IFEval-200 degeneration evaluation

The full comparison uses the same 200 prompts and 312 constraints for every condition. The
unmodified baseline is the existing matched BF16 run on the identical frozen subset.

| Condition | Strict prompt | Strict instruction | Loose prompt | Loose instruction | Repetitive | Blank / at limit |
|---|---:|---:|---:|---:|---:|---:|
| Matched BF16 base | 69.0% (138/200) | 78.21% (244/312) | 78.0% (156/200) | 85.26% (266/312) | 20/200 | 0 / 2 |
| Alpha-3 patch, top-40k | 63.0% (126/200) | 73.72% (230/312) | 70.5% (141/200) | 80.45% (251/312) | 20/200 | 0 / 0 |
| Alpha-3 patch, top-80k | 63.5% (127/200) | 74.68% (233/312) | 70.0% (140/200) | 79.81% (249/312) | 18/200 | 0 / 2 |
| Alpha-3 patch, top-160k | 64.5% (129/200) | 74.04% (231/312) | 71.0% (142/200) | 79.81% (249/312) | 16/200 | 0 / 1 |
| Alpha-3 patch, top-320k | 66.5% (133/200) | 75.00% (234/312) | 72.0% (144/200) | 80.13% (250/312) | 20/200 | 0 / 3 |

The full set reveals degeneration that the 32-prompt pilot missed. Relative to the matched BF16
base, strict prompt accuracy drops by 6.0, 5.5, 4.5, and 2.5 points at 40k, 80k, 160k, and 320k;
strict instruction accuracy drops by 4.49, 3.53, 4.17, and 3.21 points. The effect is not monotonic:
320k is both safer and better on strict IFEval than 40k--160k, but it remains below the base.

## IFEval batch benchmark

Before the first patched-model IFEval run, real IFEval prompts were benchmarked with the two-model
top-80k patch on physical GPU 1. Batch 16 was the fastest tested configuration and fit comfortably,
so future runs with this BF16 patch setup should reuse batch 16 rather than benchmark again.

| Batch | Examples/s | Peak allocated GPU memory |
|---:|---:|---:|
| 4 | 0.198 | 31.08 GiB |
| 8 | 0.351 | 32.17 GiB |
| 16 | 0.615 | 34.31 GiB |

## Artifacts

- Selection manifest: `/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/selection/circuit_breakers_heldout_seed42_n200_train256_harmbench_disjoint.jsonl`
- Selection metadata: the adjacent `.metadata.json` file
- Ranking: `/workspace/xcy/safety_repro/neurips_neuron/output/change_scores/llama3_instruct_vs_sft_snrawdot256_alpha3_snheldout_seed42_n200_raw_completion.pt`
- Top-8k run: `results/llama3_sft_patch_snraw_alpha3_heldout_top8k_raw/`
- Top-20k run: `results/llama3_sft_patch_snraw_alpha3_heldout_top20k_raw/`
- Top-40k run: `results/llama3_sft_patch_snraw_alpha3_heldout_top40k_raw/`
- Top-80k run: `results/llama3_sft_patch_snraw_alpha3_heldout_top80k_raw/`
- Top-160k run: `results/llama3_sft_patch_snraw_alpha3_heldout_top160k_raw/`
- Top-240k run: `results/llama3_sft_patch_snraw_alpha3_heldout_top240k_raw/`
- Top-320k run: `results/llama3_sft_patch_snraw_alpha3_heldout_top320k_raw/`
- Full-patch run: `results/llama3_sft_patch_snraw_alpha3_heldout_top444416_raw/`
- IFEval batch benchmark: `results/ifeval_benchmark_sft_patch_snraw_alpha3_top80k_gpu1/`
- Full IFEval base: `results/ifeval_llama3_base_bf16/`
- Full IFEval patched runs: `results/ifeval200_sft_patch_snraw_alpha3_top{40,80,160,320}k_gpu1/`
- Initial 32-prompt screening runs: `results/ifeval32_sft_patch_snraw_alpha3_top{20,40,80,160,240,320}k_gpu1/`
