# BBH-200 safety-edit capability evaluation

## What was run

Three Llama-3-8B-Instruct conditions were evaluated on the frozen seed-112,
task-stratified 200-example BBH subset:

1. Unmodified FP32 baseline.
2. Grad on-policy expanded ranking, `K=4000`, strength 1, positive-only,
   final-token scope, FP32.
3. SN-Tune alpha 6 FP32 checkpoint.

All conditions used the official raw three-shot chain-of-thought prompts, greedy
decoding, the lm-evaluation-harness 1,024-token ceiling and stop strings, and its
case-sensitive `the answer is` regex plus exact-match metric. The evaluator was
pinned at commit `8a07e1110d060de48cfc7a9a7987b7659060b60b`. Final runs used only
physical GPU 0.

## Results

| Condition | Correct | Micro accuracy | Task macro | Delta vs. baseline |
|---|---:|---:|---:|---:|
| Baseline | 127/200 | 63.5% | 63.23% | -- |
| Grad on-policy K=4000 | 119/200 | 59.5% | 59.33% | -4.0 pp |
| SN-Tune alpha 6 | 107/200 | 53.5% | 53.24% | -10.0 pp |

Paired example analysis:

| Condition | Baseline errors fixed | Baseline correct broken | Delta 95% paired bootstrap CI | Exact McNemar p |
|---|---:|---:|---:|---:|
| Grad on-policy K=4000 | 16 | 24 | [-10.0, +2.0] pp | 0.2682 |
| SN-Tune alpha 6 | 10 | 30 | [-16.0, -4.0] pp | 0.0022 |

Generation diagnostics:

| Condition | Extraction failures | Repetitive outputs | Token-limit outputs | Mean generated tokens |
|---|---:|---:|---:|---:|
| Baseline | 4 | 26 | 5 | 197.44 |
| Grad on-policy K=4000 | 14 | 32 | 9 | 220.10 |
| SN-Tune alpha 6 | 6 | 46 | 25 | 283.10 |

The Grad point estimate supports a capability trade-off, but its paired confidence
interval includes zero on this 200-example subset. The alpha-6 SN-Tune degradation
is larger and statistically clear under the paired analysis. SN-Tune also produces
substantially more repetitive and token-limit outputs, so its accuracy loss includes
generation degeneration rather than only incorrect completed reasoning.

## Artifacts

- Dataset: `/workspace/xcy/dataset/big_bench_hard/subsets/bbh_seed112_n200.jsonl`
- Baseline: `results/bbh_llama3_base_raw_cot_fp32/`
- Grad: `results/bbh_grad_onpolicy_expanded_k4000_s1_raw_cot_fp32/`
- SN-Tune: `results/bbh_sn_alpha6_raw_cot_fp32/`

Each result directory contains the semantic/runtime configuration, validation data,
raw and scored per-example responses, aggregate summary, and checksums.
