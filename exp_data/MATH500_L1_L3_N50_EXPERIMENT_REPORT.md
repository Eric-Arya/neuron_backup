# MATH-500 L1-L3 50-example evaluation

## What was run

A frozen seed-112 sample of 50 MATH-500 problems was drawn only from levels
1, 2, and 3. Largest-remainder allocation preserves their relative frequency
within the original 238-example L1-L3 pool:

| Level | Original L1-L3 pool | Original ratio | Subset | Subset ratio |
|---|---:|---:|---:|---:|
| 1 | 43 | 18.07% | 9 | 18% |
| 2 | 90 | 37.82% | 19 | 38% |
| 3 | 105 | 44.12% | 22 | 44% |

Four Llama-3-8B-Instruct FP32 conditions were evaluated:

1. Unmodified baseline.
2. Grad on-policy expanded ranking, K=4000, strength 1, positive-only,
   final-token scope.
3. SN-Tune alpha 6.
4. Raw-format IA3-SFT alpha 3, using
   `gate = 1 + 3 * (trained_gate - 1)`.

All conditions used the publisher zero-shot chain-of-thought instruction in
the model's native chat template, greedy decoding, a 1,024 generated-token
limit, and deterministic `math_verify` symbolic equivalence. Batch size 16
was selected by the existing real-prompt MATH-500 benchmark. Every model run
used only physical GPU 0.

## Results

| Condition | Correct | Accuracy | Delta vs. baseline |
|---|---:|---:|---:|
| Baseline | 24/50 | 48% | -- |
| Grad on-policy K=4000 | 18/50 | 36% | -12 pp |
| SN-Tune alpha 6 | 18/50 | 36% | -12 pp |
| IA3-SFT alpha 3 | 23/50 | 46% | -2 pp |

Accuracy by difficulty:

| Condition | Level 1 (n=9) | Level 2 (n=19) | Level 3 (n=22) |
|---|---:|---:|---:|
| Baseline | 77.78% | 36.84% | 45.45% |
| Grad on-policy K=4000 | 55.56% | 36.84% | 27.27% |
| SN-Tune alpha 6 | 55.56% | 36.84% | 27.27% |
| IA3-SFT alpha 3 | 77.78% | 47.37% | 31.82% |

Paired comparison against baseline:

| Condition | Baseline errors fixed | Baseline correct broken | Paired delta 95% bootstrap CI | Exact McNemar p |
|---|---:|---:|---:|---:|
| Grad on-policy K=4000 | 5 | 11 | [-28, +4] pp | 0.2101 |
| SN-Tune alpha 6 | 5 | 11 | [-28, +4] pp | 0.2101 |
| IA3-SFT alpha 3 | 5 | 6 | [-14, +10] pp | 1.0000 |

Generation diagnostics:

| Condition | Extraction failures | Scoring errors | Repetitive outputs | Token-limit outputs | Mean generated tokens |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | 0 | 10 | 3 | 307.48 |
| Grad on-policy K=4000 | 1 | 0 | 12 | 4 | 316.18 |
| SN-Tune alpha 6 | 0 | 0 | 9 | 2 | 278.42 |
| IA3-SFT alpha 3 | 0 | 0 | 10 | 3 | 281.32 |

Grad and SN-Tune have the same aggregate and per-level accuracy drop.
The point estimates support a safety-capability trade-off, but the paired
confidence intervals include zero on this 50-example subset. IA3-SFT alpha 3
is much closer to baseline at -2 points, with a paired interval that also
includes zero. All 200 stored generations were independently rescored
successfully, with identical example IDs and ordering across conditions.

## Artifacts

- Subset: `/workspace/xcy/dataset/math500/subsets/math500_l1_l3_seed112_n50`
- Baseline: `results/math500_l1_l3_n50_llama3_base_fp32/`
- Grad: `results/math500_l1_l3_n50_grad_onpolicy_expanded_k4000_s1_fp32/`
- SN-Tune: `results/math500_l1_l3_n50_sn_alpha6_fp32/`
- IA3-SFT alpha 3: `results/math500_l1_l3_n50_ia3_sft_snraw_alpha3_fp32/`
