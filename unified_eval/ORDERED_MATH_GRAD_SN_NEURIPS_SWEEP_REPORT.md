# Ordered MATH, Grad, SN, and NeurIPS multiplier sweeps

Updated: 2026-08-20

## What was done

The four requested stages were run in order, using both H100s whenever two
independent runs were available. Safety selection and HarmBench evaluation used
raw prompts. No BeaverTails, Beaver score, GSM8K, or MMLU evaluation was run.

1. Completed MATH-500 L1-L3 n=100 for the 12 points missing from the main-only
   IFEval–HarmBench figure.
2. Screened lower untried strengths for positive-only, final-token on-policy
   Grad at K=4000, then promoted the useful candidates to full IFEval-200.
3. Added attenuation support to the direct SN controller and tested 0.95x,
   0.9x, and 0.8x. Existing results already covered 1.05x, 1.1x, and 1.2x.
4. Added direct all-token scaling for the NeurIPS top-20k post-MLP candidate
   pool and screened 0.6x, 0.8x, 1.0x, 1.2x, and 1.4x.

## 1. Complete MATH coverage

All 18 points in the main-only figure now have matched MATH-500 L1-L3 n=100
results. The 12 newly run accuracies were:

| Family | Setting: MATH accuracy |
|---|---|
| SN-Tune | alpha 1: 53%; alpha 4: 45%; alpha 8: 46% |
| IA3-SFT | alpha 1: 50%; 1.5: 45%; 2: 43%; 2.5: 43%; 3.5: 48% |
| IA3 guide patch | K=40k: 43%; 80k: 50%; 160k: 44%; 320k: 47% |

Together with the existing baseline (48%), SN alpha 6 (46%), IA3-SFT alpha 3
(45%), and Grad K=1k/2k/4k (47%/45%/41%), the clearest dose-ordered math loss
is still on-policy Grad. The other trajectories fluctuate and should not be
interpreted as monotone effects from this n=100 screening set.

The guide-patch model had not previously been benchmarked on MATH. A real-data
8/16/32 batch test selected batch 32 (16.98 examples/s, 44.87 GiB peak); that
default is now encoded for future guide-patch MATH runs.

## 2. Lower-strength on-policy Grad at K=4000

The IFEval-32 screen tested s=0.25, 0.4, 0.5, 0.6, 0.75, and 0.85. The five
new nontrivial points were then evaluated on raw HarmBench-200; all except the
already clearly weak s=0.25 point were promoted to full IFEval-200.

| Strength | HarmBench ASR ↓ | IFEval strict prompt ↑ | Strict instruction ↑ | Repetitive HB / IFEval |
|---:|---:|---:|---:|---:|
| 0.4 | 22.0% | 67.0% | 77.24% | 78 / 20 |
| 0.5 | 18.5% | 66.0% | 75.96% | 82 / 18 |
| **0.6** | **15.0%** | **69.5%** | **78.21%** | 90 / 26 |
| **0.75** | **8.5%** | **66.5%** | **75.64%** | 91 / 24 |
| **0.85** | **6.0%** | **62.5%** | **73.40%** | 89 / 31 |
| 1.0 (existing) | 5.5% | 61.0% | 71.79% | -- |

This found a better Pareto ladder. In particular, s=0.6 strictly dominates the
old on-policy K=1000,s=1 point (15.0% vs. 17.0% ASR and 69.5% vs. 67.5%
IFEval), while s=0.75 strictly dominates K=2000,s=1 (8.5% vs. 9.5% ASR and
66.5% vs. 65.5% IFEval). Strengths 0.4 and 0.5 are dominated by s=0.6.
Repetition remains substantial, so the low-ASR points are not cleanly free of
generation degeneration.

## 3. Direct SN multipliers

The matched BF16 reference is 66.0% raw HarmBench ASR and 69.0% full IFEval
strict-prompt accuracy. Positive multipliers had already been tested: 1.05x,
1.1x, and 1.2x produced 70.0%, 73.0%, and 86.5% ASR, respectively, so the new
work focused on attenuation.

| Multiplier | HarmBench ASR ↓ | IFEval strict prompt ↑ | Evaluation scope |
|---:|---:|---:|---|
| **0.95x** | **63.5%** | **64.5%** | HB-200 + IFEval-200 |
| 0.9x | 64.5% | 56.5% | HB-200 + IFEval-200 |
| 0.8x | 79.5% | 25.0% | HB-200 + IFEval-32 screen |

Only 0.95x is potentially useful: it gains 2.5 points of ASR safety at a 4.5
point IFEval cost. The 0.9x point is dominated by 0.95x, and 0.8x degrades both
metrics. This is a weak trade-off, not a competitive replacement for trained
SN-Tune or the Grad sweep.

The first SN-direct IFEval batch benchmark tested 8/16/32 and selected batch 32
(59.06 examples/s at the 16-token benchmark length, 15.73 GiB peak). This
method-specific IFEval default is now encoded.

## 4. Direct NeurIPS candidate-pool multipliers

| Multiplier | Raw HarmBench ASR ↓ | IFEval-32 strict prompt ↑ | HB repetitive /200 | IFEval repetitive / limit |
|---:|---:|---:|---:|---:|
| 0.6x | 98.5% | 12.5% | 107 | 32 / 32 |
| 0.8x | 98.5% | 15.625% | 103 | 32 / 32 |
| 1.0x control | 98.5% | 15.625% | 95 | 32 / 32 |
| 1.2x | 98.0% | 12.5% | 107 | 32 / 32 |
| 1.4x | 98.5% | 15.625% | 99 | 32 / 32 |

No multiplier produces a useful trade-off. Every IFEval generation reaches
the 1,024-token limit and is repetitive, while raw HarmBench ASR stays near
98%. The unmodified Llama-2 base control is itself degenerate under this raw
diagnostic, so these results do not justify a full IFEval-200 promotion.

The real-prompt HarmBench benchmark selected batch 32 (36.87 examples/s,
28.55 GiB peak). A short IFEval benchmark also selected 32, but full-length
generation OOMed at roughly 78.8 GiB. Corrected 1,024-token benchmarks measured
batch 8 at 0.212 examples/s and 42.76 GiB and batch 16 at 0.302 examples/s and
60.76 GiB; batch 16 is now the encoded NeurIPS-direct IFEval default.

## Implementation and artifacts

- Direct-controller implementation: `unified_eval/methods.py`
- CLI, semantic fingerprints, and benchmarked defaults: `unified_eval/runner.py`
- Consolidated trade-off data: `results/ifeval_harmbench_tradeoff.csv`
- Updated IFEval figure: `figures/ifeval_harmbench_tradeoff_main_only.png`
- New MATH figure: `figures/math500_harmbench_tradeoff_main_only.png`
- Detailed MATH report: `MATH500_L1_L3_N100_EXPERIMENT_REPORT.md`
- Grad full runs: `results/harmbench_grad_onpolicy_expanded_k4000_s*_fp32/`
  and `results/ifeval200_grad_onpolicy_expanded_k4000_s*_fp32/`
- SN direct runs: `results/sn_direct_raw_cap25_m*_harmbench_ifeval32_bf16/`
  and `results/sn_direct_raw_cap25_m*_ifeval200_bf16/`
- NeurIPS direct screen:
  `results/neurips_direct_top20k_m*_raw_harmbench_ifeval32_fp32/`
