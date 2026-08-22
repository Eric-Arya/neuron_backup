# Zero-floor diagonal-Fisher K grid and KL-matched direct controls

## Summary

This experiment expanded the first-cue-256 Grad comparison across Fisher
controller sizes. Every Fisher controller used

\[
\delta_j=\min\left(0.75,\;c\frac{\max(g_j,0)}{F_{jj}+\lambda}\right),
\qquad \lambda=\operatorname{median}(F_{jj}>0),
\]

with floor zero, positive-only direction, and final-position intervention.
The plotted capability trajectory contains every tested size:
`K=1k, 2k, 4k, 6k, 8k, 12k, 16k`. A stronger `K=12k, c=0.48`
point is also plotted as a separate fixed-K branch.

The main result is that the larger zero-floor Fisher controllers reach about
7.5% frozen HarmBench ASR while retaining 67.0% strict IFEval, 62.5--63.5%
BBH, and 46--47% MATH100. Uniform direct controls with the same
diagonal-Fisher quadratic budget all failed the HarmBench47 advancement gate,
so none received frozen capability evaluation.

## Protocol

- Model: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct`.
- Ranking: first-cue-256 positive-gradient ranking; no refusal-sentence Grad
  results are used or plotted.
- Fisher: 2,048 WikiText contexts, raw 128-token contexts, top-16k diagonal.
- Safety selection: raw HarmBench47 development subset.
- Frozen safety: raw HarmBench200.
- Capability: IFEval200, official raw three-shot CoT BBH200, and the frozen
  seed-112 MATH100 subset containing 18/38/44 level-1/2/3 problems.
- Precision: FP32. Persisted H100 batch sizes were reused: HarmBench 16,
  IFEval 8, BBH 8, and MATH100 16.
- Advancement rule for the KL-matched direct controls: HarmBench47 ASR below
  10%. No control passed, so no direct-control frozen or capability run was
  launched.

## Fisher K grid

The per-K controller was chosen on HarmBench47 by taking the smallest tested
`c` below 10% ASR; when no value passed, the lowest-ASR value was retained to
show the trajectory. Ties favored fewer repetitive responses and then smaller
`c`.

| K | c | HB47 ASR | Frozen HB200 ASR | IFEval strict | BBH | MATH100 |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 0.64 | 40.43% | 38.5% | 67.5% | 63.5% | 49.0% |
| 2k | 0.64 | 25.53% | 21.0% | 66.5% | 61.5% | 48.0% |
| 4k | 0.40 | 17.02% | 15.0% | 65.5% | 62.0% | 46.0% |
| 6k | 0.52 | 10.64% | 7.5% | 66.0% | 61.0% | 47.0% |
| 8k | 0.48 | 6.38% | 7.0% | 63.5% | 63.5% | 45.0% |
| 12k | 0.22 | 8.51% | 7.5% | 67.0% | 62.5% | 46.0% |
| 12k | 0.48 | 0.00% | 1.5% | 64.0% | 65.0% | 49.0% |
| 16k | 0.18 | 6.38% | 7.5% | 67.0% | 63.5% | 47.0% |

The trajectory is not monotone in capability cost. The 8k point dips to 63.5%
IFEval and 45% MATH100, while the gentler 12k and 16k controllers recover to
67.0% IFEval. The stronger 12k `c=0.48` controller reaches 1.5% frozen ASR and
improves BBH/MATH100 to 65%/49%, but lowers IFEval to 64%. The 12k `c=0.22`
and 16k points coincide in the IFEval--HarmBench panel at `(67.0, 7.5)` but
remain separately identified as F6 and F8.

## KL-matched direct gate

For a uniform direct edit `delta_j=s`, the strength was chosen to match each
Fisher controller's raw diagonal-Fisher quadratic cost:

\[
s_K=\sqrt{\frac{\sum_{j=1}^{K}F_{jj}\delta_j^2}
                  {\sum_{j=1}^{K}F_{jj}}}.
\]

This is an exact match under the saved diagonal approximation, not a claim of
exact finite-radius model KL equality. All five controls were screened together
with one model load on raw HarmBench47.

| K | Fisher c | Direct s | Fisher HB47 ASR | Matched-direct HB47 ASR | Advance? |
|---:|---:|---:|---:|---:|:---:|
| 1k | 0.64 | 0.395116 | 40.43% | 51.06% | No |
| 2k | 0.64 | 0.425380 | 25.53% | 40.43% | No |
| 4k | 0.40 | 0.355565 | 17.02% | 34.04% | No |
| 12k | 0.22 | 0.226653 | 8.51% | 27.66% | No |
| 16k | 0.18 | 0.175207 | 6.38% | 31.91% | No |

At every tested K, Fisher allocation produced lower development ASR than the
uniform direct edit at the same predicted diagonal-KL budget. The gap is
especially large at 12k and 16k. Because all direct controls missed the strict
10% gate, advancing them would spend frozen evaluations on points that are not
competitive on safety.

## Figures

The main figures use compact point IDs rather than configuration annotations.
All plotted direct Grad points use first-cue-256.

- `figures/bbh_harmbench_tradeoff_main_only.{png,pdf,svg}`
- `figures/math500_harmbench_tradeoff_main_only.{png,pdf,svg}`
- `figures/ifeval_harmbench_tradeoff_main_only.{png,pdf,svg}`

The Fisher IDs are F1--F8 for `(K,c)=(1k,.64),(2k,.64),(4k,.40),
(6k,.52),(8k,.48),(12k,.22),(12k,.48),(16k,.18)`.

## Artifacts

- Fisher grid and evaluations:
  `results/grad_fisher_zero_floor_all_k_cap0p75/`
- KL-matched direct scales and HarmBench47 screen:
  `results/grad_direct_klmatched_fisher_cap0p75/`
- Fisher grid runner: `scripts/run_fisher_zero_floor_all_k_grid.sh`
- Gated direct runner: `scripts/run_kl_matched_direct_grid.sh`
- Figure generator: `scripts/plot_main_tradeoffs_with_fisher.py`
