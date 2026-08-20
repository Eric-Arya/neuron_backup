# Fixed-neuron safety–capability tradeoff

## Design

- Model: Meta-Llama-3-8B-Instruct, greedy decoding, raw chat format.
- Controller: the same fixed global top-25 neurons for every prompt. They were selected on AdvBench HarmBehavior rows 100–299, disjoint from the safety evaluation rows 0–99.
- Safety: first 100 HarmBehavior examples, 512 generated tokens.
- Capability: first 100 GSM8K test examples using the ICLR-reproduction zero-shot prompt, 256 generated tokens.
- Every condition uses the same examples, enabling paired tests.
- Strength `s` applies `positive scale = 1+s` and `negative scale = max(0,1-s)`. Negative neurons are never sign-reversed.

## Results

| `s` | Positive scale | Negative scale | HarmBehavior ASR ↓ | Δ ASR | GSM8K accuracy ↑ | Δ accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 1.00 | 34% | — | 68% | — |
| 0.25 | 1.25 | 0.75 | 32% | −2 pp | 65% | −3 pp |
| 0.50 | 1.50 | 0.50 | 30% | −4 pp | 67% | −1 pp |
| 0.75 | 1.75 | 0.25 | 22% | −12 pp | 64% | −4 pp |
| 1.00 | 2.00 | 0.00 | **19%** | **−15 pp** | 62% | −6 pp |
| 1.50 | 2.50 | 0.00 | 23% | −11 pp | 63% | −5 pp |
| 2.00 | 3.00 | 0.00 | 20% | −14 pp | **59%** | **−9 pp** |

The controller shows a clear empirical dose-level tradeoff:

- Safety has a significant paired linear trend: −7.14 ASR percentage points per unit `s` (50,000 within-example permutations, two-sided `p < 0.0001`). Differences across all strengths are significant by Cochran's Q (`p = 6.96e-5`).
- GSM8K has a significant paired degradation trend: −3.94 accuracy points per unit `s` (`p = 0.0337`). The omnibus, non-ordered Cochran's Q test is not significant (`p = 0.476`), showing that the evidence is specifically in the ordered trend and is weaker than the safety effect.
- Across the seven dose levels, safety gain and capability loss have Spearman `rho = 0.893` (exact dose-label permutation `p = 0.0123`). This is descriptive because dose levels are not independent samples.
- After Holm correction across the six comparisons, the ASR reductions at `s=1` (`p=0.0245`) and `s=2` (`p=0.0331`) remain significant. No individual GSM8K condition remains significant after correction; at `s=2`, the paired bootstrap 95% interval for the −9 point change is [−19, +1].
- Capability repetition rises from 3/100 at baseline to 8–15/100 under intervention. Thus, degradation is visible in generation quality as well as exact-match accuracy.

## Interpretation

This is strong empirical evidence for a safety–capability tradeoff on these matched 100-example evaluations, but not a universal proof. The best observed safety point is `s=1` (`positive ×2`, `negative ×0`), not the strongest intervention. Increasing positive neurons further gives no reliable safety improvement and eventually lowers GSM8K to 59%, so `s=1` remains the sensible operating point among those tested.

## Artifacts

- `results/tradeoff/fixed_top25_strength_sweep/harmbehavior_first100/`
- `results/tradeoff/fixed_top25_strength_sweep/gsm8k_first100/`
- `results/tradeoff/fixed_top25_strength_sweep/analysis/tradeoff_summary.csv`
- `results/tradeoff/fixed_top25_strength_sweep/analysis/paired_tests.csv`
- `results/tradeoff/fixed_top25_strength_sweep/analysis/analysis.json`
- `results/tradeoff/fixed_top25_strength_sweep/analysis/tradeoff_curve.png`

Re-run the statistical analysis with:

```bash
python scripts/analyze_safety_capability_tradeoff.py
```
