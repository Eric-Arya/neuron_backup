# Dense full-Fisher prefix sweep on HarmBench-47

## Result

Dense full Fisher did not produce a viable safety controller in the tested
range. The best raw HarmBench-47 result was 42.55% attack success rate (ASR),
at `k=2,000` with target positive-coordinate median 0.45 or 0.60. This is well
above the established 10% advancement gate, so none of these configurations
should advance to frozen HarmBench or capability evaluation.

Increasing `k` helped: the best ASR fell from 55.32% at `k=250` to 42.55% at
`k=2,000`. However, the gain came with more repetitive responses, reaching
14/47 for both best `k=2,000` settings. The saved dense Fisher artifact covers
only the first 2,000 ranked neurons; larger `k` would require computing a new
quadratically larger matrix.

## Protocol

- Model: `Meta-Llama-3-8B-Instruct`, FP32.
- Ranking and safe gradient: first completed refusal cue from 256 safe
  on-policy SNCorpus raw responses, tail/final-position scope, positive
  gradients only.
- Fisher: saved empirical 2,000 by 2,000 full matrix estimated from 1,024 raw
  WikiText contexts, with all off-diagonal entries retained.
- Prefixes: `k=250, 500, 1,000, 1,500, 2,000`.
- Safety screen: the raw 47-example HarmBench tuning subset, greedy decoding,
  128 generated tokens, batch size 16.
- Edit constraints: floor 0 and maximum delta 0.75.
- No BeaverTrail, Beaver score, GSM8K, MMLU, frozen HarmBench, or capability
  evaluation was run.

For each principal prefix, the controller solves

\[
v_k=\arg\min_{v\geq 0}\left(\frac12v^\top A_kv-g_k^\top v\right),
\]

where

\[
A_k=0.5F_k+0.5\operatorname{diag}(F_k)+\lambda I,
\qquad
\lambda=0.01\,\operatorname{median}_{j:F_{jj}>0}F_{jj}.
\]

Thus each solve uses the whole dense principal Fisher matrix `F_k`, including
cross-neuron curvature. The final edit is

\[
\Delta_j=\min(0.75,a v_{k,j}),
\]

with `a` chosen so the uncapped positive-coordinate median targets one of
0.10, 0.20, 0.30, 0.45, or 0.60. All five nonnegative solves converged.

The existing real-context benchmark was reused rather than repeated. For full
Fisher construction, batch size 4 was fastest at 3.108 contexts/s and used
39.3 GB on an H100; batch 16 remained the persisted safety-evaluation setting.

## HarmBench-47 sweep

`Positive` is the number of nonzero coordinates after the constrained solve;
`Capped` is the number whose delta reached 0.75.

| k | Target positive median | Successful attacks | ASR | Repetitive | Positive | Capped |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.10 | 31/47 | 65.96% | 9 | 222 | 5 |
| 250 | 0.20 | 29/47 | 61.70% | 7 | 222 | 15 |
| 250 | 0.30 | 27/47 | 57.45% | 5 | 222 | 42 |
| 250 | 0.45 | 27/47 | 57.45% | 6 | 222 | 75 |
| 250 | 0.60 | 26/47 | 55.32% | 8 | 222 | 94 |
| 500 | 0.10 | 30/47 | 63.83% | 9 | 437 | 9 |
| 500 | 0.20 | 27/47 | 57.45% | 9 | 437 | 27 |
| 500 | 0.30 | 27/47 | 57.45% | 8 | 437 | 57 |
| 500 | 0.45 | 25/47 | 53.19% | 10 | 437 | 132 |
| 500 | 0.60 | 25/47 | 53.19% | 9 | 437 | 184 |
| 1,000 | 0.10 | 29/47 | 61.70% | 8 | 826 | 18 |
| 1,000 | 0.20 | 27/47 | 57.45% | 5 | 826 | 47 |
| 1,000 | 0.30 | 27/47 | 57.45% | 8 | 826 | 125 |
| 1,000 | 0.45 | 25/47 | 53.19% | 10 | 826 | 261 |
| 1,000 | 0.60 | 25/47 | 53.19% | 13 | 826 | 345 |
| 1,500 | 0.10 | 26/47 | 55.32% | 9 | 1,214 | 21 |
| 1,500 | 0.20 | 26/47 | 55.32% | 10 | 1,214 | 72 |
| 1,500 | 0.30 | 23/47 | 48.94% | 14 | 1,214 | 185 |
| 1,500 | 0.45 | 22/47 | 46.81% | 14 | 1,214 | 356 |
| 1,500 | 0.60 | 21/47 | 44.68% | 14 | 1,214 | 501 |
| 2,000 | 0.10 | 26/47 | 55.32% | 9 | 1,598 | 23 |
| 2,000 | 0.20 | 25/47 | 53.19% | 12 | 1,598 | 82 |
| 2,000 | 0.30 | 22/47 | 46.81% | 13 | 1,598 | 225 |
| 2,000 | 0.45 | 20/47 | 42.55% | 14 | 1,598 | 455 |
| 2,000 | 0.60 | 20/47 | 42.55% | 14 | 1,598 | 659 |

## Interpretation

Within each `k`, stronger median edits usually reduce ASR, and larger prefixes
help once `k` reaches 1,500. The curve nevertheless saturates far above the
safety threshold: at `k=2,000`, increasing the target median from 0.45 to 0.60
caps 204 additional coordinates without changing either ASR or repetition.
This makes further strength escalation under the same 0.75 cap unattractive.

The practical conclusion is that the tested dense natural-gradient allocation
is not competitive for safety editing at the available `k<=2,000`. The strong
diagonal-Fisher results depend on much larger prefixes, up to 12,000 neurons;
testing whether full Fisher eventually benefits from the same scale would
require a substantially more memory-efficient structured or blockwise
approximation rather than materializing a 12,000 by 12,000 dense matrix.
