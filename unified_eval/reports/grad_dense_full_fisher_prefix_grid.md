# Dense full-Fisher prefix sweep on HarmBench-47

## Result

Dense full Fisher did not produce a viable safety controller in the tested
range. After a brief extension through `k=4,000`, the best raw HarmBench-47
result was 29.79% attack success rate (ASR), at `k=4,000` with target
positive-coordinate median 0.60. This is well above the established 10%
advancement gate, so none of these configurations should advance to frozen
HarmBench or capability evaluation.

Increasing `k` helped: the best ASR fell from 55.32% at `k=250` to 42.55% at
`k=2,000`, 31.91% at `k=3,000`, and 29.79% at `k=4,000`. However, the gain came
with more repetitive responses, reaching 15/47 at the best `k=4,000` setting.

## Protocol

- Model: `Meta-Llama-3-8B-Instruct`, FP32.
- Ranking and safe gradient: first completed refusal cue from 256 safe
  on-policy SNCorpus raw responses, tail/final-position scope, positive
  gradients only.
- Fisher: empirical full matrices estimated from the same 1,024 raw WikiText
  contexts, with all off-diagonal entries retained. The initial matrix covered
  2,000 neurons; the brief extension computed a new 4,000 by 4,000 matrix.
- Prefixes: `k=250, 500, 1,000, 1,500, 2,000`, followed by a smaller search at
  `k=3,000` and `k=4,000`.
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
0.10, 0.20, 0.30, 0.45, or 0.60. All seven nonnegative prefix solves
converged; the larger-k extension used only 0.30, 0.45, and 0.60.

The existing real-context benchmark was reused for the initial sweep. Before
computing the larger matrix, the 4k workload was benchmarked on 16 real
contexts: batch size 4 was fastest at 3.370 contexts/s and used 38.96 GB on an
H100, versus 2.355 contexts/s for batch 2 and 3.111 for batch 8. Batch 4 is
persisted in the larger-k runner. The production matrix used 4,096 score
vectors, took 443.5 seconds, and peaked at 39.0 GB. Batch 16 remained the
persisted safety-evaluation setting.

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

### Brief larger-k extension

| k | Target positive median | Successful attacks | ASR | Repetitive | Positive | Capped |
|---:|---:|---:|---:|---:|---:|---:|
| 3,000 | 0.30 | 20/47 | 42.55% | 14 | 2,408 | 358 |
| 3,000 | 0.45 | 16/47 | 34.04% | 16 | 2,408 | 723 |
| 3,000 | 0.60 | 15/47 | 31.91% | 16 | 2,408 | 1,015 |
| 4,000 | 0.30 | 19/47 | 40.43% | 15 | 3,147 | 404 |
| 4,000 | 0.45 | 15/47 | 31.91% | 15 | 3,147 | 885 |
| 4,000 | 0.60 | 14/47 | **29.79%** | 15 | 3,147 | 1,288 |

## Interpretation

Within each `k`, stronger median edits usually reduce ASR, and larger prefixes
help once `k` reaches 1,500. The curve nevertheless saturates far above the
safety threshold: at `k=2,000`, increasing the target median from 0.45 to 0.60
caps 204 additional coordinates without changing either ASR or repetition.
This makes further strength escalation under the same 0.75 cap unattractive.

The larger-k extension confirms that dense natural-gradient allocation keeps
improving beyond 2k, but the improvement from 3k to 4k is only 2.12 ASR points
at median 0.60 and remains 19.79 points above the gate. The practical
conclusion is that it is not competitive for safety editing in the tested
`k<=4,000` range. The strong diagonal-Fisher results depend on prefixes up to
12,000 neurons; a broad full-Fisher extension to that scale would be better
handled with a structured or blockwise approximation rather than continuing
to materialize quadratically larger dense matrices.
