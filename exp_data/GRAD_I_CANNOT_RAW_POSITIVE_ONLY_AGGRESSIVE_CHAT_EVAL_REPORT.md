# Aggressive raw `I cannot` Grad, positive-only experiment

Updated: 2026-08-18

## Outcome

Starting from the raw-extracted `I cannot` Grad ranking, this experiment strengthens only neurons
with a positive mean gradient. Negative-gradient neurons are never weakened. All interventions are
applied at every token position using native Llama-3 chat evaluation.

The maximum intervention strengthens all 237 positive neurons in the saved 500-neuron ranking by
5x (`strength=4`). It produces the requested ASR--capability tradeoff: HarmBench refusal-substring
ASR falls from 29.5% to 6.0%, while GSM8K falls from 77% to 56% and MMLU falls from 70.18% to
65.96%.

| Frozen condition | HarmBench ASR ↓ | Beaver cost ↓ | BeaverTails cost ↓ / reward ↑ | GSM8K-100 ↑ | MMLU-285 ↑ | Repetitive HarmBench |
|---|---:|---:|---:|---:|---:|---:|
| Prior raw Grad, signed K=25, strength 0.5 | 29.5% (59/200) | -5.183 | **-10.119 / -2.981** | 77% | 70.18% (200/285) | 11/200 |
| Positive-only K=200, strength 2 (3x) | 24.5% (49/200) | -4.028 | **-9.931 / -3.371** | 76% | 68.07% (194/285) | 14/200 |
| Positive-only K=237, strength 3 (4x) | 14.5% (29/200) | -5.073 | **-10.254 / 1.839** | 75% | 67.02% (191/285) | 118/200 |
| **Positive-only K=237, strength 4 (5x)** | **6.0% (12/200)** | **-4.080** | **-7.315 / 0.585** | **56%** | **65.96% (188/285)** | **179/200** |

Relative to the prior signed K=25 controller, the maximum positive-only intervention reduces ASR
by 23.5 percentage points, loses 21 GSM8K points, and loses 12 MMLU questions (4.21 percentage
points). This is strong evidence of an intervention-strength/capability tradeoff under the
refusal-substring ASR metric.

It is not a clean safety improvement across metrics. Mean Beaver cost becomes 1.103 points higher
(less safe on that scale), and 179/200 HarmBench responses trigger the four-word repetition
diagnostic. A typical failure repeats an `I cannot ...` sentence until the token limit. Thus the
6% ASR primarily demonstrates aggressive refusal-direction amplification and degeneration, not a
general improvement in response quality or safety.

The added BeaverTails column reports `cost / reward` on the exact frozen 200-prompt, seed-42
`round0/330k/test` sample. All four conditions use native Llama-3 chat, greedy decoding, and a
128-token ceiling; both score models consume the full decoded prompt plus completion. Lower cost
is safer and higher reward is more helpful. Physical GPU 0 was used for generation and for both
score models.

BeaverTails reinforces the degeneration caveat. The non-degenerate signed K=25 and positive-only
K=200 conditions have 0/200 and 1/200 repetitive BeaverTails responses, respectively, and similar
cost/reward scores. K=237/strength-3 reaches the lowest safety cost (-10.254) and a positive reward,
but 123/200 responses are repetitive. K=237/strength-4 worsens safety cost to -7.315 and has
174/200 repetitive responses. Its positive reward therefore does not demonstrate improved
helpfulness: common failures repeat `I cannot provide` until the token limit, which the scalar
reward model does not reliably penalize.

## Development sweep

The disjoint 47-prompt HarmBench development split and GSM8K-20 capability check were used before
the frozen test. Selected frontier points were:

| Positive neurons | Strength (multiplier) | Development ASR ↓ | GSM8K-20 ↑ | Repetitive |
|---:|---:|---:|---:|---:|
| 0 | 0 (1x) | 34.0% | 60% | -- |
| 200 | 1 (2x) | 25.5% | 65% | 2/47 |
| 200 | 2 (3x) | 23.4% | 70% | 5/47 |
| 200 | 4 (5x) | 14.9% | 30% | 39/47 |
| 237 | 3 (4x) | 21.3% | 65% | 27/47 |
| **237** | **4 (5x)** | **10.6%** | **35%** | **39/47** |

The full search covered K=25/50/100/200 at strengths 0.5/1/2/4, then K=200/237 at strengths
2.5/3/3.5, followed by the maximum K=237/strength-4 point. Physical GPU 1 was used throughout.

## Implementation and artifacts

The evaluator now accepts `--grad-direction positive-only`. In this mode K counts positive-gradient
neurons after filtering the stable ranking; selected neurons receive multiplier `1 + strength`,
and all negative-gradient neurons retain multiplier 1. The development sweeper has the matching
`--direction positive-only` option. All 30 evaluator tests pass, including a mask-level test that
negative-gradient neurons remain unchanged.

- Ranking: `results/grad_i_cannot_raw_extract_chat_patch/corpus_gradients/top_neurons_stable.csv`
- Coarse sweep: `results/grad_i_cannot_sn_corpus_chat/raw_positive_only_aggressive_all_chat/`
- Frontier sweep: `results/grad_i_cannot_sn_corpus_chat/raw_positive_only_frontier_all_chat/`
- Maximum-point development check: `results/grad_i_cannot_sn_corpus_chat/raw_positive_only_maximum_all_chat/`
- K=200, strength-2 frozen result: `results/grad_i_cannot_raw_positive_only_all_k200_s2/`
- K=237, strength-3 frozen result: `results/grad_i_cannot_raw_positive_only_all_k237_s3/`
- K=237, strength-4 frozen result: `results/grad_i_cannot_raw_positive_only_all_k237_s4/`
- Signed K=25 BeaverTails result: `results/grad_i_cannot_rawextract_chat_all_k25_s0p5_beavertails/`
- K=200, strength-2 BeaverTails result: `results/grad_i_cannot_raw_positive_only_all_k200_s2_beavertails/`
- K=237, strength-3 BeaverTails result: `results/grad_i_cannot_raw_positive_only_all_k237_s3_beavertails/`
- K=237, strength-4 BeaverTails result: `results/grad_i_cannot_raw_positive_only_all_k237_s4_beavertails/`
