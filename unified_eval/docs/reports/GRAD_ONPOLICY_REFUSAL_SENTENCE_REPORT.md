# On-policy refusal-sentence Grad: positive-only K=1000

Updated: 2026-08-19

## Method

The extraction source is the raw-format Llama-3 generation over the 197-prompt,
deduplicated complement of the frozen HarmBench-200 test manifest. Of 65 responses
classified safe by the refusal-substring rule, 54 contain the exact phrase `I cannot`.
All 54 were used, without reserving the prior tuning split.

For each response, the raw prompt plus all generated text before the first `I cannot`
is the prefill. The target is the complete first sentence beginning with that occurrence
of `I cannot`. The extraction objective is the mean teacher-forced target-token log
probability. Neuron-scale gradients are taken at the final prefill position and the
target positions, matching the established final-position Grad definition.

The resulting gradient tensor has shape `54 x 32 x 14336`. Refusal targets contain
6--29 tokens (mean 16.07). Neurons were stability-ranked inside the 4,000 largest
absolute-mean-gradient candidates. The safety controller selects the first 1,000
positive-gradient neurons from that ranking, doubles their activations, and leaves all
negative-gradient neurons unchanged (`--grad-direction positive-only`, strength 1,
scope `last`). The 1,000th selected positive neuron is rank 1,864 in the unfiltered
stability ranking.

## Frozen safety result

Safety evaluation uses the frozen raw HarmBench-200 manifest, greedy decoding, a
128-token limit, refusal-substring ASR, and Beaver-7B cost over prompt plus completion.
| Condition | HarmBench ASR lower is better | Mean Beaver cost lower is better | Repetitive responses |
|---|---:|---:|---:|
| Llama-3 baseline | 65.5% (131/200) | 0.594 | 26/200 |
| On-policy refusal Grad, positive-only K=1000, strength 1 | **20.0% (40/200)** | **-6.794** | 63/200 |
| On-policy refusal Grad, positive-only K=1000, strength 1.5 | **5.5% (11/200)** | **-8.643** | 50/200 |

Relative to baseline there are 93 unsafe-to-safe refusal-substring changes and two
safe-to-unsafe changes. ASR decreases by 45.5 percentage points and mean cost decreases
by 7.388. The edited model begins 116/200 responses with normalized `I cannot` and
contains `I cannot` somewhere in 158/200 responses.

At strength 1.5, relative to baseline there are 120 unsafe-to-safe changes and zero
safe-to-unsafe changes. Relative to strength 1, another 29 unsafe responses become
safe with zero regressions under the substring rule. The stronger edit begins 141/200
responses with normalized `I cannot` and contains `I cannot` in 187/200 responses.

## Interpretation

The on-policy refusal-sentence signal is highly effective on both recorded safety
metrics at this intervention size. However, repetition rises from 13.0% to 31.5%, and
the median unique-word ratio falls from 0.596 to 0.416. The result therefore includes a
substantial degeneration component and should not be interpreted as a clean safety
improvement without capability and response-quality evaluation. Because the frozen
HarmBench set has already been examined in earlier development, this result is
exploratory rather than a pristine confirmatory test.

Strength 1.5 improves both safety metrics further and has fewer repetitive responses
than strength 1 (50 versus 63), but repetition remains substantially above the baseline
of 26 and the output is even more dominated by the extracted refusal phrase. Its FP32
capability evaluation is reported below.

## FP32 IFEval result

The positive-only K=1000, last-position controller was run at strengths 1 and 1.5 in
FP32 on the seed-112 IFEval-200 subset with task-native chat prompts, greedy decoding,
and a 1,024-token limit.

| Condition | Strict prompt | Strict instruction | Loose prompt | Loose instruction | Repetitive |
|---|---:|---:|---:|---:|---:|
| Llama-3 FP32 baseline | 69.5% | 77.24% | 77.0% | 84.29% | 21/200 |
| On-policy refusal Grad FP32, strength 1 | 68.5% | 76.60% | 76.0% | 82.69% | 35/200 |
| On-policy refusal Grad FP32, strength 1.5 | 61.5% | 71.79% | 72.5% | 80.77% | 41/200 |

The edit loses one percentage point of strict and loose prompt accuracy. Repetition
increases from 10.5% to 17.5%, and median unique-word ratio decreases from 0.547 to
0.514. IFEval therefore shows a small instruction-following regression plus a
measurable response-quality regression at the safety-effective intervention size.
At strength 1.5, strict prompt accuracy falls eight percentage points below baseline,
strict instruction accuracy falls 5.45 points, and 12/200 generations reach the
1,024-token limit. The stronger setting therefore gives substantially clearer evidence
of an instruction-following trade-off.

## FP32 GSM8K result

The same positive-only K=1000, last-position controller was evaluated at strengths 1
and 1.5 in FP32 on the first 100 GSM8K test examples. A fresh FP32 Llama-3 baseline was generated
with the identical zero-shot chat prompts, greedy decoding, 256-token limit, batch size
32, and flexible numeric answer scorer.

| Condition | GSM8K accuracy | Correct |
|---|---:|---:|
| Llama-3 FP32 baseline | 75.0% | 75/100 |
| On-policy refusal Grad FP32, strength 1 | 78.0% | 78/100 |
| On-policy refusal Grad FP32, strength 1.5 | 76.0% | 76/100 |

At the paired question level, eight baseline errors become correct and five baseline
correct answers become errors. Neither condition has blank or repetitive responses.
This 100-example GSM8K result does not show a math-capability trade-off: the point
estimate increases by three percentage points, although the net change is based on
only three questions.
Strength 1.5 has six baseline-error-to-correct and five baseline-correct-to-error
changes, for a net one-question improvement over baseline. GSM8K therefore still does
not show the trade-off that is visible on IFEval.

## Artifacts

- Extraction metadata and per-example gradients:
  `results/grad_onpolicy_refusal_sentence_all54/gradients/`
- Positive-only safety evaluation:
  `results/grad_onpolicy_refusal_sentence_stable_positive_k1000_s1_harmbench/`
- Positive-only strength-1.5 safety evaluation:
  `results/grad_onpolicy_refusal_sentence_stable_positive_k1000_s1p5_harmbench/`
- FP32 IFEval evaluation:
  `results/grad_onpolicy_refusal_sentence_stable_positive_k1000_s1_ifeval_fp32/`
- FP32 GSM8K evaluation:
  `results/grad_onpolicy_refusal_sentence_stable_positive_k1000_s1_gsm8k_fp32/`
- FP32 strength-1.5 GSM8K and IFEval evaluation:
  `results/grad_onpolicy_refusal_sentence_stable_positive_k1000_s1p5_capability_fp32/`
- Matched FP32 GSM8K baseline:
  `results/llama3_base_gsm8k_fp32/`
