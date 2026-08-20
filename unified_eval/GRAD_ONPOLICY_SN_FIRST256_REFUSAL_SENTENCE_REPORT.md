# On-policy Grad from the SN-Tune first 256 training prompts

Updated: 2026-08-19

## Method

This experiment avoids using the HarmBench evaluation prompt set to construct the
Grad signal. The source is the exact first 256 rows of the SN-Tune training file
`circuit_breakers_train.json`, rendered in raw format. Fresh greedy BF16 responses
were generated from Llama-3-8B-Instruct with a 256-token limit and seed 112.

Of the 256 generations, 116 pass the refusal-substring safety rule and 65 contain the
exact phrase `I cannot`. All 65 are safe by that rule and all were used. Only 9/65
begin with normalized `I cannot`; for the remainder, the generated text preceding the
first occurrence is retained as response prefill. The target is the complete first
sentence beginning at that occurrence. The source prompts have zero exact normalized
overlap with the frozen HarmBench-200 evaluation prompts.

The objective is mean teacher-forced target-token log probability conditioned on the
raw prompt and generated response prefix. Gradients are taken at the final prefill
position and refusal-target positions. The tensor has shape `65 x 32 x 14336`; target
sentences contain 9--26 tokens (mean 15.09), with mean target log probability -0.400.
The 4,000-neuron stability ranking contains 1,943 objective-supporting and 2,057
objective-suppressing neurons. Positive-only K=1000 reaches original stability rank
1,960.

A major source-quality caveat is that 51/65 selected generations are repetitive by
the experiment's four-gram rule. Thus, this signal may learn generation degeneration
along with refusal behavior even though it avoids direct HarmBench selection leakage.

## Raw HarmBench safety

Safety uses the frozen raw HarmBench-200 manifest, greedy decoding, a 128-token limit,
the case-sensitive llm-attacks refusal-substring ASR, and Beaver-7B cost. The
intervention uses the final-token scope, K=1000, and `positive-only`. No BeaverTails
evaluation was run.

| Condition | ASR (lower is safer) | Mean Beaver cost | Repetitive | At token limit |
|---|---:|---:|---:|---:|
| Llama-3 BF16 raw baseline | 65.5% (131/200) | 0.594 | 26/200 | -- |
| SN-first256 Grad, strength 1 | **21.0% (42/200)** | **-6.023** | 100/200 | 194/200 |
| SN-first256 Grad, strength 1.5 | **15.5% (31/200)** | **-7.560** | 107/200 | 197/200 |

At strength 1, relative to baseline, 90 unsafe responses become safe and one safe
response becomes unsafe. At strength 1.5, the corresponding counts are 101 and one.
From strength 1 to 1.5, 12 responses change unsafe-to-safe and one changes
safe-to-unsafe.

The edit therefore greatly improves the recorded safety metrics, but half the outputs
are repetitive and nearly all hit the generation limit. This is not a clean safety
improvement: refusal behavior and output degeneration are strongly confounded.

## FP32 GSM8K and IFEval

Both strengths were evaluated with FP32 model and intervention computation. GSM8K
uses the first 100 test rows, zero-shot Llama-3 chat, greedy decoding, a 256-token
limit, and flexible numeric exact match. IFEval uses the frozen seed-112 coverage-first
200-row subset, task-native chat, greedy decoding, a 1,024-token limit, and the
official strict and loose scorers.

| Condition | GSM8K | IFEval strict prompt | Strict instruction | Loose prompt | Loose instruction | IFEval repetitive | At limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3 FP32 baseline | 75.0% | 69.5% | 77.24% | 77.0% | 84.29% | 21/200 | 2/200 |
| SN-first256 Grad, strength 1 | 75.0% | 67.0% | 76.28% | 77.5% | 84.29% | 27/200 | 7/200 |
| SN-first256 Grad, strength 1.5 | 76.0% | 60.0% | 70.83% | 67.0% | 77.24% | 36/200 | 7/200 |

On paired GSM8K examples, strength 1 changes six baseline errors to correct and six
baseline correct answers to errors. Strength 1.5 changes five errors to correct and
four correct answers to errors. The 100-example math result therefore shows no math
capability loss.

IFEval shows little aggregate change at strength 1: strict prompt accuracy falls 2.5
points, strict instruction accuracy falls 0.96 points, loose prompt accuracy rises
0.5 point, and loose instruction accuracy is unchanged. At strength 1.5 the trade-off
is clear: strict prompt accuracy falls 9.5 points, strict instruction accuracy falls
6.41 points, loose prompt accuracy falls 10 points, and loose instruction accuracy
falls 7.05 points. Repetition also increases with strength.

## Larger-K FP32 safety sweep

The stability ranking contains only 1,943 objective-supporting neurons, so K=1,943 is
the maximum possible `positive-only` intervention. Larger-K raw HarmBench runs used
FP32 and otherwise retained the preceding safety protocol. Strength is the additive
scale, so strengths 1.5, 2, and 2.5 correspond to positive activation multipliers
2.5x, 3x, and 3.5x.

| K | Strength | HarmBench ASR | Mean Beaver cost | Repetitive |
|---:|---:|---:|---:|---:|
| 1,500 | 1.5 | 10.5% (21/200) | **-8.760** | 100/200 |
| 1,943 | 1.5 | 8.0% (16/200) | -6.511 | 160/200 |
| 1,943 | 2 | 6.5% (13/200) | 1.011 | 177/200 |
| 1,943 | 2.25 | 7.5% (15/200) | 2.553 | 172/200 |
| 1,943 | 2.5 | **4.5% (9/200)** | 3.680 | 177/200 |
| 1,943 | 3 | 11.0% (22/200) | 5.546 | 139/200 |
| SN-Tune alpha=8 FP32 reference | -- | **0.5% (1/200)** | -5.009 | -- |

Increasing K lowers Grad ASR, but it does not match SN-Tune alpha=8. The response is
non-monotonic in strength: the best observed ASR is 4.5% at strength 2.5, followed by
a reversal to 11.0% at strength 3. Beaver cost also reverses direction and becomes
worse than the unedited raw baseline at strengths 2 and above. All maximal-K responses
reach the 128-token limit on average, and 69.5--88.5% are repetitive. The low lexical
ASR at these settings is therefore not supported by the learned cost metric or by
response-quality diagnostics.

Strengths 2.6 and 2.75 were stopped after 32/200 safety generations at the user's
request. Those partial artifacts are excluded from the table and all comparisons.

## Expanded-ranking, moderate-strength pilot

The preceding K=1,943 limit came only from saving 4,000 ranked candidates, not from
the model. The existing full `65 x 32 x 14336` gradient tensor was reranked with a
20,000-candidate pool; the expanded ranking contains 9,881 objective-supporting
neurons. No gradients were recomputed. A quick FP32 sweep then held strength at 1
(2x positive activation) and increased only K.

| K | Strength | HarmBench ASR | Mean Beaver cost | Repetitive |
|---:|---:|---:|---:|---:|
| 1,000 | 1 | 17.0% (34/200) | -4.807 | 85/200 |
| 2,000 | 1 | 9.5% (19/200) | -5.938 | 91/200 |
| 4,000 | 0.25 | 43.0% (86/200) | not run | 66/200 |
| 4,000 | 1 | **5.5% (11/200)** | **-7.864** | 88/200 |

K=4,000 nearly matches the aggressive K=1,943, strength-2.5 ASR of 4.5%, while
retaining a strongly negative Beaver cost instead of reversing it to +3.680. Output
repetition remains high, so this is still not a clean refusal intervention.

The K=4,000, strength-0.25 run uses the exact same 20,000-candidate ranking and
selected-neuron prefix as K=4,000, strength 1, isolating activation strength. Its
43.0% ASR improves on the matched raw baseline's 67.5% with 52 prompt-level safety
gains and 3 losses (two-sided exact McNemar p=1.54e-12), but it is substantially
less safe than strength 1 at 5.5% ASR. Moving from strength 0.25 to 1 yields 76
safety gains and 1 loss (p=1.03e-21). Repetitive responses increase from 66/200
to 88/200. Per the current evaluation protocol, the new run did not invoke Beaver
scoring.

For a quick capability anchor, both conditions were evaluated on only the fixed first
32 rows of the coverage-first IFEval subset (50 instruction constraints). These pilot
scores must not be compared as if they were full IFEval-200 estimates.

| Condition | Strict prompt | Strict instruction | Loose prompt | Loose instruction | Repetitive | At limit |
|---|---:|---:|---:|---:|---:|---:|
| FP32 baseline, first 32 | 68.75% (22/32) | 74% (37/50) | 71.88% (23/32) | 80% (40/50) | -- | 0/32 |
| K=2,000, strength 1 | 71.88% (23/32) | 80% (40/50) | 75.0% (24/32) | 82% (41/50) | 3/32 | 0/32 |
| K=4,000, strength 1 | 71.88% (23/32) | 78% (39/50) | 78.13% (25/32) | 82% (41/50) | 3/32 | 0/32 |

This small anchor shows no obvious IFEval collapse at strength 1. It is only coarse
evidence: the full 200-row evaluation would be needed to estimate the capability
change reliably.

The subsequent full IFEval-200 evaluation shows that the 32-row anchor was optimistic:

| Condition | Safety ASR | Strict prompt | Strict instruction | Loose prompt | Loose instruction | Repetitive | At limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 65.5% | 69.5% | 77.24% | 77.0% | 84.29% | 21/200 | 2/200 |
| K=1,000, strength 1 | 17.0% | 67.5% | 76.92% | 77.5% | 85.26% | 28/200 | 3/200 |
| K=2,000, strength 1 | 9.5% | 65.5% | 76.92% | 72.5% | 82.05% | 27/200 | 7/200 |
| K=4,000, strength 1 | **5.5%** | 61.0% | 71.79% | 70.5% | 78.53% | 36/200 | 3/200 |

K=1,000 loses only two strict-prompt points while reducing ASR by 48.5 points. Relative to
baseline, K=2,000 has 20 strict-prompt losses and 12 gains; K=4,000
has 30 losses and 13 gains. Thus K=2,000 preserves instruction-level compliance
fairly well but loses four strict-prompt points. K=4,000 provides the stronger safety
effect but loses 8.5 strict-prompt points, 5.45 strict-instruction points, 6.5
loose-prompt points, and 5.77 loose-instruction points. This is a clear but much less
severe capability trade-off than the aggressive-strength maximal-K settings.

## IFEval for the lowest-ASR completed settings

Only the two selected low-ASR settings, maximal K at strengths 2 and 2.5, were run on
IFEval. Evaluation is FP32 and uses the same frozen IFEval-200 protocol as above.

| Condition | Safety ASR | Strict prompt | Strict instruction | Loose prompt | Loose instruction | Repetitive | At limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3 FP32 baseline | 65.5% | 69.5% | 77.24% | 77.0% | 84.29% | 21/200 | 2/200 |
| SN-Tune alpha=8 FP32 reference | 0.5% | 55.0% | 65.06% | 61.5% | 69.87% | 31/200 | 9/200 |
| Grad K=1,943, strength 2 | 6.5% | 41.0% | 49.36% | 45.5% | 53.21% | 67/200 | 38/200 |
| Grad K=1,943, strength 2.5 | 4.5% | 19.5% | 33.01% | 23.0% | 35.90% | 52/200 | 130/200 |

Relative to the FP32 baseline, strength 2 has 68 strict-prompt losses and 11 gains;
strength 2.5 has 109 losses and 9 gains. Mean generated length rises from 295.6 tokens
at baseline to 353.7 and 703.7 tokens. These low-ASR Grad settings damage instruction
following substantially more than SN-Tune alpha=8 while still retaining higher ASR.

## Matched on-policy/off-policy FP32 math comparison

The expanded 20,000-candidate on-policy and off-policy rankings were compared on the
same first 100 GSM8K test rows. Every intervention uses strength 1, `positive-only`,
FP32, greedy decoding, a 256-token limit, batch size 32, and the flexible numeric
exact-match scorer. Matching the configurations in the main HarmBench--IFEval figure,
on-policy uses final-token scope and off-policy uses all-token scope. The off-policy
signal is the established raw-corpus `I cannot` teacher-forced gradient ranking.

| Condition | K | GSM8K accuracy | Paired baseline errors fixed | Paired baseline correct broken |
|---|---:|---:|---:|---:|
| FP32 baseline | -- | 75.0% (75/100) | -- | -- |
| On-policy SN-first256 | 1,000 | 77.0% (77/100) | 5 | 3 |
| On-policy SN-first256 | 2,000 | **81.0% (81/100)** | 8 | 2 |
| On-policy SN-first256 | 4,000 | 79.0% (79/100) | 9 | 5 |
| Off-policy raw corpus | 1,000 | 72.0% (72/100) | 3 | 6 |
| Off-policy raw corpus | 2,000 | 71.0% (71/100) | 4 | 8 |
| Off-policy raw corpus | 4,000 | 67.0% (67/100) | 7 | 15 |

On this 100-example math subset, the on-policy trajectory does not show a math
trade-off: all three point estimates exceed baseline, with K=2,000 highest at +6
questions. The off-policy trajectory declines with K, reaching 67% at K=4,000. The
earlier off-policy K=1,000/K=2,000 results of 73%/75% used final-token scope and are
not the points represented by the main figure. These are point estimates from only
100 examples and should not be interpreted as precise capability measurements.

## Conclusion

Using the SN-Tune training prompts removes direct HarmBench prompt leakage from Grad
selection. Positive-only Grad lowers HarmBench ASR as K and strength increase, but the
best completed result, 4.5%, does not reach SN-Tune alpha=8's 0.5%. The stronger edits
exhibit a large safety-versus-general-capability trade-off on IFEval, while the earlier
K=1000 experiment shows no loss on the 100-example GSM8K sample. Because the training
generations and safety outputs are highly repetitive, the apparent safety gain cannot
be separated from degeneration; at maximal K and high strength, Beaver cost directly
contradicts the improved refusal-substring ASR.

## Artifacts

- Fresh first-256 generations:
  `/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/training/circuit_breakers_train_first256_llama3_raw_regenerated_bf16_greedy256.jsonl`
- Extraction and ranking:
  `results/grad_onpolicy_sn_first256_refusal_sentence_all65/gradients/`
- Strength-1 safety:
  `results/grad_onpolicy_sn_first256_refusal_sentence_stable_positive_k1000_s1_harmbench/`
- Strength-1.5 safety:
  `results/grad_onpolicy_sn_first256_refusal_sentence_stable_positive_k1000_s1p5_harmbench/`
- Strength-1 FP32 capability:
  `results/grad_onpolicy_sn_first256_refusal_sentence_stable_positive_k1000_s1_capability_fp32/`
- Strength-1.5 FP32 capability:
  `results/grad_onpolicy_sn_first256_refusal_sentence_stable_positive_k1000_s1p5_capability_fp32/`
- Larger-K FP32 raw HarmBench runs:
  `results/grad_onpolicy_sn_first256_refusal_sentence_stable_positive_k1500_s1p5_harmbench_fp32/`
  and `results/grad_onpolicy_sn_first256_refusal_sentence_stable_positive_k1943_*/`
- Expanded 20,000-row ranking:
  `results/grad_onpolicy_sn_first256_refusal_sentence_all65_expanded20000/gradients/`
- Moderate-strength expanded-ranking K=1,000/K=2,000/K=4,000 safety and IFEval runs:
  `results/grad_onpolicy_sn_first256_expanded_positive_k1000_s1_harmbench_fp32/`,
  `results/ifeval_grad_onpolicy_sn_first256_expanded_positive_k1000_s1_fp32/`,
  `results/grad_onpolicy_sn_first256_expanded_positive_k2000_s1_*_fp32/`, and
  `results/grad_onpolicy_sn_first256_expanded_positive_k4000_s1_*_fp32/`
- Same-ranking K=4,000, strength-0.25 raw HarmBench run:
  `results/grad_onpolicy_sn_first256_expanded_positive_k4000_s0p25_harmbench_fp32/`
- K=1,943 strength-2 FP32 IFEval:
  `results/grad_onpolicy_sn_first256_refusal_sentence_stable_positive_k1943_s2_ifeval_fp32/`
- K=1,943 strength-2.5 FP32 IFEval:
  `results/grad_onpolicy_sn_first256_refusal_sentence_stable_positive_k1943_s2p5_ifeval_fp32/`
- Matched FP32 GSM8K comparison:
  `results/grad_onpolicy_sn_first256_expanded_positive_k*_s1_gsm8k_fp32/` and
  `results/grad_offpolicy_raw_positive_exp20000_k*_s1_all_gsm8k_fp32/`
- Matched FP32 baselines:
  `results/llama3_base_gsm8k_fp32/` and `results/ifeval_llama3_base_fp32/`
