# Raw-response aggressive `I cannot` Grad, positive-only sweep

Updated: 2026-08-18

## Outcome

This experiment uses the gradient ranking extracted from raw SN-corpus prompt/completion
boundaries and evaluates raw, untemplated prompts on both HarmBench and BeaverTails. The controller
strengthens only positive-gradient neurons, at every generated-token position. In the tables,
`strength=s` means a selected activation dimension is multiplied by `1+s`.

The sweep produces lower refusal-substring ASR as K and strength increase, but the aggressive
points are not clean safety improvements. Repetition rises sharply, every selected frozen run hits
the 128-token generation ceiling on average, and BeaverTails safety reverses direction at high
strength. The most defensible tested tradeoff is K=200, strength 1.25: it reduces frozen raw
HarmBench ASR from 65.5% to 41.0% and BeaverTails cost from -2.103 to -5.440, while helpfulness
reward changes from 7.862 to 6.995. Even this point has substantial repetition (59/200 HarmBench,
38/200 BeaverTails), so it should not be described as a quality-preserving safety result.

| Frozen raw condition | Development ASR ↓ | Development GSM8K-20 ↑ | HarmBench ASR ↓ | HarmBench cost ↓ | BeaverTails cost ↓ / reward ↑ | GSM8K-100 ↑ | MMLU-285 ↑ | Repetitive Harm / Beaver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unmodified Llama-3 baseline | 68.1% (32/47) | 60% | 65.5% (131/200) | 0.594 | -2.103 / 7.862 | 77% (77/100) | 69.82% (199/285) | 26 / 20 |
| Positive-only K=200, strength 1.25 (2.25x) | 48.9% (23/47) | 65% | **41.0% (82/200)** | **-2.318** | **-5.440 / 6.995** | **76% (76/100)** | **69.12% (197/285)** | 59 / 38 |
| Positive-only K=200, strength 1.75 (2.75x) | 29.8% (14/47) | 70% | **32.0% (64/200)** | **-3.812** | **-3.472 / 5.829** | **74% (74/100)** | **69.12% (197/285)** | 79 / 79 |
| Positive-only K=237, strength 3 (4x) | 17.0% (8/47) | 65% | **11.5% (23/200)** | **-3.658** | **1.320 / 2.193** | **75% (75/100)** | **67.02% (191/285)** | 139 / 168 |
| Positive-only K=237, strength 4 (5x) | 10.6% (5/47) | 35% | **7.0% (14/200)** | **-1.024** | **2.603 / -0.281** | **56% (56/100)** | **65.96% (188/285)** | 185 / 195 |

Lower cost is safer and higher reward is more helpful. K=200/strength-1.75 lowers HarmBench ASR
further than strength 1.25, but it worsens BeaverTails cost and reward and doubles BeaverTails
repetition. At K=237/strength-3, the low 11.5% ASR coexists with BeaverTails cost +1.320, worse
than the raw baseline's -2.103. At strength 4, BeaverTails cost worsens further to +2.603 and reward
falls below zero. These results show that lexical ASR alone would select a degenerate controller.

The full GSM8K and MMLU columns use the unified Llama-3 capability protocol for every row. GSM8K
uses the first 100 test questions, zero-shot native chat, greedy decoding, a 256-token limit, and
flexible deterministic numeric exact match. MMLU uses the fixed seed-112 285-question subset,
five-shot native chat, and constrained A/B/C/D next-token scoring. These capability prompts do not
change between the raw and chat safety-format experiments; `raw` here applies only to HarmBench and
BeaverTails. The K=200 points nearly preserve baseline capability, whereas K=237/strength-4 loses
21 GSM8K questions and 11 MMLU questions relative to the matched unified baseline.

The generated text explains the metrics. At strength 1.25, responses can begin with a relevant
refusal or cautious answer and then repeat whole suggestions. At strength 1.75, repeated variants
of `I cannot` are common. The K=237 points frequently repeat the same clause until the token limit.
That inserts refusal substrings and depresses ASR without reliably producing coherent, safe, or
helpful answers. The repetition diagnostic marks a response when any four-word sequence appears at
least five times.

## Expanded 4,000-candidate ranking outcome

The ranking pool was expanded to permit interventions on 500--1,000 positive neurons. Gradient
extraction was repeated with the identical 235 raw SN-corpus examples and exact `I cannot`
objective, but with a 4,000-candidate pool and 4,000 saved stable rows. The new per-example gradient
tensor is byte-identical to the original extraction (SHA-256
`b0f68bd91917f45567f5332f9933094774d8203bcad54630c1b522006e6b2514`), so only ranking breadth
changed. The expanded stable CSV contains 1,959 positive and 2,041 negative dimensions; its SHA-256
is `974c664e6520c32a96a942fae556763e6dcca6795e8ad56ee7fad29270ba6075`.

The development sweep covered K=500/750/1000 at low strengths 0.05/0.1/0.2/0.3/0.5 and aggressive
strengths 0.75/1/1.5/2. Six representative K=500 and K=1000 points were then evaluated on the full
frozen raw HarmBench-200 and BeaverTails-200 sets. A strength of 0.5 means a 1.5x activation
multiplier; strength 2 means 3x.

| Expanded-ranking raw condition | Development ASR ↓ | GSM8K-20 ↑ | HarmBench ASR ↓ | HarmBench cost ↓ | BeaverTails cost ↓ / reward ↑ | GSM8K-100 ↑ | MMLU-285 ↑ | Repetitive Harm / Beaver |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unmodified Llama-3 baseline | 68.1% (32/47) | 60% | 65.5% (131/200) | 0.594 | -2.103 / 7.862 | 77% (77/100) | 69.82% (199/285) | 26 / 20 |
| K=500, strength 0.5 (1.5x) | 57.4% (27/47) | 70% | **54.0% (108/200)** | **-1.764** | **-5.597 / 6.140** | **80% (80/100)** | **69.12% (197/285)** | 45 / 24 |
| K=500, strength 1 (2x) | 29.8% (14/47) | 65% | **27.0% (54/200)** | **-3.945** | **-4.042 / 5.373** | **76% (76/100)** | **69.12% (197/285)** | 85 / 73 |
| K=500, strength 2 (3x) | 10.6% (5/47) | 60% | **7.5% (15/200)** | **-5.819** | **1.121 / 1.811** | **72% (72/100)** | **66.67% (190/285)** | 140 / 179 |
| K=1000, strength 0.5 (1.5x) | 44.7% (21/47) | 60% | **30.5% (61/200)** | **-4.373** | **-4.507 / 5.746** | **80% (80/100)** | **68.07% (194/285)** | 73 / 64 |
| K=1000, strength 1 (2x) | 10.6% (5/47) | 65% | **12.5% (25/200)** | **-6.330** | **-0.665 / 2.337** | **77% (77/100)** | **66.67% (190/285)** | 142 / 169 |
| K=1000, strength 1.5 (2.5x) | 4.3% (2/47) | 55% | **2.0% (4/200)** | **-5.089** | **0.622 / 1.128** | **65% (65/100)** | **65.61% (187/285)** | 171 / 193 |

K=500/strength-0.5 is the least repetitive enlarged-pool point, but its 54% ASR improvement over
the 65.5% baseline is modest and HarmBench repetition still rises from 26 to 45. K=1000/strength-0.5
is the strongest lower-multiplier tradeoff: ASR falls to 30.5%, BeaverTails safety cost improves
from -2.103 to -4.507, and reward remains 5.746 versus baseline 7.862. It nevertheless has 73/200
repetitive HarmBench and 64/200 repetitive BeaverTails responses.

The aggressive controls confirm metric degeneration. K=500/strength-2 and K=1000/strength-1.5
reach 7.5% and 2.0% ASR, respectively, but BeaverTails cost becomes positive (less safe than the
raw baseline), and 179/200 or 193/200 BeaverTails responses are repetitive. Enlarging K therefore
lets a lower multiplier move the refusal direction more strongly; it does not make high-strength
amplification a clean safety intervention.

Full capability evaluation uses the unified protocol: GSM8K is the first 100 test rows with
zero-shot native Llama-3 chat, greedy 256-token generation, and flexible numeric exact match; MMLU
is the fixed seed-112 285-question subset with five-shot native chat and constrained A/B/C/D
scoring. K=500/strength-0.5 and K=1000/strength-0.5 score 80% GSM8K, while MMLU changes by only 2
and 5 questions from baseline. At K=1000/strength-1.5, GSM8K falls by 12 points and MMLU by 12
questions. Capability loss therefore becomes clear at the strongest setting, although the safety
quality diagnostics already fail at milder aggressive settings.

## Development sweep

Model selection used a test-disjoint 47-prompt HarmBench tuning split and a fixed GSM8K-20
capability check. The development pool is the ID complement of the frozen HarmBench-200 test after
excluding three prompt duplicates: 197 development prompts, split into 47 tuning and 150 selection
prompts. Test/development overlap is zero by both ID and exact prompt. The tuning-manifest SHA-256
is `0ef082a5778b1a739c2c79a168975ec95677f8f2fa9a386adb65d4583e5f5b5c`; the frozen test SHA-256
is `bb5b29ff9db15e420021aee3ad1a07d0ed1ca11a2d8faff024d786168b7be74c`.

The coarse search covered K=25/50/100/200 at strengths 0.5/1/2/4. The frontier search covered
K=200/237 at strengths 1.25/1.5/1.75/2/2.5/3/4. Selected frontier values are shown below; the full
CSV artifacts retain every configuration.

| Positive neurons | Strength (multiplier) | Development ASR ↓ | GSM8K-20 ↑ | Repetitive /47 |
|---:|---:|---:|---:|---:|
| 0 | 0 (1x) | 68.1% | 60% | 12 |
| 200 | 1.25 (2.25x) | 48.9% | 65% | 12 |
| 200 | 1.50 (2.5x) | 40.4% | 65% | 14 |
| 200 | 1.75 (2.75x) | 29.8% | 70% | 15 |
| 200 | 2.00 (3x) | 29.8% | 70% | 16 |
| 237 | 2.00 (3x) | 21.3% | 65% | 26 |
| 237 | 3.00 (4x) | 17.0% | 65% | 34 |
| 237 | 4.00 (5x) | 10.6% | 35% | 43 |

The automatic selector chose K=237/strength-3 among GSM8K-eligible frontier points because its
selection objective does not include repetition. The frozen results demonstrate why that automatic
choice is not endorsed. K=200/strength-1.25 was added as the baseline-repetition frontier point,
and K=200/strength-1.75 as the strongest development capability-preserving point before repetition
becomes dominant.

Before the sweep, a 32-example real-prompt raw-generation benchmark tested batch sizes 8, 16, and
32 at 16 new tokens on physical GPU 0. Throughput was 19.67, 49.46, and 96.73 examples/s with
15.045, 15.103, and 15.215 GiB peak allocation, respectively. Batch 32 was therefore used for
generation and both Beaver score models.

## Reproducibility and artifacts

All generation and safety/helpfulness scoring used only the first physical GPU:
`CUDA_VISIBLE_DEVICES=0`, with model, cost model, and reward model addressed as `cuda:0`.
HarmBench and BeaverTails use greedy decoding and at most 128 new tokens. BeaverTails uses the
frozen seed-42 200-prompt manifest from the final 600 rows of `round0/330k/test`, SHA-256
`eb2272513b954bf097aa645b518ffc271ecb22ec815f8db4377276e19909c5e9`. Its score text is the full
decoded raw prompt plus completion.

All six per-task files in every frozen run contain 200 rows, response IDs are unique, generated
checksum manifests verify, and all 30 unified evaluator tests pass.

- Ranking: `results/grad_i_cannot_raw_extract_chat_patch/corpus_gradients/top_neurons_stable.csv`
  (SHA-256 `54fc89f8f4bf8f281d1c69b2d21d710d6f60289aab8516e4e4931f2c28fdec06`)
- Split, benchmark, coarse sweep, and frontier sweep:
  `results/grad_i_cannot_raw_positive_only_raw_eval/`
- Expanded stable ranking:
  `results/grad_i_cannot_raw_extract_expanded_4000/corpus_gradients/`
- Expanded low-strength development sweep:
  `results/grad_i_cannot_raw_positive_only_raw_eval/expanded_4000_positive_only_low_all_raw/`
- Expanded aggressive development sweep:
  `results/grad_i_cannot_raw_positive_only_raw_eval/expanded_4000_positive_only_aggressive_all_raw/`
- Matched unified Llama-3 capability baseline:
  `results/llama3_base_chat_capability_unified_b16_m8/`
- K=200, strength-1.25 frozen safety and capability result:
  `results/grad_i_cannot_raw_positive_only_raw_all_k200_s1p25/`
- K=200, strength-1.75 frozen safety and capability result:
  `results/grad_i_cannot_raw_positive_only_raw_all_k200_s1p75/`
- K=237, strength-3 frozen safety and capability result:
  `results/grad_i_cannot_raw_positive_only_raw_all_k237_s3/`
- K=237, strength-4 frozen safety and capability result:
  `results/grad_i_cannot_raw_positive_only_raw_all_k237_s4/`
- Expanded-ranking K=500 frozen raw safety/BeaverTails results:
  `results/grad_i_cannot_raw_positive_only_exp4000_raw_all_k500_s0p5/`,
  `results/grad_i_cannot_raw_positive_only_exp4000_raw_all_k500_s1/`, and
  `results/grad_i_cannot_raw_positive_only_exp4000_raw_all_k500_s2/`
- Expanded-ranking K=1000 frozen raw safety/BeaverTails results:
  `results/grad_i_cannot_raw_positive_only_exp4000_raw_all_k1000_s0p5/`,
  `results/grad_i_cannot_raw_positive_only_exp4000_raw_all_k1000_s1/`, and
  `results/grad_i_cannot_raw_positive_only_exp4000_raw_all_k1000_s1p5/`
