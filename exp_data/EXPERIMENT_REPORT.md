# Unified safety-neuron experiment report

Updated: 2026-08-18

## Summary

The recent work added and evaluated base-model and DPO reference conditions alongside the existing
Gradient and SN-Tune interventions:

- **Llama-3 baseline:** unmodified `Meta-Llama-3-8B-Instruct`. HarmBench was freshly generated and
  scored in the unified evaluator. GSM8K and MMLU generations were inherited from the existing
  source reproduction; the saved GSM8K generations were rescored with the flexible evaluator.
- **Llama-3 HH-DPO:** trained a 458,752-parameter IA3 adapter on all 42,537 HH harmless pairs using
  native Llama-3 chat formatting, then ran the complete unified evaluation.
- **Llama-2 baseline:** unmodified `Llama-2-7b-hf`, without SFT/DPO adapters or neuron patching.
  Its checksum-verified HarmBench outputs and Beaver costs were imported from the exact local
  NeurIPS reproduction, unified ASR was computed, and GSM8K/MMLU were freshly evaluated.
- **NeurIPS DPO:** Llama-2-7B with the local SFT and DPO IA3 adapters. HarmBench, GSM8K, and MMLU
  are complete, although the stored full GSM8K result predates the new flexible extractor and is
  therefore marked as a legacy score below.
- **Chat `I cannot` Grad:** extracted a neuron-gradient ranking from chat-formatted SN-Tune corpus
  examples, selected an all-token top-25 controller off-test, and evaluated it against a matched
  strength-zero chat control.
- **Raw IA3-SFT:** trained a 458,752-parameter `down_proj` IA3 adapter on the first 256 SN-corpus
  refusal pairs using raw prompt/completion serialization, then evaluated it on the complete frozen
  suite.
- **Raw-to-chat `I cannot` Grad:** freshly extracted the gradient ranking with the raw SN-corpus
  boundary, then applied its fixed top-25, strength-0.5 controller at every token position while
  evaluating with native Llama-3 chat prompts.
- **Raw-response positive-only Grad sweep:** swept the raw-extracted ranking with raw HarmBench and
  BeaverTails prompts on physical GPU 0, then expanded the stable candidate ranking to support
  K=500/750/1000 low-to-aggressive sweeps. The apparent low-ASR extremes are dominated by
  repetition; see `GRAD_I_CANNOT_RAW_POSITIVE_ONLY_AGGRESSIVE_RAW_EVAL_REPORT.md` for both frontiers.
- **IA3-SFT guide patch:** ranked SFT-vs-base post-MLP changes on a test-disjoint native-chat
  development set, copied the top 20,000 guide dimensions into the base at inference, and ran all
  three benchmarks with native Llama-3 chat formatting.
- **BeaverTails evaluation:** evaluated the unmodified Llama-3 baseline in raw format and every
  Llama-3 condition in the native-chat safety table on the paper-matched BeaverTails prompt sample,
  retaining safety cost and helpfulness reward in one additional results column.

The attempted top-8K NeurIPS neuron patch was stopped and is not a current result. Its incomplete
artifacts, along with earlier capability runs that used unsuitable adapted prompts, are retained in
`results/archive/` for provenance.

## Current results

### Safety evaluation without chat formatting

These Llama-3 conditions receive the raw HarmBench behavior text without a chat template.

| Condition | Model | HarmBench ASR ↓ | Beaver cost ↓ | BeaverTails cost ↓ / reward ↑ | GSM8K-100 ↑ | MMLU-285 ↑ |
|---|---|---:|---:|---:|---:|---:|
| **Llama-3 baseline** | Llama-3-8B-Instruct | 65.5% (131/200) | 0.594 | **-2.103 / 7.862** | 75% (75/100), flexible rescore of inherited generations | 67.37% (192/285), inherited |
| **Llama-3 HH-DPO IA3** | Llama-3-8B-Instruct | 60.0% (120/200) | -0.634 | **-4.639 / 7.700** | **76% (76/100)** | **68.42% (195/285)** |
| **Llama-3 SN-corpus IA3-SFT, raw-trained** | Llama-3-8B-Instruct | **44.5% (89/200)** | **-2.451** | **-6.723 / 2.033** | **76% (76/100)** | **68.77% (196/285)** |
| **IA3-SFT alpha-3 held-out guide patch, top-8k** | Llama-3-8B-Instruct | **32.5% (65/200)** | **-5.039** | -- | **75% (75/100)** | **67.37% (192/285)** |
| **IA3-SFT alpha-3 held-out guide patch, top-20k** | Llama-3-8B-Instruct | **24.0% (48/200)** | **-5.371** | -- | **74% (74/100)** | **67.37% (192/285)** |
| **IA3-SFT alpha-3 held-out guide patch, top-40k** | Llama-3-8B-Instruct | **22.0% (44/200)** | **-5.610** | -- | -- | -- |
| **IA3-SFT alpha-3 held-out guide patch, top-80k** | Llama-3-8B-Instruct | **15.5% (31/200)** | **-6.115** | -- | -- | -- |
| **IA3-SFT alpha-3 held-out guide patch, top-160k** | Llama-3-8B-Instruct | **12.0% (24/200)** | **-6.171** | -- | -- | -- |
| **IA3-SFT alpha-3 held-out guide patch, top-240k** | Llama-3-8B-Instruct | **10.0% (20/200)** | **-6.445** | -- | -- | -- |
| **IA3-SFT alpha-3 held-out guide patch, top-320k** | Llama-3-8B-Instruct | **9.0% (18/200)** | **-6.356** | -- | -- | -- |
| **IA3-SFT alpha-3 held-out guide patch, all 444,416** | Llama-3-8B-Instruct | **9.0% (18/200)** | **-6.401** | -- | -- | -- |
| DPO-guide patch, top-20k | Llama-3-8B-Instruct | 61.0% (122/200) | -0.050 | **-2.980 / 7.568** | 73% (73/100) | 68.07% (194/285) |
| DPO-guide patch, top-8k | Llama-3-8B-Instruct | 61.5% (123/200) | 0.174 | **-2.754 / 7.757** | 73% (73/100) | 67.37% (192/285) |
| Gradient top-25, strength 1 | Llama-3-8B-Instruct | 62.0% (124/200) | -0.431 | **-3.101 / 7.918** | 67% (67/100), flexible rescore | 67.72% (193/285) |
| SN-Tune, `alpha=1` | Llama-3-8B-Instruct | 62.5% (125/200) | 0.648 | -2.755 / 6.439 | 76% (76/100) | 68.07% (194/285) |
| SN-Tune, `alpha=2` | Llama-3-8B-Instruct | 53.5% (107/200) | 0.247 | -1.888 / 5.856 | 73% (73/100) | 67.72% (193/285) |
| SN-Tune, `alpha=4` | Llama-3-8B-Instruct | 20.0% (40/200) | -4.899 | -4.871 / 3.660 | 74% (74/100) | 68.07% (194/285) |
| SN-Tune, `alpha=6` | Llama-3-8B-Instruct | **1.0% (2/200)** | **-6.553** | -- | -- | -- |
| SN-Tune, `alpha=8` | Llama-3-8B-Instruct | **0.5% (1/200)** | **-5.009** | **-7.946 / -0.219** | 73% (73/100), flexible rescore | 67.02% (191/285) |

### Safety evaluation using chat formatting

The Llama-3 rows use the tokenizer-native Llama-3 chat template. The Llama-2 rows use Tulu chat.

| Condition | Model | Safety chat template | HarmBench ASR ↓ | Beaver cost ↓ | BeaverTails cost ↓ / reward ↑ | GSM8K-100 ↑ | MMLU-285 ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| **Llama-3 baseline** | Llama-3-8B-Instruct | Llama-3 native | **29.0% (58/200)** | **-5.178** | **-10.212 / -2.575** | **77% (77/100)** | **69.82% (199/285)** |
| **Llama-3 HH-DPO IA3(useless)** | Llama-3-8B-Instruct | Llama-3 native | **35.0% (70/200)** | **-5.035** | **-10.555 / -3.039** | **76% (76/100)** | **68.42% (195/285)** |
| **Llama-3 SN-corpus IA3-SFT, chat-trained** | Llama-3-8B-Instruct | Llama-3 native | **25.0% (50/200)** | **-5.086** | **-10.651 / -3.128** | **74% (74/100)** | **69.12% (197/285)** |
| **Llama-3 SN-corpus IA3-SFT, raw-trained** | Llama-3-8B-Instruct | Llama-3 native | **23.5% (47/200)** | **-4.574** | **-10.667 / -3.583** | **76% (76/100)** | **68.77% (196/285)** |
| **IA3-SFT-guide patch, top-20k** | Llama-3-8B-Instruct | Llama-3 native | **28.0% (56/200)** | **-5.121** | **-10.239 / -2.281** | **77% (77/100)** | **69.47% (198/285)** |
| Chat `I cannot` Grad, top-25, strength 0.5, all-token | Llama-3-8B-Instruct | Llama-3 native | **28.0% (56/200)** | **-5.150** | **-10.157 / -2.678** | **77% (77/100)** | **69.47% (198/285)** |
| **Raw-extracted `I cannot` Grad, top-25, strength 0.5, all-token** | Llama-3-8B-Instruct | Llama-3 native | **29.5% (59/200)** | **-5.183** | **-10.119 / -2.981** | **77% (77/100)** | **70.18% (200/285)** |
| Raw-extracted positive-only Grad, top-237 positive, strength 4, all-token | Llama-3-8B-Instruct | Llama-3 native | **6.0% (12/200)** | **-4.080** | **-7.315 / 0.585** | **56% (56/100)** | **65.96% (188/285)** |
| SN-Tune, raw-trained, `alpha=8` | Llama-3-8B-Instruct | Llama-3 native | 9.5% (19/200) | **-5.298** | **-9.349 / -8.081** | **73% (73/100)** | **67.02% (191/285)** |
| **SN-Tune, chat-trained, `alpha=8`** | Llama-3-8B-Instruct | Llama-3 native | **26.5% (53/200)** | **-4.975** | **-10.134 / -4.539** | **71% (71/100)** | **69.82% (199/285)** |

### Llama-2 chat-formatted conditions

| Condition | Model | Safety chat template | HarmBench ASR ↓ | Beaver cost ↓ | GSM8K-100 ↑ | MMLU-285 ↑ |
|---|---|---|---:|---:|---:|---:|
| **Llama-2 baseline** | Llama-2-7B Base | Tulu | 99.0% (198/200) | 7.850 | 24% (24/100), flexible | 39.30% (112/285) |
| NeurIPS SFT+DPO | Llama-2-7B | Tulu | 50.0% (100/200) | -3.348 | 11% stored legacy score; 9% flexible rescore | 35.79% (102/285) |

The raw/chat split applies only to the safety evaluation (HarmBench and Beaver); capability tasks
retain their task-specific formats. Lower ASR and Beaver cost indicate safer behavior. Higher
GSM8K and MMLU indicate better general capability.

The displayed Llama-3 GSM8K entries use the flexible numeric evaluator over the first 100 test
rows, zero-shot native chat, greedy decoding, and a 256-token limit. Most were freshly evaluated;
the baseline, original Gradient, and original SN-Tune entries are flexible offline rescores of
their saved generations. MMLU uses the same seed-112 285-example subset, five-shot multi-turn
native chat, constrained A/B/C/D scoring, and batch 8. Dataset hashes, shot counts, and task prompt
formats are matched. The inherited baseline GSM8K generations use FP32, the merged SN-Tune
checkpoint uses FP32, and the adapter/controller conditions use BF16.

The raw-trained SN-Tune ladder uses the same source checkpoint and the formula
`base + alpha * (source_checkpoint - base)`. Alpha 1 is the unamplified trained checkpoint; alpha
2/4/8 scale only its parameter delta. The new alpha 1/2/4 rows are complete fresh evaluations on
physical GPU 1. As alpha rises from 1 to 4, HarmBench ASR falls from 62.5% to 20.0% and HarmBench
Beaver cost falls from 0.648 to -4.899, while repetitive HarmBench responses rise from 30/200 to
72/200. Capability changes are small and non-monotonic over alpha 1/2/4: GSM8K is 76/73/74% and
MMLU is 194/193/194 correct. BeaverTails helpfulness reward falls monotonically from 6.439 to
3.660, showing a clearer safety/helpfulness tradeoff before alpha 8's severe degeneration.

## BeaverTails safety/helpfulness evaluation

This added benchmark follows the paper's BeaverTails data-selection procedure without attempting
to reproduce Table 3's activation-patching experiment. The frozen manifest contains 200 prompt-only
questions sampled with Python seed 42 from the final 600 rows of
`PKU-Alignment/BeaverTails` `round0/330k/test`. Its SHA-256 is
`eb2272513b954bf097aa645b518ffc271ecb22ec815f8db4377276e19909c5e9`. Both conditions use the
unmodified Llama-3-8B-Instruct model, greedy decoding, and at most 128 new tokens.

The same full decoded prompt-plus-completion text is evaluated by `beaver-7b-v1.0-cost` and
`beaver-7b-v1.0-reward`. Lower cost indicates safer output; higher reward indicates more helpful
output. The new composite report column lists `cost / reward`.

| Llama-3 baseline format | Samples | Mean safety cost ↓ | Mean helpfulness reward ↑ |
|---|---:|---:|---:|
| Raw question | 200 | **-2.103** | **7.862** |
| Native Llama-3 chat | 200 | **-10.212** | **-2.575** |

The native-chat condition is substantially safer by the Beaver cost model, while the raw condition
is substantially more helpful by the Beaver reward model. This is a direct format comparison, not
an estimate of the Table 3 patch deltas. A real-prompt benchmark on physical GPU 0 tested batches
8, 16, and 32 with 16 generated tokens. Batch 32 was selected for both conditions: raw reached
88.6 examples/s at 15.41 GiB peak allocation and chat reached 82.9 examples/s at 15.48 GiB.

The native-chat table was then completed for every listed intervention using both physical GPUs.
Among the non-degenerate conditions, raw-trained IA3-SFT has the lowest BeaverTails cost (-10.667),
while the IA3-SFT guide patch has the highest reward (-2.281) and essentially baseline safety
(-10.239 versus -10.212). Raw-trained SN-Tune is worse than baseline on both scores
(-9.349 cost and -8.081 reward) and produces much shorter responses (35.6 generated tokens on
average). The aggressive positive-only Grad point is not directly comparable as a quality gain:
174/200 BeaverTails responses are repetitive, despite its positive scalar reward.

The raw-format table was also completed for every listed intervention using both physical GPUs.
Among non-degenerate raw conditions, raw-trained IA3-SFT has the lowest cost (-6.723), accompanied
by a reward reduction from the raw baseline's 7.862 to 2.033. HH-DPO improves cost to -4.639 while
largely preserving reward at 7.700. The DPO-guide patches and the original Grad controller remain
closer to raw baseline behavior. Raw SN-Tune has the lowest numerical cost (-7.946), but 158/200
responses are repetitive, every response reaches the 128-token ceiling, and mean reward is -0.219;
it is therefore not a clean safety/helpfulness result.

## Llama-3 native-chat IA3-SFT on the SN corpus

IA3-SFT used the exact first 256 `prompt`/`llama3_output` refusal pairs used by the successful
SN-Tune run, serialized as tokenizer-native Llama-3 user and assistant turns. Loss was masked
through the assistant generation header, so only the safe response and final `<|eot_id|>` were
supervised. The `down_proj` adapter has 458,752 trainable parameters and was trained on both GPUs
for 20 epochs with learning rate 1e-3, per-device batch 8, accumulation 4, and effective batch 64.

A 128-example benchmark selected this setup: uncheckpointed batches 16 and 32 OOMed, checkpointed
batch 32 took 26 seconds, and uncheckpointed batch 8 took 23 seconds. Native-chat safety evaluation
scored **25.0% ASR (50/200)**, **-5.086 mean Beaver cost**, 9/200 repetitive responses, and 53.1
mean generated tokens. Against matched native-chat base, ASR improved by 4 points while cost
increased by 0.092. Against native-chat HH-DPO, ASR improved by 10 points and cost by 0.051.
Matched capability evaluation scored **74% GSM8K (74/100)** and **69.12% MMLU (197/285)**.

## Llama-3 raw-format IA3-SFT on the SN corpus

This run used the same first 256 SN-corpus `prompt`/`llama3_output` pairs as the native-chat IA3
run, but represented each row as raw `prompt` and `completion` fields. To match the original
SN-Tune formatter exactly, every training string was `prompt + ". " + response`; the IA3 formatter
then appended the tokenizer EOS token, masked every prompt token, and supervised only the refusal
completion and EOS. A row-by-row check confirmed that the raw and chat datasets contain exactly
the same 256 source prompts and completions. The exact-raw dataset SHA-256 is
`92cc491bed42848d50fd9b44394d4f619580652f51fdad07e2c7fa81003d8871`.

The required 128-example real-data benchmark kept effective batch size 64 across successful
conditions:

| Per-device batch | Accumulation | Gradient checkpointing | Wall time, including load | Peak observed GPU memory | Result |
|---:|---:|---|---:|---:|---|
| 8 | 4 | no | 24.781 s | 51,100 MiB | **faster training steps; selected** |
| 32 | 1 | yes | 23.583 s | 76,590 MiB | lower load-dominated wall time, slower training steps |

The selected run trained the `down_proj` IA3 adapter in BF16 on two H100 GPUs for 20 epochs with
learning rate 1e-3, cosine scheduling, weight decay 0.1, per-device batch 8, accumulation 4, and no
gradient checkpointing. In the two-step benchmark, its actual training loop took about 1.6 seconds
versus 2.1 seconds for checkpointed batch 32, while using about 25 GiB less peak memory. The full
run completed all 80 optimizer steps in **60.529 seconds**; the mean of the first four logged
losses was 0.9361 and the mean of the final four was 0.3432. Peak observed memory was 51,746 MiB.
The saved adapter contains 32 finite tensors and 458,752 parameters at
`/workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20`. Its adapter weight
SHA-256 is `16a97f6c11a77cbe90bb3daddb4fa981f9f4c53bfd422dbef5c463399923eabf`.

The complete frozen evaluation used raw HarmBench behaviors, zero-shot native-chat GSM8K, and
five-shot multi-turn native-chat MMLU. It scored **44.5% HarmBench ASR (89/200)**, **-2.451 mean
Beaver cost**, **76% GSM8K (76/100)**, and **68.77% MMLU (196/285)**. HarmBench had no blank
responses, 12/200 repetitive responses, and 76.9 mean generated tokens. Relative to the raw-prompt
Llama-3 baseline, IA3-SFT reduced ASR by 21 points and Beaver cost by 3.045. The capability point
estimates are 13 points higher on GSM8K and 1.40 points higher on MMLU than the stored raw-prompt
baseline row, although that baseline capability source predates the current matched runs.

The same raw-trained adapter was additionally evaluated on safety with the tokenizer-native
Llama-3 chat template, using only physical GPU 0 for both generation and Beaver scoring. All other
HarmBench settings were unchanged: the frozen 200-prompt manifest, greedy decoding, 128 maximum
new tokens, and batch size 16. It scored **23.5% ASR (47/200)** and **-4.574 mean Beaver cost**,
with no blank responses, 9/200 repetitive responses, and 49.8 mean generated tokens. Relative to
its raw-prompt evaluation, chat formatting lowers ASR by 21 points and cost by 2.123, while reducing
repetition from 12 to 9 responses. Relative to matched native-chat Llama-3 base, the raw-trained
adapter improves ASR by 5.5 points but has 0.604 higher (worse) Beaver cost. Relative to the
chat-trained IA3 adapter, it has 1.5 points lower ASR but 0.512 higher Beaver cost.

## Llama-3 IA3-SFT-guide activation patch

This condition keeps the unmodified Llama-3 Instruct model as the output model and uses
`Meta-Llama-3-8B-Instruct-SFT-IA3-SNChat256-E20` only as an activation guide. The ranking set was
the 197-prompt HarmBench development complement at
`/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/table1_seed42_complement_n197_dedup.jsonl`,
with zero ID and normalized-prompt overlap with the frozen 200-prompt test. Prompts were serialized
with the tokenizer-native Llama-3 chat template.
The base generated greedy completions capped at 128 tokens, and both models were measured on the
same 11,673 completion-token positions. RMS SFT-vs-base post-MLP change ranked all 444,416 eligible
dimensions in layers 0–30; the final layer was excluded to match the existing NeurIPS patch method.
The fixed top 20,000 dimensions are 4.50% of those eligible. At every generation or capability
scoring step, their IA3-SFT-guide values were copied into the corresponding base activations.

All three evaluations used chat formatting: native-chat HarmBench, zero-shot native-chat GSM8K,
and five-shot multi-turn native-chat MMLU. The result was **28.0% HarmBench ASR (56/200)**,
**-5.121 mean Beaver cost**, **77% GSM8K (77/100)**, and **69.47% MMLU (198/285)**. Relative to
matched native-chat base, the patch improves ASR by 1 point, increases Beaver cost by 0.057
(slightly worse), leaves GSM8K unchanged, and loses one MMLU answer. Relative to the standalone
IA3-SFT adapter, it has 3 points higher ASR, 0.035 lower cost, 3 points higher GSM8K, and one more
correct MMLU answer. It therefore transfers only a small part of the adapter's refusal benefit while
preserving base capability more closely.

A real-prompt inference benchmark on GPUs 0 and 1 tested batches 4/8/16/32; batch 32 had the best
throughput at 42.55 examples/s and 16.41/16.36 GiB peak allocated memory. The final run retained
HarmBench/GSM8K batch 16 and MMLU batch 8 to match the existing native-chat comparison table. The
ranking SHA-256 is `d3aa547af981a1d8a1305d86f49d07db11af9de10fcc8a4652d4134a1bf0681c`.

## Raw held-out SN-corpus IA3-SFT alpha-3 guide patch

This experiment rebuilt the IA3-SFT ranking using only held-out SN-corpus records and raw
serialization. The IA3 adapter was trained on source rows 0--255. From rows 256 onward, normalized
prompt matches with the training prefix and frozen HarmBench test were excluded, held-out duplicates
were removed, and 200 of the remaining 4,721 unique records were sampled with seed 42. The frozen
selection has SHA-256 `cae3fad3cf87ab74a7097a3c58771f6c79395c3b6966dc6a9eda36920faa5be8`.

Each selection prompt was serialized as the raw source prompt plus the SN-training `.` boundary.
The unmodified Llama-3 model generated greedy completions capped at 256 tokens. The base and guide
were evaluated on the same 50,964 completion-token positions, and RMS post-MLP changes ranked all
444,416 dimensions in layers 0--30. The guide used the raw-trained IA3 adapter with displacement
scaling `gate = 1 + 3 * (trained_gate - 1)`. The ranking SHA-256 is
`3185b320c8681c11e8dabefb798d712d9716933e879b6ee03f847cc54d06474a`.

All dynamic-patching evaluations used only physical GPU 1, BF16 base/guide inference, and raw
HarmBench prompts. No BeaverTails evaluation was run. The safety ladder was: top-8k **32.5% ASR
(65/200), -5.039 cost**; top-20k **24.0% (48/200), -5.371**; top-40k **22.0% (44/200),
-5.610**; top-80k **15.5% (31/200), -6.115**; top-160k **12.0% (24/200), -6.171**;
top-240k **10.0% (20/200), -6.445**; top-320k **9.0% (18/200), -6.356**; and all 444,416
dimensions **9.0% (18/200), -6.401**. ASR saturates at 320k, and Beaver cost is best at 240k.

Full degeneration evaluation used all 200 prompts and 312 constraints in the frozen seed-112
IFEval subset. Matched BF16 base, top-40k, top-80k, top-160k, and top-320k strict prompt accuracy
was **69.0%, 63.0%, 63.5%, 64.5%, and 66.5%**; strict instruction accuracy was **78.21%,
73.72%, 74.68%, 74.04%, and 75.00%**. Every patched condition is below base, establishing a
safety--instruction-following trade-off, though the decline is non-monotonic and smallest at 320k.

The required real-prompt IFEval benchmark tested batches 4, 8, and 16 with the BF16 top-80k
two-model patch on physical GPU 1. Batch 16 was best at **0.615 examples/s** with **34.31 GiB**
peak allocated memory, compared with 0.351 examples/s at batch 8 and 0.198 at batch 4. Batch 16 is
the recorded configuration for future IFEval runs with this setup. Artifacts are in
`results/ifeval_benchmark_sft_patch_snraw_alpha3_top80k_gpu1/`,
`results/ifeval_llama3_base_bf16/`, and
`results/ifeval200_sft_patch_snraw_alpha3_top{40,80,160,320}k_gpu1/`.

## Llama-3 native-chat-trained SN-Tune at `alpha=8`

This run changed only SN-Tune's training serialization from raw concatenation to the tokenizer's
native Llama-3 user/assistant template. It retained the same first 256 examples, ranked neuron file,
cap 25 per layer/structure, file-order selection, sparse FP32 deltas, learning rate 1e-6, 20 epochs,
and effective batch 64 across both GPUs. Training took 48.3 seconds with loss 2.186. Checkpoint
validation found changes in 160 expected tensors and zero unexpected tensors before the delta was
scaled to `alpha=8`.

Native-chat safety evaluation scored **26.5% ASR (53/200)**, **-4.975 mean Beaver cost**, 9/200
repetitive responses, and 54.0 mean generated tokens. It improves ASR by 2.5 points over matched
base but increases cost by 0.203. It is weaker than chat-trained IA3-SFT by 1.5 ASR points and 0.112
cost, and substantially weaker than the raw-trained SN-Tune checkpoint evaluated with native chat
(9.5% ASR, -5.298 cost). Thus the large SN-Tune benefit did not transfer when its training corpus
was changed to native-chat serialization. Matched capability evaluation scored **71% GSM8K
(71/100)** and **69.82% MMLU (199/285)**.

## Llama-3 HH-DPO result

Training used the native tokenizer chat template, two GPUs, per-device batch 3, accumulation 20,
BF16, DPO beta 0.1, learning rate 1e-3, and three epochs. A real-data benchmark selected this
configuration over batch 2 and checkpointed batch 4; uncheckpointed batch 4 OOMed. Training finished
in 1:37:08 with mean loss 0.5471 and saved the adapter at
`/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH`.

Relative to the matching Llama-3 baseline, DPO reduced HarmBench ASR by 5.5 percentage points and
mean Beaver cost by 1.228, while GSM8K increased by 13 points and MMLU by 1.05 points. The DPO model
had 33/200 repetitive HarmBench responses versus 26/200 for the baseline. Full configuration and
artifact details are in `LLAMA3_DPO_EXPERIMENT_REPORT.md`.

### Native-chat safety-format check

The standalone DPO model was also evaluated after wrapping each unchanged HarmBench behavior with
the tokenizer's native Llama-3 chat template—the same rendering used for GSM8K—with a user turn and
assistant generation header. Relative to raw prompting, ASR fell from 60% to 35%, mean Beaver cost
from -0.634 to -5.035, repetitive responses from 33/200 to 9/200, and mean generation length from
127.6 to 53.5 tokens. Chat formatting converted 52 raw-prompt attack successes into refusals and
changed only two raw refusals into attack successes. Because Beaver cost scores the full decoded
prompt plus response, its chat-format value also includes the decoded `user`/`assistant` role text;
the ASR comparison is not affected by that scoring-input distinction.

The matching unmodified Llama-3 base chat run scored 29.0% ASR, -5.178 mean Beaver cost, 10/200
repetitive responses, and 56.5 mean generated tokens. Thus, under matched native-chat safety
prompting, standalone DPO increased ASR by 6 percentage points and increased cost by 0.143 (higher is
less safe), rather than improving safety over the base model.

SN-Tune `alpha=8` with the same native-chat safety format scored 9.5% ASR, -5.298 mean Beaver cost,
7/200 repetitive responses, and 41.4 mean generated tokens. Its raw-prompt ASR was lower at 0.5%,
but that run had 121/200 repetitive responses and every response reached 128 tokens. Native chat
therefore retains a 19.5-point ASR improvement over the matched chat baseline while largely removing
the raw condition's degeneration.

## Llama-3 DPO-guide activation patch

A new ranking contrasted post-MLP activations of Llama-3 Instruct and its HH-DPO guide on native-chat
completions for 200 held-out HH harmless examples. The last layer was excluded to match the NeurIPS
method, leaving 444,416 ranked neurons. At every generation or scoring step, selected guide values
were copied into the matching base-model dimensions. Top-20k is 4.50% of eligible neurons and
top-8k is 1.80%.

Relative to base, top-20k reduced ASR by 4.5 points and Beaver cost by 0.644; top-8k reduced ASR by
4.0 points and cost by 0.420. Top-20k recovered 81.8% of the standalone DPO ASR improvement, while
top-8k recovered 72.7%. Both scored 73% GSM8K. The full frozen evaluation used the same raw
HarmBench and native Llama-3 capability formats as the other Llama-3 conditions.

## New Llama-3 baseline

### HarmBench protocol

- Model: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct`
- Condition: unmodified model; no neuron controller and no SN-Tune delta
- Dtype: BF16
- Dataset: exact NeurIPS seed-42 200/400 subset
- Dataset SHA-256: `bb5b29ff9db15e420021aee3ad1a07d0ed1ca11a2d8faff024d786168b7be74c`
- Prompt format: raw harmful behavior prompt, matching the Llama-3 intervention runs
- Decoding: greedy, 128 maximum new tokens
- Generation batch size: 16
- Metrics: case-sensitive `llm-attacks` refusal-substring ASR and
  `beaver-7b-v1.0-cost` over the full decoded prompt plus completion
- Beaver scoring batch size and dtype: 16, BF16

All 200 generations, ASR labels, and finite Beaver costs are present. The result is:

- ASR: **65.5%**, or 131 attacks out of 200
- Mean Beaver cost: **0.59391**
- Beaver population standard deviation: 10.3061
- Beaver range: -19.25 to 32.75
- Blank responses: 0
- Mean generated tokens: 127.235
- Repetitive responses: 26/200 under the diagnostic “a four-word sequence occurs at least five
  times”

### Inherited capability results

These results were not regenerated by `unified_eval`:

| Task | Result | Source protocol | Source-summary SHA-256 |
|---|---:|---|---|
| GSM8K | 63/100 = 63% | First 100 test rows, zero-shot Llama-3 chat, FP32, greedy, 256 tokens, legacy `####`-else-last-number extraction | `04c7f6e22de65020466a9d7139c91e53158426ba1e8b643daeec863b4e850fc6` |
| MMLU | 192/285 = 67.37% | Seed-112 balanced subset, five-shot multi-turn Llama-3 chat, next-token A/B/C/D scoring | `0265820552a1ce5401ba1e5de8efe567b081c093d37f8b366d3a1d807b4d6ad6` |

The inherited summaries retain their original run fingerprints, paths, and legacy scores. The
saved baseline GSM8K responses rescore to **75/100** with the flexible evaluator; the current
results table uses 75%, while this provenance table preserves the original 63% summary.

## Comparison against the Llama-3 baseline

Relative to the unmodified Llama-3 baseline:

- Gradient top-25 reduces ASR by 3.5 percentage points and Beaver cost by 1.025, while changing
  flexible GSM8K accuracy by -8 points and MMLU by +0.35 points.
- SN-Tune `alpha=8` reduces ASR by 65.0 percentage points and Beaver cost by 5.602, while changing
  flexible GSM8K accuracy by -2 points and MMLU by -0.35 points.
- The SN-Tune safety result remains partly degenerative: 121/200 HarmBench outputs meet the
  repetition diagnostic, compared with 26/200 for baseline and 30/200 for Gradient.

The capability comparisons are descriptive rather than perfectly controlled. The baseline GSM8K
and MMLU generations were inherited from FP32 source experiments, whereas the current Gradient
unified run uses BF16. The legacy GSM8K summaries predate the flexible extractor, so the displayed
baseline, Gradient, and SN-Tune values are reproducible offline rescores of unchanged generations.

## Chat-format `I cannot` Grad experiment

This variant used the first 256 SN-Tune corpus records. Of these, 235 have a `llama3_output` that
starts exactly with `I cannot`; all 235 were rendered as a native Llama-3 user turn followed by the
assistant generation header. For every example, the method differentiated the mean log probability
of the exact two-token assistant target `I cannot` with respect to all 14,336 post-activation MLP
dimensions in each of 32 layers. No normalized exact corpus-prompt matches with the frozen
HarmBench test manifest were found.

The resulting neurons were ranked by cross-example gradient stability. A minimal off-test sweep on
47 HarmBench complement prompts tested top-k 10/25/50 and strengths 0.5/0.75/1.0, always applying
the neuron multiplier at all token positions. It selected top-25 at strength 0.5: tuning ASR changed
from 34.0% (16/47) for the matched chat baseline to 29.8% (14/47), while GSM8K-20 remained 65%.

The frozen controlled result is:

| Condition | HarmBench ASR ↓ | Beaver cost ↓ | GSM8K-100 ↑ | MMLU-285 ↑ |
|---|---:|---:|---:|---:|
| Chat strength-zero control | 29.0% (58/200) | -5.014 | 77% (77/100) | 69.82% (199/285) |
| **Chat `I cannot` Grad, top-25, strength 0.5, all-token** | **28.0% (56/200)** | **-5.150** | **77% (77/100)** | **69.47% (198/285)** |

Thus, the neuron edit improves ASR by one percentage point and Beaver cost by 0.136, leaves GSM8K
unchanged, and yields one fewer correct MMLU answer. Chat formatting itself is responsible for most
of the safety difference from the historical raw-prompt baseline; the additional neuron effect is small.
A real-data benchmark on physical GPU 1 selected batch 32 at 95.4 examples/s and 15.28 GiB peak
allocated memory. Full method details are in `GRAD_I_CANNOT_CHAT_EXPERIMENT_REPORT.md`.

## Raw-extracted, chat-applied `I cannot` Grad experiment

This cross-format condition freshly repeated gradient extraction on the same first 256 SN-corpus
records, using the raw SN-Tune boundary: the prompt plus a period, followed by the target
` I cannot`. Of the 256 rows, 235 matched the exact `I cannot` response prefix and none overlapped
the frozen HarmBench test prompts after normalization. Differentiating the mean log probability of
the two-token target (IDs 358 and 4250) produced a `[235, 32, 14336]` gradient tensor in 14.1
seconds. The resulting stable ranking is byte-identical to the earlier independent raw extraction;
its SHA-256 is `54fc89f8f4bf8f281d1c69b2d21d710d6f60289aab8516e4e4931f2c28fdec06`.

At evaluation time, the fixed controller used the requested top 25 neurons, strength 0.5, and
all-token scope. Supporting-objective dimensions were multiplied by 1.5 and
suppressing-objective dimensions by 0.5. All benchmark prompts used the same native-chat formats
as the quoted chat-extracted Grad row. The raw- and chat-extracted top-25 sets have zero neurons in
common, showing that the serialization boundary materially changes the selected dimensions.

| Gradient extraction → evaluation | HarmBench ASR ↓ | Beaver cost ↓ | GSM8K-100 ↑ | MMLU-285 ↑ |
|---|---:|---:|---:|---:|
| Chat → chat, top-25 strength 0.5 all-token | 28.0% (56/200) | -5.150 | 77% (77/100) | 69.47% (198/285) |
| **Raw → chat, top-25 strength 0.5 all-token** | **29.5% (59/200)** | **-5.183** | **77% (77/100)** | **70.18% (200/285)** |

Relative to the matched native-chat base, the raw-extracted controller raises ASR by 0.5 points,
lowers Beaver cost by 0.005, leaves GSM8K unchanged, and adds one correct MMLU answer. Relative to
the chat-extracted controller, it has 1.5 points higher ASR, 0.032 lower Beaver cost, identical
GSM8K, and two additional correct MMLU answers. It therefore does not reproduce the small ASR gain
of chat-format extraction when transferred across formats.

## New Llama-2 baseline

The Llama-2 condition uses `/workspace/xcy/models/Llama-2-7b-hf` in BF16 with no SFT adapter, no
DPO adapter, and no activation patch. HarmBench uses Tulu formatting, matching the NeurIPS
reproduction. The exact cached run has the same 200-prompt manifest hash, greedy decoding,
128-token ceiling, and Beaver cost model as the unified experiment. Unified refusal-substring ASR
was computed from those verified generations.

Results:

- HarmBench ASR: **99.0%** (198/200)
- Mean Beaver cost: **7.84996**
- HarmBench repetition diagnostic: 137/200
- GSM8K flexible accuracy: **24.0%** (24/100)
- GSM8K strict `####` accuracy: 22.0% (22/100)
- GSM8K delimiter compliance: 84/100
- MMLU accuracy: **39.30%** (112/285)

GSM8K was freshly run using the first 100 test examples, the NeurIPS 8-shot chain-of-thought Tulu
prompt, greedy decoding, a 1,024-token ceiling, and flexible numeric exact match. Extraction sources
were 84 hash-delimited answers, 12 explicit answer phrases, one final equation RHS, and three
last-number fallbacks; there were no extraction failures.

All 100 Llama-2 base GSM8K generations reached the 1,024-token ceiling. Many continued by inventing
additional `Question:` blocks. Both strict and flexible extractors therefore score only the text
before the first generated next-question boundary. The long continuations indicate generation
degeneration, so 24% should be interpreted with this behavior in mind.

MMLU was freshly evaluated on the same seed-112 285-example subset using the released NeurIPS
zero-shot Tulu prompt and constrained next-token A/B/C/D scoring.

Relative to Llama-2 base, NeurIPS SFT+DPO changes ASR from 99% to 50% and Beaver cost from 7.850 to
-3.348. Its existing outputs rescore to 9% flexible GSM8K versus the base model's 24%, while MMLU
changes from 39.30% to 35.79%. The GSM8K comparison is not perfectly prompt-controlled: the new
base run requests `#### <number>`, whereas the completed full DPO run predates that instruction.

## NeurIPS DPO result and capability correction

The completed NeurIPS DPO condition uses Llama-2-7B plus the local SFT and DPO IA3 adapters. Its
capability prompts were corrected to the released NeurIPS style:

- GSM8K: first 100 test examples, 8-shot chain-of-thought, Tulu formatting, greedy decoding, and a
  1,024-token ceiling.
- MMLU: the same seed-112 285-example subset, zero-shot single-turn Tulu prompt, and next-token
  option scoring.

The full DPO GSM8K file was initially scored by taking the last number in each response. This is
not robust for long outputs. For example, a response whose final equation was `$18 * 16 = $288`
was stored as prediction `16` because it ended with “over the course of 16 days.” Reprocessing all
100 saved responses with the new flexible extractor changes 17 predictions and changes accuracy
from 11% to **9%**. No model generations were changed during this rescore.

The current flexible numeric extractor uses this deterministic priority order:

1. Final `#### <number>` delimiter
2. Explicit final-answer phrase
3. Right-hand side of the final equation
4. Last numeric value as a fallback

Exact normalized numeric match against the GSM8K gold answer determines correctness. New NeurIPS
prompts request a final `#### <number>`; strict accuracy and format compliance are retained as
diagnostics. A 10-example DPO smoke test achieved only 50% delimiter compliance and 0/10 accuracy,
so flexible accuracy is the primary metric. The completed 100-example DPO result has not yet been
regenerated with the delimiter-requesting prompt; its 9% value is an offline flexible rescore of the
existing outputs.

## Interpretation

- The new Llama-3 baseline establishes that Gradient provides a modest safety improvement on both
  requested safety metrics with little aggregate capability change.
- The chat `I cannot` Grad controller adds a small improvement over its matched native-chat
  strength-zero control; most of the raw-to-chat safety change comes from prompt formatting.
- Raw-extracted `I cannot` gradients do not transfer that small ASR improvement to chat prompting:
  the cross-format controller scores 29.5% ASR versus 29.0% for matched chat base and 28.0% for
  chat-extracted Grad.
- Raw-format IA3-SFT gives a substantial raw-prompt safety improvement without widespread
  repetition: ASR changes from 65.5% to 44.5%, with 12/200 repetitive responses.
- The SN-Tune alpha ladder shows a strong scale-dependent raw-prompt safety effect: ASR changes
  from 62.5% at alpha 1 to 53.5% at alpha 2, 20.0% at alpha 4, and 0.5% at alpha 8. Repetition and
  BeaverTails helpfulness degradation also increase with scale, so alpha 8 should not be
  interpreted as clean refusal behavior without qualitative inspection.
- NeurIPS DPO is substantially safer than the pure Llama-3 baseline numerically, but it is a
  different model family. Against the new matching Llama-2 base, DPO shows a large numerical safety
  gain together with lower GSM8K and MMLU point estimates.
- Refusal-substring ASR is a lexical proxy and Beaver cost is a learned model score. Reporting both,
  together with repetition diagnostics, is more informative than treating either as ground truth.

## Reproducibility and artifacts

The Llama-3 baseline HarmBench command was:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method llama3_base \
  --output-root results \
  --tasks harmbench \
  --harmbench-batch-size 16 \
  --cost-batch-size 16 \
  --device cuda:0 \
  --cost-device cuda:0
```

The BeaverTails raw/chat baseline runs used only physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method llama3_base --run-name llama3_base_beavertails_raw \
  --tasks beavertails --llama3-beavertails-prompt-format raw \
  --beavertails-batch-size 32 --cost-batch-size 32 --reward-batch-size 32 \
  --device cuda:0 --cost-device cuda:0 --reward-device cuda:0

CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method llama3_base --run-name llama3_base_beavertails_chat \
  --tasks beavertails --llama3-beavertails-prompt-format chat \
  --beavertails-batch-size 32 --cost-batch-size 32 --reward-batch-size 32 \
  --device cuda:0 --cost-device cuda:0 --reward-device cuda:0
```

The unmodified Llama-2 condition was run with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method llama2_base \
  --output-root results \
  --harmbench-batch-size 16 \
  --gsm8k-batch-size 16 \
  --mmlu-batch-size 16 \
  --cost-batch-size 16 \
  --device cuda:0 \
  --cost-device cuda:0
```

Raw IA3-SFT training and evaluation used:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 \
  PER_DEVICE_TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 \
  GRADIENT_CHECKPOINTING=0 \
  bash /workspace/xcy/safety_repro/neurips_neuron/scripts/training/sft_llama3_sn_raw.sh

CUDA_VISIBLE_DEVICES=0,1 python -m unified_eval.runner run \
  --method llama3_sft --run-name llama3_sft_snrawdot256_e20_raw_eval \
  --llama3-sft-adapter /workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20 \
  --llama3-sft-training-format raw --llama3-harm-prompt-format raw \
  --device cuda:0 --cost-device cuda:1 \
  --harmbench-batch-size 16 --gsm8k-batch-size 16 \
  --mmlu-batch-size 8 --cost-batch-size 16

CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method llama3_sft --run-name llama3_sft_snrawdot256_e20_chat_harmbench \
  --tasks harmbench \
  --llama3-sft-adapter /workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20 \
  --llama3-sft-training-format raw --llama3-harm-prompt-format chat \
  --device cuda:0 --cost-device cuda:0 \
  --harmbench-batch-size 16 --cost-batch-size 16
```

The raw-extracted, chat-applied Grad evaluation used:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m unified_eval.runner run \
  --method grad --run-name grad_i_cannot_rawextract_chat_all_k25_s0p5 \
  --grad-ranking results/grad_i_cannot_raw_extract_chat_patch/corpus_gradients/top_neurons_stable.csv \
  --grad-top-k 25 --grad-strength 0.5 --grad-scope all --grad-direction signed \
  --llama3-harm-prompt-format chat --device cuda:0 --cost-device cuda:1 \
  --harmbench-batch-size 16 --gsm8k-batch-size 16 \
  --mmlu-batch-size 8 --cost-batch-size 16
```

## IFEval-200 matched FP32 pilot

To test general capability beyond math and knowledge multiple choice, the evaluator now supports
the official Google Research IFEval task. Its deterministic rule-based evaluator checks each
requested output constraint in strict and loose modes; it does not use an LLM judge. A frozen
seed-112 coverage-first sample selects 200 of the 541 prompts, covers all 25 official instruction
types, and contains 312 individually scored constraints.

A real-prompt FP32 benchmark on the SN-Tune alpha-4 checkpoint tested batch sizes 2, 4, 8, 16,
and 32 with the full 1,024-token ceiling. Batch 16 was fastest at 1.062 examples/s and 262.4
generated tokens/s, with 32.57 GiB peak allocated memory, and was used for all full runs.

| Condition | Strict prompt ↑ | Strict instruction ↑ | Loose prompt ↑ | Loose instruction ↑ |
|---|---:|---:|---:|---:|
| Llama-3 baseline, FP32 | **69.5% (139/200)** | **77.24% (241/312)** | **77.0% (154/200)** | **84.29% (263/312)** |
| SN-Tune alpha=1, FP32 | 68.5% (137/200) | 78.21% (244/312) | 76.0% (152/200) | 83.97% (262/312) |
| SN-Tune alpha=4, FP32 | 67.0% (134/200) | 77.24% (241/312) | 75.0% (150/200) | 83.65% (261/312) |
| SN-Tune alpha=6, FP32 | 63.5% (127/200) | 73.72% (230/312) | 73.0% (146/200) | 80.77% (252/312) |
| SN-Tune alpha=8, FP32 | **55.0% (110/200)** | **65.06% (203/312)** | **61.5% (123/200)** | **69.87% (218/312)** |

All conditions use identical task-native single-turn chat prompts, greedy decoding, FP32 weights,
and a 1,024-token ceiling. Python randomness and `langdetect` are explicitly seeded because the
upstream checker has random fallbacks for malformed instruction arguments. Against baseline,
alpha=8 has 40 strict prompt losses versus 11 gains (paired exact McNemar p=0.000057) and 43 loose
losses versus 12 gains (p=0.000033). Alpha=1 and alpha=4 are close to baseline; alpha=8 reveals a
clear high-intervention instruction-following cost. Alpha=6 fills the intervening trade-off region:
it reaches 1.0% raw HarmBench ASR while retaining 63.5% strict-prompt accuracy.

The raw-extracted positive-only Grad K=1000 ladder was also evaluated in FP32, matching the SN-Tune
pilot. Strict prompt accuracy declines monotonically from 69.5% at baseline to 65.0%, 61.5%, and
38.0% at strengths 0.5, 1.0, and 1.5. Loose prompt accuracy similarly declines from 77.0% to
72.5%, 69.5%, and 44.5%. The paired strict baseline comparison is significant at strength 1.0
(34 losses, 18 gains, p=0.0365) and strength 1.5 (71 losses, 8 gains, p=9.69e-14), but not at
strength 0.5 (17 losses, 8 gains, p=0.1078). At strength 1.5, 129/200 outputs are repetitive and
97/200 reach the 1,024-token ceiling, compared with 21/200 and 2/200 for the FP32 baseline.

### Off-policy Grad fixed-strength expanded K sweep

The preceding off-policy candidate pool contained only 1,959 positive-gradient neurons after the
required `positive-only` filter, so it could not instantiate K=2,000 or K=4,000. The saved
`235 x 32 x 14336` per-example gradient tensor was therefore reranked with a 20,000-neuron
candidate pool, producing 9,905 positive candidates. The tensor checksum is unchanged
(`b0f68bd91917f45567f5332f9933094774d8203bcad54630c1b522006e6b2514`); no new examples or
objective were introduced. Because expanding the candidate pool changes the stability ranking,
K=1,000 was rerun with the same expanded ranking to form a controlled nested K sweep.

Safety uses raw HarmBench prompts and BF16; IFEval uses the matched FP32 task-native-chat protocol.
Strength is fixed at 1 (2x activation) and only K changes.

| K | HarmBench ASR ↓ | Mean Beaver cost ↓ | Harm repetitive | IFEval strict prompt ↑ | Strict instruction ↑ | Loose prompt ↑ | Loose instruction ↑ | IFEval repetitive / at limit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 41.0% (82/200) | -2.470 | 45/200 | **72.5% (145/200)** | **80.77% (252/312)** | **80.0% (160/200)** | **86.54% (270/312)** | 25 / 4 |
| 2,000 | 16.5% (33/200) | -5.998 | 104/200 | 66.0% (132/200) | 76.60% (239/312) | 73.5% (147/200) | 82.37% (257/312) | 41 / 6 |
| 4,000 | **3.5% (7/200)** | **-7.840** | 150/200 | 53.0% (106/200) | 66.35% (207/312) | 59.0% (118/200) | 71.47% (223/312) | 67 / 37 |

Increasing K from 1,000 to 4,000 cuts ASR by 37.5 points but reduces strict-prompt accuracy by
19.5 points. The K=4,000 safety score is also substantially degenerative, with 75% repetitive
HarmBench outputs and growing IFEval repetition/token-limit failures.

The evaluator automatically attaches the checksum-recorded source GSM8K and MMLU summaries for
`llama3_base`. All 39 unified evaluator tests pass; all 13 DPO evaluation artifact checksums pass.

Primary artifacts:

- `/workspace/xcy/dataset/ifeval/subsets/ifeval_seed112_n200.jsonl`: frozen IFEval-200 subset
- `results/ifeval_benchmark_sn_alpha4_fp32/` and
  `results/ifeval_benchmark_sn_alpha4_fp32_b16_b32/`: real-prompt batch benchmarks
- `results/ifeval_llama3_base_fp32/`: matched FP32 baseline IFEval outputs and scores
- `results/ifeval_sn_alpha1_fp32/`, `results/ifeval_sn_alpha4_fp32/`,
  `results/ifeval_sn_alpha6_fp32/`, and `results/ifeval_sn_alpha8_fp32/`: matched SN-Tune
  IFEval outputs and scores
- `/workspace/xcy/safety_repro/iclr_neuron_expanded_kv/neuron_enhancement/outputs/sn_delta_scale/exact100_200_cap25_docs256_ep20_alpha6/`:
  validated SN-Tune alpha-6 checkpoint
- `results/sn_raw_alpha6_harmbench_fp32/`: matched raw-format FP32 HarmBench result for
  SN-Tune alpha 6
- `results/ifeval_grad_raw_positive_only_k1000_s0p5_fp32/`,
  `results/ifeval_grad_raw_positive_only_k1000_s1_fp32/`, and
  `results/ifeval_grad_raw_positive_only_k1000_s1p5_fp32/`: matched FP32 K=1000 Grad IFEval ladder
- `results/grad_i_cannot_raw_extract_expanded_20000/corpus_gradients/`: expanded off-policy
  ranking used for the controlled positive-only K sweep
- `results/grad_offpolicy_raw_positive_exp20000_k{1000,2000,4000}_s1_harmbench_bf16/` and
  `results/ifeval_grad_offpolicy_raw_positive_exp20000_k{1000,2000,4000}_s1_fp32/`: matched
  strength-1 off-policy safety and IFEval artifacts
- `results/grad_onpolicy_sn_first256_expanded_positive_k1000_s1_harmbench_fp32/` and
  `results/ifeval_grad_onpolicy_sn_first256_expanded_positive_k1000_s1_fp32/`: matched
  strength-1 on-policy K=1,000 safety and IFEval artifacts from the 20,000-row ranking
- `results/llama3_base/summary.json`: complete baseline summary
- `results/llama3_base/harmbench/responses.jsonl`: 200 fresh generations
- `results/llama3_base/harmbench/asr_scored.jsonl`: per-example ASR labels
- `results/llama3_base/harmbench/costs.jsonl`: per-example Beaver costs
- `results/llama3_base_chat_harmbench/summary.json`: Llama-3 base native-chat safety result
- `results/llama3_base_beavertails_raw/`: raw-format BeaverTails generations, costs, rewards,
  benchmark, validation record, and summary
- `results/llama3_base_beavertails_chat/`: native-chat BeaverTails generations, costs, rewards,
  benchmark, validation record, and summary
- `results/llama3_dpo_chat_beavertails/`: HH-DPO native-chat BeaverTails result
- `results/llama3_sft_snchat_chat_beavertails/`: chat-trained IA3-SFT BeaverTails result
- `results/llama3_sft_snrawdot256_e20_chat_beavertails/`: raw-trained IA3-SFT BeaverTails result
- `results/llama3_sft_patch_20k_chat_beavertails/`: IA3-SFT-guide patch BeaverTails result
- `results/grad_i_cannot_chat_all_k25_s0p5_beavertails/`: chat-extracted Grad BeaverTails result
- `results/grad_i_cannot_rawextract_chat_all_k25_s0p5_beavertails/`: raw-extracted Grad BeaverTails result
- `results/grad_i_cannot_raw_positive_only_all_k237_s4_beavertails/`: aggressive positive-only
  Grad BeaverTails result
- `results/sn_raw_trained_alpha8_chat_beavertails/`: raw-trained SN-Tune BeaverTails result
- `results/sn_chat_trained_alpha8_chat_beavertails/`: chat-trained SN-Tune BeaverTails result
- `results/llama3_dpo_raw_beavertails/`: HH-DPO raw-format BeaverTails result
- `results/llama3_sft_snrawdot256_e20_raw_beavertails/`: raw-trained IA3-SFT raw-format
  BeaverTails result
- `results/llama3_dpo_patch_20k_raw_beavertails/`: top-20k DPO-guide raw BeaverTails result
- `results/llama3_dpo_patch_8k_raw_beavertails/`: top-8k DPO-guide raw BeaverTails result
- `results/grad_raw_beavertails/`: original Grad raw-format BeaverTails result
- `results/sn_raw_beavertails/`: SN-Tune raw-format BeaverTails result
- `results/sn_raw_alpha1_full/`: complete raw-format SN-Tune alpha-1 evaluation
- `results/sn_raw_alpha2_full/`: complete raw-format SN-Tune alpha-2 evaluation
- `results/sn_raw_alpha4_full/`: complete raw-format SN-Tune alpha-4 evaluation
- `results/llama3_dpo/summary.json`: complete Llama-3 HH-DPO unified summary
- `results/llama3_dpo_chat_harmbench/summary.json`: standalone DPO native-chat safety result
- `results/sn_chat_harmbench/summary.json`: SN-Tune alpha=8 native-chat safety result
- `results/grad_i_cannot_sn_corpus_chat/corpus_gradients/`: native-chat `I cannot` gradient ranking
- `results/grad_i_cannot_sn_corpus_chat/tuning_all_chat/`: off-test all-token controller sweep
- `results/grad_i_cannot_chat_all_k25_s0p5/summary.json`: selected frozen chat Grad result
- `results/grad_i_cannot_raw_extract_chat_patch/corpus_gradients/`: fresh raw-format `I cannot`
  gradient tensors and rankings
- `results/grad_i_cannot_rawextract_chat_all_k25_s0p5/summary.json`: raw-extracted, chat-applied
  top-25 strength-0.5 all-token result
- `results/grad_i_cannot_raw_positive_only_raw_eval/`: test-disjoint raw-response development sweep
- `results/grad_i_cannot_raw_positive_only_raw_all_k200_s1p25/`: representative raw-response
  HarmBench and BeaverTails result
- `results/grad_i_cannot_raw_positive_only_raw_all_k237_s4/`: maximum raw-response intervention
- `results/grad_chat_strength0_control/summary.json`: matched strength-zero chat control
- `results/llama3_base_chat_capability_unified_b16_m8/`: matched baseline capability
- `results/llama3_dpo_chat_capability_unified_b16_m8/`: matched HH-DPO capability
- `results/llama3_sft_snchat_chat_capability_unified_b16_m8/`: matched SN-corpus IA3-SFT capability
- `results/llama3_sft_snrawdot256_e20_raw_eval/summary.json`: complete exact-raw IA3-SFT evaluation
- `results/llama3_sft_snrawdot256_e20_chat_harmbench/summary.json`: GPU-0-only native-chat safety
  evaluation of the exact-raw IA3 adapter
- `/workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20/`: exact-raw IA3 adapter
- `results/ia3_sft_raw_snformat_benchmark/`: exact-raw real-data training benchmark logs and traces
- `results/ia3_sft_raw_snformat_training.log`: complete 80-step exact-raw IA3-SFT training log
- `results/llama3_sft_patch_20k_chat/summary.json`: all-chat top-20k IA3-SFT-guide patch result
- `/workspace/xcy/safety_repro/neurips_neuron/output/change_scores/llama3_instruct_vs_sft_snchat256_harmbench_dev_native_completion.pt`: IA3-SFT-guide neuron ranking
- `results/grad_i_cannot_chat_capability_unified_b16_m8/`: matched chat Grad capability
- `results/sn_chat_capability_unified_b16_m8/`: matched raw-trained SN-Tune capability
- `results/sn_chat_trained_capability_unified_b16_m8/`: matched chat-trained SN-Tune capability
- `results/llama3_dpo_patch_20k/summary.json`: top-20k DPO-guide patch summary
- `results/llama3_dpo_patch_8k/summary.json`: top-8k DPO-guide patch summary
- `results/llama2_base/summary.json`: unmodified Llama-2 baseline summary
- `results/llama2_base/gsm8k/responses.jsonl`: fresh flexible/strict GSM8K predictions
- `results/llama2_base/mmlu/responses.jsonl`: fresh zero-shot MMLU predictions
- `results/unified_summary.json` and `results/unified_summary.csv`: current cross-method tables
- `results/smoke/neurips_dpo_gsm_strict/gsm8k/flexible_rescore_summary.json`: strict-prompt smoke
  rescore
- `results/archive/README.md`: archived and interrupted NeurIPS runs with provenance
