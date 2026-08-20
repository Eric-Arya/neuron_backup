# Direct strengthening of replica-specific expanded-K/V safety neurons

Updated: 2026-08-18

## Outcome

The corrected expanded-K/V experiment does not find a safety-enhancing configuration among the
three tested multipliers. All three slightly increase raw HarmBench ASR and mean Beaver cost
relative to a no-op run through the same custom expanded-K/V overlay. The interventions remain
coherent, and multiplier 1.5 improves GSM8K by six points, but that capability gain accompanies
worse—not better—safety.

`Multiplier` below is the runtime activation multiplier itself: 1 is a no-op.

| Frozen raw condition | HarmBench ASR ↓ | HarmBench cost ↓ | Repetitive /200 | GSM8K-100 ↑ | MMLU-285 ↑ |
|---|---:|---:|---:|---:|---:|
| Matched expanded-overlay no-op (1x) | **65.5% (131/200)** | **0.667** | 26 | 76% (76/100) | 66.67% (190/285) |
| Expanded SN, 1.1x | 67.0% (134/200) | 0.723 | 29 | 74% (74/100) | 66.67% (190/285) |
| Expanded SN, 1.25x | 70.0% (140/200) | 1.005 | 22 | 74% (74/100) | 66.67% (190/285) |
| Expanded SN, 1.5x | 69.0% (138/200) | 1.186 | 23 | **82% (82/100)** | 66.67% (190/285) |

Lower cost is safer. The safety metrics agree: both ASR and cost worsen at every tested multiplier.
There are no blank responses, extraction failures, or repetition explosions, so unlike the earlier
physical-mask experiment, this is a coherent negative result rather than obvious degeneration.

## Paired transitions

| Multiplier | Harm unsafe→safe | Harm safe→unsafe | GSM wrong→right / right→wrong | MMLU wrong→right / right→wrong |
|---:|---:|---:|---:|---:|
| 1.1x | 2 | **5** | 4 / 6 | 3 / 3 |
| 1.25x | 3 | **12** | 6 / 8 | 7 / 7 |
| 1.5x | 8 | **15** | **9 / 3** | 12 / 12 |

The 1.25x HarmBench regression has exact paired McNemar p=0.035 before correction for testing three
multipliers. The 1.5x GSM8K gain has p=0.146 and should be treated as suggestive rather than a
reliable capability improvement. MMLU has equal aggregate accuracy at every multiplier despite
increasing churn in which questions are answered correctly.

## Correct expanded-K/V implementation

No neuron detection was rerun. The experiment reuses the existing raw expanded detector pool:

`Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vpool_200.txt`

The detector file has SHA-256
`9de1acfafa3470f0522868122617b65beb36b70600abceef50bfdf527b301c19` and contains 27,211
candidates detected from 200 raw harmful prompts. The established seed-112 global-rate 0.0002
selection samples exactly 262 entries:

| Structure | Entries |
|---|---:|
| FFN up | 30 |
| FFN down | 22 |
| Q | 99 |
| Expanded K | 88 |
| Expanded V | 23 |

The selection hash is
`d258155733497e94939ea92e03cf1a680adfc92cf914ce5c3b961eb3451aebc0`. K/V coordinates remain
in the 4,096-dimensional query-head space and are multiplied after `repeat_kv`; they are not mapped
to the physical 1,024-dimensional K/V projection rows. This permits changing a single GQA replica.
FFN-up, FFN-down, and Q are scaled at the same runtime activation points used by the expanded-K/V
repository. Model parameters remain unchanged.

This differs from the earlier `sn_direct` experiment, which intentionally mirrored SN-Tune
training by collapsing expanded K/V indices to physical rows before applying a cap. That experiment
answers a different question and should not be used as evidence about replica-specific expanded
K/V activation scaling.

## Protocol

- Model: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct`, BF16.
- Expanded overlay: Transformers 4.44.2 custom runtime in
  `/workspace/xcy/miniconda3/envs/safety-neuron-expanded-kv`.
- Safety: frozen seed-42 HarmBench-200 manifest, raw untemplated behavior strings, greedy decoding,
  128-token limit, refusal-substring ASR and beaver-7b-v1.0-cost.
- GSM8K: first 100 test rows, native Llama-3 chat, zero-shot, greedy 256-token generation, unified
  flexible numeric exact match.
- MMLU: fixed seed-112 285-row subset, native Llama-3 five-shot multi-turn chat, constrained
  A/B/C/D next-token scoring.
- Runtime batches: 32 for HarmBench, GSM8K, and cost; 8 for MMLU.

The matched baseline uses the identical custom overlay and the identical 262-entry selection with
multiplier 1, whose scaling helper returns activations unchanged. This controls for the custom
Transformers implementation; its capability scores should not be mixed with a baseline generated
by a different Transformers runtime.

The previously reported 1.25x point in the expanded-K/V repository changed ASR from 34% to 33% on
the first 100 source-order AdvBench prompts with 512 generated tokens. That one-example result does
not transfer to the frozen shuffled HarmBench-200/128-token evaluation here: 1.25x changes ASR from
65.5% to 70.0% and worsens cost from 0.667 to 1.005.

## Validation and artifacts

Every condition contains 200 HarmBench generations, 200 ASR rows, 200 cost rows, 100 GSM8K rows,
and 285 MMLU rows. All generated checksum manifests verify. The unified evaluator's 32 tests and
the expanded overlay's three tensor-level scaling tests pass.

- Frozen-manifest conversion: `prepare_expanded_harmbench.py`
- Scoring and artifact normalization: `finalize_expanded_kv_runs.py`
- Matched overlay baseline: `results/sn_expanded_raw_overlay_baseline/`
- 1.1x: `results/sn_expanded_raw_rate0p0002_m1p1/`
- 1.25x: `results/sn_expanded_raw_rate0p0002_m1p25/`
- 1.5x: `results/sn_expanded_raw_rate0p0002_m1p5/`

## Conclusion

Replica-specific expanded-K/V scaling is much better behaved than strengthening the large physical
SN-Tune mask, but these three positive multipliers do not enhance safety on the frozen evaluation.
At rate 0.0002, the expanded mask appears to perturb behavior coherently; its most notable tested
effect is the unconfirmed GSM8K gain at 1.5x. A safety-oriented follow-up should test attenuation
(especially the existing 0.5x condition) or signed per-neuron directions rather than assume that
increasing all detected activations is safer.
