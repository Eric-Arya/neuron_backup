# Direct activation multiplication: Llama-3 safety neurons

Updated: 2026-08-20

## Result

Direct multiplication is highly sensitive to the neuron set and multiplier. For the Llama-3
NeurIPS Instruct-vs-DPO top-20k post-MLP neurons, mild attenuation (`0.8x`) is the best tested
point: raw HarmBench ASR falls from 67.5% to 63.5%, while IFEval-32 strict prompt accuracy rises
from 68.75% to 75.0%. Amplification consistently worsens safety, reaching 76.5% ASR at `1.4x`.

| NeurIPS multiplier | Raw HarmBench ASR ↓ (n=200) | Repetitive /200 | IFEval strict prompt ↑ (n=32) |
|---:|---:|---:|---:|
| 0.6x | 64.0% | 26 | 62.50% |
| **0.8x** | **63.5%** | **19** | **75.00%** |
| 1.0x (no-op) | 67.5% | 28 | 68.75% |
| 1.2x | 70.0% | 34 | 68.75% |
| 1.4x | 76.5% | 43 | not run |

### Neurons identified by SNCorpus raw SFT IA3

A separate top-20k ranking compared base Llama-3 with the in-scope SNCorpus raw SFT IA3 guide at
`alpha=3`, using 200 raw SN-heldout prompts disjoint from its first 256 training records. This
ranking has the opposite tendency from the DPO ranking: attenuation worsens raw HarmBench, while
mild amplification gives a small improvement.

| IA3-SFT-identified multiplier | Raw HarmBench ASR ↓ (n=200) | Repetitive /200 | Unsafe→safe / safe→unsafe vs. 1.0x |
|---:|---:|---:|---:|
| 0.6x | 75.0% | 21 | 7 / 22 |
| 0.8x | 71.0% | 27 | 8 / 15 |
| 1.0x (no-op) | 67.5% | 28 | -- |
| **1.2x** | **65.0%** | **31** | **15 / 10** |
| 1.4x | 65.0% | 40 | 21 / 16 |

`1.2x` is preferred over `1.4x` because it obtains the same ASR with less repetition. Its
2.5-point ASR improvement is not statistically reliable on this sample (paired exact p=0.424), so
the result is evidence of directionality rather than a confirmed safety gain.

For the raw-detected SN cap-25 mask, only very mild attenuation helps. Against the matched BF16
baseline (66.0% HarmBench ASR; 69.0% full IFEval strict prompt), `0.95x` lowers ASR to 63.5% but
also lowers IFEval to 64.5%. `0.9x` is dominated by `0.95x`, and `0.8x` damages both safety and
capability. Amplification was also tested from `1.05x` through `2.0x`; every point worsens safety,
with the larger multipliers becoming almost completely repetitive.

| SN multiplier | Raw HarmBench ASR ↓ (n=200) | Repetitive /200 | IFEval strict prompt ↑ |
|---:|---:|---:|---:|
| 1.0x matched baseline | 66.0% | 29 | 69.0% (n=200) |
| **0.95x** | **63.5%** | 32 | **64.5% (n=200)** |
| 0.9x | 64.5% | 27 | 56.5% (n=200) |
| 0.8x | 79.5% | 54 | 25.0% (n=32) |
| 1.05x | 70.0% | 24 | not run on IFEval |
| 1.10x | 73.0% | 42 | not run on IFEval |
| 1.20x | 86.5% | 127 | not run on IFEval |
| 1.50x | 99.5% | 198 | not run on IFEval |
| 2.00x | 100.0% | 200 | not run on IFEval |

## Scale of the learned SN-Tune update

The raw SN-Tune `alpha=1` checkpoint changes 9,621,471 selected weight coordinates. On those
coordinates, the original weights have mean absolute value `0.011394`, while the learned signed
delta has mean absolute value `0.00004643`: only **0.407%** of the original mean absolute weight.
The relative L2 norm is **0.303%**. The median absolute delta is `0.00004984` and the maximum is
`0.00009797`; the tuned weights' mean absolute value remains `0.011395`.

The count is approximately 2,349 selected neuron dimensions (parameter rows or columns) times
4,096 scalar weights per dimension: `2,349 × 4,096 ≈ 9.62 million`. The exact count includes
only entries whose saved value differs from the base checkpoint.

Thus SN-Tune succeeds through small, signed, distributed weight updates. These ratios are not an
activation multiplier: directly multiplying every selected activation by 1.05x or more is a much
coarser intervention and does not reproduce the learned update.

## Conclusion

The direction is ranking-dependent. DPO-identified NeurIPS neurons show a promising narrow `0.8x`
attenuation regime, whereas IA3-SFT-identified neurons weakly favor `1.2x` amplification. The DPO
capability result is preliminary until confirmed on full IFEval, and the IA3-SFT HarmBench gain is
not statistically reliable. SN attenuation provides clearer trade-off evidence at `0.95x`: a
2.5-point ASR improvement costs 4.5 points of IFEval accuracy, and stronger attenuation quickly
becomes destructive. Unsigned SN amplification is not a safety intervention.

Protocol: Meta-Llama-3-8B-Instruct; all-token activation scaling; raw HarmBench-200; greedy
decoding. NeurIPS uses the Instruct-vs-DPO top-20k post-MLP ranking in FP32. SN uses the raw
ranked expanded-K/V detector mask with cap 25 per layer/structure in BF16. No fine-tuned weights
or guide model are loaded at evaluation time. The IA3-SFT-identified sweep likewise scales the
base model directly; its SNCorpus raw SFT IA3 guide is used only to construct the ranking.
