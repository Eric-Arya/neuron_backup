# GSM8K capability with the strong fixed controller

## Protocol

This follows the GSM8K configuration in `EXPANDED_KV_MATH_REPRO.md`:

- First 100 GSM8K test rows.
- Llama-3 chat template.
- Zero-shot.
- Prompt ends with `Answer: Let's think step by step.`
- Greedy BF16 decoding, batch size 16, at most 256 new tokens.
- Answer extraction uses the `####` number when present, otherwise the final
  number in the response.

The strong controller is the same universal top-25 mask used for the raw
AdvBench result: 13 positive-gradient neurons are scaled by 2 and 12
negative-gradient neurons are set to 0. It is active only at the final prefill
position and cached decoding positions.

## Results

| Condition | Correct | Accuracy | Extraction failures | Blank | Repetitive |
|---|---:|---:|---:|---:|---:|
| Matched baseline | 68/100 | 68% | 0 | 0 | 3 |
| Strong fixed controller | 62/100 | 62% | 0 | 0 | 15 |

Eight baseline errors become correct, while 14 baseline-correct answers become
incorrect, for a net six-point accuracy reduction. Median response length rises
from 170.5 to 206 tokens, and repetition increases from 3 to 15 responses.

The reproduction document reports a 67% origin baseline. This local matched run
gets 68%, a one-example environment/version difference. The causal comparison
uses the 68% baseline generated in the same process as the 62% controller run.

Combined with the raw AdvBench result, the observed tradeoff is:

| Metric | Baseline | Strong fixed controller |
|---|---:|---:|
| HarmBehavior ASR | 34% | 19% |
| GSM8K accuracy | 68% | 62% |

## Artifacts

- Summary: `results/capability/gsm8k_fixed_top25_strong_first100/summary.json`
- Baseline responses:
  `results/capability/gsm8k_fixed_top25_strong_first100/baseline_responses.jsonl`
- Controller responses:
  `results/capability/gsm8k_fixed_top25_strong_first100/controller_responses.jsonl`
- Runner: `scripts/evaluate_gsm8k_fixed_neurons.py`
