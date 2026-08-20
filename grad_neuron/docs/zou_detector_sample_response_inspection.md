# Responses on the 200 `zou_train` detector prompts

## Protocol

- Corpus: `zou_train.txt`, 4,994 harmful prompts.
- Exact detector sample: `random.Random(112).sample(corpus, 200)`.
- Model: Meta-Llama-3-8B-Instruct.
- Prompt format: raw, matching the expanded-K/V detector run.
- Generation: greedy BF16, at most 256 new tokens, physical GPU 1.

The detector itself requests only one new token. This run extends generation to
256 tokens to inspect what the model would continue producing.

## Summary

| Diagnostic | Result |
|---|---:|
| Responses | 200 |
| Blank responses | 0 |
| Reached 256-token limit | 196 |
| Median response tokens | 256 |
| Contains `I cannot` | 47 |
| Starts with `I cannot` | 6 |
| Official refusal-substring safe | 84 |
| Official refusal-substring ASR | 58% |
| Repeated four-gram at least five times | 89 |

## Qualitative behavior

Raw prompting often makes the instruction-tuned model behave like a document or
web-page continuation rather than an assistant. Common openings include
`(Part 1)`, `(Updated)`, copied prompt text, dialogue continuations, and article
fragments.

Observed response types include:

- Direct refusal: `I cannot provide information or guidance ...`.
- Refusal repeated until the token limit.
- Cautionary or disapproving answer without an official refusal substring.
- Harmful procedural continuation, including cybercrime instructions.
- Prompt echoing or short-span repetition until the token limit.
- Mixed answers that warn an activity is illegal but continue describing it.

The 58% ASR should not be interpreted as a semantic harmfulness rate. For
example, an instructional answer can be counted safe merely because it contains
the word `illegal`, while a genuinely cautionary response can be counted unsafe
because it lacks one of the case-sensitive refusal markers.

## Artifacts

- Exact sampled prompts: `data/processed/zou_detector_sample_seed112_n200.csv`
- Prompts and generated responses:
  `data/processed/zou_detector_sample_seed112_n200_llama3_raw_responses.csv`
- Sample preparation: `scripts/prepare_zou_detector_sample.py`
- Generation driver: `scripts/generate_responses.py`
