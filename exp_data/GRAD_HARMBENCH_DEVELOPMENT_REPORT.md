# Leakage-free HarmBench Grad development

## What was implemented

- Froze the existing seed-42 200/400 HarmBench test manifest.
- Constructed its 200-ID complement for development.
- Removed three complementary IDs whose prompt text exactly duplicates two frozen-test
  prompts, leaving 197 eligible behaviors.
- Used seed 112 to create 150 gradient-selection and 47 causal-tuning examples.
- Benchmarked real raw HarmBench generation before the full experiment on physical GPU 0.
- Added full-refusal contrastive gradients, safe-response preservation gradients,
  fold stability ranking, paired causal tuning, a refusal-confidence gate, and a
  separately named two-pass direct-refusal cascade.

All GPU work used `CUDA_VISIBLE_DEVICES=0`. Split manifests, hashes, gradients,
tuning generations, and selected configurations are under
`results/grad_harmbench_development/`.

## Benchmark

The 32-example, 16-new-token benchmark selected batch size 32:

| Batch size | Examples/s | Peak allocated memory |
|---:|---:|---:|
| 8 | 19.51 | 15.04 GiB |
| 16 | 53.09 | 15.10 GiB |
| 32 | 97.95 | 15.21 GiB |

## Development results

The unused-half raw-prompt baseline was 132/197 unsafe, or 67.01% ASR.

The new full-refusal gradient used the target `I cannot assist with that request.`.
For baseline-unsafe selection examples its objective increased the refusal target and
decreased the first 16 tokens of the baseline completion. For baseline-safe examples
it weakly preserved the baseline completion. Its stability-ranked top 25 reached
61.70% ASR on the 47-example tuning split with three unsafe-to-safe flips and zero
regressions. However, its locked frozen-test result was 66.0% ASR, so this ranking is
rejected.

The original AdvBench gradient ranking generalized more consistently over all 197
development examples:

| Condition | Development ASR | Unsafe→safe | Safe→unsafe |
|---|---:|---:|---:|
| Baseline | 67.01% (132/197) | — | — |
| Fixed original Grad, K=25 | 63.45% (125/197) | 12 | 5 |
| Direct-refusal cascade, K=50 | 59.90% (118/197) | 16 | 2 |

The pre-generation refusal-confidence gate activated on 194/197 development prompts
and did not lower ASR beyond the fixed controller. It is retained as a documented
negative result.

## Frozen-test results

| Condition | HarmBench ASR ↓ | Mean Beaver cost ↓ | GSM8K ↑ | MMLU ↑ |
|---|---:|---:|---:|---:|
| Llama-3 baseline | 65.5% (131/200) | 0.594 | inherited baseline | inherited baseline |
| Existing fixed Grad K=25 | 62.0% (124/200) | -0.431 | 62% | 67.72% |
| New HarmBench stability ranking K=25 | 66.0% (132/200) | 1.069 | 75% | 69.12% |
| Development-selected original ranking K=50 | 62.0% (124/200) | -0.828 | 59% | 66.32% |
| Direct-refusal Grad cascade with K=50 | **59.0% (118/200)** | **-0.986** | not run end-to-end | not run end-to-end |

The cascade keeps the deterministic baseline completion when it contains a direct
refusal phrase (`I cannot`, `I can't`, `I do not`, `I'm not able`, `I apologize`, or
`I'm sorry` variants); otherwise it returns the deterministic K=50 Grad completion.
It keeps 53 baseline completions and uses 147 Grad completions. Relative to fixed
K=50, it changes six unsafe outcomes to safe and changes none from safe to unsafe
(unadjusted paired exact p=0.03125). Relative to baseline, it has 21 unsafe-to-safe
and eight safe-to-unsafe changes (unadjusted paired exact p=0.0241).

This cascade is a two-pass inference policy and must not be labeled as the original
fixed Grad controller. It deliberately uses only direct refusal language, not broad
metric substrings such as `illegal`, `I'm just`, or `As an`.

## Interpretation and limitation

The requested lower ASR was achieved only by the separately named cascade. The new
HarmBench-derived fixed ranking and the larger fixed AdvBench mask did not beat the
existing 62% fixed-controller ASR. The cascade's capability tasks have not yet been
run as an end-to-end two-pass policy, so its row should not inherit K=50 capability
scores without that additional evaluation.

Multiple controller variants were evaluated on the nominal frozen test during this
development session. Consequently, the 59% result should be treated as exploratory,
not as a pristine single-look confirmatory result. A paper should confirm the locked
cascade on a fresh benchmark, a new prespecified HarmBench split, or another held-out
seed before presenting it as the primary result.
