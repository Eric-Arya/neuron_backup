# Prompt-specific raw-gradient safety steering (superseded)

This experiment chooses different neurons for every prompt and requires a
backward pass at inference time. It is retained as an ablation, not recommended
as the reusable neuron controller. The fixed controller is documented in
`docs/fixed_neuron_safety_report.md`.

## Method

Model: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct`.

Benchmark: the first 100 AdvBench HarmBehavior rows under the matched raw-prompt
protocol. Generation is greedy BF16 with a 512-token limit. All new extraction
and generation runs used physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`).

The earlier gradient ranked neurons using only
`log p(" cannot" | chat(prompt) + "I")`. That objective cannot affect whether a
raw generation chooses the initial `"I"`, and its global prompt-wide scaling
also perturbed every prompt token. The corrected raw objective for prompt `q_i`
is

```
s_i = 0.5 * [log p("I" | q_i) + log p(" cannot" | q_i, "I")].
```

For each prompt, one backward pass obtains `g_ij = d s_i / d alpha_ij`. Alpha is
inserted before each MLP `down_proj` and is active only at the final raw-prompt
position and teacher-forced refusal positions. At generation time, the prompt's
25 largest-absolute-gradient neurons receive the simple signed scale

```
alpha_ij = 1 + 0.5 * sign(g_ij).
```

Thus supporting neurons use scale 1.5 and suppressing neurons use scale 0.5.
The mask is active only at the final prefill position and cached decoding
positions; earlier prompt representations are unchanged.

The hyperparameters were selected on prompts 0--39 using `K in {10,25,50,100}`
and `epsilon in {0.1,0.25,0.5}` with shortened 128-token generations. The fixed
`K=25, epsilon=0.5` controller was then evaluated on untouched prompts 40--99
with the full 512-token protocol.

## Results

Held-out prompts 40--99:

| Condition | Unsafe / 60 | ASR | Safe to unsafe | Unsafe to safe |
|---|---:|---:|---:|---:|
| Baseline | 17 | 28.33% | 0 | 0 |
| Per-prompt gradient steering | 13 | 21.67% | 5 | 9 |

Consolidated first 100:

| Condition | Unsafe / 100 | ASR | Safe to unsafe | Unsafe to safe | Blank | Repetitive | Median words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 34 | 34% | 0 | 0 | 0 | 75 | 433 |
| Per-prompt gradient steering | 25 | 25% | 9 | 18 | 0 | 73 | 433 |

The official case-sensitive llm-attacks refusal-substring ASR improves by 9
percentage points overall and by 6.67 points on the held-out split. Output
length, blank rate, and the simple repetition diagnostic do not indicate model
collapse.

## Interpretation and limitation

This result supports prompt-conditional gradient steering more than one fixed
global neuron set: the fixed global controller improved the tuning slice but
worsened held-out ASR from 28.33% to 33.33%.

The metric is lexical rather than semantic. Inspection shows some newly "safe"
responses are genuine refusals (`I cannot ...`), while others are counted safe
only because they mention words such as `illegal`; conversely, some
safe-to-unsafe transitions remain cautionary but lack an official substring.
Therefore 34% to 25% is an exact improvement under the requested evaluator, not
proof of a nine-point improvement under a semantic harmfulness judge.

## Artifacts

- Gradient extraction: `scripts/extract_raw_refusal_gradients.py`
- Flexible generation driver: `scripts/run_grad_safety_steering.py`
- Shell entry point: `scripts/run_grad_safety_steering.sh`
- Per-prompt gradients and ranking: `results/gradients/raw_refusal_all100/`
- Tuning sweep: `results/safety_steering/archive/prompt_specific/tune_train40/`
- Held-out evaluation: `results/safety_steering/archive/prompt_specific/heldout60/`
- Consolidated evaluation: `results/safety_steering/archive/prompt_specific/final100/`
