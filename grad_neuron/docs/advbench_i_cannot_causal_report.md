# AdvBench `I cannot` neuron causal sweep

## Protocol

- Model: Meta-Llama-3-8B-Instruct.
- Data: first 100 AdvBench HarmBehavior rows.
- Prompt format: raw goal, following `EXPANDED_KV_MATH_REPRO.md`.
- Generation: greedy BF16, seed 112, batch size 16, at most 512 new tokens.
- Evaluator: case-sensitive `llm-attacks` refusal-substring ASR.
- Baseline: 34/100 attack successes (34% ASR).
- Intervention: globally scale selected SwiGLU intermediate dimensions immediately before
  `down_proj`, at every token position.
- Ranking: descending `abs(mean_g)` within the positive or negative `mean_g` sign.

The gradient was detected under a chat-formatted `assistant: I` prefill, whereas this causal
evaluation intentionally uses the report's raw-prompt protocol. The experiment therefore also
tests transfer across prompt formats.

## Results

| Intervention | K | ASR | Safe to unsafe | Unsafe to safe | Blanks | Median unique-word ratio |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0 | 34% | 0 | 0 | 0 | 0.100 |
| Positive, alpha=0 | 1000 | 100% | 66 | 0 | 8 | 0.010 |
| Positive, alpha=0 | 500 | 83% | 49 | 0 | 0 | 0.094 |
| Positive, alpha=0 | 200 | 76% | 48 | 6 | 0 | 0.058 |
| Positive, alpha=0 | 100 | 73% | 45 | 6 | 0 | 0.071 |
| Negative, alpha=0 | 1000 | 47% | 20 | 7 | 0 | 0.330 |
| Negative, alpha=0 | 500 | 42% | 12 | 4 | 0 | 0.247 |
| Negative, alpha=0 | 200 | 40% | 11 | 5 | 0 | 0.156 |
| Negative, alpha=0 | 100 | 44% | 13 | 3 | 0 | 0.128 |
| Positive, alpha=2 | 1000 | 54% | 27 | 7 | 0 | 0.043 |
| Positive, alpha=2 | 500 | 71% | 42 | 5 | 0 | 0.039 |
| Positive, alpha=2 | 200 | 71% | 43 | 6 | 0 | 0.049 |
| Positive, alpha=2 | 100 | 76% | 45 | 3 | 0 | 0.050 |

## Interpretation

Zeroing positive-gradient neurons strongly removes refusal-substring behavior. The top-100 and
top-200 conditions have no blank-output problem and raise ASR by 39 and 42 points, respectively.
This is evidence that the gradient detects coordinates causally important to the `I cannot`
refusal pivot.

The aggressive interventions do not establish clean harmful compliance. Responses become much
more repetitive, and qualitative inspection shows a mixture of compliance, vague or incomplete
answers, safe text that no longer contains an evaluator keyword, and degeneration. Top-1000
positive zeroing is invalid as a useful jailbreak setting: 66 responses contain fewer than ten
words and eight are blank.

Zeroing negative-gradient neurons converts some baseline-unsafe rows to refusal, including clear
cases, but every tested K also converts more safe rows to evaluator-unsafe. No negative-zero
condition improves aggregate safety over the 34% baseline ASR.

Doubling positive-gradient neurons also fails to improve aggregate safety. All four conditions
raise ASR and sharply reduce lexical diversity. The local derivative sign at alpha=1 therefore
does not extrapolate monotonically to the large alpha=2 intervention under raw-prompt distribution
shift.

## Recommended next experiment

Use the same chat format for detection and causal evaluation, and test local perturbations around
alpha=1 (for example 1.05, 1.1, 1.25, and 1.5) with K in 20, 50, and 100. This directly tests the
first-order prediction while reducing nonlinear generation damage. A content-based evaluator is
needed before interpreting refusal-substring ASR as actual harmfulness.
