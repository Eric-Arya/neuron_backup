# Fixed neuron controller selected on evaluation rows (superseded)

This historical run selected neurons using rows 0--39, which overlap the
consolidated rows 0--99 evaluation. The current disjoint experiment is documented
in `docs/fixed_neuron_disjoint_advbench_report.md`.

## Goal

Use one neuron set and one intervention for every raw prompt. No prompt-specific
gradient, neuron selection, or parameter fitting occurs at inference time.

## Selection and intervention

The training set is AdvBench HarmBehavior prompts 0--39. For each prompt, the
raw refusal score is

```
s_i = 0.5 * [log p("I" | q_i) + log p(" cannot" | q_i, "I")].
```

The gradients are averaged over the 40 training prompts. The fixed controller
contains the 25 neurons with largest `abs(mean_g)`. For every prompt, the exact
same scale is applied:

```
alpha_j = 1 + 0.5 * sign(mean_g_j)
```

Positive-gradient neurons therefore use 1.5 and negative-gradient neurons use
0.5. Scaling is restricted to the final prefill token and cached decoding
positions. Earlier prompt-token representations are unchanged.

The complete fixed neuron list is stored in
`results/safety_steering/archive/fixed_overlap_eval/final100/selected_neurons.json`.

## Results

Held-out prompts 40--99, full 512-token raw protocol:

| Condition | Unsafe / 60 | ASR | Safe to unsafe | Unsafe to safe |
|---|---:|---:|---:|---:|
| Baseline | 17 | 28.33% | 0 | 0 |
| Fixed top-25 | 11 | 18.33% | 3 | 9 |

Consolidated prompts 0--99:

| Condition | Unsafe / 100 | ASR | Safe to unsafe | Unsafe to safe | Blank | Repetitive | Median words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 34 | 34% | 0 | 0 | 0 | 75 | 433 |
| Fixed top-25 | 31 | 31% | 8 | 11 | 0 | 67 | 432 |

The held-out improvement is 10 percentage points. The consolidated improvement
is 3 points. There is no blank-output collapse, and the simple repetition count
decreases.

The previously selected fixed top-100 mask failed held-out evaluation
(28.33% to 33.33% ASR), showing that broader intervention is not better.

## Scope and limitation

This is a genuinely reusable controller for this model and intervention point:
the neuron IDs, signs, and scales are constant across prompts. It is not yet
established that the same coordinates transfer to other model checkpoints or
prompt formats.

The requested llm-attacks evaluator is a case-sensitive refusal-substring
metric. Its improvement does not by itself establish an equal improvement under
a semantic harmfulness judge.

## Artifacts

- Training gradients: `results/gradients/raw_refusal_train40/`
- Tuning sweep: `results/safety_steering/archive/fixed_overlap_eval/tune_train40/`
- Held-out evaluation: `results/safety_steering/archive/fixed_overlap_eval/heldout60/`
- Consolidated evaluation: `results/safety_steering/archive/fixed_overlap_eval/final100/`
- Implementation: `scripts/run_grad_safety_steering.py`
