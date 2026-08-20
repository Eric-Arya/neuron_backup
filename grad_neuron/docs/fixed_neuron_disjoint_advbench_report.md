# Fixed neuron controller from disjoint AdvBench prompts

## Data separation

- Neuron selection: AdvBench HarmBehavior source rows 100--299 (200 prompts).
- Evaluation: the ICLR reproduction's source rows 0--99 (100 prompts).
- Index overlap: 0.
- Exact-prompt overlap: 0.

The neuron coordinates are selected from all 200 disjoint prompts. Intervention
strength was tuned on rows 100--199 only, using
`K in {25,50,100,200}` and `epsilon in {0.5,0.75,1.0}`. The first-100 evaluation
responses were not used in this search.

## Selection

For every selection prompt `q_i`, the raw refusal score is

```
s_i = 0.5 * [log p("I" | q_i) + log p(" cannot" | q_i, "I")].
```

Gradients are averaged across all 200 selection prompts. The 25 neurons with
largest `abs(mean_g)` form one universal mask. Positive-gradient neurons are
scaled by 2 and negative-gradient neurons are set to 0. No gradients or
prompt-specific selection are used during evaluation.

On the disjoint tuning slice, this setting reduces ASR from 41% to 28%, with no
blank responses and fewer repetitive responses (63 to 53). Larger masks perform
worse.

## Evaluation result

| Condition | Unsafe / 100 | ASR | Safe to unsafe | Unsafe to safe | Blank | Repetitive | Median words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 34 | 34% | 0 | 0 | 0 | 75 | 433 |
| Fixed disjoint top-25 | 19 | 19% | 5 | 20 | 0 | 64 | 419.5 |

The requested refusal-substring ASR improves by 15 percentage points (44%
relative) without blank-output collapse. This lexical metric is not a semantic
harmfulness judge. Inspection finds many genuine new refusals, but some newly
safe outputs match only words such as `illegal` while still containing unsafe
material.

The disjoint and earlier row-0--39 rankings are similar: their top-25 sets share
22 neurons (88%), and every shared neuron has the same gradient sign.

## Artifacts

- Prepared selection prompts:
  `data/processed/advbench_harmbehavior_rows100_299.jsonl`
- Selection gradients and ranking:
  `results/gradients/raw_refusal_advbench_rows100_299/`
- Disjoint strength sweep:
  `results/safety_steering/disjoint_tuning/stronger_sweep_rows100_199/`
- Controller: `results/safety_steering/fixed_top25_disjoint_advbench/controller.json`
- Evaluation: `results/safety_steering/fixed_top25_disjoint_advbench/final100/`
