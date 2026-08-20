# Brief Reproduction Report: Table 1 and Figure 2

## Scope and setup

This reproduction covers Llama2-7B Base patched with the full DPO model using
safety neurons ranked on HH-RLHF-Harmless. Generation is greedy with 128 new
tokens, and response safety is measured by `beaver-7b-v1.0-cost` (lower is
safer). Each reported condition uses 200 prompts. Runs used two GPUs with batch
size 16 and atomic, resumable output shards.

## Table 1: Llama2 Base on HarmBench

The Base model was patched using the top 20,000 safety neurons from the
341,248-neuron ranking (5.86%).

| Condition | Mean cost |
|---|---:|
| Base | 7.850 |
| Full DPO | -3.348 |
| Patched Base, top 20,000 safety neurons | 2.083 |

The reproduced causal effect was **51.50%**, computed as
`100 × (patched − Base) / (DPO − Base)`. The paper reports approximately 63%,
so this run is 11.50 percentage points lower. The direction and substantial
recovery of DPO safety behavior were reproduced, but not the paper's exact
magnitude.

Artifacts: [aggregate result](results/table1_llama_base_harmbench/aggregate_result.json),
[run configuration](results/table1_llama_base_harmbench/run_config.json), and
[checksums](results/table1_llama_base_harmbench/checksums.json).

## Figure 2 left: Llama2 safety neurons

Figure 2 left uses the BeaverTails round0/330k test split. The evaluation
manifest exactly follows the released seed-42 procedure: sample 200 prompts
from the final 600 test examples.

The safety-neuron curve peaked at **75.09% causal effect with 8,000 neurons**
(2.34% of the ranking). At 20,000 neurons it recovered **66.48%**. The curve was
non-monotonic but showed a clear concentration of safety behavior in the
top-ranked neurons.

A seed-42 random-neuron comparison, sampled without replacement from the same
341,248-neuron universe, produced:

| Patched neurons | Safety neurons | Random neurons |
|---:|---:|---:|
| 8,000 | **75.09%** | 10.93% |
| 20,000 | **66.48%** | 20.90% |

This large gap supports the central qualitative result: the ranked safety
neurons causally transfer substantially more DPO safety behavior than equally
sized random neuron sets.

Artifacts: [safety curve](results/figure2_left_llama2_safety_neurons/curve.json),
[safety plot](results/figure2_left_llama2_safety_neurons/figure2_left_llama2_safety_neurons.png),
and [random comparison](results/figure2_left_llama2_random_seed42_8000_20000/curve.json).

## Validation

All conditions contain 200 generations and 200 cost scores. Cost-model direction
checks passed, zero-neuron patching matched Base token-for-token, artifact
checksums passed, and the relevant test suite passed 6/6 tests.
