# PKU contrastive Grad experiment

## Objective

Test whether a neuron controller derived from full safe-versus-unsafe response
pairs improves HarmBench refusal-substring ASR and Beaver cost without using an
`I cannot` target or Beaver cost during gradient extraction.

## Data and contrastive objective

- Source subset: 150 PKU-SafeRLHF pairs with exactly one safe and one unsafe
  response.
- Pair file:
  `/workspace/xcy/dataset/pku_saferlhf/contrastive_pairs_seed112_n150.jsonl`
- Pair SHA-256:
  `f5ce9a7b937e627f610c085669c4f0d11748f65a823f7598e463276074e38ddf`
- Objective for each pair:

  `mean_token_logp(safe | prompt) - mean_token_logp(unsafe | prompt)`

- The alpha gates multiply individual Llama-3 MLP intermediate dimensions at
  the final prompt position and all teacher-forced response prediction
  positions. Model parameters remain frozen.

## Runtime benchmark

Four real pairs were used to compare standard backpropagation with non-reentrant
gradient checkpointing on separate H100 GPUs.

| Mode | Pairs/s | Peak CUDA memory |
|---|---:|---:|
| Standard | 3.83 | 16.11 GiB |
| Gradient checkpointing | 3.50 | 16.11 GiB |

Standard backpropagation was selected. The full 150-pair extraction took 18.45
seconds at 8.13 pairs/s and peaked at 17.03 GiB.

## Gradient diagnostics

- Gradient tensor shape: `150 x 32 x 14336`.
- Mean safe completion log-probability: -2.1603.
- Mean unsafe completion log-probability: -2.1930.
- Mean safe-minus-unsafe contrast: 0.0327.
- Top-100 split-half overlap: 36/100; Jaccard 0.220.
- Sign agreement among overlapping neurons: 80.6%.
- Full gradient-vector split-half Pearson correlation: 0.223.
- Only 3/25 neurons overlap the prior raw `I cannot` top-25 ranking.

The raw absolute-mean ranking is heterogeneous, so a stability ranking by
`abs(mean gradient) / gradient standard deviation` was evaluated as a
pre-specified robustness variant.

## Development selection

Nine top-k/strength configurations for each ranking were evaluated on the
47-prompt non-test HarmBench tuning manifest. Beaver cost was computed only
after generation and was not used to extract gradients.

| Ranking/configuration | ASR | Mean Beaver cost | GSM8K-20 |
|---|---:|---:|---:|
| Development baseline | 68.1% | 0.028 | 60% |
| Raw top-50, strength 0.25 | 68.1% | -0.976 | 55% |
| Stable top-50, strength 1.0 | 63.8% | -0.596 | 65% |

The raw controller was the best cost-only candidate. The stable controller was
the development winner: two unsafe-to-safe ASR flips, zero safe-to-unsafe flips,
lower Beaver cost, and no observed capability regression on the small GSM8K
check.

## Frozen HarmBench-200 results

All conditions use raw prompts, 128-token greedy decoding, the case-sensitive
llm-attacks refusal-substring ASR, and Beaver cost over prompt plus completion.
Lower values are better for both metrics.

| Condition | ASR | Mean Beaver cost | Unsafe-to-safe | Safe-to-unsafe |
|---|---:|---:|---:|---:|
| Llama-3 baseline | 65.5% | 0.594 | -- | -- |
| Raw top-50, strength 0.25 | 67.0% | 0.409 | 7 | 10 |
| Stable top-50, strength 1.0 | 66.5% | 0.767 | 3 | 5 |
| Prior `I cannot` Grad, for reference | 62.0% | -0.431 | 17 | 10 |

For the raw controller, the paired mean Beaver-cost change relative to baseline
is -0.185; 102/200 prompts have lower cost. A 10,000-sample paired bootstrap
95% interval is [-1.010, 0.643], so this small aggregate improvement is not
statistically resolved. The stable controller worsens mean cost by 0.173 and
ASR by one percentage point.

## Conclusion

This first PKU full-response contrastive Grad construction does not improve
frozen-test ASR. The raw ranking gives a small Beaver-cost decrease, but it
worsens ASR and the paired cost interval includes zero. The development gains
of the stability-ranked controller do not generalize. Consequently, these
results do not support replacing the existing controller with this PKU
contrastive controller in its current form.

The most likely limitations are heterogeneous PKU safe-response strategies,
low cross-pair gradient consistency, and mean-log-probability gradients that
mix safety with response style and content. Further work should change the
contrastive objective or pair curation rather than tune more configurations on
the frozen test set.
