# BBH-200 safety-edit capability evaluation

## What was run

Seventeen Llama-3-8B-Instruct conditions from the main safety--MATH figure were
evaluated on the frozen seed-112, task-stratified 200-example BBH subset:

- Unmodified baseline.
- SN-Tune at `alpha={1,4,6,8}`.
- Raw-format IA3-SFT at `alpha={1,1.5,2,2.5,3,3.5}`.
- Raw-format IA3 guide patch at `K={40000,80000}` with IA3 alpha 3.
- On-policy Grad at `K={1000,2000,4000}`, strength 1, plus `K=4000` at
  strength 0.75, all with positive-only direction and final-token scope.

Guide-patch `K=160000` and `K=320000` were intentionally stopped before BBH
evaluation at the user's request. All completed conditions used the official raw
three-shot chain-of-thought prompts, greedy decoding, the lm-evaluation-harness
1,024-token ceiling and stop strings, and its case-sensitive `the answer is`
regex plus exact-match metric. The evaluator was pinned at commit
`8a07e1110d060de48cfc7a9a7987b7659060b60b`.

Standard single-model runs used FP32 and batch size 8. Guide-patch runs require
the base and guide model together and used BF16. A prior real-BBH benchmark found
batch size 16 to be the fastest safe guide-patch setting: 0.170 examples/s and
54.35 GiB peak allocation, versus 0.139 examples/s and 39.30 GiB at batch size 8.
Batch size 32 OOMed, so the runner now defaults guide-patch BBH to batch size 16.
The `K=4000, s=0.75` Grad point was evaluated as two deterministic interleaved
100-example shards on physical GPUs 0 and 1, then merged and rescored in the
canonical 200-example order.

## Results

Lower HarmBench ASR is safer; higher BBH accuracy is better.

| Method | Setting | HarmBench ASR | BBH correct | BBH micro | BBH task macro | BBH delta vs. baseline |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Unmodified | 65.5% | 127/200 | 63.5% | 63.23% | -- |
| SN-Tune | alpha=1 | 62.5% | 124/200 | 62.0% | 61.57% | -1.5 pp |
| SN-Tune | alpha=4 | 20.0% | 121/200 | 60.5% | 60.19% | -3.0 pp |
| SN-Tune | alpha=6 | 1.0% | 107/200 | 53.5% | 53.24% | -10.0 pp |
| SN-Tune | alpha=8 | 0.5% | 79/200 | 39.5% | 39.09% | -24.0 pp |
| IA3-SFT | alpha=1 | 44.5% | 127/200 | 63.5% | 63.23% | 0.0 pp |
| IA3-SFT | alpha=1.5 | 33.5% | 129/200 | 64.5% | 64.22% | +1.0 pp |
| IA3-SFT | alpha=2 | 24.5% | 129/200 | 64.5% | 64.22% | +1.0 pp |
| IA3-SFT | alpha=2.5 | 16.0% | 127/200 | 63.5% | 63.23% | 0.0 pp |
| IA3-SFT | alpha=3 | 8.0% | 128/200 | 64.0% | 63.69% | +0.5 pp |
| IA3-SFT | alpha=3.5 | 6.5% | 132/200 | 66.0% | 65.74% | +2.5 pp |
| IA3 guide patch | K=40k | 22.0% | 133/200 | 66.5% | 66.47% | +3.0 pp |
| IA3 guide patch | K=80k | 15.5% | 132/200 | 66.0% | 65.87% | +2.5 pp |
| Grad (on-policy) | K=1000, s=1 | 17.0% | 121/200 | 60.5% | 60.05% | -3.0 pp |
| Grad (on-policy) | K=2000, s=1 | 9.5% | 115/200 | 57.5% | 57.14% | -6.0 pp |
| Grad (on-policy) | K=4000, s=0.75 | 8.5% | 123/200 | 61.5% | 61.18% | -2.0 pp |
| Grad (on-policy) | K=4000, s=1 | 5.5% | 119/200 | 59.5% | 59.33% | -4.0 pp |

Generation diagnostics:

| Method | Setting | Extraction failures | Repetitive outputs | Token-limit outputs | Mean generated tokens |
|---|---|---:|---:|---:|---:|
| Baseline | Unmodified | 4 | 26 | 5 | 197.44 |
| SN-Tune | alpha=1 | 2 | 26 | 5 | 200.67 |
| SN-Tune | alpha=4 | 4 | 30 | 9 | 220.03 |
| SN-Tune | alpha=6 | 6 | 46 | 25 | 283.10 |
| SN-Tune | alpha=8 | 10 | 90 | 74 | 487.37 |
| IA3-SFT | alpha=1 | 4 | 28 | 2 | 191.41 |
| IA3-SFT | alpha=1.5 | 5 | 29 | 2 | 190.18 |
| IA3-SFT | alpha=2 | 5 | 27 | 3 | 192.35 |
| IA3-SFT | alpha=2.5 | 5 | 27 | 3 | 192.86 |
| IA3-SFT | alpha=3 | 6 | 24 | 2 | 191.26 |
| IA3-SFT | alpha=3.5 | 5 | 26 | 3 | 193.74 |
| IA3 guide patch | K=40k | 5 | 100 | 80 | 579.33 |
| IA3 guide patch | K=80k | 6 | 106 | 85 | 594.92 |
| Grad (on-policy) | K=1000, s=1 | 5 | 23 | 6 | 206.34 |
| Grad (on-policy) | K=2000, s=1 | 6 | 26 | 6 | 206.91 |
| Grad (on-policy) | K=4000, s=0.75 | 9 | 27 | 4 | 198.98 |
| Grad (on-policy) | K=4000, s=1 | 14 | 32 | 9 | 220.10 |

## Interpretation

- SN-Tune shows the clearest safety--capability trade-off. Moving from baseline
  to alpha 8 reduces HarmBench ASR from 65.5% to 0.5%, while BBH falls from
  63.5% to 39.5%. The increasingly repetitive and token-limited generations show
  that the largest-alpha loss includes output degeneration.
- On-policy Grad also trades BBH capability for safety: its edited points reduce
  ASR to 17.0--5.5% and score 57.5--61.5% on BBH. The BBH K trajectory is
  not monotonic in K on this 200-example subset. At fixed `K=4000`, reducing
  strength from 1 to 0.75 raises BBH from 59.5% to 61.5%, while ASR rises from
  5.5% to 8.5%, giving a clean local safety--capability trade-off.
- The tested IA3-SFT trajectory does not show a BBH trade-off. Alpha 3.5 reaches
  6.5% ASR and 66.0% BBH, 2.5 points above baseline. This small-subset improvement
  should be interpreted alongside the other capability evaluations rather than as
  proof of a general capability gain.
- The two completed guide-patch points also score above baseline on BBH, but their
  generations are pathological: 100--106 repetitive responses and 80--85 outputs
  at the token ceiling. Their aggregate exact-match scores therefore need this
  diagnostic caveat.

## Validation and artifacts

Every completed result contains exactly 200 responses across all 27 BBH tasks.
All 17 ordered response-ID streams have the same SHA-256 digest,
`af7b17da13c2ad37b38a75a042421327e371679fd5014bf59a0ad0b8108c6a48`,
confirming matched examples and order.

- Dataset: `/workspace/xcy/dataset/big_bench_hard/subsets/bbh_seed112_n200.jsonl`
- Figure: `figures/bbh_harmbench_tradeoff_main_only.{png,pdf,svg}`
- Baseline: `results/bbh_llama3_base_raw_cot_fp32/`
- SN-Tune: `results/bbh_sn_alpha{1,4,6,8}_raw_cot_fp32/`
- IA3-SFT: `results/bbh_ia3_sft_snraw_alpha*_raw_cot_fp32/`
- Guide patch: `results/bbh_sft_patch_snraw_alpha3_top{40000,80000}_raw_cot_bf16/`
- Grad K sweep: `results/bbh_grad_onpolicy_expanded_k{1000,2000,4000}_s1_raw_cot_fp32/`
- Grad strength point: `results/bbh_grad_onpolicy_expanded_k4000_s0p75_raw_cot_fp32/`

Each completed result directory contains the semantic/runtime configuration,
validation data, raw and scored per-example responses, and aggregate summary.
