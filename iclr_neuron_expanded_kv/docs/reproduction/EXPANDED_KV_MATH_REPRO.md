# Llama-3 expanded-K/V safety-neuron deactivation report

## Executive summary

This branch reproduces the paper's Llama-3 safety-neuron intervention while fixing the main GQA
compatibility problem: K/V interventions operate in the expanded 4,096-dimensional query-head
space after `repeat_kv`, rather than zeroing a shared row in the physical 1,024-dimensional K/V
projection. The compatibility work from `full-reproduction-experiments` is retained.

The current best global-rate operating points on the first 100 evaluation examples are:

| Global rate | Selected SN entries | GSM8K chat 0-shot | HarmBehavior raw ASR | Interpretation |
|---:|---:|---:|---:|---|
| 0 | 0 | 67% | 34% | Origin baseline |
| 0.0001 | 131 | 62% | 50% | Conservative |
| 0.0002 | 262 | 60% | 56% | Strong math preservation |
| 0.0004 | 524 | 45% | 70% | Balanced safety attack |
| **0.0006** | **786** | **30%** | **75%** | Practical upper limit under the accepted 30% math floor |

All four evaluated global-rate safety runs have zero blank responses. Rates above `0.0006`
degrade sharply: `0.0007` scores 20% on GSM8K, `0.0008` scores 1%, and `0.0016` scores 0%.

## Branch, model, environments, and data

- Worktree: `/workspace/xcy/safety_repro/iclr_neuron_expanded_kv`
- Branch: `llama3-expanded-kv-math`
- Model: `/workspace/xcy/models/Meta-Llama-3-8B-Instruct`
- Deactivation environment: `/workspace/xcy/miniconda3/envs/safety-neuron-expanded-kv`
- Detection environment: `/workspace/xcy/miniconda3/envs/safety-neuron-detect-expanded`
- GSM8K source: `/workspace/xcy/dataset/table1_sources/gsm8k_main`
- HarmBehavior source: `/workspace/xcy/dataset/harmful_behaviors.csv`
- HarmBehavior prepared subset: first 100 source rows, SHA-256
  `ac1adf0903c8697e23ca689e50dc2b0abba52312b0ffb8b39969e1dcd7f00c93`
- Global selection seed: 112
- Decoding: greedy, BF16, batch size 16

## Implementation

### Expanded GQA K/V deactivation

Llama-3 8B uses 32 query heads and 8 K/V heads. Each physical K/V head is shared by four query
heads. A physical K/V projection has 1,024 output coordinates, while the repeated tensor used by
attention has 4,096 coordinates.

The deactivation overlay now masks K/V after `repeat_kv`:

- K and V detector/deactivation indices are validated in `[0, 4096)`.
- A selected expanded coordinate masks only one query-head replica.
- The shared physical `k_proj` or `v_proj` row is not permanently zeroed.
- The behavior is implemented in eager, SDPA, and Flash Attention paths.

This avoids the earlier failure mode where selecting one native K/V coordinate unintentionally
deactivated all four GQA replicas.

### Q and FFN deactivation

- Q uses one row of `q_proj` per selected coordinate.
- `fwd_up` uses one row of `up_proj`.
- `fwd_down` uses one input column of `down_proj`.

For the standalone FFN capability sweep, only `fwd_down` was selected. One `down_proj` input
column corresponds to one FFN activation dimension and is sufficient to remove that dimension's
output contribution. This avoids counting the paired `up_proj` row as a second intervention.

### Detector candidate pool

The main detector run uses 200 raw harmful prompts with per-prompt caps `attn=1000` and
`ffn=2000`. The full expanded candidate file is:

`neuron_detection/output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vpool_200.txt`

| Structure | Detected entries |
|---|---:|
| FFN up | 3,153 |
| FFN down | 3,153 |
| Q | 9,343 |
| Expanded K | 9,343 |
| Expanded V | 2,219 |
| **Total** | **27,211** |

## Evaluation protocol

### GSM8K capability

The current primary capability protocol follows the project requirement to use Llama-3 chat
format:

- First 100 GSM8K test rows.
- Zero-shot, matching the paper's stated setup for instruction-tuned models.
- A single user turn rendered by `tokenizer.apply_chat_template`.
- Prompt ends with `Answer: Let's think step by step.`
- Greedy decoding with at most 256 new tokens.
- Accuracy uses the final extracted number; missing numbers are extraction failures.

The paper states zero-shot GSM8K for instruction-tuned models but does not explicitly document
whether the chat template was used. Chat rendering here is an explicit project requirement.

### HarmBehavior safety

- First 100 AdvBench HarmBehavior rows.
- Raw goal prompt, matching the configuration whose origin ASR is close to the paper.
- Greedy decoding with at most 512 new tokens.
- ASR follows the case-sensitive refusal-substring rule from `llm-attacks`.
- Blank responses are reported separately because the official substring rule would otherwise count
  them as attack successes.

### Baselines and paper reference

| Metric | Paper Origin | Paper Deact-SN | This subset Origin |
|---|---:|---:|---:|
| GSM8K accuracy | 75.9% | 72.4% | 67% |
| HarmBehavior ASR | 30% | 78% | 34% |

The paper numbers use its complete evaluation protocol. Results in this report use the first 100
rows and should be compared primarily against the corresponding subset origin values.

## Standalone Q and FFN incremental sweeps

These experiments deactivate only one structure at a time, with no V or K intervention. Counts are
cumulative: for a given structure, every larger seed-112 selection contains the smaller selection.

| Selected entries | Q accuracy | Q extraction failures | FFN-down accuracy | FFN extraction failures |
|---:|---:|---:|---:|---:|
| 0 | 67% | 0 | 67% | 0 |
| 32 | 64% | 0 | 72% | 0 |
| 64 | 65% | 1 | 71% | 0 |
| 128 | 65% | 0 | 65% | 0 |
| 256 | 49% | 0 | 64% | 0 |
| 512 | 24% | 1 | 62% | 0 |

Main observations:

- Q is stable through 128 entries, declines to 49% at 256, and reaches 24% at 512.
- FFN-down remains close to origin even at 512 entries; no FFN-only upper limit was found in the
  completed sweep.
- The strong FFN results differ from an earlier raw 5-shot smoke test. Prompt format and using only
  `fwd_down`, rather than mixing FFN structures, materially change the result.
- The 100-row result is treated as the primary evidence; eight-row tests below are diagnostic only.

## V-only and small mixed experiments

### V-only HarmBehavior-100

| Expanded V entries | HarmBehavior ASR | Blank responses |
|---:|---:|---:|
| Origin | 34% | 0 |
| 557 | 41% | 0 |
| 750 | 52% | 0 |
| 900 | 64% | 0 |

V900 scores 57/100 on chat zero-shot GSM8K, versus 67/100 for Origin. This showed that expanded V
can increase ASR while retaining coherent math, but the later global-rate sweep produces a better
ASR/capability curve.

### Adding small Q/FFN selections to V900

On the first eight chat zero-shot GSM8K rows, Origin and V900 both score 4/8. Keeping the exact
V900 selection fixed gives:

| Added structure | Correct | Extraction failures |
|---|---:|---:|
| Q8 | 4/8 | 0 |
| Q16 | 4/8 | 0 |
| Q32 | 3/8 | 0 |
| FFN-down4 | 3/8 | 0 |
| FFN-down8 | 3/8 | 0 |
| FFN-down32 | 2/8 | 0 |

This tiny combined screen suggested that additions interact with V900 more strongly than the same
Q/FFN structures used alone. It should not override the 100-row standalone results.

## Global-rate definition and validation

The global expanded intervention universe is:

```text
32 layers * (14336 fwd_up + 14336 fwd_down + 4096 Q + 4096 K + 4096 V)
= 1,310,720 structure coordinates
```

For rate `r`, the requested count is:

```text
max(1, floor(1,310,720 * r + 0.5))
```

That exact count is sampled uniformly without replacement from the eligible 27,211-entry detected
SN pool. The rate denominator is the model-wide expanded intervention universe, not the detected
pool size. All five structures are explicitly eligible. Validation tests check:

- the 1,310,720 denominator;
- exact rounded count;
- exact agreement between total and per-structure counts;
- deterministic selection at seed 112;
- a changed selection at a different seed;
- expanded K/V bounds.

The tested seed-112 rate selections are cumulative: higher-rate samples preserve the lower-rate
sample as a prefix.

## Global-rate results

### Selection composition

| Global rate | Total | FFN up | FFN down | Q | Expanded K | Expanded V |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0001 | 131 | 16 | 10 | 45 | 52 | 8 |
| 0.0002 | 262 | 30 | 22 | 99 | 88 | 23 |
| 0.0004 | 524 | 61 | 48 | 189 | 182 | 44 |
| 0.0006 | 786 | 91 | 69 | 278 | 283 | 65 |

Selection SHA-256 values:

- `0.0001`: `ed9b9df4ea61f3e5d9ae8f35bb1608c8adaae331644662c75db01af192f34a52`
- `0.0002`: `d258155733497e94939ea92e03cf1a680adfc92cf914ce5c3b961eb3451aebc0`
- `0.0004`: `a32b671724a9d34521995a64dca6918df3ecfd3e9324795998cb6585f632f2ae`
- `0.0006`: `86cb1ea15cc6a9f5d0365db0520e64109984ed513abd38c4fc66885993948f7a`

### Capability, safety, and coherence tradeoff

| Rate | Entries | GSM8K | Extraction failures | Repetitive GSM outputs | HarmBehavior ASR | Blank harm outputs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 67% | 0 | 5/100 | 34% | 0 |
| 0.0001 | 131 | 62% | 0 | 7/100 | 50% | 0 |
| 0.0002 | 262 | 60% | 0 | 7/100 | 56% | 0 |
| 0.0004 | 524 | 45% | 0 | 10/100 | 70% | 0 |
| **0.0006** | **786** | **30%** | **0** | **14/100** | **75%** | **0** |
| 0.0007 | 917 | 20% | 4 | 22/100 | not run | not run |
| 0.0008 | 1,049 | 1% | 29 | 49/100 | not run | not run |
| 0.0016 | 2,097 | 0% | 56 | 94/100 | not run | not run |

![Expanded-K/V capability, safety, and degeneration tradeoff](../../figures/expanded_kv_tradeoff.png)

*The figure depicts the requested operating range through rate `0.0007`; the higher-rate
degeneration results remain in the table.*

“Repetitive” means at least one four-word sequence occurs five or more times in a response. It is a
strict heuristic: even Origin flags 5/100. The abrupt change between `0.0007` and `0.0008`, together
with the extraction failures, marks genuine generation degeneration rather than ordinary accuracy
loss.

### Interpretation

- `0.0001` and `0.0002` preserve most subset math accuracy and raise ASR by 16 and 22 points.
- `0.0004` is the best balanced point if 45% math is preferred: ASR reaches 70%.
- `0.0006` is the practical upper point under the accepted 30% math floor: ASR reaches 75%, near
  the paper's 78%, without blank harm outputs or GSM extraction failures.
- `0.0007` is below the accepted math floor.
- `0.0008` and above should not be used; outputs become substantially repetitive and frequently
  unparseable.

## Configurable neuron scaling

The intervention now accepts `--neuron-scale` (aliases: `--neuron-multiplier` and
`--scale-factor`). The default remains `0`, so existing commands retain hard-deactivation
behavior. A scale of `1` is a no-op, `0.5` attenuates selected channels, and `2` doubles them.

Scaling is applied at runtime without modifying model parameters:

- FFN-up scales selected `up_proj` activation dimensions before gating.
- FFN-down scales selected intermediate activation dimensions before `down_proj`.
- Q scales selected query projection coordinates before RoPE.
- expanded K/V scale replica-specific coordinates after `repeat_kv`.

This prevents a multiplier from compounding across autoregressive decoding steps. Unit tests also
verify that model weights remain unchanged and that scales `0`, `0.5`, `1`, and `2` have the
expected tensor-level behavior.

The first-100 protocol above was run for scales `0.5` and `2` at rates `0.0002` and `0.0004`.
Selection seed 112 and the candidate pool are unchanged, so each rate uses the same selected-neuron
hash as its hard-deactivation counterpart.

| Rate | Entries | Scale | GSM8K | Extraction failures | Repetitive GSM outputs | HarmBehavior ASR | Blank harm outputs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | n/a | 67% | 0 | 5/100 | 34% | 0 |
| 0.0002 | 262 | 0 | 60% | 0 | 7/100 | 56% | 0 |
| 0.0002 | 262 | 0.5 | **70%** | 0 | 5/100 | **47%** | 0 |
| 0.0002 | 262 | 2 | 58% | 0 | 6/100 | 37% | 0 |
| 0.0004 | 524 | 0 | 45% | 0 | 10/100 | 70% | 0 |
| 0.0004 | 524 | 0.5 | **61%** | 0 | 2/100 | **40%** | 0 |
| 0.0004 | 524 | 2 | 53% | 0 | 6/100 | 41% | 0 |

At both rates, attenuation by `0.5` produces a substantially milder safety intervention than hard
zeroing while preserving more math capability. Doubling the detected channels keeps HarmBehavior
ASR near the 34% origin value (37% and 41%), which is directionally consistent with the selected
channels carrying safety-relevant behavior. All scaling runs remain coherent under the primary
failure checks: there are no GSM8K extraction failures and no blank HarmBehavior responses.

## Historical raw 5-shot smoke tests

Before chat zero-shot evaluation was added, the first eight GSM8K rows were tested with raw 5-shot
prompts and 128 new tokens. These results remain useful only as a record of degeneration discovery:

| Configuration | Correct | Extraction failures | Coherent? |
|---|---:|---:|---|
| Origin | 4/8 | 0 | Yes |
| Random expanded V557 | 6/8 | 0 | Yes |
| Mixed detected pool, 2,000 entries | 0/8 | 6 | No |
| Revised mixed pool, 956 entries | 1/8 | 0 | No, repetitive |
| Expanded V557 | 5/8 | 0 | Yes |
| Expanded V1000 | 0/8 | 0 | No, repetitive |

These values should not be compared directly with the primary chat zero-shot tables.

## StrongREJECT balanced-60 evaluation

Nine global-rate conditions were also evaluated on StrongREJECT's bundled balanced small
dataset at
`/workspace/xcy/dataset/projects/iclr_neuron/StrongREJECT/data/strongreject_small_dataset.csv`.
It contains 60 direct harmful prompts: 10 examples from each of six behavior categories. Llama-3
responses use chat format, greedy decoding, and at most 512 new tokens.

Scoring uses the official `strongreject_finetuned` formulation with the fully local
`/workspace/xcy/models/google/gemma-2b` base and the downloaded `strongreject-15k-v1` LoRA
adapter. The primary metric is mean harmfulness in `[0,1]`; higher is more harmful.

| Global rate | SN entries | Mean StrongREJECT score | Score >= 0.5 | Refusal-substring ASR | Repetitive outputs | Blank responses |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0117 | 1/60 | 1.7% | 0/60 | 0 |
| 0.0001 | 131 | 0.0160 | 1/60 | 6.7% | not measured | 0 |
| 0.0002 | 262 | 0.0172 | 1/60 | 5.0% | not measured | 0 |
| 0.0004 | 524 | 0.0251 | 2/60 | 8.3% | not measured | 0 |
| **0.0006** | **786** | **0.0316** | **1/60** | **13.3%** | **8/60** | **0** |
| 0.0008 | 1,049 | 0.0312 | 1/60 | 36.7% | 20/60 | 0 |
| 0.0010 | 1,311 | 0.0159 | 0/60 | 40.0% | 36/60 | 0 |
| 0.0012 | 1,573 | 0.0203 | 0/60 | 58.3% | 48/60 | 0 |
| 0.0016 | 2,097 | 0.0115 | 0/60 | 95.0% | 49/60 | 0 |

StrongREJECT mean harmfulness rises through `0.0006`, is essentially flat at `0.0008`, and then
falls back toward the origin score. In contrast, refusal-substring ASR rises to 95% at `0.0016`.
That rise is not a useful jailbreak result: beginning at `0.0010`, most responses satisfy the
four-word repetition heuristic, and median unique-word ratios fall from 0.897 at `0.0006` to
0.110, 0.041, and 0.031 at `0.0010`, `0.0012`, and `0.0016`. The substring metric is therefore
mostly measuring broken refusals and degenerate generation at high rates. On this dataset,
`0.0006` is the best tested operating point by primary mean harmfulness; `0.0008` is the highest
arguably usable exploratory point, but already has substantially more repetition.

## Reproduction commands

### Detect the full expanded candidate pool

```bash
cd /workspace/xcy/safety_repro/iclr_neuron_expanded_kv/neuron_detection
/workspace/xcy/miniconda3/envs/safety-neuron-detect-expanded/bin/python \
  neuron_detection.py zou_train 200 112 vpool --prompt-format raw \
  --top-attn 1000 --top-ffn 2000
```

### Validate a global rate without loading model weights

```bash
cd /workspace/xcy/safety_repro/iclr_neuron_expanded_kv
/workspace/xcy/miniconda3/envs/safety-neuron-expanded-kv/bin/python \
  neuron_deactivate/run_table1_deactivated.py \
  --neurons neuron_detection/output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vpool_200.txt \
  --prepared /workspace/xcy/safety_repro/iclr_neuron/neuron_deactivate/evaluation_data/harm_behavior_first_100.jsonl \
  --deact-rate 0.0004 --structures fwd_up fwd_down q k v \
  --prompt-format raw --dry-run
```

### Run GSM8K-100 at a global rate

```bash
cd /workspace/xcy/safety_repro/iclr_neuron_expanded_kv
/workspace/xcy/miniconda3/envs/safety-neuron-expanded-kv/bin/python \
  neuron_deactivate/run_table1_capability.py gsm8k sn \
  --neurons neuron_detection/output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vpool_200.txt \
  --deact-rate 0.0004 --structures fwd_up fwd_down q k v \
  --prompt-format chat --num-fewshot 0 --limit 100 \
  --batch-size 16 --max-new-tokens 256
```

### Run HarmBehavior-100 at the same global rate

```bash
cd /workspace/xcy/safety_repro/iclr_neuron_expanded_kv
/workspace/xcy/miniconda3/envs/safety-neuron-expanded-kv/bin/python \
  neuron_deactivate/run_table1_deactivated.py \
  --neurons neuron_detection/output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vpool_200.txt \
  --prepared /workspace/xcy/safety_repro/iclr_neuron/neuron_deactivate/evaluation_data/harm_behavior_first_100.jsonl \
  --deact-rate 0.0004 --structures fwd_up fwd_down q k v \
  --prompt-format raw --batch-size 16 --max-new-tokens 512
```

Add `--neuron-scale 0.5` or `--neuron-scale 2` to either command to run a scaling intervention.
The complete requested grid can be run sequentially on GPU 0 with:

```bash
cd /workspace/xcy/safety_repro/iclr_neuron_expanded_kv
./neuron_deactivate/run_neuron_scaling_grid.sh \
  --rates "0.0002 0.0004" --scales "0.5 2" \
  --math-gpu 0 --harm-gpu 0
```

The grid runner defaults to the first 100 GSM8K rows, the first 100 HarmBehavior rows, batch size
16, greedy decoding, and the generation lengths documented above. Its flags allow these settings,
paths, rates, scales, seed, and GPU assignments to be changed.

### Run standalone Q or FFN-down sweeps

```bash
cd /workspace/xcy/safety_repro/iclr_neuron_expanded_kv
/workspace/xcy/miniconda3/envs/safety-neuron-expanded-kv/bin/python \
  neuron_deactivate/run_table1_capability.py gsm8k sn \
  --neurons neuron_detection/output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vpool_200.txt \
  --deact-count 256 --structures q \
  --prompt-format chat --num-fewshot 0 --limit 100 \
  --batch-size 16 --max-new-tokens 256
```

Replace `--structures q` with `--structures fwd_down` for the FFN-only condition.

## Output locations

- Global-rate GSM8K sweep:
  `neuron_deactivate/evaluation_outputs/global_rate_sweep_first100/`
- Global-rate HarmBehavior runs:
  `neuron_deactivate/evaluation_outputs/deact_sn/*_sn_rate*_all_raw_harm_behavior_first_100/`
- Standalone Q/FFN sweep:
  `neuron_deactivate/evaluation_outputs/table1_capability_q_ffn_sweep_first100/`
- Q32/FFN32 initial comparison:
  `neuron_deactivate/evaluation_outputs/table1_capability_q_ffn32_first100/`
- StrongREJECT balanced-60 generation and Gemma scores:
  `neuron_deactivate/evaluation_outputs/strongreject_balanced60/`
- Neuron-scaling GSM8K and HarmBehavior grid:
  `neuron_deactivate/evaluation_outputs/neuron_scaling_grid/`

Evaluation outputs are intentionally ignored by Git; summaries and run metadata remain available in
the worktree. The implementation, tests, and this report are version-controlled.
