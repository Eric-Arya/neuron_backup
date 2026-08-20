# Reproduce Table 1: Llama2 Base, Ours, HarmBench (+63)

## Target

Reproduce only the Llama2 **Base → Ours** HarmBench result from Table 1: approximately **+63% causal effect**.

HarmBench is only the evaluation dataset. Safety-neuron ranks remain those already computed on HH-RLHF-Harmless, preserving the paper's transfer setting.

## Paper configuration

- HarmBench: deterministically sample 200 of the 400 behaviors with seed 42
- Prompt format: Tulu chat format
- Decoding: greedy
- Maximum new tokens: 128
- Safety neurons: top 20,000 from the existing change-score ranking
- Base: pretrained Llama2 without adapters
- DPO: Base → SFT adapter → DPO adapter
- Patched Base: dynamically replace the selected Base neuron activations with DPO activations at every generated token
- Metric: `beaver-7b-v1.0-cost`; lower cost is safer

The released Table 1 script uses 20,000 neurons, approximately 5.86% of the 341,248 scored neurons.

## Plan

1. Prepare HarmBench.
   - Convert the shared 400-row CSV into
     `/workspace/xcy/dataset/projects/neurips_neuron/harmbench/processed/prompts_all_400.jsonl`.
   - Store the frozen seed-42 sample at
     `/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/table1_seed42_n200.jsonl`.
   - Freeze the sampled 200 prompt IDs so all conditions use identical inputs.

2. Prepare the cost model. --- model is already present
   - Obtain the currently missing `beaver-7b-v1.0-cost` model. 
   - Validate loading, finite scores, and the score direction.

3. Modernize the released evaluation.
   - Support the current separate SFT and DPO adapters.
   - Repair dynamic activation patching for the installed Transformers version.
   - Add flexible paths/hyperparameters, multi-GPU defaults, logging, and resumable per-batch outputs.

4. Smoke-test and benchmark.
   - Test two real HarmBench prompts with eight generated tokens.
   - Verify zero-neuron patching exactly matches Base.
   - Verify top-20,000 patching changes model behavior.
   - Benchmark safe batch sizes on both H100 GPUs.

5. Run the three required conditions on the same 200 prompts.
   - Base
   - Full DPO
   - Base dynamically patched from DPO using the top 20,000 neurons

6. Compute and validate the causal effect.

   ```text
   causal_effect = 100 * (patched_cost - base_cost) / (dpo_cost - base_cost)
   ```

   Paper reference values:

   - Base cost: 8.0
   - Patched Base cost: -3.9
   - DPO cost: -11.0
   - Causal effect: approximately +63%

7. Save the prompt manifest, generations, per-example costs, aggregate result, logs, and checksums.

## Scope and estimate

This first run excludes other datasets, the SFT panel, random-neuron baselines, and the full Figure 2 sweep. Estimated time after obtaining the cost model: **3–6 hours**.

## Implementation result (2026-08-15)

Implemented in `src/eval/table1_harmbench.py` with flexible two-GPU shell entry
points in `scripts/eval/table1_llama_base_harmbench.sh` and
`scripts/eval/table1_llama_base_harmbench_pipeline.sh`.

- Frozen manifest: 200/200 unique seed-42 IDs; SHA-256
  `bb5b29ff9db15e420021aee3ad1a07d0ed1ca11a2d8faff024d786168b7be74c`
- Smoke test: zero-neuron output exactly matched Base; top-20,000 changed both
  tested generations; Beaver scores were finite and lower was confirmed safer.
- Benchmark: batch size 16 was fastest among 2, 4, 8, and 16 for both patched
  generation and cost scoring on the two H100s.
- Full run: Base `7.84996`, patched Base `2.08289`, DPO `-3.34752`, giving a
  causal effect of `51.5033%` over the identical 200 prompts.

The run is complete but does **not** reproduce the paper's approximate `+63%`:
the Base value matches the paper reference (`8.0`) closely, while the locally
trained DPO and patched checkpoints are substantially less safe than the paper
references (`-11.0` and `-3.9`). Complete artifacts and verified checksums are
under `results/table1_llama_base_harmbench/`.
