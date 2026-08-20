# Gradient-neuron safety experiments

## Layout

- `data/processed/`: local processed AdvBench prompts and model responses.
- `scripts/`: generation, gradient extraction, IG, causal sweep, and safety-steering entry points.
- `docs/`: research plan and experiment reports.
- `results/gradients/`: canonical local-gradient, IG, and raw-refusal gradient artifacts.
- `results/safety_steering/`: prompt-conditional controller results and archived failed global controls.
- `results/causal_sweeps/`: zeroing and fixed-scaling experiments.
- `results/smoke/`: small validation runs retained for provenance.

## Current main result

The reusable fixed top-25 raw-gradient controller selected from disjoint
AdvBench rows 100--299 is documented in
`docs/fixed_neuron_disjoint_advbench_report.md`. Its evaluation output is in
`results/safety_steering/fixed_top25_disjoint_advbench/`.

Prompt-specific and failed fixed-mask experiments are retained under
`results/safety_steering/archive/`.

The response inspection for the exact 200-prompt expanded-K/V detector sample is
in `docs/zou_detector_sample_response_inspection.md`.
Dataset roles and current usage are summarized in `docs/dataset_usage.md`.
The strong controller's GSM8K capability result is documented in
`docs/gsm8k_strong_fixed_controller_report.md`.

Run scripts from this directory, for example:

```bash
CUDA_DEVICE=1 K_VALUES="25" EPSILONS="1.0" \
  RANKING="results/gradients/raw_refusal_advbench_rows100_299/top_neurons.csv" \
  bash scripts/run_grad_safety_steering.sh
```
