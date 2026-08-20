# Unified safety-neuron evaluation

This package evaluates safety-neuron interventions on Llama-3. **Grad is the original research
method proposed and developed in this project; it is evaluated here, not reproduced from prior
work.** IA3 post-training and SN-Tune are reproduced comparison methods from prior papers.

The active research scope is:

| Method | Runner name | Description |
|---|---|---|
| Baseline | `llama3_base` | Unmodified Meta-Llama-3-8B-Instruct |
| Grad (ours) | `grad` | Our positive-gradient activation-editing method; `positive-only` is the default direction |
| IA3 post-training | `llama3_sft` | SNCorpus raw-format SFT IA3, with displacement scaling around identity |
| IA3 activation patch | `llama3_sft_patch` | Patch ranked post-MLP activations from the SNCorpus raw-SFT IA3 guide |
| SN-Tune | `sn` | Merged SN-Tune checkpoint |
| Direct diagnostics | `sn_direct`, `neurips_direct` | Direct activation scaling for an explicit neuron ranking |

Safety-neuron selection and safety evaluation use raw prompts. The currently relevant evaluation
tasks are HarmBench, IFEval, BBH, and MATH-500. Legacy BeaverTails, Llama-2-7B, and HH-corpus DPO
support and artifacts have been removed from this directory.

## Repository layout

| Path | Contents |
|---|---|
| `unified_eval/` | Evaluator package and intervention implementations |
| `tests/` | Unit and protocol tests |
| `scripts/` | Dataset preparation, plotting, analysis, and launch scripts |
| `docs/reports/` | Current experiment reports and research notes |
| `docs/archive/` | Historical reports outside the active experiment scope |
| `figures/` | Generated paper figures |
| `results/` | Configuration-fingerprinted experiment artifacts |
| `papers/` | Local paper references |

## Commands

Validate the default raw HarmBench configuration:

```bash
python -m unified_eval.runner validate --method grad
python -m unified_eval.runner validate --method llama3_sft
python -m unified_eval.runner validate --method sn
```

Before the first full run for a method/task combination, benchmark a minimal slice of the real
dataset and retain the selected batch-size default in `resolve_method_defaults`:

```bash
python -m unified_eval.runner benchmark --method grad \
  --tasks harmbench --benchmark-task harmbench
```

Run raw HarmBench with the default positive-only Grad direction:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method grad --tasks harmbench
```

Run a capability evaluation explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m unified_eval.runner run \
  --method grad --tasks ifeval bbh math500
```

The launcher can run independent Grad and SN-Tune jobs concurrently:

```bash
COMMAND=run METHODS="grad sn" bash scripts/run_unified_eval.sh
```

Outputs are resumable under `results/<run-name>/`. Each run records semantic configuration,
runtime settings, validation metadata, summaries, and checksums.

See [the report index](docs/README.md) for completed experiments and [the script index](scripts/README.md)
for maintenance utilities.
