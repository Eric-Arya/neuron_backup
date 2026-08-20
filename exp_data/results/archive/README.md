# Archived NeurIPS runs

These directories are retained for provenance but excluded from the current
top-level unified summary:

- `neurips_20k_table2_adapted_256`: completed top-20K run using the obsolete
  adapted capability prompts and 256-token GSM8K limit. Its HarmBench outputs
  remain valid, but its capability results are not the NeurIPS paper protocol.
- `neurips_dpo_table2_adapted_256`: completed DPO run with the same obsolete
  capability settings.
- `neurips_8k_interrupted_table2_adapted`: interrupted at 128/200 HarmBench
  responses.
- `neurips_8k_interrupted_paper`: stopped after completing HarmBench generation;
  Beaver scoring and capability evaluation were not run.

The canonical `../neurips_dpo` directory is the completed DPO evaluation using
the NeurIPS paper capability prompts and a 1,024-token GSM8K limit.
