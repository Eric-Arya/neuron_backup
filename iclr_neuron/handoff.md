# Reproduction handoff

## Git state

- Original paper code: `main`, tag `original-paper-code`, commit
  `06e2e70957d6e9dadfb5e94116ccd4bd63f07d0b`.
- Reproduction code: branch `full-reproduction-experiments`, tag
  `full-reproduction-baseline`.
- Inspect or compare an original file with `git show original-paper-code:path/to/file.py` and
  `git diff original-paper-code -- path/to/file.py`.
- Stage named files, not `git add -A`; datasets, outputs, caches, and unrelated user changes must
  remain uncommitted.

## Environments and patched files

The repository copy is the source of truth. Each file is copied only to its matching environment;
the Transformers versions are incompatible across phases.

| Phase | Environment (Transformers) | Repository file | Path below environment `transformers/` |
|---|---|---|---|
| Detection | `iclr_neuron_detection` (4.42.4) | `neuron_detection/transformers/models/llama/modeling_llama.py` | `models/llama/modeling_llama.py` |
| Deactivation | `iclr_neuron_deactivation` (4.44.2) | `neuron_deactivate/transformers/generation/utils.py` | `generation/utils.py` |
| Deactivation | `iclr_neuron_deactivation` (4.44.2) | `neuron_deactivate/transformers/models/llama/modeling_llama.py` | `models/llama/modeling_llama.py` |
| Enhancement | `iclr_neuron_enhancement` (4.38.2) | `neuron_enhancement/transformers/trainer.py` | `trainer.py` |

Environment roots are
`/workspace/xcy/miniconda3/envs/<environment>/lib/python3.9/site-packages/transformers/`.
Full package snapshots are in `environment_snapshots/`. Phase drivers are
`neuron_detection/neuron_detection.py`, `neuron_deactivate/run_table1_deactivated.py`, and
`neuron_enhancement/train_neuron.py`.

## Editing workflow

1. Edit the repository file and inspect `git diff -- <file>`.
2. Copy it to the matching environment path in the table.
3. Run `cmp <repository-file> <environment-file>`; status 0 is required.
4. Start a new Python process and run the relevant test/experiment.
5. Stage the named source file, commit it, and record `git rev-parse HEAD` with the result.

If a file was edited directly in `site-packages`, copy it back to the phase-specific repository path
before further editing, inspect the diff, and commit it. Never leave the only copy in the environment.

## Current reproduction constants

- Models: `/workspace/xcy/models`; dataset root: `/workspace/xcy/dataset`.
- Shared raw/source datasets live under `shared/`, project-generated datasets under
  `projects/iclr_neuron/`, and the Hugging Face cache under `_cache/huggingface/`.
- Deactivation rate: `0.0005`; seed: `112`.
- Neuron file:
  `neuron_detection/output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn100_ffn200_kvnative_chat_native_200.txt`.
- Neuron-file SHA-256:
  `9e532e49ac7bc9a089264bf3676d6c9b8cadaf429fa91406e34db86f34f7771c`.
- Chat is the primary safety protocol. Raw is diagnostic; the two must use separate output paths.
