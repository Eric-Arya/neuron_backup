# Scripts

Run scripts from the repository root. Python scripts add the repository root to `sys.path`, so direct
invocation remains supported after the directory reorganization.

Grad scripts implement and evaluate the original method developed in this project. IA3
post-training and SN-Tune scripts support comparisons with methods from prior papers.

| Script group | Files |
|---|---|
| Dataset preparation | `prepare_bbh_subset.py`, `prepare_math500_subset.py`, `prepare_expanded_harmbench.py`, `prepare_sn_heldout_selection.py`, `construct_pku_contrastive_pairs.py` |
| Grad method development and analysis | `extract_pku_contrastive_gradients.py` |
| Result normalization | `finalize_expanded_kv_runs.py` |
| Plotting | `plot_ifeval_harmbench_tradeoff.py`, `plot_bbh_harmbench_tradeoff.py` |
| Launchers | `run_unified_eval.sh`, `run_grad_harmbench_development.sh` |

Examples:

```bash
python scripts/prepare_math500_subset.py --help
python scripts/plot_ifeval_harmbench_tradeoff.py
bash scripts/run_unified_eval.sh
```
