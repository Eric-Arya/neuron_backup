#!/usr/bin/env bash
# Example: COMMAND=run TOP_K_VALUES="20000 8000" CUDA_VISIBLE_DEVICES=0,1 bash run_llama3_dpo_patch_sweep.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

COMMAND="${COMMAND:-run}"
TOP_K_VALUES="${TOP_K_VALUES:-20000 8000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RANKING="${RANKING:-/workspace/xcy/safety_repro/neurips_neuron/output/change_scores/llama3_instruct_vs_dpo_hh_harmless_native_completion.pt}"
LLAMA3_DPO_ADAPTER="${LLAMA3_DPO_ADAPTER:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH}"
HARMBENCH_BATCH_SIZE="${HARMBENCH_BATCH_SIZE:-16}"
GSM8K_BATCH_SIZE="${GSM8K_BATCH_SIZE:-16}"
MMLU_BATCH_SIZE="${MMLU_BATCH_SIZE:-8}"
COST_BATCH_SIZE="${COST_BATCH_SIZE:-16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

read -r -a TOP_K_ARRAY <<< "${TOP_K_VALUES}"
read -r -a EXTRA_ARRAY <<< "${EXTRA_ARGS}"
for top_k in "${TOP_K_ARRAY[@]}"; do
  if ! [[ "${top_k}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'TOP_K_VALUES entries must be positive integers; got %s\n' "${top_k}" >&2
    exit 2
  fi
  run_name="llama3_dpo_patch_$((top_k / 1000))k"
  "${PYTHON_BIN}" -m unified_eval.runner "${COMMAND}" \
    --method llama3_dpo_patch \
    --run-name "${run_name}" \
    --llama3-dpo-patch-top-k "${top_k}" \
    --llama3-dpo-patch-ranking "${RANKING}" \
    --llama3-dpo-adapter "${LLAMA3_DPO_ADAPTER}" \
    --output-root "${OUTPUT_ROOT}" \
    --harmbench-batch-size "${HARMBENCH_BATCH_SIZE}" \
    --gsm8k-batch-size "${GSM8K_BATCH_SIZE}" \
    --mmlu-batch-size "${MMLU_BATCH_SIZE}" \
    --cost-batch-size "${COST_BATCH_SIZE}" \
    --base-device cuda:0 --guide-device cuda:1 --cost-device cuda:0 \
    "${EXTRA_ARRAY[@]}"
done

"${PYTHON_BIN}" -m unified_eval.runner summarize --output-root "${OUTPUT_ROOT}"
