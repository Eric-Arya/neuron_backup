#!/usr/bin/env bash
# Example: COMMAND=run METHODS="llama3_dpo" LLAMA3_DPO_ADAPTER=/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH HARMBENCH_BATCH_SIZE=16 bash run_unified_eval.sh
# Smoke: COMMAND=run METHODS="grad" EXTRA_ARGS="--harmbench-limit 2 --gsm8k-limit 2 --mmlu-limit 2 --harmbench-max-new-tokens 8 --gsm8k-max-new-tokens 8" bash run_unified_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

COMMAND="${COMMAND:-run}"
METHODS="${METHODS:-grad sn neurips}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results}"
HARMBENCH_BATCH_SIZE="${HARMBENCH_BATCH_SIZE:-16}"
GSM8K_BATCH_SIZE="${GSM8K_BATCH_SIZE:-16}"
MMLU_BATCH_SIZE="${MMLU_BATCH_SIZE:-8}"
MATH500_BATCH_SIZE="${MATH500_BATCH_SIZE:-8}"
COST_BATCH_SIZE="${COST_BATCH_SIZE:-16}"
LLAMA3_DPO_ADAPTER="${LLAMA3_DPO_ADAPTER:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${OUTPUT_ROOT}"

read -r -a METHOD_LIST <<< "${METHODS}"
read -r -a EXTRA_ARRAY <<< "${EXTRA_ARGS}"

run_method() {
  local method="$1"
  local visible_gpus="$2"
  local guide_device="cuda:0"
  if [[ "${method}" == "neurips" ]]; then
    guide_device="cuda:1"
  fi
  CUDA_VISIBLE_DEVICES="${visible_gpus}" "${PYTHON_BIN}" -m unified_eval.runner "${COMMAND}" \
    --method "${method}" \
    --output-root "${OUTPUT_ROOT}" \
    --harmbench-batch-size "${HARMBENCH_BATCH_SIZE}" \
    --gsm8k-batch-size "${GSM8K_BATCH_SIZE}" \
    --mmlu-batch-size "${MMLU_BATCH_SIZE}" \
    --math500-batch-size "${MATH500_BATCH_SIZE}" \
    --cost-batch-size "${COST_BATCH_SIZE}" \
    --llama3-dpo-adapter "${LLAMA3_DPO_ADAPTER}" \
    --device cuda:0 --base-device cuda:0 --guide-device "${guide_device}" --cost-device cuda:0 \
    "${EXTRA_ARRAY[@]}"
}

want_method() {
  local wanted="$1"
  local method
  for method in "${METHOD_LIST[@]}"; do
    [[ "${method}" == "${wanted}" ]] && return 0
  done
  return 1
}

# The two independent Llama-3 approaches occupy one GPU each and run together.
pids=()
if want_method grad; then
  run_method grad 0 >"${OUTPUT_ROOT}/grad_${COMMAND}.log" 2>&1 &
  pids+=("$!")
fi
if want_method sn; then
  run_method sn 1 >"${OUTPUT_ROOT}/sn_${COMMAND}.log" 2>&1 &
  pids+=("$!")
fi
for pid in "${pids[@]}"; do
  wait "${pid}"
done

# The standalone Llama-3 DPO adapter uses one GPU and runs after any parallel pair.
if want_method llama3_dpo; then
  run_method llama3_dpo 0 | tee "${OUTPUT_ROOT}/llama3_dpo_${COMMAND}.log"
fi

# Dynamic NeurIPS patching needs a base model and DPO guide on separate GPUs.
if want_method neurips; then
  run_method neurips 0,1 | tee "${OUTPUT_ROOT}/neurips_${COMMAND}.log"
fi

"${PYTHON_BIN}" -m unified_eval.runner summarize --output-root "${OUTPUT_ROOT}"
