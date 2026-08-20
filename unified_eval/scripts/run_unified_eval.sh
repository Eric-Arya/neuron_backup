#!/usr/bin/env bash
# Example: COMMAND=run METHODS="grad sn" bash scripts/run_unified_eval.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COMMAND="${COMMAND:-run}"
METHODS="${METHODS:-grad sn}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results}"
HARMBENCH_BATCH_SIZE="${HARMBENCH_BATCH_SIZE:-16}"
GSM8K_BATCH_SIZE="${GSM8K_BATCH_SIZE:-16}"
MMLU_BATCH_SIZE="${MMLU_BATCH_SIZE:-8}"
MATH500_BATCH_SIZE="${MATH500_BATCH_SIZE:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${OUTPUT_ROOT}"

read -r -a METHOD_LIST <<< "${METHODS}"
read -r -a EXTRA_ARRAY <<< "${EXTRA_ARGS}"

run_method() {
  local method="$1"
  local visible_gpus="$2"
  CUDA_VISIBLE_DEVICES="${visible_gpus}" "${PYTHON_BIN}" -m unified_eval.runner "${COMMAND}" \
    --method "${method}" \
    --output-root "${OUTPUT_ROOT}" \
    --harmbench-batch-size "${HARMBENCH_BATCH_SIZE}" \
    --gsm8k-batch-size "${GSM8K_BATCH_SIZE}" \
    --mmlu-batch-size "${MMLU_BATCH_SIZE}" \
    --math500-batch-size "${MATH500_BATCH_SIZE}" \
    --device cuda:0 --base-device cuda:0 --guide-device cuda:1 \
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

for method in llama3_base llama3_sft llama3_sft_patch sn_direct neurips_direct; do
  if want_method "${method}"; then
    visible_gpus=0
    [[ "${method}" == "llama3_sft_patch" ]] && visible_gpus=0,1
    run_method "${method}" "${visible_gpus}" | tee "${OUTPUT_ROOT}/${method}_${COMMAND}.log"
  fi
done

"${PYTHON_BIN}" -m unified_eval.runner summarize --output-root "${OUTPUT_ROOT}"
