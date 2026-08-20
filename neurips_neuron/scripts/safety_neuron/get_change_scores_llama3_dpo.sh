#!/usr/bin/env bash
# Example: CUDA_VISIBLE_DEVICES=0,1 EVAL_BATCH_SIZE=10 NUM_SAMPLES=200 bash scripts/safety_neuron/get_change_scores_llama3_dpo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/workspace/xcy/dataset/_cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_MODEL="${BASE_MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
DPO_ADAPTER="${DPO_ADAPTER:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
DATASET="${DATASET:-/workspace/xcy/dataset/shared/hh_rlhf/harmless_base/test.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${REPO_ROOT}/output/change_scores/llama3_instruct_vs_dpo_hh_harmless_native_completion.pt}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TOKEN_TYPE="${TOKEN_TYPE:-completion}"
LOG_FILE="${LOG_FILE:-${OUTPUT_FILE%.pt}.log}"

for path in "${BASE_MODEL}" "${DPO_ADAPTER}" "${TOKENIZER_PATH}" "${DATASET}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'Required path does not exist: %s\n' "${path}" >&2
    exit 1
  fi
done

IFS=',' read -r -a GPU_ID_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_ID_ARRAY[@]} < 2 )); then
  printf 'Change-score computation expects multiple GPUs; got CUDA_VISIBLE_DEVICES=%s\n' \
    "${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")" "${HF_HOME}" "${HF_DATASETS_CACHE}"
exec > >(tee "${LOG_FILE}") 2>&1

printf '===== Llama-3 change scores started: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'Base: %s\nDPO guide: %s\nDataset: %s\nOutput: %s\n' \
  "${BASE_MODEL}" "${DPO_ADAPTER}" "${DATASET}" "${OUTPUT_FILE}"
printf 'GPUs=%s samples=%s batch=%s max_new_tokens=%s format=native\n' \
  "${CUDA_VISIBLE_DEVICES}" "${NUM_SAMPLES}" "${EVAL_BATCH_SIZE}" "${MAX_NEW_TOKENS}"

python -m src.change_scores \
  --dataset "${DATASET}" \
  --output_file "${OUTPUT_FILE}" \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --first_peft_path "${DPO_ADAPTER}" \
  --second_peft_path \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --token_type "${TOKEN_TYPE}" \
  --chat_format native \
  "$@"

sha256sum "${OUTPUT_FILE}" > "${OUTPUT_FILE}.sha256"
printf '===== Llama-3 change scores finished: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
