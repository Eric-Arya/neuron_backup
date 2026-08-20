#!/usr/bin/env bash
# Example: CUDA_VISIBLE_DEVICES=0,1 EVAL_BATCH_SIZE=10 NUM_SAMPLES=200 bash scripts/safety_neuron/get_change_scores.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/workspace/xcy/dataset/_cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_MODEL="${BASE_MODEL:-/workspace/xcy/models/Llama-2-7b-hf}"
SFT_ADAPTER="${SFT_ADAPTER:-${REPO_ROOT}/output/real_run}"
DPO_ADAPTER="${DPO_ADAPTER:-${REPO_ROOT}/output/dpo_real_run}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${SFT_ADAPTER}}"
DATASET="${DATASET:-/workspace/xcy/dataset/shared/hh_rlhf/harmless_base/test.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${REPO_ROOT}/output/change_scores/llama2_sft_vs_dpo_hh_harmless_sft_completion.pt}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TOKEN_TYPE="${TOKEN_TYPE:-completion}"
GENERATION_STARTSWITH="${GENERATION_STARTSWITH:-}"
LOG_FILE="${LOG_FILE:-${OUTPUT_FILE%.pt}.log}"

for path in "${BASE_MODEL}" "${SFT_ADAPTER}" "${DPO_ADAPTER}" "${TOKENIZER_PATH}" "${DATASET}"; do
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

printf '===== Change scores started: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'GPUs: %s | samples: %s | batch: %s | max new tokens: %s\n' \
  "${CUDA_VISIBLE_DEVICES}" "${NUM_SAMPLES}" "${EVAL_BATCH_SIZE}" "${MAX_NEW_TOKENS}"
printf 'M1/SFT: Base + %s\nM2/DPO: Base + %s + %s\n' \
  "${SFT_ADAPTER}" "${SFT_ADAPTER}" "${DPO_ADAPTER}"
printf 'Dataset: %s\nOutput: %s\n' "${DATASET}" "${OUTPUT_FILE}"

python -m src.change_scores \
  --dataset "${DATASET}" \
  --output_file "${OUTPUT_FILE}" \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --first_peft_path "${SFT_ADAPTER}" "${DPO_ADAPTER}" \
  --second_peft_path "${SFT_ADAPTER}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --token_type "${TOKEN_TYPE}" \
  --generation_startswith "${GENERATION_STARTSWITH}" \
  "$@"

printf '===== Change scores finished: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
