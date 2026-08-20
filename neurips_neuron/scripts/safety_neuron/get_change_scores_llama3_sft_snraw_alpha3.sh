#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-/workspace/xcy/dataset/_cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_MODEL="${BASE_MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20}"
DATASET="${DATASET:-/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/selection/circuit_breakers_heldout_seed42_n200_train256_harmbench_disjoint.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${REPO_ROOT}/output/change_scores/llama3_instruct_vs_sft_snrawdot256_alpha3_snheldout_seed42_n200_raw_completion.pt}"
EXPECTED_DATASET_SHA256="${EXPECTED_DATASET_SHA256:-cae3fad3cf87ab74a7097a3c58771f6c79395c3b6966dc6a9eda36920faa5be8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
IA3_ALPHA="${IA3_ALPHA:-3}"
LOG_FILE="${LOG_FILE:-${OUTPUT_FILE%.pt}.log}"

for path in "${BASE_MODEL}" "${SFT_ADAPTER}" "${DATASET}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'Required path does not exist: %s\n' "${path}" >&2
    exit 1
  fi
done

actual_dataset_sha256="$(sha256sum "${DATASET}" | cut -d' ' -f1)"
if [[ "${actual_dataset_sha256}" != "${EXPECTED_DATASET_SHA256}" ]]; then
  printf 'Selection dataset checksum mismatch: expected %s, got %s\n' \
    "${EXPECTED_DATASET_SHA256}" "${actual_dataset_sha256}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")" "${HF_HOME}" "${HF_DATASETS_CACHE}"
exec > >(tee "${LOG_FILE}") 2>&1

printf '===== Raw SN-heldout alpha-%s change scores started: %s =====\n' \
  "${IA3_ALPHA}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'Base: %s\nGuide: %s\nDataset: %s\nOutput: %s\n' \
  "${BASE_MODEL}" "${SFT_ADAPTER}" "${DATASET}" "${OUTPUT_FILE}"
printf 'GPUs=%s samples=%s batch=%s max_new_tokens=%s format=raw suffix=. alpha=%s\n' \
  "${CUDA_VISIBLE_DEVICES}" "${NUM_SAMPLES}" "${EVAL_BATCH_SIZE}" \
  "${MAX_NEW_TOKENS}" "${IA3_ALPHA}"

python -m src.change_scores \
  --dataset "${DATASET}" \
  --output_file "${OUTPUT_FILE}" \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name_or_path "${BASE_MODEL}" \
  --first_peft_path "${SFT_ADAPTER}" \
  --second_peft_path \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --token_type completion \
  --chat_format raw \
  --generation_startswith . \
  --first_ia3_alpha "${IA3_ALPHA}" \
  "$@"

sha256sum "${OUTPUT_FILE}" > "${OUTPUT_FILE}.sha256"
printf '===== Change scores finished: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
