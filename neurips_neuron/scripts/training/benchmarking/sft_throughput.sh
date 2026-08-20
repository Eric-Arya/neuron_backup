#!/usr/bin/env bash
# Example: NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=15 GRADIENT_CHECKPOINTING=1 bash scripts/training/benchmarking/sft_throughput.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# The default benchmark processes exactly 240 examples. Set NUM_EXAMPLES only
# when doing a smaller smoke check or a larger throughput run.
NUM_EXAMPLES="${NUM_EXAMPLES:-240}"
SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE:-${REPO_ROOT}/data/processed/sharegpt/sharegpt_data.jsonl}"
BENCHMARK_FILE="${BENCHMARK_FILE:-${SCRIPT_DIR}/sharegpt_${NUM_EXAMPLES}.jsonl}"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
VISIBLE_GPU_COUNT="${#GPU_ID_ARRAY[@]}"
NUM_GPUS="${NUM_GPUS:-${VISIBLE_GPU_COUNT}}"

if ! [[ "${NUM_EXAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NUM_EXAMPLES must be a positive integer (got %s)\n' "${NUM_EXAMPLES}" >&2
  exit 2
fi
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || (( NUM_GPUS > VISIBLE_GPU_COUNT )); then
  printf 'NUM_GPUS must be an integer between 1 and %d (got %s)\n' \
    "${VISIBLE_GPU_COUNT}" "${NUM_GPUS}" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-/workspace/xcy/models/Llama-2-7b-hf}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/batch${PER_DEVICE_TRAIN_BATCH_SIZE:-8}_accum${GRADIENT_ACCUMULATION_STEPS:-15}_ckpt${GRADIENT_CHECKPOINTING:-1}}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-15}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SEED="${SEED:-42}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
SAVE_OUTPUT="${SAVE_OUTPUT:-0}"

if ! [[ "${PER_DEVICE_TRAIN_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo 'PER_DEVICE_TRAIN_BATCH_SIZE and GRADIENT_ACCUMULATION_STEPS must be positive integers.' >&2
  exit 2
fi
if ! [[ "${PREPROCESSING_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo 'PREPROCESSING_NUM_WORKERS must be a non-negative integer.' >&2
  exit 2
fi
if [[ "${SAVE_OUTPUT}" != 0 && "${SAVE_OUTPUT}" != 1 && "${SAVE_OUTPUT}" != true && "${SAVE_OUTPUT}" != false ]]; then
  echo 'SAVE_OUTPUT must be 0, 1, true, or false.' >&2
  exit 2
fi

SOURCE_COUNT="$(wc -l < "${SOURCE_TRAIN_FILE}")"
if (( SOURCE_COUNT < NUM_EXAMPLES )); then
  printf 'Source file has %d examples, but %d were requested: %s\n' \
    "${SOURCE_COUNT}" "${NUM_EXAMPLES}" "${SOURCE_TRAIN_FILE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${BENCHMARK_FILE}")"
head -n "${NUM_EXAMPLES}" "${SOURCE_TRAIN_FILE}" > "${BENCHMARK_FILE}.tmp"
mv "${BENCHMARK_FILE}.tmp" "${BENCHMARK_FILE}"

EXTRACTED_COUNT="$(wc -l < "${BENCHMARK_FILE}")"
if (( EXTRACTED_COUNT != NUM_EXAMPLES )); then
  printf 'Expected %d extracted examples, found %d.\n' \
    "${NUM_EXAMPLES}" "${EXTRACTED_COUNT}" >&2
  exit 1
fi

TRAIN_ARGS=(
  --train_file "${BENCHMARK_FILE}"
  --model_name_or_path "${MODEL_PATH}"
  --tokenizer_name "${TOKENIZER_PATH}"
  --use_ia3
  --ia3_module down_proj
  --feedforward_modules down_proj
  --torch_dtype "${TORCH_DTYPE}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs 1
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
  --logging_steps "${LOGGING_STEPS}"
  --seed "${SEED}"
)

if [[ "${GRADIENT_CHECKPOINTING}" == 1 || "${GRADIENT_CHECKPOINTING}" == true ]]; then
  TRAIN_ARGS+=(--gradient_checkpointing)
fi
if [[ "${USE_FLASH_ATTN}" == 1 || "${USE_FLASH_ATTN}" == true ]]; then
  TRAIN_ARGS+=(--use_flash_attn)
fi
if [[ "${SAVE_OUTPUT}" == 1 || "${SAVE_OUTPUT}" == true ]]; then
  mkdir -p "${OUTPUT_DIR}"
  TRAIN_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))
printf 'Benchmarking %d examples on %d GPU(s): %s\n' \
  "${NUM_EXAMPLES}" "${NUM_GPUS}" "${CUDA_VISIBLE_DEVICES}"
printf 'Per-device batch=%d, accumulation=%d, effective batch=%d, checkpointing=%s\n' \
  "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" \
  "${EFFECTIVE_BATCH_SIZE}" "${GRADIENT_CHECKPOINTING}"
printf 'Benchmark slice: %s\n' "${BENCHMARK_FILE}"
if [[ "${SAVE_OUTPUT}" == 1 || "${SAVE_OUTPUT}" == true ]]; then
  printf 'Output directory: %s\n' "${OUTPUT_DIR}"
else
  printf 'Model output saving: disabled\n'
fi

START_SECONDS="${SECONDS}"
accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --module src.training.finetune \
  "${TRAIN_ARGS[@]}" "$@"
ELAPSED_SECONDS=$((SECONDS - START_SECONDS))

if (( ELAPSED_SECONDS > 0 )); then
  EXAMPLES_PER_SECOND="$(awk -v examples="${NUM_EXAMPLES}" -v seconds="${ELAPSED_SECONDS}" \
    'BEGIN { printf "%.2f", examples / seconds }')"
else
  EXAMPLES_PER_SECOND="n/a"
fi
printf 'Benchmark complete: %d examples in %d seconds (%s examples/second, including load/preprocessing).\n' \
  "${NUM_EXAMPLES}" "${ELAPSED_SECONDS}" "${EXAMPLES_PER_SECOND}"
