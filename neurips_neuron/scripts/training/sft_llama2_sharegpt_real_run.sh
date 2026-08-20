#!/usr/bin/env bash
# Example: NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=4 bash scripts/training/sft_llama2_sharegpt_real_run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# CUDA_VISIBLE_DEVICES selects which GPUs are used; NUM_GPUS can restrict the
# launch to the first N selected devices. The defaults use both GPUs here.
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
VISIBLE_GPU_COUNT="${#GPU_ID_ARRAY[@]}"
NUM_GPUS="${NUM_GPUS:-${VISIBLE_GPU_COUNT}}"

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || (( NUM_GPUS > VISIBLE_GPU_COUNT )); then
  printf 'NUM_GPUS must be an integer between 1 and %d (got %s)\n' \
    "${VISIBLE_GPU_COUNT}" "${NUM_GPUS}" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-/workspace/xcy/models/Llama-2-7b-hf}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/processed/sharegpt/sharegpt_data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/real_run}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-15}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-epoch}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/training.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

printf '\n===== Training started: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'Using %s GPU(s): %s\n' "${NUM_GPUS}" "${CUDA_VISIBLE_DEVICES}"

TRAIN_ARGS=(
  --train_file "${TRAIN_FILE}"
  --model_name_or_path "${MODEL_PATH}"
  --tokenizer_name "${TOKENIZER_PATH}"
  --use_ia3
  --ia3_module down_proj
  --feedforward_modules down_proj
  --use_flash_attn
  --torch_dtype bfloat16
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --gradient_checkpointing
  --quiet_preprocessing
  --logging_steps "${LOGGING_STEPS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --output_dir "${OUTPUT_DIR}"
  --seed "${SEED}"
)

# Extra arguments are appended, so this launcher remains usable with options
# added by newer versions of src.training.finetune.
accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --module src.training.finetune \
  "${TRAIN_ARGS[@]}" "$@"

printf '===== Training finished: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
