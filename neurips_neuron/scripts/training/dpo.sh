#!/usr/bin/env bash
# Example: NUM_GPUS=2 SAVE_STEPS=100 bash scripts/training/dpo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/workspace/xcy/dataset/_cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

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

BASE_MODEL="${BASE_MODEL:-/workspace/xcy/models/Llama-2-7b-hf}"
SFT_ADAPTER="${SFT_ADAPTER-${REPO_ROOT}/output/real_run}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${SFT_ADAPTER:-${BASE_MODEL}}}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/xcy/dataset/shared/hh_rlhf/harmless_base/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/dpo_real_run}"
CHAT_FORMAT="${CHAT_FORMAT:-tulu}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-15}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
BETA="${BETA:-0.1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
SEED="${SEED:-42}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29510}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
REPORT_TO="${REPORT_TO:-none}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-8}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"

for path in "${BASE_MODEL}" "${TOKENIZER_PATH}" "${TRAIN_FILE}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'Required path does not exist: %s\n' "${path}" >&2
    exit 1
  fi
done
if [[ -n "${SFT_ADAPTER}" && ! -e "${SFT_ADAPTER}" ]]; then
  printf 'Required SFT adapter does not exist: %s\n' "${SFT_ADAPTER}" >&2
  exit 1
fi
if [[ "${CHAT_FORMAT}" != tulu && "${CHAT_FORMAT}" != native ]]; then
  printf 'CHAT_FORMAT must be tulu or native (got %s)\n' "${CHAT_FORMAT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/training.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

TRAIN_ARGS=(
  --model_name_or_path "${BASE_MODEL}"
  --use_ia3
  --ia3_module down_proj
  --feedforward_modules down_proj
  --tokenizer_name "${TOKENIZER_PATH}"
  --chat_format "${CHAT_FORMAT}"
  --train_file "${TRAIN_FILE}"
  --do_train
  --bf16 true
  --torch_dtype "${TORCH_DTYPE}"
  --beta "${BETA}"
  --max_length "${MAX_LENGTH}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --warmup_ratio "${WARMUP_RATIO}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --logging_steps "${LOGGING_STEPS}"
  --save_strategy "${SAVE_STRATEGY}"
  --save_steps "${SAVE_STEPS}"
  --remove_unused_columns false
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}"
  --dataset_num_proc "${PREPROCESSING_NUM_WORKERS}"
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}"
  --output_dir "${OUTPUT_DIR}"
  --report_to "${REPORT_TO}"
  --seed "${SEED}"
)

if [[ -n "${SFT_ADAPTER}" ]]; then
  TRAIN_ARGS+=(--peft_name_or_path "${SFT_ADAPTER}")
fi

if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
  TRAIN_ARGS+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    printf 'RESUME_FROM_CHECKPOINT is not a directory: %s\n' \
      "${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ "${GRADIENT_CHECKPOINTING}" == 1 || "${GRADIENT_CHECKPOINTING}" == true ]]; then
  TRAIN_ARGS+=(--gradient_checkpointing)
fi
if [[ "${USE_FLASH_ATTN}" == 1 || "${USE_FLASH_ATTN}" == true ]]; then
  TRAIN_ARGS+=(--use_flash_attn true)
else
  TRAIN_ARGS+=(--use_flash_attn false)
fi

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
printf '\n===== DPO started: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'GPUs=%d (%s), per-device batch=%d, accumulation=%d, effective batch=%d\n' \
  "${NUM_GPUS}" "${CUDA_VISIBLE_DEVICES}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  "${GRADIENT_ACCUMULATION_STEPS}" "${EFFECTIVE_BATCH_SIZE}"
printf 'Reference: %s%s\nChat format: %s\nPreference data: %s\nOutput: %s\n' \
  "${BASE_MODEL}" "${SFT_ADAPTER:+ + ${SFT_ADAPTER}}" "${CHAT_FORMAT}" \
  "${TRAIN_FILE}" "${OUTPUT_DIR}"
printf 'Checkpoints: strategy=%s, every=%s step(s), limit=%s, resume=%s\n' \
  "${SAVE_STRATEGY}" "${SAVE_STEPS}" "${SAVE_TOTAL_LIMIT:-none}" \
  "${RESUME_FROM_CHECKPOINT:-none}"

accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --module src.training.dpo \
  "${TRAIN_ARGS[@]}" "$@"

printf '===== DPO finished: %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
