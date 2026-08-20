#!/usr/bin/env bash
# Example: CUDA_VISIBLE_DEVICES=1,3 NUM_GPUS=2 bash scripts/training/sft_llama2_sharegpt_brief.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/processed/sharegpt/sharegpt_data.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/llama2_sharegpt_ia3_ff_42}"

accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29500}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --module src.training.finetune \
  --train_file "${TRAIN_FILE}" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name "${TOKENIZER_PATH:-${MODEL_PATH}}" \
  --use_ia3 \
  --ia3_module down_proj \
  --feedforward_modules down_proj \
  --use_flash_attn \
  --torch_dtype "${TORCH_DTYPE:-bfloat16}" \
  --max_seq_length "${MAX_SEQ_LENGTH:-4096}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-15}" \
  --learning_rate "${LEARNING_RATE:-1e-3}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --weight_decay "${WEIGHT_DECAY:-0.1}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --gradient_checkpointing \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-epoch}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED:-42}" \
  "$@"
