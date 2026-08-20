#!/usr/bin/env bash
# Raw prompt/completion IA3 SFT on the refusal corpus used by SN-Tune.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
VISIBLE_GPU_COUNT="${#GPU_ID_ARRAY[@]}"
NUM_GPUS="${NUM_GPUS:-${VISIBLE_GPU_COUNT}}"

MODEL_PATH="${MODEL_PATH:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/training/circuit_breakers_train_first256_llama3_raw_snformat.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29513}"

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || (( NUM_GPUS > VISIBLE_GPU_COUNT )); then
  printf 'NUM_GPUS must be between 1 and %d (got %s)\n' "${VISIBLE_GPU_COUNT}" "${NUM_GPUS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
TRAIN_ARGS=(
  --train_file "${TRAIN_FILE}"
  --model_name_or_path "${MODEL_PATH}"
  --tokenizer_name "${MODEL_PATH}"
  --use_ia3
  --ia3_module down_proj
  --feedforward_modules down_proj
  --use_flash_attn
  --torch_dtype bfloat16
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type cosine
  --weight_decay 0.1
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --quiet_preprocessing
  --logging_steps 1
  --output_dir "${OUTPUT_DIR}"
  --seed 112
)
if [[ "${GRADIENT_CHECKPOINTING}" == 1 ]]; then
  TRAIN_ARGS+=(--gradient_checkpointing)
fi

exec accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --mixed_precision bf16 \
  --module src.training.finetune \
  "${TRAIN_ARGS[@]}" "$@"
