#!/usr/bin/env bash
# Example: GPUS=2 BATCH_SIZE=8 GRAD_ACCUM=2 LR=1e-6 DATA_LIMIT=50 ./run_sn_tune.sh
# Override any setting below through an environment variable or append extra train_neuron.py flags.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/xcy/miniconda3/envs/iclr_neuron_enhancement/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/workspace/xcy/miniconda3/envs/iclr_neuron_enhancement/bin/torchrun}"
GPUS="${GPUS:-2}"
MODEL="${MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
DATASET="${DATASET:-/workspace/xcy/dataset/projects/iclr_neuron/safety_neuron/training/circuit_breakers_train.json}"
NEURONS="${NEURONS:-${SCRIPT_DIR}/../neuron_detection/output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vpool_200.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/llama3_sn_tune_raw_50_fp32}"
DATA_LIMIT="${DATA_LIMIT:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
NEURON_CAP="${NEURON_CAP:-100}"
SELECTION_METHOD="${SELECTION_METHOD:-set-order}"
KV_MAP="${KV_MAP:-head-aware}"
SEED="${SEED:-112}"
PREPARE_FOR_KBIT="${PREPARE_FOR_KBIT:-0}"
SPARSE_FP32_DELTAS="${SPARSE_FP32_DELTAS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"

PREPARE_ARGS=(--no-prepare-for-kbit-training)
if [[ "${PREPARE_FOR_KBIT}" == "1" ]]; then
  PREPARE_ARGS=(--prepare-for-kbit-training)
fi
SPARSE_ARGS=(--no-sparse-fp32-deltas)
if [[ "${SPARSE_FP32_DELTAS}" == "1" ]]; then
  SPARSE_ARGS=(--sparse-fp32-deltas)
fi
CHECKPOINT_ARGS=(--no-gradient-checkpointing)
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  CHECKPOINT_ARGS=(--gradient-checkpointing)
fi

exec "${TORCHRUN_BIN}" --standalone --nproc-per-node="${GPUS}" \
  "${SCRIPT_DIR}/train_neuron.py" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --neurons "${NEURONS}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-limit "${DATA_LIMIT}" \
  --per-device-batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --learning-rate "${LR}" \
  --epochs "${EPOCHS}" \
  --max-seq-length "${MAX_SEQ_LENGTH}" \
  --neuron-cap "${NEURON_CAP}" \
  --selection-method "${SELECTION_METHOD}" \
  --kv-index-space expanded \
  --kv-map "${KV_MAP}" \
  "${PREPARE_ARGS[@]}" \
  "${SPARSE_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  --prompt-format "${PROMPT_FORMAT}" \
  --seed "${SEED}" \
  "$@"
