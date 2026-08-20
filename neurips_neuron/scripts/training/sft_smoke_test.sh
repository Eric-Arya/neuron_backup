#!/usr/bin/env bash
# Example: NUM_GPUS=2 SMOKE_EXAMPLES=8 bash scripts/training/sft_smoke_test.sh
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

# One-step SFT smoke test on a small slice of the real ShareGPT data.
MODEL_PATH="${MODEL_PATH:-/workspace/xcy/models/Llama-2-7b-hf}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/processed/sharegpt/sharegpt_data.jsonl}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/xcy/tmp/safety_neuron_sft_smoke}"
SMOKE_FILE="${SMOKE_FILE:-/workspace/xcy/tmp/safety_neuron_sft_smoke/train.jsonl}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$SMOKE_FILE")"
head -n "$SMOKE_EXAMPLES" "$TRAIN_FILE" > "$SMOKE_FILE"

accelerate launch \
  --num_machines 1 \
  --num_processes "$NUM_GPUS" \
  --main_process_port "${MAIN_PROCESS_PORT:-29501}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --module src.training.finetune \
  --train_file "$SMOKE_FILE" \
  --model_name_or_path "$MODEL_PATH" \
  --use_lora \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0 \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 1 \
  --logging_steps 1 \
  --learning_rate 1e-4 \
  --output_dir "$OUTPUT_DIR" \
  --seed "${SEED:-0}" \
  "$@"

test -f "$OUTPUT_DIR/adapter_config.json" || {
  echo "SFT smoke test did not produce $OUTPUT_DIR/adapter_config.json" >&2
  exit 1
}
echo "SFT smoke test passed: $OUTPUT_DIR"
