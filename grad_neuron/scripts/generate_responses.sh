#!/usr/bin/env bash
# Example override:
# NUM_GPUS=2 NUM_EXAMPLES=200 BATCH_SIZE=16 MAX_NEW_TOKENS=256 bash scripts/generate_responses.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
INPUT_CSV="${INPUT_CSV:-/workspace/xcy/dataset/shared/advbench/raw/harmful_behaviors.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/data/processed/llama3_8b_instruct_first200.csv}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
START_INDEX="${START_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-42}"
PROMPT_FORMAT="${PROMPT_FORMAT:-chat}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" "${SCRIPT_DIR}/generate_responses.py" \
  --model "${MODEL_PATH}" \
  --input-csv "${INPUT_CSV}" \
  --output-csv "${OUTPUT_CSV}" \
  --num-examples "${NUM_EXAMPLES}" \
  --start-index "${START_INDEX}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --seed "${SEED}" \
  --prompt-format "${PROMPT_FORMAT}"
