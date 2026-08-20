#!/usr/bin/env bash
# Example override:
# NUM_GPUS=2 MAX_EXAMPLES=20 TOP_K=500 OUTPUT_DIR=/path/to/output bash scripts/extract_i_cannot_gradients.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
INPUT_CSV="${INPUT_CSV:-${PROJECT_ROOT}/data/processed/llama3_8b_instruct_first200.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/gradients/i_cannot}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
TOP_K="${TOP_K:-2000}"
TOP_PER_LAYER="${TOP_PER_LAYER:-50}"
SEED="${SEED:-42}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  "${SCRIPT_DIR}/extract_i_cannot_gradients.py" \
  --model "${MODEL_PATH}" \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-examples "${MAX_EXAMPLES}" \
  --top-k "${TOP_K}" \
  --top-per-layer "${TOP_PER_LAYER}" \
  --seed "${SEED}"
