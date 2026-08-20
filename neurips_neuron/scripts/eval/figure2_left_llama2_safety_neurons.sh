#!/usr/bin/env bash
# Example: TOP_K="0 2000 4000 6000 8000 10000" GENERATION_BATCH_SIZE=16 bash scripts/eval/figure2_left_llama2_safety_neurons.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-/workspace/xcy/dataset/projects/neurips_neuron/beavertails/splits/figure2_seed42_n200.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/figure2_left_llama2_safety_neurons}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TOP_K="${TOP_K:-0 200 400 600 800 1000 1200 1500 2000 3000 4000 5000 6000 7000 8000 9000 10000 12000 14000 16000 18000 20000}"
read -r -a TOP_K_VALUES <<< "${TOP_K}"

mkdir -p "${OUTPUT_DIR}/logs"
"${PYTHON_BIN}" -m eval.figure2_llama2 \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --score-batch-size "${SCORE_BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --top-k "${TOP_K_VALUES[@]}" \
  "$@" 2>&1 | tee -a "${OUTPUT_DIR}/logs/run.log"
