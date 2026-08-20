#!/usr/bin/env bash
# Example: RANDOM_SEED=7 TOP_K="8000 20000" GENERATION_BATCH_SIZE=16 bash scripts/eval/figure2_left_llama2_random_neurons.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-/workspace/xcy/dataset/projects/neurips_neuron/beavertails/splits/figure2_seed42_n200.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/figure2_left_llama2_random_seed42_8000_20000}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
RANDOM_SEED="${RANDOM_SEED:-42}"
TOP_K="${TOP_K:-8000 20000}"
read -r -a TOP_K_VALUES <<< "${TOP_K}"

mkdir -p "${OUTPUT_DIR}/logs"
"${PYTHON_BIN}" -m eval.figure2_llama2 \
  --selection random \
  --random-seed "${RANDOM_SEED}" \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --score-batch-size "${SCORE_BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --top-k "${TOP_K_VALUES[@]}" \
  "$@" 2>&1 | tee -a "${OUTPUT_DIR}/logs/run.log"
