#!/usr/bin/env bash
# Example: COMMAND=all GPU=0 BATCH_SIZE=32 bash scripts/run_grad_harmbench_development.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COMMAND="${COMMAND:-prepare}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/grad_harmbench_development}"
MODEL="${MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
SEED="${SEED:-112}"
TUNING_COUNT="${TUNING_COUNT:-47}"
SELECTION_MANIFEST="${SELECTION_MANIFEST:-}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REFUSAL_TARGET="${REFUSAL_TARGET:-I cannot assist with that request.}"
CONTRAST_TOKENS="${CONTRAST_TOKENS:-16}"
CONTRAST_WEIGHT="${CONTRAST_WEIGHT:-0.5}"
SAFE_PRESERVATION_WEIGHT="${SAFE_PRESERVATION_WEIGHT:-0.25}"
CANDIDATE_POOL="${CANDIDATE_POOL:-2000}"
RANKING_TOP_K="${RANKING_TOP_K:-500}"
K_VALUES="${K_VALUES:-10 25 50}"
STRENGTHS="${STRENGTHS:-0.5 0.75 1.0}"
SCOPES="${SCOPES:-last all}"
GSM8K_LIMIT="${GSM8K_LIMIT:-20}"
SWEEP_NAME="${SWEEP_NAME:-tuning}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

read -r -a K_ARRAY <<< "${K_VALUES}"
read -r -a STRENGTH_ARRAY <<< "${STRENGTHS}"
read -r -a SCOPE_ARRAY <<< "${SCOPES}"
read -r -a EXTRA_ARRAY <<< "${EXTRA_ARGS}"

prepare() {
  "${PYTHON_BIN}" -m unified_eval.grad_development prepare \
    --output-dir "${OUTPUT_DIR}" --seed "${SEED}" \
    --tuning-count "${TUNING_COUNT}"
}

benchmark() {
  "${PYTHON_BIN}" -m unified_eval.grad_development benchmark \
    --output-dir "${OUTPUT_DIR}" --model "${MODEL}" --device cuda:0 \
    --batch-sizes 8 16 32 --examples 32 --max-new-tokens 16
}

baseline() {
  "${PYTHON_BIN}" -m unified_eval.grad_development baseline \
    --output-dir "${OUTPUT_DIR}" --model "${MODEL}" --device cuda:0 \
    --batch-size "${BATCH_SIZE}" --max-new-tokens 128 "${EXTRA_ARRAY[@]}"
}

extract() {
  if [[ -z "${SELECTION_MANIFEST}" ]]; then
    echo "SELECTION_MANIFEST must name explicit training-only data for legacy extract" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m unified_eval.grad_development extract \
    --output-dir "${OUTPUT_DIR}" --model "${MODEL}" --device cuda:0 \
    --refusal-target "${REFUSAL_TARGET}" --contrast-tokens "${CONTRAST_TOKENS}" \
    --contrast-weight "${CONTRAST_WEIGHT}" \
    --safe-preservation-weight "${SAFE_PRESERVATION_WEIGHT}" \
    --candidate-pool "${CANDIDATE_POOL}" --top-k "${RANKING_TOP_K}" \
    --selection-manifest "${SELECTION_MANIFEST}" \
    "${EXTRA_ARRAY[@]}"
}

sweep() {
  "${PYTHON_BIN}" -m unified_eval.grad_development sweep \
    --output-dir "${OUTPUT_DIR}" --model "${MODEL}" --device cuda:0 \
    --sweep-name "${SWEEP_NAME}" \
    --batch-size "${BATCH_SIZE}" --k-values "${K_ARRAY[@]}" \
    --strengths "${STRENGTH_ARRAY[@]}" --scopes "${SCOPE_ARRAY[@]}" \
    --gsm8k-limit "${GSM8K_LIMIT}" "${EXTRA_ARRAY[@]}"
}

case "${COMMAND}" in
  prepare) prepare ;;
  benchmark) prepare; benchmark ;;
  baseline) baseline ;;
  extract) extract ;;
  sweep) sweep ;;
  smoke)
    prepare
    benchmark
    EXTRA_ARRAY=(--limit 3 --overwrite)
    baseline
    extract
    ;;
  all)
    prepare
    benchmark
    baseline
    extract
    sweep
    ;;
  *)
    echo "Unknown COMMAND=${COMMAND}; use prepare, benchmark, baseline, extract, sweep, smoke, or all" >&2
    exit 2
    ;;
esac
