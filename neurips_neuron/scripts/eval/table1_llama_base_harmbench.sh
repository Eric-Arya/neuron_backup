#!/usr/bin/env bash
# Example: COMMAND=smoke TOP_K=20000 MAX_NEW_TOKENS=8 GENERATION_BATCH_SIZE=2 bash scripts/eval/table1_llama_base_harmbench.sh
# Full run: COMMAND=run GENERATION_BATCH_SIZE=8 SCORE_BATCH_SIZE=16 bash scripts/eval/table1_llama_base_harmbench.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Evaluation uses one generation model per GPU. The cost model runs after both
# generation models are released, on the first GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND="${COMMAND:-run}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/table1_seed42_n200.jsonl}"
MODEL="${MODEL:-/workspace/xcy/models/Llama-2-7b-hf}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
SFT_ADAPTER="${SFT_ADAPTER:-${REPO_ROOT}/output/real_run}"
DPO_ADAPTER="${DPO_ADAPTER:-${REPO_ROOT}/output/dpo_real_run}"
RANKING="${RANKING:-${REPO_ROOT}/output/change_scores/llama2_sft_vs_dpo_hh_harmless_sft_completion.pt}"
COST_MODEL="${COST_MODEL:-/workspace/xcy/models/beaver-7b-v1.0-cost}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/table1_llama_base_harmbench}"
EXPECTED_PROMPTS="${EXPECTED_PROMPTS:-200}"
TOP_K="${TOP_K:-20000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
BASE_DEVICE="${BASE_DEVICE:-cuda:0}"
GUIDE_DEVICE="${GUIDE_DEVICE:-cuda:1}"
COST_DEVICE="${COST_DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
SMOKE_NUM_PROMPTS="${SMOKE_NUM_PROMPTS:-2}"
SMOKE_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-8}"
BENCHMARK_BATCH_SIZES="${BENCHMARK_BATCH_SIZES:-2 4 8 16}"
BENCHMARK_MAX_NEW_TOKENS="${BENCHMARK_MAX_NEW_TOKENS:-8}"

ARGS=(
  "${COMMAND}"
  --dataset "${DATASET}"
  --model "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --sft-adapter "${SFT_ADAPTER}"
  --dpo-adapter "${DPO_ADAPTER}"
  --ranking "${RANKING}"
  --cost-model "${COST_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --expected-prompts "${EXPECTED_PROMPTS}"
  --top-k "${TOP_K}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --generation-batch-size "${GENERATION_BATCH_SIZE}"
  --score-batch-size "${SCORE_BATCH_SIZE}"
  --base-device "${BASE_DEVICE}"
  --guide-device "${GUIDE_DEVICE}"
  --cost-device "${COST_DEVICE}"
  --dtype "${DTYPE}"
)

if [[ "${COMMAND}" == "smoke" ]]; then
  ARGS+=(--smoke-num-prompts "${SMOKE_NUM_PROMPTS}" --smoke-max-new-tokens "${SMOKE_MAX_NEW_TOKENS}")
elif [[ "${COMMAND}" == "benchmark" ]]; then
  read -r -a BATCH_CANDIDATES <<< "${BENCHMARK_BATCH_SIZES}"
  ARGS+=(--benchmark-batch-sizes "${BATCH_CANDIDATES[@]}" --benchmark-max-new-tokens "${BENCHMARK_MAX_NEW_TOKENS}")
fi

mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/${COMMAND}.log}"
printf 'Command: %s | GPUs: %s | generation batch: %s | score batch: %s | top-k: %s\n' \
  "${COMMAND}" "${CUDA_VISIBLE_DEVICES}" "${GENERATION_BATCH_SIZE}" "${SCORE_BATCH_SIZE}" "${TOP_K}"
"${PYTHON_BIN}" -m eval.table1_harmbench "${ARGS[@]}" "$@" 2>&1 | tee -a "${LOG_FILE}"
