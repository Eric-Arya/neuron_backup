#!/usr/bin/env bash
# SNCorpus-raw SFT IA3 guide-patch capability runs at K=160k or 320k.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-bbh}"
TOP_K="${TOP_K:-160000}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
ADAPTER="${ADAPTER:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20}"
RANKING="${RANKING:-/workspace/xcy/safety_repro/neurips_neuron/output/change_scores/llama3_instruct_vs_sft_snrawdot256_alpha3_snheldout_seed42_n200_raw_completion.pt}"
MATH100="${MATH100:-/workspace/xcy/dataset/math500/subsets/math500_l1_l3_seed112_n100}"
MATH100_SOURCE="${MATH100_SOURCE:-${MATH100}/SOURCE.json}"

if [[ "${TOP_K}" != "160000" && "${TOP_K}" != "320000" ]]; then
  echo "TOP_K must be 160000 or 320000" >&2
  exit 2
fi

# Persisted real-prompt benchmarks for the two-model BF16 guide patch:
# BBH batch 16 is fastest (0.170 examples/s, 54.35 GiB); MATH batch 32 is
# fastest (16.98 examples/s at the benchmark decode length, 44.87 GiB).
if [[ "${TOP_K}" == "320000" ]]; then
  # The 320k activation bridge exceeded H100 memory on a long real-prompt
  # batch at 16 after 32 examples; batch 8 is the persisted safe setting.
  BBH_BATCH_SIZE="${BBH_BATCH_SIZE:-8}"
else
  BBH_BATCH_SIZE="${BBH_BATCH_SIZE:-16}"
fi
MATH_BATCH_SIZE="${MATH_BATCH_SIZE:-32}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

common=(
  --method llama3_sft_patch
  --output-root "${REPO_ROOT}/results"
  --llama3-model "${MODEL}"
  --llama3-sft-adapter "${ADAPTER}"
  --llama3-sft-training-format raw
  --llama3-sft-ia3-alpha 3
  --llama3-sft-patch-ranking "${RANKING}"
  --llama3-sft-patch-top-k "${TOP_K}"
  --llama3-sft-patch-dtype bfloat16
  --base-device cuda:0
  --guide-device cuda:0
  --bbh-batch-size "${BBH_BATCH_SIZE}"
  --math500-batch-size "${MATH_BATCH_SIZE}"
  --seed 112
)

run_bbh() {
  "${PYTHON_BIN}" -m unified_eval.runner run \
    "${common[@]}" \
    --run-name "bbh_sft_patch_snraw_alpha3_top${TOP_K}_raw_cot_bf16" \
    --tasks bbh
}

run_math100() {
  "${PYTHON_BIN}" -m unified_eval.runner run \
    "${common[@]}" \
    --run-name "math500_l1_l3_n100_sft_patch_snraw_alpha3_top${TOP_K}_bf16" \
    --tasks math500 \
    --math500 "${MATH100}" \
    --math500-source "${MATH100_SOURCE}"
}

case "${TASK}" in
  bbh) run_bbh ;;
  math100) run_math100 ;;
  all) run_bbh; run_math100 ;;
  *)
    echo "TASK must be bbh, math100, or all" >&2
    exit 2
    ;;
esac
