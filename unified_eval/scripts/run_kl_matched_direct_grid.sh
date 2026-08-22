#!/usr/bin/env bash
# KL-matched direct Grad controls for the selected zero-floor Fisher points.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COMMAND="${COMMAND:-generate}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/grad_direct_klmatched_fisher_cap0p75}"
MODEL="${MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
RANKING="${RANKING:-${REPO_ROOT}/results/grad_onpolicy_sn_safe256_first_cue_tail_expanded50000/gradients/top_neurons_stable.csv}"
TUNING_MANIFEST="${TUNING_MANIFEST:-${REPO_ROOT}/results/grad_harmbench_development/tuning_manifest.jsonl}"
FROZEN_MANIFEST="${FROZEN_MANIFEST:-/workspace/xcy/dataset/projects/neurips_neuron/harmbench/splits/table1_seed42_n200.jsonl}"
MATH100="${MATH100:-/workspace/xcy/dataset/math500/subsets/math500_l1_l3_seed112_n100}"
MATH100_SOURCE="${MATH100_SOURCE:-${MATH100}/SOURCE.json}"

# Raw-Fisher quadratic-KL matches for the locked Fisher controllers.
# K=6k and K=8k are intentionally excluded for now.
DIRECT_SPECS=(
  "1000 0.395116"
  "2000 0.42538"
  "4000 0.355565"
  "12000 0.226653"
  "16000 0.175207"
)
ADVANCE_KS="${ADVANCE_KS:-}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

fisher_scale_for_k() {
  case "$1" in
    1000) echo "${REPO_ROOT}/results/grad_fisher_zero_floor_all_k_cap0p75/scales/k1000/floorfisher_k1000_floor0p0_c0p64_cap0p75_damp1p0.json" ;;
    2000) echo "${REPO_ROOT}/results/grad_fisher_zero_floor_all_k_cap0p75/scales/k2000/floorfisher_k2000_floor0p0_c0p64_cap0p75_damp1p0.json" ;;
    4000) echo "${REPO_ROOT}/results/grad_fisher_zero_floor_all_k_cap0p75/scales/k4000/floorfisher_k4000_floor0p0_c0p4_cap0p75_damp1p0.json" ;;
    12000) echo "${REPO_ROOT}/results/grad_fisher_zero_floor_all_k_cap0p75/scales/k12000/floorfisher_k12000_floor0p0_c0p22_cap0p75_damp1p0.json" ;;
    16000) echo "${REPO_ROOT}/results/grad_fisher_zero_floor_all_k_cap0p75/scales/k16000/floorfisher_k16000_floor0p0_c0p18_cap0p75_damp1p0.json" ;;
    *) echo "Unknown K=$1" >&2; return 2 ;;
  esac
}

scale_for_k() {
  local active_k="$1"
  local strength="$2"
  local encoded_strength="${strength//./p}"
  echo "${OUTPUT_DIR}/scales/k${active_k}/direct_k${active_k}_s${encoded_strength}.json"
}

selected_specs() {
  local spec
  local active_k
  for spec in "${DIRECT_SPECS[@]}"; do
    read -r active_k _ <<<"${spec}"
    if [[ -z "${ADVANCE_KS}" || " ${ADVANCE_KS} " == *" ${active_k} "* ]]; then
      echo "${spec}"
    fi
  done
}

generate() {
  local spec
  local active_k
  local strength
  for spec in "${DIRECT_SPECS[@]}"; do
    read -r active_k strength <<<"${spec}"
    "${PYTHON_BIN}" -m unified_eval.fisher_grad make-variants \
      --ranking "${RANKING}" \
      --base-individual-scales "$(fisher_scale_for_k "${active_k}")" \
      --output-dir "${OUTPUT_DIR}/scales/k${active_k}" \
      --direct-top-k "${active_k}" \
      --direct-strengths "${strength}"
  done
}

screen() {
  local spec
  local active_k
  local strength
  local scales=()
  for spec in "${DIRECT_SPECS[@]}"; do
    read -r active_k strength <<<"${spec}"
    scales+=("$(scale_for_k "${active_k}" "${strength}")")
  done
  "${PYTHON_BIN}" -m unified_eval.fisher_grad evaluate-safety \
    --model "${MODEL}" \
    --ranking "${RANKING}" \
    --manifest "${TUNING_MANIFEST}" \
    --scale-files "${scales[@]}" \
    --output-dir "${OUTPUT_DIR}/tuning47" \
    --batch-size 16 \
    --max-new-tokens 128 \
    --dtype float32 \
    --device cuda:0
}

frozen_harmbench() {
  if [[ -z "${ADVANCE_KS}" ]]; then
    echo "Set ADVANCE_KS to the K values that passed tuning ASR < 10%." >&2
    return 2
  fi
  local spec
  local active_k
  local strength
  local scales=()
  while read -r active_k strength; do
    scales+=("$(scale_for_k "${active_k}" "${strength}")")
  done < <(selected_specs)
  "${PYTHON_BIN}" -m unified_eval.fisher_grad evaluate-safety \
    --model "${MODEL}" \
    --ranking "${RANKING}" \
    --manifest "${FROZEN_MANIFEST}" \
    --scale-files "${scales[@]}" \
    --output-dir "${OUTPUT_DIR}/frozen_harmbench" \
    --batch-size 16 \
    --max-new-tokens 128 \
    --dtype float32 \
    --device cuda:0
}

capability() {
  if [[ -z "${ADVANCE_KS}" ]]; then
    echo "Set ADVANCE_KS to the K values that passed tuning ASR < 10%." >&2
    return 2
  fi
  local active_k
  local strength
  local encoded_strength
  while read -r active_k strength; do
    encoded_strength="${strength//./p}"
    "${PYTHON_BIN}" -m unified_eval.runner run \
      --method grad \
      --run-name "direct_klmatched_math100_k${active_k}_s${encoded_strength}" \
      --tasks ifeval bbh math500 \
      --output-root "${OUTPUT_DIR}/capability" \
      --llama3-model "${MODEL}" \
      --grad-ranking "${RANKING}" \
      --grad-top-k "${active_k}" \
      --grad-strength "${strength}" \
      --grad-scope last \
      --grad-direction positive-only \
      --grad-dtype float32 \
      --math500 "${MATH100}" \
      --math500-source "${MATH100_SOURCE}" \
      --ifeval-batch-size 8 \
      --bbh-batch-size 8 \
      --math500-batch-size 16 \
      --device cuda:0 \
      --seed 112
  done < <(selected_specs)
}

case "${COMMAND}" in
  generate) generate ;;
  screen) screen ;;
  frozen) frozen_harmbench ;;
  capability) capability ;;
  *) echo "Unknown COMMAND=${COMMAND}; use generate, screen, frozen, or capability" >&2; exit 2 ;;
esac
