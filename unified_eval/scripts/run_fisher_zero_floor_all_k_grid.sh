#!/usr/bin/env bash
# Zero-floor diagonal-Fisher grid for first-cue-256 Grad.
#
# Examples:
#   COMMAND=generate bash scripts/run_fisher_zero_floor_all_k_grid.sh
#   COMMAND=screen GPU=0 SHARD_INDEX=0 SHARD_COUNT=2 \
#     bash scripts/run_fisher_zero_floor_all_k_grid.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COMMAND="${COMMAND:-generate}"
GPU="${GPU:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
FINALISTS_ONLY="${FINALISTS_ONLY:-0}"
SCREEN_NAME="${SCREEN_NAME:-tuning47}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/grad_fisher_zero_floor_all_k_cap0p75}"
MODEL="${MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
FISHER="${FISHER:-${REPO_ROOT}/results/grad_firstcue_fisher_diag_wikitext2048_k16000/fisher.pt}"
RANKING="${RANKING:-${REPO_ROOT}/results/grad_onpolicy_sn_safe256_first_cue_tail_expanded50000/gradients/top_neurons_stable.csv}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/results/grad_harmbench_development/tuning_manifest.jsonl}"
MATH100="${MATH100:-/workspace/xcy/dataset/math500/subsets/math500_l1_l3_seed112_n100}"
MATH100_SOURCE="${MATH100_SOURCE:-${MATH100}/SOURCE.json}"

# Persisted H100 benchmark: batch 16 is the selected safety-evaluation default.
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CAPABILITY_KS="${CAPABILITY_KS:-1000 2000 4000 6000 8000 16000}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

make_variants() {
  local active_k="$1"
  shift
  "${PYTHON_BIN}" -m unified_eval.fisher_grad make-floor-fisher-variants \
    --fisher "${FISHER}" \
    --ranking "${RANKING}" \
    --output-dir "${OUTPUT_DIR}/scales/k${active_k}" \
    --pool-k 16000 \
    --active-k "${active_k}" \
    --floors 0 \
    --score-scales "$@" \
    --caps 0.75 \
    --damping-ratios 1
}

generate() {
  mkdir -p "${OUTPUT_DIR}/scales"
  make_variants 1000 0.32 0.48 0.64
  make_variants 2000 0.32 0.48 0.64
  make_variants 4000 0.32 0.40 0.48 0.64
  make_variants 6000 0.28 0.36 0.44 0.52
  make_variants 8000 0.24 0.32 0.40 0.48
  make_variants 12000 0.18 0.22 0.28 0.40 0.48
  make_variants 16000 0.14 0.18 0.22 0.30
}

screen() {
  if (( SHARD_COUNT <= 0 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
    echo "Require 0 <= SHARD_INDEX < SHARD_COUNT" >&2
    exit 2
  fi
  mapfile -t all_scales < <(
    find "${OUTPUT_DIR}/scales" -type f -name 'floorfisher_k*.json' | sort -V
  )
  local selected=()
  local index
  for index in "${!all_scales[@]}"; do
    if [[ "${FINALISTS_ONLY}" == "1" ]]; then
      case "$(basename "${all_scales[index]}")" in
        floorfisher_k1000_floor0p0_c0p64_cap0p75_damp1p0.json | \
        floorfisher_k2000_floor0p0_c0p64_cap0p75_damp1p0.json | \
        floorfisher_k4000_floor0p0_c0p4_cap0p75_damp1p0.json | \
        floorfisher_k6000_floor0p0_c0p52_cap0p75_damp1p0.json | \
        floorfisher_k8000_floor0p0_c0p48_cap0p75_damp1p0.json | \
        floorfisher_k12000_floor0p0_c0p22_cap0p75_damp1p0.json | \
        floorfisher_k16000_floor0p0_c0p18_cap0p75_damp1p0.json) ;;
        *) continue ;;
      esac
    fi
    if (( index % SHARD_COUNT == SHARD_INDEX )); then
      selected+=("${all_scales[index]}")
    fi
  done
  if (( ${#selected[@]} == 0 )); then
    echo "No scale files selected for shard ${SHARD_INDEX}/${SHARD_COUNT}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m unified_eval.fisher_grad evaluate-safety \
    --model "${MODEL}" \
    --ranking "${RANKING}" \
    --manifest "${MANIFEST}" \
    --scale-files "${selected[@]}" \
    --output-dir "${OUTPUT_DIR}/${SCREEN_NAME}/shard${SHARD_INDEX}-of-${SHARD_COUNT}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype float32 \
    --device cuda:0
}

finalist_scale() {
  case "$1" in
    1000) echo "${OUTPUT_DIR}/scales/k1000/floorfisher_k1000_floor0p0_c0p64_cap0p75_damp1p0.json" ;;
    2000) echo "${OUTPUT_DIR}/scales/k2000/floorfisher_k2000_floor0p0_c0p64_cap0p75_damp1p0.json" ;;
    4000) echo "${OUTPUT_DIR}/scales/k4000/floorfisher_k4000_floor0p0_c0p4_cap0p75_damp1p0.json" ;;
    6000) echo "${OUTPUT_DIR}/scales/k6000/floorfisher_k6000_floor0p0_c0p52_cap0p75_damp1p0.json" ;;
    8000) echo "${OUTPUT_DIR}/scales/k8000/floorfisher_k8000_floor0p0_c0p48_cap0p75_damp1p0.json" ;;
    12000) echo "${OUTPUT_DIR}/scales/k12000/floorfisher_k12000_floor0p0_c0p22_cap0p75_damp1p0.json" ;;
    16000) echo "${OUTPUT_DIR}/scales/k16000/floorfisher_k16000_floor0p0_c0p18_cap0p75_damp1p0.json" ;;
    *) echo "Unknown finalist K=$1" >&2; return 2 ;;
  esac
}

capability() {
  local active_k
  local scale_file
  local encoded_c
  local run_name
  for active_k in ${CAPABILITY_KS}; do
    scale_file="$(finalist_scale "${active_k}")"
    encoded_c="$(basename "${scale_file}" | sed -E 's/^floorfisher_k[0-9]+_floor[^_]+_c([^_]+)_.*/\1/')"
    run_name="fisher_zero_floor_math100_k${active_k}_c${encoded_c}_cap0p75"
    "${PYTHON_BIN}" -m unified_eval.runner run \
      --method grad \
      --run-name "${run_name}" \
      --tasks ifeval bbh math500 \
      --output-root "${OUTPUT_DIR}/capability" \
      --llama3-model "${MODEL}" \
      --grad-ranking "${RANKING}" \
      --grad-top-k "${active_k}" \
      --grad-scale-file "${scale_file}" \
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
  done
}

case "${COMMAND}" in
  generate) generate ;;
  screen) screen ;;
  capability) capability ;;
  all) generate; screen ;;
  *)
    echo "Unknown COMMAND=${COMMAND}; use generate, screen, capability, or all" >&2
    exit 2
    ;;
esac
