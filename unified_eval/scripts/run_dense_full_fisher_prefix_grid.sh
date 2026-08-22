#!/usr/bin/env bash
# Dense/full-Fisher prefix grid for first-cue-256 Grad on the raw HarmBench-47 subset.
#
# Examples:
#   COMMAND=generate bash scripts/run_dense_full_fisher_prefix_grid.sh
#   COMMAND=screen GPU=0 bash scripts/run_dense_full_fisher_prefix_grid.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COMMAND="${COMMAND:-generate}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/grad_dense_full_fisher_prefix_grid}"
MODEL="${MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
FISHER="${FISHER:-${REPO_ROOT}/results/grad_firstcue_fisher_wikitext1024_k2000/fisher.pt}"
RANKING="${RANKING:-${REPO_ROOT}/results/grad_onpolicy_sn_safe256_first_cue_tail_expanded50000/gradients/top_neurons_stable.csv}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/results/grad_harmbench_development/tuning_manifest.jsonl}"

# Reuse the persisted real-context benchmarks. Full-Fisher construction selected
# batch 4 (3.108 contexts/s, 39.3 GB), while safety evaluation selected batch 16.
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

generate() {
  "${PYTHON_BIN}" -m unified_eval.fisher_grad make-dense-prefix-variants \
    --fisher "${FISHER}" \
    --ranking "${RANKING}" \
    --output-dir "${OUTPUT_DIR}/scales" \
    --active-k 250 500 1000 1500 2000 \
    --target-positive-medians 0.1 0.2 0.3 0.45 0.6 \
    --cap 0.75 \
    --shrinkage 0.5 \
    --damping-ratio 0.01
}

screen() {
  mapfile -t scales < <(
    find "${OUTPUT_DIR}/scales" -maxdepth 1 -type f -name 'densefisher_k*.json' | sort -V
  )
  if (( ${#scales[@]} != 25 )); then
    echo "Expected 25 dense-Fisher scale files; found ${#scales[@]}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m unified_eval.fisher_grad evaluate-safety \
    --model "${MODEL}" \
    --ranking "${RANKING}" \
    --manifest "${MANIFEST}" \
    --scale-files "${scales[@]}" \
    --output-dir "${OUTPUT_DIR}/tuning47" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype float32 \
    --device cuda:0
}

case "${COMMAND}" in
  generate) generate ;;
  screen) screen ;;
  all) generate; screen ;;
  *)
    echo "Unknown COMMAND=${COMMAND}; use generate, screen, or all" >&2
    exit 2
    ;;
esac
