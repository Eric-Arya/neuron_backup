#!/usr/bin/env bash
# Brief dense/full-Fisher extension to K=3k and 4k for first-cue-256 Grad.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

COMMAND="${COMMAND:-compute}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
RANKING="${RANKING:-${REPO_ROOT}/results/grad_onpolicy_sn_safe256_first_cue_tail_expanded50000/gradients/top_neurons_stable.csv}"
CONTEXTS="${CONTEXTS:-/workspace/xcy/dataset/wikitext/wikitext-2-raw-v1/firstcue_fisher_seed42/fisher_contexts.jsonl}"
FISHER_DIR="${FISHER_DIR:-${REPO_ROOT}/results/grad_firstcue_fisher_wikitext1024_k4000}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/grad_dense_full_fisher_larger_k}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/results/grad_harmbench_development/tuning_manifest.jsonl}"

# Persisted 4k benchmark on 16 real contexts: batch 4 was fastest at 3.370
# contexts/s and used 38.96 GB; batch 2 reached 2.355 and batch 8 reached 3.111.
FISHER_BATCH_SIZE="${FISHER_BATCH_SIZE:-4}"
SAFETY_BATCH_SIZE="${SAFETY_BATCH_SIZE:-16}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

compute() {
  "${PYTHON_BIN}" -m unified_eval.fisher_grad compute \
    --model "${MODEL}" \
    --ranking "${RANKING}" \
    --contexts "${CONTEXTS}" \
    --output-dir "${FISHER_DIR}" \
    --top-k 4000 \
    --batch-size "${FISHER_BATCH_SIZE}" \
    --continuation-tokens 32 \
    --probes 4 \
    --matrix-mode dense \
    --shrinkage 0.5 \
    --damping-ratio 0.01 \
    --dtype float32 \
    --device cuda:0 \
    --seed 112
}

generate() {
  "${PYTHON_BIN}" -m unified_eval.fisher_grad make-dense-prefix-variants \
    --fisher "${FISHER_DIR}/fisher.pt" \
    --ranking "${RANKING}" \
    --output-dir "${OUTPUT_DIR}/scales" \
    --active-k 3000 4000 \
    --target-positive-medians 0.3 0.45 0.6 \
    --cap 0.75 \
    --shrinkage 0.5 \
    --damping-ratio 0.01
}

screen() {
  mapfile -t scales < <(
    find "${OUTPUT_DIR}/scales" -maxdepth 1 -type f -name 'densefisher_k*.json' | sort -V
  )
  if (( ${#scales[@]} != 6 )); then
    echo "Expected 6 dense-Fisher scale files; found ${#scales[@]}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m unified_eval.fisher_grad evaluate-safety \
    --model "${MODEL}" \
    --ranking "${RANKING}" \
    --manifest "${MANIFEST}" \
    --scale-files "${scales[@]}" \
    --output-dir "${OUTPUT_DIR}/tuning47" \
    --batch-size "${SAFETY_BATCH_SIZE}" \
    --max-new-tokens 128 \
    --dtype float32 \
    --device cuda:0
}

case "${COMMAND}" in
  compute) compute ;;
  generate) generate ;;
  screen) screen ;;
  all) compute; generate; screen ;;
  *)
    echo "Unknown COMMAND=${COMMAND}; use compute, generate, screen, or all" >&2
    exit 2
    ;;
esac
