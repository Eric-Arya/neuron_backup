#!/usr/bin/env bash
# Example: CUDA_VISIBLE_DEVICES=0,1 NUM_SAMPLES=2 MAX_NEW_TOKENS=8 bash scripts/safety_neuron/change_scores_smoke_test.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NUM_SAMPLES="${NUM_SAMPLES:-2}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}" \
OUTPUT_FILE="${OUTPUT_FILE:-${REPO_ROOT}/output/change_scores/smoke_test.pt}" \
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/output/change_scores/smoke_test.log}" \
bash "${REPO_ROOT}/scripts/safety_neuron/get_change_scores.sh" "$@"
