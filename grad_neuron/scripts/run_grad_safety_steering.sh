#!/usr/bin/env bash
# Example: CUDA_DEVICE=1 K_VALUES="25" EPSILONS="1.0" bash scripts/run_grad_safety_steering.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
K_VALUES="${K_VALUES:-25}"
EPSILONS="${EPSILONS:-1.0}"
SCOPE="${SCOPE:-last}"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-40}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RANKING="${RANKING:-${PROJECT_ROOT}/results/gradients/raw_refusal_advbench_rows100_299/top_neurons.csv}"
PER_EXAMPLE_GRADIENTS="${PER_EXAMPLE_GRADIENTS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/safety_steering/fixed_top25_disjoint_advbench/run}"

read -r -a K_ARGS <<< "${K_VALUES}"
read -r -a EPSILON_ARGS <<< "${EPSILONS}"

EXTRA_ARGS=()
if [[ -n "${PER_EXAMPLE_GRADIENTS}" ]]; then
  EXTRA_ARGS+=(--per-example-gradients "${PER_EXAMPLE_GRADIENTS}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python "${SCRIPT_DIR}/run_grad_safety_steering.py" \
  --ranking "${RANKING}" \
  --k-values "${K_ARGS[@]}" \
  --epsilons "${EPSILON_ARGS[@]}" \
  --scope "${SCOPE}" \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
