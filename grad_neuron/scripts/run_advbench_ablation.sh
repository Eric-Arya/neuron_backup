#!/usr/bin/env bash
# Example override:
# NUM_GPUS=2 SIGNS="positive" NEURON_SCALE=2 K_VALUES="1000 500" bash scripts/run_advbench_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NUM_GPUS="${NUM_GPUS:-2}"
K_VALUES="${K_VALUES:-1000 500 200 100}"
SIGNS="${SIGNS:-positive negative}"
NEURON_SCALE="${NEURON_SCALE:-0}"
LIMIT="${LIMIT:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/causal_sweeps/i_cannot_zero}"

read -r -a K_VALUE_ARGS <<< "${K_VALUES}"
read -r -a SIGN_ARGS <<< "${SIGNS}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  "${SCRIPT_DIR}/run_advbench_ablation.py" \
  --k-values "${K_VALUE_ARGS[@]}" \
  --signs "${SIGN_ARGS[@]}" \
  --neuron-scale "${NEURON_SCALE}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --output-dir "${OUTPUT_DIR}"
