#!/usr/bin/env bash
# Example override:
# NUM_GPUS=2 BASELINE_ALPHA=0.9 INTEGRATION_STEPS=16 MAX_EXAMPLES=20 bash scripts/extract_integrated_gradients.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NUM_GPUS="${NUM_GPUS:-2}"
BASELINE_ALPHA="${BASELINE_ALPHA:-0.9}"
INTEGRATION_STEPS="${INTEGRATION_STEPS:-16}"
DTYPE="${DTYPE:-float32}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/gradients/integrated_gradients}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  "${SCRIPT_DIR}/extract_integrated_gradients.py" \
  --baseline-alpha "${BASELINE_ALPHA}" \
  --integration-steps "${INTEGRATION_STEPS}" \
  --dtype "${DTYPE}" \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "${OUTPUT_DIR}"
