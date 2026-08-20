#!/usr/bin/env bash
# Example: BENCHMARK_BATCH_SIZES="2 4 8" bash scripts/eval/table1_llama_base_harmbench_pipeline.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${REPO_ROOT}/scripts/eval/table1_llama_base_harmbench.sh"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/table1_llama_base_harmbench}"
export OUTPUT_DIR

# Benchmark first on a minimal real-data slice, then use the measured fastest
# successful batch sizes. Existing batch shards make the subsequent run safe to
# resume after interruption.
COMMAND=benchmark bash "${RUNNER}" "$@"

read -r GENERATION_BATCH_SIZE SCORE_BATCH_SIZE < <(
  "${PYTHON_BIN:-python}" - "${OUTPUT_DIR}/benchmark.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
print(result["recommended_generation_batch_size"], result["recommended_score_batch_size"])
PY
)
export GENERATION_BATCH_SIZE SCORE_BATCH_SIZE
COMMAND=smoke bash "${RUNNER}" "$@"
COMMAND=run bash "${RUNNER}" "$@"
