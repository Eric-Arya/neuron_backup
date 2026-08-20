#!/usr/bin/env bash
# Example: NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_CHECKPOINTING=0 NUM_EXAMPLES=16 NUM_STEPS=2 bash scripts/training/benchmarking/dpo_throughput.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NUM_EXAMPLES="${NUM_EXAMPLES:-16}"
NUM_STEPS="${NUM_STEPS:-2}"
SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE:-/workspace/xcy/dataset/shared/hh_rlhf/harmless_base/train.jsonl}"
BENCHMARK_FILE="${BENCHMARK_FILE:-/workspace/xcy/dataset/projects/neurips_neuron/hh_rlhf/processed/benchmark_${NUM_EXAMPLES}.jsonl}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
SAMPLE_MODE="${SAMPLE_MODE:-longest}"
CONFIG_NAME="batch${PER_DEVICE_TRAIN_BATCH_SIZE}_accum${GRADIENT_ACCUMULATION_STEPS}_ckpt${GRADIENT_CHECKPOINTING}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${CONFIG_NAME}}"

for value in "${NUM_EXAMPLES}" "${NUM_STEPS}" "${PER_DEVICE_TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Example, step, batch, and accumulation values must be positive integers (got %s).\n' "${value}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "${BENCHMARK_FILE}")" "${OUTPUT_DIR}"
if [[ "${SAMPLE_MODE}" == longest ]]; then
  # Keep the longest JSONL records so the minimal benchmark still exercises
  # memory pressure representative of the configured sequence length.
  awk -v n="${NUM_EXAMPLES}" '
    {
      size = length($0)
      for (i = 1; i <= n; i++) {
        if (size > sizes[i]) {
          for (j = n; j > i; j--) {
            sizes[j] = sizes[j - 1]
            lines[j] = lines[j - 1]
          }
          sizes[i] = size
          lines[i] = $0
          break
        }
      }
    }
    END { for (i = 1; i <= n && lines[i] != ""; i++) print lines[i] }
  ' "${SOURCE_TRAIN_FILE}" > "${BENCHMARK_FILE}.tmp"
elif [[ "${SAMPLE_MODE}" == first ]]; then
  head -n "${NUM_EXAMPLES}" "${SOURCE_TRAIN_FILE}" > "${BENCHMARK_FILE}.tmp"
else
  printf 'SAMPLE_MODE must be longest or first (got %s).\n' "${SAMPLE_MODE}" >&2
  exit 2
fi
mv "${BENCHMARK_FILE}.tmp" "${BENCHMARK_FILE}"
if (( $(wc -l < "${BENCHMARK_FILE}") != NUM_EXAMPLES )); then
  printf 'Could not extract %d real examples from %s\n' "${NUM_EXAMPLES}" "${SOURCE_TRAIN_FILE}" >&2
  exit 1
fi

START_SECONDS="${SECONDS}"
TRAIN_FILE="${BENCHMARK_FILE}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING}" \
MAX_LENGTH="${MAX_LENGTH}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
NUM_TRAIN_EPOCHS=1 \
LOGGING_STEPS=1 \
SAVE_STRATEGY=no \
REPORT_TO=none \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29512}" \
bash "${REPO_ROOT}/scripts/training/dpo.sh" \
  --max_steps "${NUM_STEPS}" \
  --save_final_model false
ELAPSED_SECONDS=$((SECONDS - START_SECONDS))

printf 'Benchmark wall time: %d seconds. Trainer metrics: %s/train_results.json\n' \
  "${ELAPSED_SECONDS}" "${OUTPUT_DIR}"
