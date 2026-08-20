#!/usr/bin/env bash
# Example: NUM_GPUS=2 SMOKE_EXAMPLES=4 MAX_LENGTH=256 bash scripts/training/dpo_smoke_test.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE:-/workspace/xcy/dataset/shared/hh_rlhf/harmless_base/train.jsonl}"
SFT_ADAPTER="${SFT_ADAPTER:-${REPO_ROOT}/output/real_run}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-4}"
SMOKE_FILE="${SMOKE_FILE:-/workspace/xcy/dataset/projects/neurips_neuron/hh_rlhf/processed/smoke_${SMOKE_EXAMPLES}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/xcy/tmp/safety_neuron_dpo_separate_smoke}"
SFT_MODEL_FILE="${SFT_ADAPTER}/adapter_model.safetensors"

mkdir -p "$(dirname "${SMOKE_FILE}")" "${OUTPUT_DIR}"
head -n "${SMOKE_EXAMPLES}" "${SOURCE_TRAIN_FILE}" > "${SMOKE_FILE}.tmp"
mv "${SMOKE_FILE}.tmp" "${SMOKE_FILE}"
SFT_HASH_BEFORE="$(sha256sum "${SFT_MODEL_FILE}" | awk '{print $1}')"

TRAIN_FILE="${SMOKE_FILE}" \
SFT_ADAPTER="${SFT_ADAPTER}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}" \
MAX_LENGTH="${MAX_LENGTH:-256}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-128}" \
NUM_TRAIN_EPOCHS=1 \
LOGGING_STEPS=1 \
SAVE_STRATEGY=steps \
SAVE_STEPS=1 \
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}" \
REPORT_TO=none \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}" \
bash "${REPO_ROOT}/scripts/training/dpo.sh" --max_steps 1

test -f "${OUTPUT_DIR}/adapter_config.json" || {
  printf 'DPO smoke test did not produce %s/adapter_config.json\n' "${OUTPUT_DIR}" >&2
  exit 1
}
for checkpoint_file in adapter_config.json adapter_model.safetensors optimizer.pt scheduler.pt trainer_state.json; do
  test -f "${OUTPUT_DIR}/checkpoint-1/${checkpoint_file}" || {
    printf 'DPO smoke checkpoint is missing %s\n' "${checkpoint_file}" >&2
    exit 1
  }
done
SFT_HASH_AFTER="$(sha256sum "${SFT_MODEL_FILE}" | awk '{print $1}')"
if [[ "${SFT_HASH_BEFORE}" != "${SFT_HASH_AFTER}" ]]; then
  printf 'The frozen SFT adapter changed during DPO smoke testing.\n' >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}/reference/adapter_model.safetensors" ]]; then
  printf 'DPO output unexpectedly contains an embedded reference adapter.\n' >&2
  exit 1
fi
printf 'DPO smoke test passed: %s\n' "${OUTPUT_DIR}"
