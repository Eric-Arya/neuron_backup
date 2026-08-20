#!/usr/bin/env bash
# Example: CUDA_VISIBLE_DEVICES=0,1 PER_DEVICE_TRAIN_BATCH_SIZE=3 GRADIENT_ACCUMULATION_STEPS=20 LEARNING_RATE=1e-3 bash scripts/training/dpo_llama3_8b_instruct.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export BASE_MODEL="${BASE_MODEL:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
# Llama-3-8B-Instruct is the already-SFT reference checkpoint, so no separate
# SFT adapter is loaded unless the caller explicitly supplies one.
export SFT_ADAPTER="${SFT_ADAPTER-}"
export TRAIN_FILE="${TRAIN_FILE:-/workspace/xcy/dataset/shared/hh_rlhf/harmless_base/train.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/xcy/models/Meta-Llama-3-8B-Instruct-DPO-IA3-HH}"
export CHAT_FORMAT="${CHAT_FORMAT:-native}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-3}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-20}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export BETA="${BETA:-0.1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
export USE_FLASH_ATTN="${USE_FLASH_ATTN:-1}"
export REPORT_TO="${REPORT_TO:-none}"

exec bash "${REPO_ROOT}/scripts/training/dpo.sh" "$@"
