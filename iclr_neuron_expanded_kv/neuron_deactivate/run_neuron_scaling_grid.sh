#!/usr/bin/env bash
# Example: ./neuron_deactivate/run_neuron_scaling_grid.sh --rates "0.0002 0.0004" --scales "0.5 2" --batch-size 16 --math-gpu 0 --harm-gpu 0

set -uo pipefail

PROJECT_ROOT="/workspace/xcy/safety_repro/iclr_neuron_expanded_kv"
PYTHON_BIN="/workspace/xcy/miniconda3/envs/safety-neuron-expanded-kv/bin/python"
MODEL_PATH="/workspace/xcy/models/Meta-Llama-3-8B-Instruct"
DATA_ROOT="/workspace/xcy/dataset/shared"
PREPARED_PATH="/workspace/xcy/safety_repro/iclr_neuron/neuron_deactivate/evaluation_data/harm_behavior_first_100.jsonl"
NEURON_PATH="${PROJECT_ROOT}/neuron_detection/output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn1000_ffn2000_kvexpanded_raw_vpool_200.txt"
OUTPUT_ROOT="${PROJECT_ROOT}/neuron_deactivate/evaluation_outputs/neuron_scaling_grid"
RATES=(0.0002 0.0004)
SCALES=(0.5 2)
STRUCTURES=(fwd_up fwd_down q k v)
BATCH_SIZE=16
MATH_MAX_NEW_TOKENS=256
HARM_MAX_NEW_TOKENS=512
SEED=112
MATH_GPU=0
HARM_GPU=0
MATH_LIMIT=100
HARM_STOP_AFTER=""

usage() {
    sed -n '2,3p' "$0"
    echo "Flags: --rates LIST --scales LIST --batch-size N --math-max-new-tokens N"
    echo "       --harm-max-new-tokens N --math-gpu ID --harm-gpu ID --seed N"
    echo "       --math-limit N --harm-stop-after N --output-root PATH --neurons PATH"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rates) read -r -a RATES <<< "$2"; shift 2 ;;
        --scales) read -r -a SCALES <<< "$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --math-max-new-tokens) MATH_MAX_NEW_TOKENS="$2"; shift 2 ;;
        --harm-max-new-tokens) HARM_MAX_NEW_TOKENS="$2"; shift 2 ;;
        --math-gpu) MATH_GPU="$2"; shift 2 ;;
        --harm-gpu) HARM_GPU="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --math-limit) MATH_LIMIT="$2"; shift 2 ;;
        --harm-stop-after) HARM_STOP_AFTER="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --neurons) NEURON_PATH="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ${#RATES[@]} -eq 0 || ${#SCALES[@]} -eq 0 ]]; then
    echo "At least one rate and scale are required." >&2
    exit 2
fi

cd "$PROJECT_ROOT" || exit 1

for rate in "${RATES[@]}"; do
    for scale in "${SCALES[@]}"; do
        rate_tag=${rate//./p}
        scale_tag=${scale//-/neg}
        scale_tag=${scale_tag//./p}
        condition="rate${rate_tag}_scale${scale_tag}"
        harm_dir="${OUTPUT_ROOT}/harm_behavior/${condition}"
        mkdir -p "$harm_dir"

        math_command=(
            "$PYTHON_BIN" neuron_deactivate/run_table1_capability.py gsm8k sn
            --model "$MODEL_PATH"
            --data-root "$DATA_ROOT"
            --output-root "${OUTPUT_ROOT}/math"
            --neurons "$NEURON_PATH"
            --deact-rate "$rate"
            --structures "${STRUCTURES[@]}"
            --neuron-scale "$scale"
            --prompt-format chat
            --num-fewshot 0
            --batch-size "$BATCH_SIZE"
            --max-new-tokens "$MATH_MAX_NEW_TOKENS"
            --seed "$SEED"
        )
        if [[ -n "$MATH_LIMIT" ]]; then
            math_command+=(--limit "$MATH_LIMIT")
        fi

        harm_command=(
            "$PYTHON_BIN" neuron_deactivate/run_table1_deactivated.py
            --model "$MODEL_PATH"
            --prepared "$PREPARED_PATH"
            --neurons "$NEURON_PATH"
            --deact-rate "$rate"
            --structures "${STRUCTURES[@]}"
            --neuron-scale "$scale"
            --prompt-format raw
            --batch-size "$BATCH_SIZE"
            --max-new-tokens "$HARM_MAX_NEW_TOKENS"
            --seed "$SEED"
            --responses "${harm_dir}/responses.jsonl"
            --scored "${harm_dir}/scored.jsonl"
            --summary "${harm_dir}/summary.json"
            --run-metadata "${harm_dir}/run.json"
        )
        if [[ -n "$HARM_STOP_AFTER" ]]; then
            harm_command+=(--stop-after "$HARM_STOP_AFTER")
        fi

        echo "Starting ${condition}: GSM8K on GPU ${MATH_GPU}, HarmBehavior on GPU ${HARM_GPU}"
        if [[ "$MATH_GPU" == "$HARM_GPU" ]]; then
            CUDA_VISIBLE_DEVICES="$MATH_GPU" "${math_command[@]}"
            math_status=$?
            if [[ $math_status -eq 0 ]]; then
                CUDA_VISIBLE_DEVICES="$HARM_GPU" "${harm_command[@]}"
                harm_status=$?
            else
                harm_status=1
            fi
        else
            CUDA_VISIBLE_DEVICES="$MATH_GPU" "${math_command[@]}" &
            math_pid=$!
            CUDA_VISIBLE_DEVICES="$HARM_GPU" "${harm_command[@]}" &
            harm_pid=$!
            wait "$math_pid"
            math_status=$?
            wait "$harm_pid"
            harm_status=$?
        fi
        if [[ $math_status -ne 0 || $harm_status -ne 0 ]]; then
            echo "Condition ${condition} failed: math=${math_status}, harm=${harm_status}" >&2
            exit 1
        fi
    done
done
