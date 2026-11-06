#!/usr/bin/env bash
# Shared launch plumbing; source this file from a recipe.
set -euo pipefail

RECIPE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${RECIPE_DIR}/../.." && pwd)"
# Recipe paths (including forwarded CLI paths) are relative to the repository.
cd -- "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-manage_table}"
DATA_ROOT="${DATA_ROOT:-data}"

recipe_has_option() {
    local option="$1"
    shift
    local argument
    for argument in "$@"; do
        if [[ "${argument}" == "${option}" || "${argument}" == "${option}="* ]]; then
            return 0
        fi
    done
    return 1
}

recipe_launch() {
    local method="$1"
    shift
    local -a common_args=(
        --task "${TASK_NAME}"
        --method "${method}"
        --epochs "${NUM_EPOCHS}"
        --batch-size "${BATCH_SIZE}"
        --num-workers "${NUM_WORKERS}"
        --val-num-workers "${VAL_NUM_WORKERS:-4}"
        --lr "${LEARNING_RATE}"
        --seed "${SEED:-42}"
        --port "${MAIN_PROCESS_PORT}"
        --precision "${MIXED_PRECISION:-bf16}"
        --accelerate-config "${ACCELERATE_CONFIG_FILE:-${RECIPE_DIR}/accelerate.yaml}"
    )
    if [[ -n "${GPU_LIST:-}" ]]; then
        common_args+=(--gpus "${GPU_LIST}")
    fi
    if [[ -n "${NUM_PROCESSES:-}" ]]; then
        common_args+=(--num-processes "${NUM_PROCESSES}")
    fi
    if [[ -n "${CACHE_DIR:-}" ]]; then
        common_args+=(--cache-dir "${CACHE_DIR}")
    fi
    if [[ -n "${RUN_DIR:-}" ]]; then
        common_args+=(--run-dir "${RUN_DIR}")
    fi
    case "${USE_WANDB:-true}" in
        true|True|1) common_args+=(--wandb) ;;
        false|False|0) common_args+=(--no-wandb) ;;
        *) echo 'USE_WANDB must be true or false' >&2; return 2 ;;
    esac
    if [[ -n "${MODEL_PRETRAINED:-}" ]]; then
        case "${MODEL_PRETRAINED}" in
            true|True|1) common_args+=(--pretrained) ;;
            false|False|0) common_args+=(--no-pretrained) ;;
            *) echo 'MODEL_PRETRAINED must be true or false' >&2; return 2 ;;
        esac
    fi
    if [[ -n "${HF_OFFLINE:-}" ]]; then
        export HF_HUB_OFFLINE="${HF_OFFLINE}"
        export TRANSFORMERS_OFFLINE="${HF_OFFLINE}"
    fi
    exec bash "${RECIPE_DIR}/../launch.sh" "${common_args[@]}" "$@"
}
