#!/usr/bin/env bash
set -uo pipefail

##############################
# Configurable Constants
##############################
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL1_ROOT="$(dirname "${SCRIPT_DIR}")"
export SKILL1_ROOT

# ---- API Keys ----
export OPENAI_BASE_URL="https://aigc.sankuai.com/v1/openai/native"
export OPENAI_API_KEY="1985951821684846684"
export ANTHROPIC_BASE_URL="https://aigc.sankuai.com/v1/anthropic/"
export ANTHROPIC_AUTH_TOKEN="2038469073881301067"

# ---- Model (comma-separated for multi-model) ----
export EVAL_MODEL="${EVAL_MODEL:-gpt-4.1,qwen-vl-max-latest,glm-4.6v,kimi-k2.5}"

# ---- Environment data paths ----
export ALFWORLD_DATA="${SKILL1_ROOT}/data/alfworld"
export WEBSHOP_DATA="${SKILL1_ROOT}/data/datasets/webshop/webshop_data"
export PYTHONPATH="${SKILL1_ROOT}/agent_system/environments/env_package/alfworld:${SKILL1_ROOT}/agent_system/environments/env_package/webshop/webshop:${SKILL1_ROOT}:${PYTHONPATH:-}"

# ---- CLI args ----
ENV_NAME="${1:-alfworld}"
NUM_TASKS="${2:-10}"
MAX_TRIALS="${3:-3}"
MAX_STEPS="${4:-50}"

# ---- Output ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SKILL1_ROOT}/eval_results/memory_retry_${ENV_NAME}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/eval.log"

echo "=========================================="
echo " Memory Ablation (Multi-Trial Retry)"
echo " Env:        ${ENV_NAME}"
echo " Model(s):   ${EVAL_MODEL}"
echo " Tasks:      ${NUM_TASKS}"
echo " Max trials: ${MAX_TRIALS}"
echo " Max steps:  ${MAX_STEPS}"
echo " Output:     ${OUTPUT_DIR}"
echo " Log:        ${LOG_FILE}"
echo "=========================================="

python3 "${SCRIPT_DIR}/eval_memory_ablation.py" \
    --env "${ENV_NAME}" \
    --num_tasks "${NUM_TASKS}" \
    --max_trials "${MAX_TRIALS}" \
    --max_steps "${MAX_STEPS}" \
    --models "${EVAL_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
