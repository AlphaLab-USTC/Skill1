#!/usr/bin/env bash
set -uo pipefail

##############################
# Configurable Constants
##############################
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL1_ROOT="$(dirname "${SCRIPT_DIR}")"
export SKILL1_ROOT

# ---- API / Model (change EVAL_MODEL to swap models) ----
export ANTHROPIC_BASE_URL="https://aigc.sankuai.com/v1/anthropic/"
export ANTHROPIC_AUTH_TOKEN="2038469073881301067"
export EVAL_MODEL="${EVAL_MODEL:-aws.claude-opus-4.6-b}"

# ---- Environment data paths ----
export ALFWORLD_DATA="${SKILL1_ROOT}/data/alfworld"
export WEBSHOP_DATA="${SKILL1_ROOT}/data/datasets/webshop/webshop_data"
export PYTHONPATH="${SKILL1_ROOT}/agent_system/environments/env_package/alfworld:${SKILL1_ROOT}/agent_system/environments/env_package/webshop/webshop:${SKILL1_ROOT}:${PYTHONPATH:-}"

# ---- CLI args ----
ENV_NAME="${1:-alfworld}"
NUM_SAMPLES="${2:-10}"
MAX_STEPS="${3:-50}"

# ---- Output ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_SHORT="${EVAL_MODEL//\./_}"
OUTPUT_DIR="${SKILL1_ROOT}/eval_results/${ENV_NAME}_${MODEL_SHORT}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/eval.log"

echo "=========================================="
echo " Claude Eval: ${ENV_NAME}"
echo " Model:       ${EVAL_MODEL}"
echo " Samples:     ${NUM_SAMPLES}"
echo " Max steps:   ${MAX_STEPS}"
echo " Output:      ${OUTPUT_DIR}"
echo " Log:         ${LOG_FILE}"
echo "=========================================="

python3 "${SCRIPT_DIR}/eval_claude.py" \
    --env "${ENV_NAME}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_steps "${MAX_STEPS}" \
    --model "${EVAL_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
