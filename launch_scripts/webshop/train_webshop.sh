#!/usr/bin/env bash
# ==============================================================================
# Train Skill1 on WebShop
# Full pipeline: No split + Top-1 rerank + Pure similarity retrieval
# ==============================================================================
set -uo pipefail

export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
export WANDB_API_KEY="${WANDB_API_KEY:?Please set WANDB_API_KEY}"
# export http_proxy="http://your-proxy:port"
# export https_proxy="http://your-proxy:port"
ulimit -u unlimited 2>/dev/null || ulimit -u 65535 2>/dev/null || true
ulimit -n 65536 2>/dev/null || true

SKILL1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE=vllm
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

num_cpus_per_env_worker=0.1
BASE_TRAIN_BATCH=16
BASE_VAL_BATCH=64
group_size=8
n_gpus_per_node=8

N_NODE="${N_NODE:-1}"
train_data_size=${BASE_TRAIN_BATCH}
val_data_size=${BASE_VAL_BATCH}

RUN_NAME="${RUN_NAME:-skill1_webshop}"
export MODEL_PATH="${MODEL_PATH:-${SKILL1_ROOT}/Qwen/Qwen2.5-7B-Instruct}"
export HDFS_CHECKPOINT_PATH="${SKILL1_ROOT}/trained_models/skill1/webshop/ckpts"
export WEBSHOP_DATA="${SKILL1_ROOT}/data/datasets/webshop/webshop_data"
export HYDRA_FULL_ERROR=1

MEMORY_CACHE_DIR="${SKILL1_ROOT}/trained_models/skill1/webshop/memory_cache/${RUN_NAME}"
export SKILL_LIBRARY_FILE="${MEMORY_CACHE_DIR}/skill_library.json"
export SKILL_ANALYSIS_FILE="${MEMORY_CACHE_DIR}/skill_analysis.jsonl"
export TRAJECTORY_ANALYSIS_FILE="${MEMORY_CACHE_DIR}/trajectory_analysis.jsonl"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RAY_num_server_call_thread=1
export RAY_worker_niceness=10
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
wandb online

n_gpus_per_node="${n_gpus_per_node:-8}"

cd "${SKILL1_ROOT}"
export PYTHONPATH="${SKILL1_ROOT}/agent_system/environments/env_package/webshop/webshop:${SKILL1_ROOT}:${PYTHONPATH:-}"
pip3 install -e . 2>&1 | tail -1
pip3 install "gym==0.24.0" thefuzz python-Levenshtein beautifulsoup4 selenium clean-text rank_bm25 flask werkzeug 2>&1 | tail -1

mkdir -p "${HDFS_CHECKPOINT_PATH}" "${MEMORY_CACHE_DIR}"

ray stop --force 2>/dev/null || true
sleep 5

# ── Multi-node ray cluster setup ──
RAY_PORT=8278
METRICS_EXPORT_PORT=20541
DASHBOARD_AGENT_HTTP_PORT=52365
DASHBOARD_AGENT_GRPC_PORT=53589
RUNTIME_ENV_AGENT_PORT=48869
MIN_WORKER_PORT=10002
MAX_WORKER_PORT=12001
DASHBOARD_PORT=8265

if [ "${N_NODE}" -gt 1 ]; then
    if [ -n "${AFO_ENV_CLUSTER_SPEC:-}" ]; then
        HEAD_ADDR=$(python3 -c "import os, json, socket; spec = json.loads(os.environ['AFO_ENV_CLUSTER_SPEC']); role = spec['role']; master = spec[role][0]; addr, _ = master.split(':'); print(socket.gethostbyname(addr))")
        NODE_RANK=$(python3 -c "import os, json; spec = json.loads(os.environ['AFO_ENV_CLUSTER_SPEC']); print(spec.get('index', 0))")
    elif [ -f "$(pwd)/connection/hope_info.py" ]; then
        read -r HEAD_ADDR NODE_RANK <<< "$(python3 $(pwd)/connection/hope_info.py)"
    else
        HEAD_ADDR=$(hostname -i)
        NODE_RANK=0
    fi
    echo "====[Multi-node] HEAD_ADDR=${HEAD_ADDR}, NODE_RANK=${NODE_RANK}, N_NODE=${N_NODE} ===="
else
    NODE_RANK=0
fi

_SIGNAL_DIR="${MEMORY_CACHE_DIR}/cluster_signal"
mkdir -p "${_SIGNAL_DIR}"

LOG_FILE="${MEMORY_CACHE_DIR}/train_${TIMESTAMP}.log"
echo "=============================================="
echo "  Experiment : ${RUN_NAME}"
echo "  Log        : ${LOG_FILE}"
echo "  N_NODE     : ${N_NODE} (rank=${NODE_RANK})"
echo "=============================================="

TRAIN_CMD="python3 -X faulthandler -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$SKILL1_ROOT/data/datasets/webshop/train_verl.parquet \
    data.val_files=$SKILL1_ROOT/data/datasets/webshop/test_verl.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=16384 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.distill_log_path=$SKILL_ANALYSIS_FILE \
    +data.trajectory_log_path=$TRAJECTORY_ANALYSIS_FILE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    +actor_rollout_ref.actor.tis_imp_ratio_cap=-1 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    +algorithm.intrinsic_reward_coefficient=1.0 \
    +algorithm.intrinsic_hard_cutoff=False \
    +algorithm.distill_reference_policy=False \
    +algorithm.credit_assignment=True \
    env.env_name=Webshop \
    env.skill_library.top_k=3 \
    +env.skill_library.filepath=$SKILL_LIBRARY_FILE \
    env.skill_library.alpha=0.05 \
    +env.skill_library.enable_memory=True \
    +env.skill_library.retrieve_mode=both \
    +env.skill_library.memory_start_cutoff=0.0 \
    +env.skill_library.skill_decay=False \
    +env.skill_library.potential_based_on_binary_success=False \
    +env.skill_library.group_relative_intrinsic_rewards=False \
    +env.skill_library.enable_query_generation=True \
    +env.skill_library.enable_description_head=True \
    +env.skill_library.enable_rerank=True \
    +env.skill_library.distill_reward_type=first_order_diff \
    +env.skill_library.u_hat_aggregation=max \
    +env.skill_library.rerank_train_top1=True \
    +env.skill_library.full_group_memory=True \
    +env.skill_library.relevance_weight=1.0 \
    +env.train_retrieve_type=ucb \
    +env.eval_retrieve_type=greedy \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='skill1_webshop' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$N_NODE \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
    trainer.total_epochs=150 \
    trainer.val_before_train=True"

if [ "${N_NODE}" -eq 1 ]; then
    eval ${TRAIN_CMD} 2>&1 | tee "${LOG_FILE}"
else
    if [ "${NODE_RANK}" -eq 0 ]; then
        ray start --head \
            --port=${RAY_PORT} \
            --dashboard-host='0.0.0.0' \
            --metrics-export-port=${METRICS_EXPORT_PORT} \
            --dashboard-agent-grpc-port=${DASHBOARD_AGENT_GRPC_PORT} \
            --dashboard-agent-listen-port=${DASHBOARD_AGENT_HTTP_PORT} \
            --runtime-env-agent-port=${RUNTIME_ENV_AGENT_PORT} \
            --dashboard-port=${DASHBOARD_PORT} \
            --min-worker-port=${MIN_WORKER_PORT} \
            --max-worker-port=${MAX_WORKER_PORT}

        echo "Waiting for ${N_NODE} nodes to join ray cluster..."
        start_time=$(date +%s)
        timeout=600
        while true; do
            active_nodes=$(ray status 2>/dev/null | grep -c "1 node_" || echo 0)
            echo "  Active nodes: ${active_nodes}/${N_NODE}"
            [ "${active_nodes}" -ge "${N_NODE}" ] && break
            elapsed=$(( $(date +%s) - start_time ))
            if [ "${elapsed}" -ge "${timeout}" ]; then
                echo "ERROR: Timeout (${timeout}s) waiting for workers!"
                exit 1
            fi
            sleep 10
        done
        echo "All nodes joined! Submitting training job..."

        ray job submit --verbose -- ${TRAIN_CMD} 2>&1 | tee "${LOG_FILE}"

        touch "${_SIGNAL_DIR}/head_done.txt"
        sleep 15
    else
        sleep 30
        ray start --address="${HEAD_ADDR}:${RAY_PORT}" \
            --metrics-export-port=${METRICS_EXPORT_PORT} \
            --dashboard-agent-grpc-port=${DASHBOARD_AGENT_GRPC_PORT} \
            --runtime-env-agent-port=${RUNTIME_ENV_AGENT_PORT} \
            --dashboard-agent-listen-port=${DASHBOARD_AGENT_HTTP_PORT} \
            --dashboard-port=${DASHBOARD_PORT} \
            --min-worker-port=${MIN_WORKER_PORT} \
            --max-worker-port=${MAX_WORKER_PORT}
        echo "Worker rank=${NODE_RANK} joined head at ${HEAD_ADDR}:${RAY_PORT}"

        while [ ! -f "${_SIGNAL_DIR}/head_done.txt" ]; do
            echo "Worker waiting for head to finish... ($(date))"
            sleep 300
        done
    fi
fi
