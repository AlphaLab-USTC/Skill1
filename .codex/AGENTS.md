# Skill1 Project Guide

## What is this project?

**Skill1** trains a single LLM policy (Qwen2.5-7B) via RL (GRPO) to co-evolve three capabilities of skill-augmented agents:
1. **Skill Selection** — generate query + rerank retrieved skills
2. **Skill Utilization** — multi-turn env interaction conditioned on selected skill
3. **Skill Distillation** — reflect on trajectory, write new reusable skill to library

All learning comes from a single binary task-outcome reward, decomposed into:
- Low-frequency trend (EMA per-skill utility) → credits selection
- High-frequency variation (outcome - trend) → credits distillation

## Repo Structure

```
.
├── agent_system/               # Core agent logic
│   ├── environments/           # Env wrappers (ALFWorld, WebShop)
│   │   └── env_package/        # Vendored env source code
│   ├── memory/                 # Memory/history management
│   ├── multi_turn_rollout/     # Trajectory collection & credit assignment
│   └── reward_manager/         # Reward computation
├── verl/                       # RL training framework (forked from verl)
├── launch_scripts/             # Training entry points
│   ├── alfworld/               # ALFWorld training configs
│   └── webshop/                # WebShop training configs
├── data/                       # Datasets (parquet) and env data
├── trained_models/             # Checkpoints & skill library caches
├── huggingface.co/             # Local model weights (Qwen2.5-7B-Instruct)
└── 3rdparty/                   # Third-party dependencies
```

## Key Files

- `agent_system/multi_turn_rollout/rollout_loop.py` — TrajectoryCollector: handles credit assignment, token-level reward shaping
- `agent_system/multi_turn_rollout/metarl_rollout_loop.py` — Skill1-specific rollout with selection/utilization/distillation stages
- `agent_system/environments/env_manager.py` — Manages parallel env instances
- `verl/trainer/main_ppo.py` — Main training entry point (GRPO algorithm)
- `launch_scripts/alfworld/train_alfworld.sh` — Full training script for ALFWorld
- `launch_scripts/webshop/train_webshop.sh` — Full training script for WebShop

## Environments

### ALFWorld
- Conda env: `alfworld-qwen2` (from `agent-alfworld-env.yaml`)
- Text-based household tasks: Pick, Look, Clean, Heat, Cool, Pick2
- Data: `data/alfworld/` and `data/datasets/alfworld/`
- PYTHONPATH must include `agent_system/environments/env_package/alfworld`

### WebShop
- Conda env: from `agent-webshop-env.yaml`
- Online shopping simulator: search + purchase matching user specs
- Data: `data/datasets/webshop/webshop_data/`
- PYTHONPATH must include `agent_system/environments/env_package/webshop/webshop`

## Training

Built on **verl** (Volcano Engine RL for LLM). Uses:
- GRPO algorithm with group_size=8 or 16
- vLLM for rollout inference (tensor_model_parallel_size=4)
- FSDP for actor training
- Ray for distributed orchestration
- WandB for logging

Key hyperparameters (in launch scripts):
- `lr=1e-6`, `kl_loss_coef=0.01`
- `env.skill_library.alpha=0.05` (EMA smoothing for utility)
- `env.skill_library.lambda_distill=0.3`, `lambda_rerank=0.3`
- `env.skill_library.top_k=3` (retrieve 3 candidates for reranking)
- `env.max_steps=50` (max interaction turns per episode)

## Requirements

- Python 3.12, CUDA 12.x
- vLLM 0.11.0, flash-attn 2.7.4
- 8x GPUs per node (A100/H100 recommended)
- `WANDB_API_KEY` must be set for training
