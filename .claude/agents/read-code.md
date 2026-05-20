---
name: read-code
description: Guide for understanding the Skill1 codebase architecture and key modules
---

# Code Reading Guide

You are helping the user understand the Skill1 codebase.

## Architecture Overview

Skill1 extends the **verl** RL framework with a skill-augmented agent system. The main addition is `agent_system/` which implements the three-stage lifecycle.

## Core Modules

### 1. Skill Library & Memory (`agent_system/memory/`)

- `base.py` — BaseMemory interface
- `memory.py` — SimpleMemory: stores per-env step history (obs, action pairs)

The **skill library** itself is managed within the rollout loop (not a separate module). It's a JSON file containing skills with fields:
- `desc` — scenario description (used for retrieval)
- `strat` — strategy text (used during utilization)
- `utility` — EMA utility score U(s)
- `n_selected` — selection count

### 2. Multi-Turn Rollout (`agent_system/multi_turn_rollout/`)

- `rollout_loop.py` — `TrajectoryCollector`: the workhorse class
  - Handles credit assignment (lambda_rerank, lambda_distill)
  - Computes token-level advantages via GRPO
  - Manages the selection → utilization → distillation sequence
- `metarl_rollout_loop.py` — Skill1-specific rollout logic
  - Query generation stage
  - Reranking stage
  - Conditioned multi-turn interaction
  - Post-episode skill distillation
- `utils.py` — Helpers for image processing, data conversion

### 3. Environments (`agent_system/environments/`)

- `base.py` — Base environment interface
- `env_manager.py` — Parallel environment manager (Ray actors)
- `env_package/alfworld/` — Vendored ALFWorld source
- `env_package/webshop/` — Vendored WebShop source

### 4. Reward Manager (`agent_system/reward_manager/`)

- `episode.py` — Episode-level reward computation

### 5. RL Training Framework (`verl/`)

Forked from [verl](https://github.com/verl-project/verl). Key paths:
- `verl/trainer/main_ppo.py` — Main training loop
- `verl/trainer/ppo/` — PPO/GRPO algorithm implementation
- `verl/workers/` — Actor, rollout, ref model workers
- `verl/utils/dataset/rl_dataset.py` — Data loading

## Data Flow

```
1. main_ppo.py loads prompts from parquet
2. Prompts dispatched to rollout workers (vLLM)
3. metarl_rollout_loop orchestrates:
   a. Generate query → retrieve from skill library → rerank
   b. Multi-turn interaction with env (up to max_steps)
   c. Generate distilled skill from trajectory
4. TrajectoryCollector computes rewards:
   - R_util = r(tau) (binary task outcome)
   - R_rerank = lambda_rerank * U(selected_skill)
   - R_distill = lambda_distill * (r(tau) - U_hat)
5. GRPO update on actor parameters
6. Skill library updated (add new skills if r=1, update utilities)
```

## Config System

Uses **Hydra** (OmegaConf). Base configs in `verl/trainer/config/`. Override via command-line in launch scripts. Key config groups:
- `data.*` — dataset paths, lengths, batch sizes
- `actor_rollout_ref.*` — model, optimizer, rollout engine
- `env.*` — environment name, skill library settings
- `algorithm.*` — GRPO, credit assignment, intrinsic reward
- `trainer.*` — logging, checkpointing, epochs
