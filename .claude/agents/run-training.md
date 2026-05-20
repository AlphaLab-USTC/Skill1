---
name: run-training
description: Guide for launching Skill1 training on ALFWorld or WebShop
---

# Training Guide

You are helping the user launch or debug Skill1 training.

## Prerequisites

1. Base conda env `skill1` is activated
2. `WANDB_API_KEY` is set
3. Model weights exist at `huggingface.co/Qwen/Qwen2.5-7B-Instruct/` (or set `MODEL_PATH`)
4. 8 GPUs available per node

## Quick Launch

### ALFWorld
```bash
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/shiyaorui/code/skill1_private
bash launch_scripts/alfworld/train_alfworld.sh
```

### WebShop
```bash
bash launch_scripts/webshop/train_webshop.sh
```

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | (required) | WandB API key |
| `MODEL_PATH` | `huggingface.co/Qwen/Qwen2.5-7B-Instruct` | Base model path |
| `RUN_NAME` | `skill1_alfworld` / `skill1_webshop` | Experiment name |
| `N_NODE` | `1` | Number of nodes (multi-node with Ray) |

## Training Outputs

- Checkpoints: `trained_models/skill1/{alfworld,webshop}/ckpts/{RUN_NAME}/`
- Skill library: `trained_models/skill1/{env}/memory_cache/{RUN_NAME}/skill_library.json`
- Skill analysis: `...memory_cache/{RUN_NAME}/skill_analysis.jsonl`
- Trajectory log: `...memory_cache/{RUN_NAME}/trajectory_analysis.jsonl`
- Training log: `...memory_cache/{RUN_NAME}/train_{TIMESTAMP}.log`

## Important Hyperparameters

These can be overridden via command-line Hydra overrides:

- `data.train_batch_size=16` — prompts per training step
- `env.rollout.n=8` — rollouts per prompt (group size for GRPO)
- `actor_rollout_ref.actor.optim.lr=1e-6` — learning rate
- `env.skill_library.alpha=0.05` — EMA smoothing for skill utility
- `env.skill_library.lambda_distill=0.3` — weight for distillation reward
- `env.skill_library.lambda_rerank=0.3` — weight for rerank reward
- `env.skill_library.top_k=3` — number of candidates to retrieve for reranking
- `env.max_steps=50` — max env interaction turns
- `trainer.total_epochs=150` — total training epochs
- `trainer.test_freq=20` — evaluate every N epochs
- `trainer.save_freq=50` — save checkpoint every N epochs

## Multi-Node Training

Set `N_NODE=2` (or more). The script auto-detects head/worker roles via:
1. `AFO_ENV_CLUSTER_SPEC` env var (cluster scheduler)
2. `connection/hope_info.py` (custom cluster)
3. Falls back to single-node

## Debugging Tips

- Set `HYDRA_FULL_ERROR=1` (already set in scripts) for full stack traces
- Check WandB dashboard for training curves (project: `skill1_alfworld` or `skill1_webshop`)
- `ray status` to check cluster health
- Kill zombies: `ray stop --force`
- The skill library JSON grows over training — inspect it to see what skills are being learned
