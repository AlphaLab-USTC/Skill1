---
name: setup-env
description: Guide for setting up the Skill1 development environment (base + task-specific)
---

# Environment Setup Guide

You are helping the user set up the Skill1 development environment.

## Base Environment (skill1)

```bash
conda create -n skill1 python==3.12 -y
conda activate skill1

# Core dependencies
pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# Install the project
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/LA/shiyaorui/code/skill1_private
pip install -e .
```

## ALFWorld Task Environment

```bash
conda env create -f agent-alfworld-env.yaml
# This creates env named "alfworld-qwen2" with Python 3.11
# Key packages: alfworld==0.4.2, textworld==1.6.2, vllm==0.8.5.post1
```

Required PYTHONPATH for ALFWorld:
```bash
export PYTHONPATH="${SKILL1_ROOT}/agent_system/environments/env_package/alfworld:${SKILL1_ROOT}:${PYTHONPATH:-}"
```

ALFWorld data must exist at: `data/alfworld/`

## WebShop Task Environment

```bash
conda env create -f agent-webshop-env.yaml
```

Required PYTHONPATH for WebShop:
```bash
export PYTHONPATH="${SKILL1_ROOT}/agent_system/environments/env_package/webshop/webshop:${SKILL1_ROOT}:${PYTHONPATH:-}"
```

WebShop data must exist at: `data/datasets/webshop/webshop_data/`

## Model Weights

The base model (Qwen2.5-7B-Instruct) should be at:
```
huggingface.co/Qwen/Qwen2.5-7B-Instruct/
```

Override via `MODEL_PATH` env var if stored elsewhere.

## Common Issues

1. **CUDA OOM**: Reduce `actor_rollout_ref.rollout.gpu_memory_utilization` (default 0.7)
2. **Ray cluster errors**: Run `ray stop --force` before starting training
3. **textworld import errors**: Make sure the alfworld conda env has `textworld[pddl]` installed
4. **vLLM version mismatch**: The base env uses vllm==0.11.0; the alfworld env uses 0.8.5 — use the base env for training
