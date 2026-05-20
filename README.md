# Skill1

**Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning**

<p align="center">
  <a href="https://arxiv.org/abs/2605.06130"><img src="https://img.shields.io/badge/arXiv-2605.06130-b31b1b?style=for-the-badge&logo=arxiv" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2605.06130"><img src="https://img.shields.io/badge/HuggingFace-Paper-ffd21e?style=for-the-badge&logo=huggingface" alt="HuggingFace"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge" alt="License"></a>
</p>

---

## TL;DR

LLM agents can be augmented with a **skill library** — a persistent memory of reusable strategies. Using such a library requires three coupled capabilities: **selecting** a relevant skill, **utilizing** it during execution, and **distilling** new skills from experience. Prior methods optimize these in isolation with separate reward signals, causing conflicting evolution.

**Skill1** trains a single policy (Qwen2.5-7B) via RL (GRPO) to co-evolve all three capabilities using only one task-outcome reward. Credit assignment is achieved by decomposing the reward into a low-frequency trend (credits selection) and high-frequency variation (credits distillation).

---

## How It Works

```
Task → [Selection] → [Utilization] → [Distillation] → Skill Library
         ↑                                                   |
         └───────────────────────────────────────────────────┘
```

1. **Skill Selection** — Policy generates a query, retrieves candidates via a frozen encoder, and re-ranks them.
2. **Skill Utilization** — Policy interacts with the environment conditioned on the selected skill.
3. **Skill Distillation** — Policy reflects on the trajectory and writes a new reusable skill (strategy + scenario description) into the library.

All three stages are produced by the same policy and optimized by the same task-outcome signal — no auxiliary models, no hand-crafted rewards.

---

## Quick Start

### 1. Install Base Environment

```bash
conda create -n skill1 python==3.12 -y
conda activate skill1

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

### 2. Install Task Environments

ALFWorld:
```bash
conda env create -f agent-alfworld-env.yaml
```

WebShop:
```bash
conda env create -f agent-webshop-env.yaml
```

### 3. Download Data

We use download the Alfworld and WebShop data from the original sources: [alfworld/alfworld](https://github.com/alfworld/alfworld) | [princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop)

### 4. Run Training

```bash
# ALFWorld
bash launch_scripts/alfworld/train_alfworld.sh

# WebShop
bash launch_scripts/webshop/train_webshop.sh
```

---

## Acknowledgments

This code is built upon several open-source projects. We thank the authors and contributors of: [verl](https://github.com/verl-project/verl), [verl-agent](https://github.com/langfengQ/verl-agent/tree/master), and [LaMer](https://github.com/mlbio-epfl/LaMer).

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{shi2026skill1,
  title={Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning},
  author={Shi, Yaorui and Chen, Yuxin and Lu, Zhengxi and Miao, Yuchun and Liu, Shugui and Gu, Qi and Cai, Xunliang and Wang, Xiang and Zhang, An},
  journal={arXiv preprint arXiv:2605.06130},
  year={2026}
}
```
