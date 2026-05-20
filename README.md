# Skill1

**Skill1: Unified Co-Evolution of Skill-Augmented Agents**

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <em>A framework that trains a single policy to co-evolve skill selection, utilization, and distillation toward a shared task-outcome objective.</em>
</p>

---

## Overview

**Skill1** achieves unified evolution of skill-augmented agents by co-evolving three capabilities simultaneously:

- **Skill Selection** — The policy generates a query and re-ranks retrieved candidates from the skill library.
- **Skill Utilization** — The policy performs multi-turn interaction conditioned on the selected skill.
- **Skill Distillation** — The policy reflects on trajectories and distills reusable skills into the library.

All learning signals are derived from a single task-outcome reward, enabling coherent co-evolution without auxiliary models or hand-crafted rewards.

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

### 3. Run Training

```bash
# ALFWorld
bash launch_scripts/alfworld/skill1_full.sh

# WebShop
bash launch_scripts/webshop/skill1_full_webshop.sh
```

---

## Acknowledgments

This code is built upon several open-source projects. We thank the authors and contributors of: [verl](https://github.com/verl-project/verl), [verl-agent](https://github.com/langfengQ/verl-agent/tree/master), and [LaMer](https://github.com/mlbio-epfl/LaMer).

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{skill1,
  title={Skill1: Unified Co-Evolution of Skill-Augmented Agents},
  year={2026}
}
```
