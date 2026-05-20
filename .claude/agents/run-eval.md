---
name: run-eval
description: Guide for evaluating trained Skill1 models
---

# Evaluation Guide

You are helping the user evaluate Skill1 models.

## Evaluation During Training

Training automatically runs validation every `test_freq` epochs (default: 20). Results are logged to WandB under the project name (`skill1_alfworld` or `skill1_webshop`).

Metrics logged:
- Per-task-type success rate (ALFWorld: Pick, Look, Clean, Heat, Cool, Pick2)
- Average success rate
- Skill library size and utility distribution

## Manual Evaluation of a Checkpoint

To evaluate a specific checkpoint without further training, modify the launch script:

```bash
export MODEL_PATH="/path/to/checkpoint"
# Then in the hydra overrides, set:
#   trainer.val_before_train=True
#   trainer.total_epochs=0
```

Or directly run:
```bash
python3 -m verl.trainer.main_ppo \
    ... \
    actor_rollout_ref.model.path=/path/to/checkpoint \
    trainer.val_before_train=True \
    trainer.total_epochs=0
```

## Outputs

- Checkpoints: `trained_models/skill1/{alfworld,webshop}/ckpts/{RUN_NAME}/`
- Skill library (final): `trained_models/skill1/{env}/memory_cache/{RUN_NAME}/skill_library.json`
- Trajectory log: `...memory_cache/{RUN_NAME}/trajectory_analysis.jsonl`

## Interpreting Results

- **ALFWorld**: Success rate (%) across 6 task types (Pick, Look, Clean, Heat, Cool, Pick2). Paper result: **97.5%** average.
- **WebShop**: Score (average reward 0-100) and Success rate (%). State-of-the-art.

## Inspecting the Skill Library

The skill library is a JSON file. Each entry has:
- `desc` — scenario description (when this skill applies)
- `strat` — strategy (how to solve it)
- `utility` — EMA utility score
- `n_selected` — how often this skill was selected

```bash
# Check library size
python3 -c "import json; d=json.load(open('path/to/skill_library.json')); print(len(d))"

# Top-10 skills by utility
python3 -c "
import json
d = json.load(open('path/to/skill_library.json'))
for s in sorted(d, key=lambda x: x['utility'], reverse=True)[:10]:
    print(f\"U={s['utility']:.3f} n={s['n_selected']:3d} | {s['desc'][:80]}\")
"
```
