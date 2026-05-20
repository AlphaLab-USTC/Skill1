#!/usr/bin/env python3
"""
Standalone evaluation of Claude models on ALFWorld / WebShop environments.

Bypasses the verl/Ray training infrastructure and directly talks to the
environment + Anthropic API.

Usage:
    python3 eval_claude.py --env alfworld --num_samples 10
    python3 eval_claude.py --env webshop  --num_samples 10
"""

import argparse
import json
import os
import re
import sys
import time
import difflib
import random
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import anthropic

##############################
# Configurable Constants
##############################
SKILL1_ROOT = os.environ.get(
    "SKILL1_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
ANTHROPIC_BASE_URL = os.environ.get(
    "ANTHROPIC_BASE_URL",
    "https://aigc.sankuai.com/v1/anthropic/",
)
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "2038469073881301067")
DEFAULT_MODEL = os.environ.get("EVAL_MODEL", "aws.claude-opus-4.6-b")
HISTORY_LENGTH = 2  # rolling window of recent (obs, action) pairs

##############################
# Prompt Templates (copied inline to avoid import-time side effects)
##############################
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.

Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.

Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.

Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


##############################
# Helpers
##############################

def parse_action(response_text: str) -> str:
    """Extract the content between <action> and </action> tags."""
    m = re.search(r"<action>(.*?)</action>", response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: try last line
    lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def match_alfworld_action(parsed: str, admissible: list) -> tuple:
    """Fuzzy-match parsed action to admissible commands. Returns (action, is_valid)."""
    if parsed in admissible:
        return parsed, True
    matches = difflib.get_close_matches(parsed, admissible, n=1, cutoff=0.5)
    if matches:
        return matches[0], True
    return "look", False


def call_claude(client, model: str, prompt: str, max_tokens: int = 2048) -> str:
    """Send a single prompt to Claude and return the text response."""
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def format_history(history: list) -> str:
    """Format a rolling history list of (obs, action) into a readable string."""
    parts = []
    for obs, act in history:
        obs_short = obs[:300] + "..." if len(obs) > 300 else obs
        parts.append(f"Observation: {obs_short}\nAction: {act}")
    return "\n---\n".join(parts)


##############################
# ALFWorld Evaluation
##############################

def eval_alfworld(client, model, num_samples, max_steps, output_dir):
    import yaml
    # Add alfworld package to path
    alf_pkg = os.path.join(SKILL1_ROOT, "agent_system/environments/env_package/alfworld")
    if alf_pkg not in sys.path:
        sys.path.insert(0, alf_pkg)

    from alfworld.agents.environment import get_environment

    config_path = os.path.join(alf_pkg, "configs/config_tw.yaml")
    with open(config_path) as f:
        alf_config = yaml.safe_load(f)

    base_env = get_environment(alf_config["env"]["type"])(alf_config, train_eval="eval_in_distribution")
    env = base_env.init_env(batch_size=1)

    results = []
    success_count = 0
    task_type_stats = defaultdict(lambda: {"total": 0, "success": 0})

    for ep in range(num_samples):
        env.seed(ep)
        obs, infos = env.reset()
        obs_text = obs[0]
        admissible = infos["admissible_commands"][0]

        # Extract task
        task_start = obs_text.find("Your task is to: ")
        task_desc = obs_text[task_start + len("Your task is to: "):].strip() if task_start != -1 else obs_text

        # Detect task type from gamefile
        gamefile = infos.get("extra.gamefile", ["unknown"])[0]
        task_types = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
                      "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_clean_then_place_in_recep"]
        task_type = "unknown"
        for tt in task_types:
            if tt in str(gamefile):
                task_type = tt
                break

        print(f"\n{'='*60}")
        print(f"[Episode {ep+1}/{num_samples}] Task: {task_desc[:100]}...")
        print(f"  Type: {task_type}")
        print(f"{'='*60}")

        history = []
        trajectory = []
        done = False
        won = False

        for step in range(max_steps):
            # Build prompt
            formatted_admissible = "\n ".join(f"'{s}'" for s in admissible if s != "help")
            if not history:
                prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=obs_text,
                    admissible_actions=formatted_admissible,
                )
            else:
                prompt = ALFWORLD_TEMPLATE.format(
                    task_description=task_desc,
                    step_count=step,
                    history_length=min(len(history), HISTORY_LENGTH),
                    action_history=format_history(history[-HISTORY_LENGTH:]),
                    current_step=step + 1,
                    current_observation=obs_text,
                    admissible_actions=formatted_admissible,
                )

            # Call Claude
            t0 = time.time()
            response = call_claude(client, model, prompt)
            api_time = time.time() - t0

            parsed_action = parse_action(response)
            action, is_valid = match_alfworld_action(parsed_action, admissible)

            print(f"  Step {step+1}: action='{action}' (valid={is_valid}, api={api_time:.1f}s)")

            trajectory.append({
                "step": step + 1,
                "observation": obs_text[:500],
                "admissible": admissible,
                "claude_response": response,
                "parsed_action": parsed_action,
                "executed_action": action,
                "is_valid": is_valid,
                "api_time": api_time,
            })

            history.append((obs_text, action))

            # Step env
            obs_list, scores, dones, infos_step = env.step([action])
            obs_text = obs_list[0]
            for k in infos_step:
                infos_step[k] = infos_step[k][0] if isinstance(infos_step[k], list) else infos_step[k]
            admissible = infos_step["admissible_commands"]
            done = dones[0]
            won = bool(infos_step.get("won", False))

            if done:
                print(f"  >>> DONE at step {step+1}, won={won}")
                break

        success_count += int(won)
        task_type_stats[task_type]["total"] += 1
        task_type_stats[task_type]["success"] += int(won)

        episode_result = {
            "episode": ep + 1,
            "task": task_desc,
            "task_type": task_type,
            "gamefile": str(gamefile),
            "num_steps": len(trajectory),
            "won": won,
            "trajectory": trajectory,
        }
        results.append(episode_result)

        # Write incrementally
        with open(os.path.join(output_dir, "trajectories.jsonl"), "a") as f:
            f.write(json.dumps(episode_result, ensure_ascii=False) + "\n")

    return results, success_count, num_samples, dict(task_type_stats)


##############################
# WebShop Evaluation
##############################

def eval_webshop(client, model, num_samples, max_steps, output_dir):
    import gym
    # Add webshop package to path
    webshop_pkg = os.path.join(SKILL1_ROOT, "agent_system/environments/env_package/webshop/webshop")
    if webshop_pkg not in sys.path:
        sys.path.insert(0, webshop_pkg)

    from web_agent_site.envs import WebAgentTextEnv  # noqa: triggers gym registration

    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None)

    # Test session indices (first 500 are test set)
    test_indices = list(range(500))
    random.seed(42)
    selected = random.sample(test_indices, min(num_samples, len(test_indices)))

    results = []
    success_count = 0
    total_score = 0.0

    for ep_idx, session_idx in enumerate(selected):
        obs, info = env.reset(session=session_idx)
        info = dict(info or {})
        avail = env.get_available_actions()

        # Parse task from obs
        parts = obs.split(" [SEP] ")
        task_desc = parts[2] if len(parts) > 2 and parts[1] == "Instruction:" else obs
        # Format observation (remove task prefix)
        try:
            idx = parts.index(task_desc)
            obs_formatted = " [SEP] ".join(parts[idx + 1:])
        except (ValueError, IndexError):
            obs_formatted = obs

        print(f"\n{'='*60}")
        print(f"[Episode {ep_idx+1}/{num_samples}] Session {session_idx}: {task_desc[:100]}...")
        print(f"{'='*60}")

        history = []
        trajectory = []
        done = False
        final_reward = 0.0

        for step in range(max_steps):
            # Format available actions
            actions_list = []
            if avail.get("has_search_bar"):
                actions_list.append("search[<your query>]")
            for txt in avail.get("clickables", []):
                actions_list.append(f"click[{txt}]")
            formatted_avail = "\n".join(f"'{s}'," for s in actions_list)

            # Build prompt
            if not history:
                prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_desc,
                    current_observation=obs_formatted,
                    available_actions=formatted_avail,
                )
            else:
                prompt = WEBSHOP_TEMPLATE.format(
                    task_description=task_desc,
                    step_count=step,
                    history_length=min(len(history), HISTORY_LENGTH),
                    action_history=format_history(history[-HISTORY_LENGTH:]),
                    current_step=step + 1,
                    current_observation=obs_formatted,
                    available_actions=formatted_avail,
                )

            # Truncate if too long
            if len(prompt) > 13000:
                prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_desc,
                    current_observation=obs_formatted,
                    available_actions=formatted_avail,
                )

            # Call Claude
            t0 = time.time()
            response = call_claude(client, model, prompt)
            api_time = time.time() - t0

            action = parse_action(response)
            # Validate: must be search[...] or click[...]
            is_valid = bool(re.match(r"(search\[.*\]|click\[.*\])", action))
            if not is_valid and actions_list:
                # Try to extract from response more aggressively
                m = re.search(r"(search\[.*?\]|click\[.*?\])", response)
                if m:
                    action = m.group(1)
                    is_valid = True
                else:
                    action = actions_list[0] if actions_list else "click[back to search]"

            print(f"  Step {step+1}: action='{action[:80]}' (valid={is_valid}, api={api_time:.1f}s)")

            trajectory.append({
                "step": step + 1,
                "observation": obs_formatted[:500],
                "available_actions": actions_list[:20],
                "claude_response": response,
                "parsed_action": action,
                "is_valid": is_valid,
                "api_time": api_time,
            })

            history.append((obs_formatted[:300], action))

            # Step env
            obs_raw, reward, done, info_step = env.step(action)
            info_step = dict(info_step or {})
            avail = env.get_available_actions()
            final_reward = info_step.get("task_score", reward)

            # Re-format obs
            parts_new = obs_raw.split(" [SEP] ")
            try:
                idx_new = parts_new.index(task_desc)
                obs_formatted = " [SEP] ".join(parts_new[idx_new + 1:])
            except (ValueError, IndexError):
                obs_formatted = obs_raw

            if done:
                won = (reward == 1.0) or (final_reward == 1.0)
                print(f"  >>> DONE at step {step+1}, score={final_reward:.3f}, won={won}")
                break

        won = (final_reward == 1.0)
        success_count += int(won)
        total_score += final_reward

        episode_result = {
            "episode": ep_idx + 1,
            "session_idx": session_idx,
            "task": task_desc,
            "num_steps": len(trajectory),
            "final_score": final_reward,
            "won": won,
            "trajectory": trajectory,
        }
        results.append(episode_result)

        with open(os.path.join(output_dir, "trajectories.jsonl"), "a") as f:
            f.write(json.dumps(episode_result, ensure_ascii=False) + "\n")

    env.close()
    return results, success_count, num_samples, total_score


##############################
# Main
##############################

def main():
    parser = argparse.ArgumentParser(description="Evaluate Claude on ALFWorld/WebShop")
    parser.add_argument("--env", required=True, choices=["alfworld", "webshop"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    args = parser.parse_args()

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.replace(".", "_").replace("/", "_")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(SKILL1_ROOT, "eval_results", f"{args.env}_{model_short}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = timestamp
    config["anthropic_base_url"] = ANTHROPIC_BASE_URL
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model:       {args.model}")
    print(f"Environment: {args.env}")
    print(f"Samples:     {args.num_samples}")
    print(f"Max steps:   {args.max_steps}")
    print(f"Output:      {output_dir}")
    print()

    client = anthropic.Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_AUTH_TOKEN,
    )

    t_start = time.time()

    if args.env == "alfworld":
        results, successes, total, task_stats = eval_alfworld(
            client, args.model, args.num_samples, args.max_steps, output_dir
        )
        summary = {
            "env": "alfworld",
            "model": args.model,
            "num_samples": total,
            "successes": successes,
            "success_rate": successes / total if total > 0 else 0,
            "task_type_breakdown": {
                k: {**v, "rate": v["success"] / v["total"] if v["total"] > 0 else 0}
                for k, v in task_stats.items()
            },
            "total_time_s": time.time() - t_start,
        }

    elif args.env == "webshop":
        results, successes, total, total_score = eval_webshop(
            client, args.model, args.num_samples, args.max_steps, output_dir
        )
        summary = {
            "env": "webshop",
            "model": args.model,
            "num_samples": total,
            "successes": successes,
            "success_rate": successes / total if total > 0 else 0,
            "avg_score": total_score / total if total > 0 else 0,
            "total_time_s": time.time() - t_start,
        }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
