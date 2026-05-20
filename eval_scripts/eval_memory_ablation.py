#!/usr/bin/env python3
"""
Memory Ablation: No Memory vs Text Memory vs OCR Visual Memory (Multi-Trial)

Each task is attempted up to N trials. After each failed trial, a skill is
generated. The next trial uses accumulated skills as memory. This design
ensures memory is directly relevant (same task) and measures retry improvement.

Conditions:
  no_memory:    Every trial is independent — no skill library carried over
  text_memory:  Reflections from prior trials injected as plain text
  ocr_memory:   Reflections rendered as images and sent as multimodal input

Usage:
    python3 eval_memory_ablation.py --env alfworld --num_tasks 10 --max_trials 3
    python3 eval_memory_ablation.py --env alfworld --num_tasks 10 --models gpt-4.1,kimi-k2.5
"""

import argparse
import base64
import difflib
import json
import os
import random
import re
import sys
import textwrap
import time
from collections import defaultdict
from datetime import datetime
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

##############################
# Configurable Constants
##############################
SKILL1_ROOT = os.environ.get(
    "SKILL1_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://aigc.sankuai.com/v1/openai/native")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "1985951821684846684")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://aigc.sankuai.com/v1/anthropic/")
ANTHROPIC_AUTH_TOKEN = os.environ.get("ANTHROPIC_AUTH_TOKEN", "2038469073881301067")

MD_API_URL = os.environ.get("MD_API_URL", "http://33.235.246.42:8000/render")
MEMORY_MODES = ["no_memory", "text_memory", "ocr_memory"]
HISTORY_LENGTH = 2
MAX_RETRIES_PER_CALL = 2  # retry on transient API errors

##############################
# Prompt Templates
##############################
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
{skills}
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
{skills}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_REFLECT_TEMPLATE = """
You are an expert evaluating an ALFRED Embodied Environment task attempt.
Your task is to: {task_description}

You have just completed attempt #{trial_num} at this task. The task was {success} completed.

Trajectory of the attempt:
{current_trajectory}

<think>
Given the task outcome, analyze the trajectory to understand:
1. What subtasks were attempted? (pick up, navigate, use appliance, place object)
2. Which subtasks succeeded vs failed based on the observations?
3. What specific actions or decisions led to this outcome?
4. What is the most valuable lesson from this attempt?
</think>

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "pick_up_object", "description": "[describe pickup action]", "status": "[completed or incomplete]"}},
{{"name": "navigate_to_location", "description": "[describe navigation]", "status": "[completed or incomplete]"}},
{{"name": "use_appliance", "description": "[describe appliance use if applicable]", "status": "[completed or incomplete or N/A]"}},
{{"name": "place_object", "description": "[describe placement]", "status": "[completed or incomplete]"}}
],
"task_success": [true or false],
"action_lesson": "[key action insight with specific object/location references]",
"navigation_lesson": "[spatial insight with specific location references]"
}}

Output ONLY the JSON evaluation.
"""

WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
{skills}
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
{skills}
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

WEBSHOP_REFLECT_TEMPLATE = """
You are an expert evaluating a WebShop shopping attempt.
Your task is to: {task_description}

You have just completed attempt #{trial_num} at this task. The task was {success} completed.

Trajectory of the attempt:
{current_trajectory}

<think>
Given the task outcome, analyze the trajectory to understand:
1. What subtasks were attempted? (search, filter, select, purchase)
2. Which subtasks succeeded vs failed based on the observations?
3. What specific actions or decisions led to this outcome?
4. What are the 1-2 most valuable lessons from this attempt?
</think>

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "search_product", "description": "[describe actual search]", "status": "[completed or incomplete]"}},
{{"name": "apply_filters", "description": "[describe filters used]", "status": "[completed or incomplete]"}},
{{"name": "select_item", "description": "[describe selection]", "status": "[completed or incomplete]"}},
{{"name": "complete_purchase", "description": "[describe purchase]", "status": "[completed or incomplete]"}}
],
"task_success": [true or false],
"action_lesson": "[key action insight]",
"navigation_lesson": "[navigation insight]"
}}

Output ONLY the JSON evaluation.
"""

##############################
# Unified LLM Client
##############################

def make_client(model: str):
    if "claude" in model.lower() or model.startswith("aws."):
        import anthropic
        return anthropic.Anthropic(base_url=ANTHROPIC_BASE_URL, api_key=ANTHROPIC_AUTH_TOKEN), "anthropic"
    else:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL), "openai"


def call_llm(client, client_type: str, model: str, prompt: str,
             images: list = None, max_tokens: int = 2048) -> str:
    for attempt in range(MAX_RETRIES_PER_CALL + 1):
        try:
            if client_type == "openai":
                if images:
                    content = []
                    for img_bytes in images:
                        b64 = base64.b64encode(img_bytes).decode()
                        content.append({"type": "image_url",
                                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
                    content.append({"type": "text", "text": prompt})
                    messages = [{"role": "user", "content": content}]
                else:
                    messages = [{"role": "user", "content": prompt}]
                resp = client.chat.completions.create(
                    model=model, max_tokens=max_tokens, messages=messages)
                return resp.choices[0].message.content
            else:
                if images:
                    content = []
                    for img_bytes in images:
                        b64 = base64.b64encode(img_bytes).decode()
                        content.append({"type": "image",
                                        "source": {"type": "base64", "media_type": "image/png", "data": b64}})
                    content.append({"type": "text", "text": prompt})
                    messages = [{"role": "user", "content": content}]
                else:
                    messages = [{"role": "user", "content": prompt}]
                resp = client.messages.create(
                    model=model, max_tokens=max_tokens, messages=messages)
                return resp.content[0].text
        except Exception as e:
            if attempt < MAX_RETRIES_PER_CALL:
                wait = 5 * (attempt + 1)
                print(f"    [API error, retry {attempt+1}/{MAX_RETRIES_PER_CALL} in {wait}s]: {e}")
                time.sleep(wait)
            else:
                print(f"    [API error, giving up]: {e}")
                return ""


##############################
# Markdown → Image Rendering
##############################

_FONTS = None

def get_fonts():
    global _FONTS
    if _FONTS is None:
        def try_load(candidates, size):
            for p in candidates:
                if os.path.exists(p):
                    return ImageFont.truetype(p, size)
            return ImageFont.load_default()
        _FONTS = {
            'regular': try_load(['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'], 15),
            'bold': try_load(['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'], 15),
            'h2': try_load(['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'], 20),
            'mono': try_load(['/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'], 14),
        }
    return _FONTS


def render_markdown_to_image(md_text: str, width: int = 900) -> bytes:
    fonts = get_fonts()
    padding, bg, text_c, accent_c, line_c = 24, 'white', '#1a1a1a', '#2563eb', '#e5e7eb'
    cw = width - 2 * padding

    elements, in_table, table_rows = [], False, []
    for line in md_text.strip().split('\n'):
        line = line.strip()
        if not line:
            if in_table and table_rows:
                elements.append(('table', table_rows)); table_rows = []; in_table = False
            elements.append(('space', 8)); continue
        if line.startswith('## '):
            if in_table and table_rows:
                elements.append(('table', table_rows)); table_rows = []; in_table = False
            elements.append(('h2', line[3:]))
        elif line.startswith('| '):
            if '---' in line: continue
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells: table_rows.append(cells); in_table = True
        elif line.startswith(('- ', '* ')):
            if in_table and table_rows:
                elements.append(('table', table_rows)); table_rows = []; in_table = False
            elements.append(('bullet', line[2:]))
        elif line.startswith('**') and '**' in line[2:]:
            if in_table and table_rows:
                elements.append(('table', table_rows)); table_rows = []; in_table = False
            elements.append(('bold_text', re.sub(r'\*\*', '', line)))
        else:
            if in_table and table_rows:
                elements.append(('table', table_rows)); table_rows = []; in_table = False
            elements.append(('text', line))
    if table_rows: elements.append(('table', table_rows))

    y, ops = padding, []
    wrap_w = int(cw / 8.5)
    for kind, data in elements:
        if kind == 'h2':
            ops.append(('h2', data, y)); y += 30
            ops.append(('line', None, y)); y += 12
        elif kind == 'space': y += data
        elif kind == 'bullet':
            for i, wl in enumerate(textwrap.wrap(data, width=wrap_w)):
                ops.append(('text', ('  \u2022  ' if i == 0 else '     ') + wl, y)); y += 22
        elif kind == 'bold_text':
            for wl in textwrap.wrap(data, width=wrap_w):
                ops.append(('bold', wl, y)); y += 22
        elif kind == 'table':
            hdr, rows = data[0], data[1:]
            col_w = cw // max(len(hdr), 1)
            ops.append(('thdr', hdr, y, col_w)); y += 26
            ops.append(('line', None, y)); y += 4
            for row in rows:
                ops.append(('trow', row, y, col_w)); y += 24
            y += 8
        elif kind == 'text':
            for wl in textwrap.wrap(data, width=wrap_w):
                ops.append(('text', wl, y)); y += 22

    img = Image.new('RGB', (width, y + padding), bg)
    draw = ImageDraw.Draw(img)
    for op in ops:
        k = op[0]
        if k == 'h2': draw.text((padding, op[2]), op[1], fill=accent_c, font=fonts['h2'])
        elif k == 'line': draw.line([(padding, op[2]), (width-padding, op[2])], fill=line_c, width=1)
        elif k == 'text': draw.text((padding, op[2]), op[1], fill=text_c, font=fonts['regular'])
        elif k == 'bold': draw.text((padding, op[2]), op[1], fill=text_c, font=fonts['bold'])
        elif k == 'thdr':
            for j, c in enumerate(op[1]): draw.text((padding+j*op[3], op[2]), c, fill=accent_c, font=fonts['bold'])
        elif k == 'trow':
            for j, c in enumerate(op[1]):
                cl = c.lower().strip()
                color = '#059669' if cl=='completed' else ('#dc2626' if cl=='incomplete' else text_c)
                draw.text((padding+j*op[3], op[2]), c, fill=color, font=fonts['mono'])
    buf = BytesIO(); img.save(buf, format='PNG'); return buf.getvalue()


def try_render_image(md_text: str) -> bytes:
    try:
        import requests
        resp = requests.post(MD_API_URL, json={"content": md_text, "format": "png"}, timeout=5)
        if resp.status_code == 200: return resp.content
    except Exception: pass
    return render_markdown_to_image(md_text)


##############################
# Action Parsing
##############################

def parse_action(text: str) -> str:
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if m: return m.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""

def match_alfworld_action(parsed: str, admissible: list) -> tuple:
    if parsed in admissible: return parsed, True
    matches = difflib.get_close_matches(parsed, admissible, n=1, cutoff=0.5)
    if matches: return matches[0], True
    return "look", False

def format_history(history: list) -> str:
    parts = []
    for obs, act in history:
        obs_short = obs[:300] + "..." if len(obs) > 300 else obs
        parts.append(f"Observation: {obs_short}\nAction: {act}")
    return "\n---\n".join(parts)


##############################
# Reflection Generation & Formatting
##############################

def generate_skill(client, client_type, model, env_name, task_desc, trajectory_text, won, trial_num):
    success_str = "successfully" if won else "unsuccessfully"
    tmpl = ALFWORLD_REFLECT_TEMPLATE if env_name == "alfworld" else WEBSHOP_REFLECT_TEMPLATE
    prompt = tmpl.format(task_description=task_desc, success=success_str,
                         current_trajectory=trajectory_text, trial_num=trial_num)
    response = call_llm(client, client_type, model, prompt, max_tokens=1024)
    json_str = ""
    cb = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if cb: json_str = cb.group(1)
    else:
        s, e = response.find('{'), response.rfind('}')
        if s != -1 and e != -1: json_str = response[s:e+1]
    if json_str:
        try: return json.loads(json_str)
        except json.JSONDecodeError: pass
    return {"subtasks": [], "task_success": won,
            "action_lesson": response[:200] if response else "No skill.",
            "navigation_lesson": None}


def format_skill_markdown(refl: dict, task_desc: str, trial: int) -> str:
    lines = [f"## Attempt {trial} Reflection: {task_desc[:80]}", ""]
    subtasks = refl.get("subtasks", [])
    if subtasks:
        lines += ["| Subtask | Description | Status |", "|---------|-------------|--------|"]
        for st in subtasks:
            lines.append(f"| {st.get('name','')} | {st.get('description','')[:60]} | {st.get('status','')} |")
        lines.append("")
    lines.append(f"**Result**: {'Success' if refl.get('task_success') else 'Failure'}")
    lines.append("")
    al = refl.get("action_lesson")
    if al: lines.append(f"**Action Lesson**: {al}")
    nl = refl.get("navigation_lesson")
    if nl and nl != "null": lines.append(f"**Navigation Lesson**: {nl}")
    return "\n".join(lines)


def format_skill_text(refl: dict, trial: int) -> str:
    parts = [f"[Attempt {trial}]"]
    for st in refl.get("subtasks", []):
        parts.append(f"  - {st.get('name','')}: {st.get('status','')}")
    parts.append(f"  Result: {'Success' if refl.get('task_success') else 'Failure'}")
    al = refl.get("action_lesson")
    if al: parts.append(f"  Lesson: {al}")
    nl = refl.get("navigation_lesson")
    if nl and nl != "null": parts.append(f"  Navigation: {nl}")
    return "\n".join(parts)


def build_memory_context(mode, text_refls, img_refls):
    if mode == "no_memory": return "", None
    if mode == "text_memory":
        if not text_refls: return "", None
        text = "Reflections from your previous attempts at THIS SAME TASK:\n" + "\n---\n".join(text_refls)
        text += "\nUse these lessons to avoid repeating past mistakes."
        return text, None
    if mode == "ocr_memory":
        if not img_refls: return "", None
        # Add a short text note telling the model to look at the images
        note = "The images above contain skills from your previous attempts at this same task. Study them carefully to avoid repeating past mistakes."
        return note, list(img_refls)
    return "", None


##############################
# Single Trial Runner (ALFWorld)
##############################

def run_alfworld_trial(env, client, client_type, model, max_steps,
                       task_desc, skills_str, images_list):
    """Run one trial of an ALFWorld task. Returns (won, trajectory, traj_text)."""
    obs, infos = env.reset()
    obs_text = obs[0]
    admissible = infos["admissible_commands"][0]
    history, trajectory, traj_parts = [], [], []

    for step in range(max_steps):
        fmt_adm = "\n ".join(f"'{s}'" for s in admissible if s != "help")
        if not history:
            prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
                skills=skills_str, current_observation=obs_text,
                admissible_actions=fmt_adm)
        else:
            prompt = ALFWORLD_TEMPLATE.format(
                task_description=task_desc, skills=skills_str,
                step_count=step,
                history_length=min(len(history), HISTORY_LENGTH),
                action_history=format_history(history[-HISTORY_LENGTH:]),
                current_step=step+1, current_observation=obs_text,
                admissible_actions=fmt_adm)

        t0 = time.time()
        response = call_llm(client, client_type, model, prompt, images=images_list)
        api_time = time.time() - t0

        parsed = parse_action(response)
        action, valid = match_alfworld_action(parsed, admissible)
        print(f"    Step {step+1}: '{action}' (valid={valid}, {api_time:.1f}s)")

        trajectory.append({"step": step+1, "obs": obs_text[:500], "action": action,
                           "valid": valid, "api_time": api_time})
        traj_parts.append(f"[Obs {step+1}: '{obs_text[:300]}', Act {step+1}: '{action}']")
        history.append((obs_text, action))

        obs_list, _, dones, infos_step = env.step([action])
        obs_text = obs_list[0]
        for k in infos_step:
            infos_step[k] = infos_step[k][0] if isinstance(infos_step[k], list) else infos_step[k]
        admissible = infos_step["admissible_commands"]
        won = bool(infos_step.get("won", False))
        if dones[0]:
            print(f"    >>> {'WON' if won else 'LOST'} at step {step+1}")
            break

    return won, trajectory, "\n".join(traj_parts)


##############################
# Single Trial Runner (WebShop)
##############################

def run_webshop_trial(env, client, client_type, model, max_steps,
                      session_idx, task_desc, obs_formatted,
                      skills_str, images_list):
    """Run one trial of a WebShop task. Returns (won, final_score, trajectory, traj_text)."""
    avail = env.get_available_actions()
    history, trajectory, traj_parts = [], [], []
    final_reward = 0.0

    for step in range(max_steps):
        actions_list = []
        if avail.get("has_search_bar"):
            actions_list.append("search[<your query>]")
        for txt in avail.get("clickables", []):
            actions_list.append(f"click[{txt}]")
        fmt_avail = "\n".join(f"'{s}'," for s in actions_list)

        if not history:
            prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                skills=skills_str, task_description=task_desc,
                current_observation=obs_formatted, available_actions=fmt_avail)
        else:
            prompt = WEBSHOP_TEMPLATE.format(
                skills=skills_str, task_description=task_desc,
                step_count=step,
                history_length=min(len(history), HISTORY_LENGTH),
                action_history=format_history(history[-HISTORY_LENGTH:]),
                current_step=step+1, current_observation=obs_formatted,
                available_actions=fmt_avail)
        if len(prompt) > 13000:
            prompt = WEBSHOP_TEMPLATE_NO_HIS.format(
                skills=skills_str, task_description=task_desc,
                current_observation=obs_formatted, available_actions=fmt_avail)

        t0 = time.time()
        response = call_llm(client, client_type, model, prompt, images=images_list)
        api_time = time.time() - t0

        action = parse_action(response)
        valid = bool(re.match(r"(search\[.*\]|click\[.*\])", action))
        if not valid:
            m = re.search(r"(search\[.*?\]|click\[.*?\])", response)
            if m: action = m.group(1); valid = True
            elif actions_list: action = actions_list[0]

        print(f"    Step {step+1}: '{action[:80]}' (valid={valid}, {api_time:.1f}s)")
        trajectory.append({"step": step+1, "obs": obs_formatted[:500],
                           "action": action, "valid": valid, "api_time": api_time})
        traj_parts.append(f"[Obs {step+1}: '{obs_formatted[:300]}', Act {step+1}: '{action}']")
        history.append((obs_formatted[:300], action))

        obs_raw, reward, done, info_step = env.step(action)
        info_step = dict(info_step or {})
        avail = env.get_available_actions()
        final_reward = info_step.get("task_score", reward)

        parts_new = obs_raw.split(" [SEP] ")
        try:
            idx_new = parts_new.index(task_desc)
            obs_formatted = " [SEP] ".join(parts_new[idx_new+1:])
        except (ValueError, IndexError):
            obs_formatted = obs_raw

        if done:
            won = (reward == 1.0) or (final_reward == 1.0)
            print(f"    >>> score={final_reward:.3f}, won={won}")
            break

    won = (final_reward == 1.0)
    return won, final_reward, trajectory, "\n".join(traj_parts)


##############################
# ALFWorld Multi-Trial Evaluation
##############################

def eval_alfworld(client, client_type, model, num_tasks, max_trials, max_steps,
                  memory_mode, output_dir):
    import yaml
    alf_pkg = os.path.join(SKILL1_ROOT, "agent_system/environments/env_package/alfworld")
    if alf_pkg not in sys.path:
        sys.path.insert(0, alf_pkg)
    from alfworld.agents.environment import get_environment

    config_path = os.path.join(alf_pkg, "configs/config_tw.yaml")
    with open(config_path) as f:
        alf_config = yaml.safe_load(f)

    base_env = get_environment(alf_config["env"]["type"])(alf_config, train_eval="eval_in_distribution")
    env = base_env.init_env(batch_size=1)

    os.makedirs(os.path.join(output_dir, "skills"), exist_ok=True)
    if memory_mode == "ocr_memory":
        os.makedirs(os.path.join(output_dir, "skill_images"), exist_ok=True)

    task_results = []
    first_trial_wins, final_wins = 0, 0
    task_type_stats = defaultdict(lambda: {"total": 0, "first_win": 0, "final_win": 0})

    for task_idx in range(num_tasks):
        env.seed(task_idx)
        obs, infos = env.reset()
        obs_text = obs[0]

        task_start = obs_text.find("Your task is to: ")
        task_desc = obs_text[task_start + len("Your task is to: "):].strip() if task_start != -1 else obs_text
        gamefile = infos.get("extra.gamefile", ["unknown"])[0]
        task_types = ["pick_and_place", "pick_two_obj_and_place", "look_at_obj_in_light",
                      "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
                      "pick_clean_then_place_in_recep"]
        task_type = next((tt for tt in task_types if tt in str(gamefile)), "unknown")

        print(f"\n{'='*60}")
        print(f"[{memory_mode}] Task {task_idx+1}/{num_tasks}: {task_desc[:100]}...")
        print(f"  Type: {task_type}")
        print(f"{'='*60}")

        text_refls, img_refls = [], []
        trial_records = []

        for trial in range(1, max_trials + 1):
            # Re-seed and reset to get the same task state
            env.seed(task_idx)
            env.reset()

            refls_str, imgs_list = build_memory_context(memory_mode, text_refls, img_refls)

            print(f"  --- Trial {trial}/{max_trials} (memory items: {len(text_refls)}) ---")
            won, trajectory, traj_text = run_alfworld_trial(
                env, client, client_type, model, max_steps,
                task_desc, refls_str, imgs_list)

            trial_records.append({
                "trial": trial, "won": won, "num_steps": len(trajectory),
                "num_skills": len(text_refls), "trajectory": trajectory,
            })

            if trial == 1:
                first_trial_wins += int(won)
                task_type_stats[task_type]["first_win"] += int(won)

            if won:
                print(f"  ==> Task SOLVED on trial {trial}!")
                final_wins += int(won)
                task_type_stats[task_type]["final_win"] += 1
                break

            # Generate skill for failed trial
            if trial < max_trials and memory_mode != "no_memory":
                print(f"  Generating skill for failed trial {trial}...")
                t0 = time.time()
                refl = generate_skill(client, client_type, model, "alfworld",
                                           task_desc, traj_text, won, trial)
                print(f"  Reflection done in {time.time()-t0:.1f}s")
                with open(os.path.join(output_dir, "skills",
                                       f"task{task_idx+1:03d}_trial{trial}.json"), "w") as f:
                    json.dump(refl, f, indent=2, ensure_ascii=False)
                text_refls.append(format_skill_text(refl, trial))
                if memory_mode == "ocr_memory":
                    md = format_skill_markdown(refl, task_desc, trial)
                    img_bytes = try_render_image(md)
                    img_refls.append(img_bytes)
                    with open(os.path.join(output_dir, "skill_images",
                                           f"task{task_idx+1:03d}_trial{trial}.png"), "wb") as f:
                        f.write(img_bytes)
        else:
            # All trials exhausted without success
            final_wins += 0

        task_type_stats[task_type]["total"] += 1
        task_results.append({
            "task_idx": task_idx + 1, "task": task_desc, "task_type": task_type,
            "gamefile": str(gamefile), "trials": trial_records,
            "first_trial_won": trial_records[0]["won"],
            "finally_won": any(t["won"] for t in trial_records),
            "solved_on_trial": next((t["trial"] for t in trial_records if t["won"]), None),
        })
        with open(os.path.join(output_dir, "task_results.jsonl"), "a") as f:
            f.write(json.dumps(task_results[-1], ensure_ascii=False) + "\n")

    summary = {
        "env": "alfworld", "model": model, "memory_mode": memory_mode,
        "num_tasks": num_tasks, "max_trials": max_trials, "max_steps": max_steps,
        "first_trial_success": first_trial_wins, "final_success": final_wins,
        "first_trial_rate": first_trial_wins / num_tasks if num_tasks else 0,
        "final_rate": final_wins / num_tasks if num_tasks else 0,
        "improvement": (final_wins - first_trial_wins) / num_tasks if num_tasks else 0,
        "task_type_breakdown": {k: {**v, "first_rate": v["first_win"]/v["total"] if v["total"] else 0,
                                     "final_rate": v["final_win"]/v["total"] if v["total"] else 0}
                                for k, v in task_type_stats.items()},
    }
    return task_results, summary


##############################
# WebShop Multi-Trial Evaluation
##############################

def eval_webshop(client, client_type, model, num_tasks, max_trials, max_steps,
                 memory_mode, output_dir):
    import gym
    webshop_pkg = os.path.join(SKILL1_ROOT, "agent_system/environments/env_package/webshop/webshop")
    if webshop_pkg not in sys.path:
        sys.path.insert(0, webshop_pkg)
    from web_agent_site.envs import WebAgentTextEnv  # noqa

    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None)
    random.seed(42)
    selected = random.sample(list(range(500)), min(num_tasks, 500))

    os.makedirs(os.path.join(output_dir, "skills"), exist_ok=True)
    if memory_mode == "ocr_memory":
        os.makedirs(os.path.join(output_dir, "skill_images"), exist_ok=True)

    task_results = []
    first_trial_wins, final_wins = 0, 0
    first_total_score, final_total_score = 0.0, 0.0

    for task_idx, session_idx in enumerate(selected):
        obs, info = env.reset(session=session_idx)
        parts = obs.split(" [SEP] ")
        task_desc = parts[2] if len(parts) > 2 and parts[1] == "Instruction:" else obs
        try:
            idx = parts.index(task_desc)
            obs_formatted = " [SEP] ".join(parts[idx+1:])
        except (ValueError, IndexError):
            obs_formatted = obs

        print(f"\n{'='*60}")
        print(f"[{memory_mode}] Task {task_idx+1}/{num_tasks} Session {session_idx}: {task_desc[:100]}...")
        print(f"{'='*60}")

        text_refls, img_refls = [], []
        trial_records = []
        best_score = 0.0

        for trial in range(1, max_trials + 1):
            # Reset to same session
            obs, info = env.reset(session=session_idx)
            parts = obs.split(" [SEP] ")
            try:
                idx = parts.index(task_desc)
                obs_fmt = " [SEP] ".join(parts[idx+1:])
            except (ValueError, IndexError):
                obs_fmt = obs

            refls_str, imgs_list = build_memory_context(memory_mode, text_refls, img_refls)

            print(f"  --- Trial {trial}/{max_trials} (memory items: {len(text_refls)}) ---")
            won, score, trajectory, traj_text = run_webshop_trial(
                env, client, client_type, model, max_steps,
                session_idx, task_desc, obs_fmt, refls_str, imgs_list)

            trial_records.append({
                "trial": trial, "won": won, "score": score,
                "num_steps": len(trajectory), "num_skills": len(text_refls),
                "trajectory": trajectory,
            })

            if trial == 1:
                first_trial_wins += int(won)
                first_total_score += score

            best_score = max(best_score, score)

            if won:
                print(f"  ==> Task SOLVED on trial {trial}!")
                final_wins += 1
                break

            if trial < max_trials and memory_mode != "no_memory":
                print(f"  Generating skill for trial {trial}...")
                t0 = time.time()
                refl = generate_skill(client, client_type, model, "webshop",
                                           task_desc, traj_text, won, trial)
                print(f"  Reflection done in {time.time()-t0:.1f}s")
                with open(os.path.join(output_dir, "skills",
                                       f"task{task_idx+1:03d}_trial{trial}.json"), "w") as f:
                    json.dump(refl, f, indent=2, ensure_ascii=False)
                text_refls.append(format_skill_text(refl, trial))
                if memory_mode == "ocr_memory":
                    md = format_skill_markdown(refl, task_desc, trial)
                    img_bytes = try_render_image(md)
                    img_refls.append(img_bytes)
                    with open(os.path.join(output_dir, "skill_images",
                                           f"task{task_idx+1:03d}_trial{trial}.png"), "wb") as f:
                        f.write(img_bytes)

        final_total_score += best_score
        if not any(t["won"] for t in trial_records):
            pass  # final_wins already not incremented

        task_results.append({
            "task_idx": task_idx+1, "session_idx": session_idx, "task": task_desc,
            "trials": trial_records, "first_trial_won": trial_records[0]["won"],
            "first_trial_score": trial_records[0]["score"],
            "best_score": best_score, "finally_won": any(t["won"] for t in trial_records),
            "solved_on_trial": next((t["trial"] for t in trial_records if t["won"]), None),
        })
        with open(os.path.join(output_dir, "task_results.jsonl"), "a") as f:
            f.write(json.dumps(task_results[-1], ensure_ascii=False) + "\n")

    env.close()
    summary = {
        "env": "webshop", "model": model, "memory_mode": memory_mode,
        "num_tasks": num_tasks, "max_trials": max_trials, "max_steps": max_steps,
        "first_trial_success": first_trial_wins, "final_success": final_wins,
        "first_trial_rate": first_trial_wins / num_tasks if num_tasks else 0,
        "final_rate": final_wins / num_tasks if num_tasks else 0,
        "first_avg_score": first_total_score / num_tasks if num_tasks else 0,
        "final_avg_score": final_total_score / num_tasks if num_tasks else 0,
        "improvement": (final_wins - first_trial_wins) / num_tasks if num_tasks else 0,
    }
    return task_results, summary


##############################
# Main
##############################

def main():
    parser = argparse.ArgumentParser(description="Memory Ablation (Multi-Trial)")
    parser.add_argument("--env", required=True, choices=["alfworld", "webshop"])
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--max_trials", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--models", default=None,
                        help="Comma-separated models (default: gpt-4o-2024-11-20)")
    parser.add_argument("--modes", default=None,
                        help="Comma-separated memory modes (default: all three)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    models = args.models.split(",") if args.models else [os.environ.get("EVAL_MODEL", "gpt-4o-2024-11-20")]
    modes = args.modes.split(",") if args.modes else MEMORY_MODES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = args.output_dir or os.path.join(
        SKILL1_ROOT, "eval_results", f"memory_retry_{args.env}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    config = vars(args) | {"models": models, "modes": modes, "timestamp": timestamp}
    with open(os.path.join(base_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 70)
    print(" Memory Ablation Experiment (Multi-Trial Retry)")
    print("=" * 70)
    print(f"  Environment:  {args.env}")
    print(f"  Models:       {models}")
    print(f"  Modes:        {modes}")
    print(f"  Tasks:        {args.num_tasks}")
    print(f"  Max trials:   {args.max_trials}")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Output:       {base_dir}")
    print("=" * 70)

    all_summaries = {}

    for model in models:
        model_safe = model.replace(".", "_").replace("/", "_")
        client, client_type = make_client(model)

        for mode in modes:
            mode_dir = os.path.join(base_dir, model_safe, mode)
            os.makedirs(mode_dir, exist_ok=True)

            print(f"\n{'#'*70}")
            print(f" model={model}  mode={mode}")
            print(f"{'#'*70}")

            t0 = time.time()
            if args.env == "alfworld":
                _, summary = eval_alfworld(
                    client, client_type, model, args.num_tasks, args.max_trials,
                    args.max_steps, mode, mode_dir)
            else:
                _, summary = eval_webshop(
                    client, client_type, model, args.num_tasks, args.max_trials,
                    args.max_steps, mode, mode_dir)

            summary["total_time_s"] = time.time() - t0
            with open(os.path.join(mode_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            all_summaries[f"{model_safe}/{mode}"] = summary

            fr = summary.get('first_trial_rate', 0)
            fnr = summary.get('final_rate', 0)
            imp = summary.get('improvement', 0)
            print(f"\n  >>> {mode}: trial1={fr:.0%}  final={fnr:.0%}  improvement={imp:+.0%}  time={summary['total_time_s']:.0f}s")

    # Comparison table
    print(f"\n{'='*90}")
    print(f" MULTI-TRIAL MEMORY ABLATION: {args.env} (max_trials={args.max_trials})")
    print(f"{'='*90}")
    print(f" {'Model':<25} {'Mode':<15} {'Trial-1':>8} {'Final':>8} {'Improv':>8} {'Time':>8}")
    print(f" {'-'*25} {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key, s in all_summaries.items():
        m = s["model"][:23]
        mode = s["memory_mode"]
        fr = f"{s.get('first_trial_rate',0):.0%}"
        fnr = f"{s.get('final_rate',0):.0%}"
        imp = f"{s.get('improvement',0):+.0%}"
        t = f"{s.get('total_time_s',0):.0f}s"
        print(f" {m:<25} {mode:<15} {fr:>8} {fnr:>8} {imp:>8} {t:>8}")

    print(f"{'='*90}")

    with open(os.path.join(base_dir, "comparison.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results saved to: {base_dir}")


if __name__ == "__main__":
    main()
