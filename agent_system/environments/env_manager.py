# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory, SkillLibrary, WebshopSimpleMemory
from omegaconf import OmegaConf
import re
import copy

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv with full Skill1 support:
    SkillLibrary, group-split, distill/step_distill, rerank, query gen.
    """
    def __init__(self, envs, projection_f, config, retrieve_type):
        self.memory = SearchMemory()
        self.group_n = config.env.rollout.n

        mem_config = config.env.get('skill_library', {})
        filepath = mem_config.get('filepath', "search_skill_library.json")
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        relevance_weight = mem_config.get('relevance_weight', 0.7)
        alpha = mem_config.get('alpha', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)

        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0)
        self.current_progress_ratio = 0.0
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        assert not (self.full_group_memory and self.group_outperformance), \
            "group_outperformance requires split group (full_group_memory must be False)"
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)
        self.distill_reward_type = mem_config.get('distill_reward_type', 'self_assess')
        self.u_hat_aggregation = mem_config.get('u_hat_aggregation', 'max')

        self.enable_query_generation = mem_config.get('enable_query_generation', False)
        self.enable_description_head = mem_config.get('enable_description_head', False)
        self.enable_rerank = mem_config.get('enable_rerank', False)
        self.selection_trainable = mem_config.get('selection_trainable', True)

        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        self.single_distill_per_group = mem_config.get('single_distill_per_group', False)
        self.ema_gamma = 0.9
        self.rerank_train_top1 = mem_config.get('rerank_train_top1', False)
        self.retriever_type = mem_config.get('retriever_type', 'dense')

        print(f"[Search] memory retrieve_type: {retrieve_type}")
        print(f"[Search] memory retrieve_mode: {self.retrieve_mode}")
        print(f"[Search] top_k_retrieved_memory: {self.top_k}")
        print(f"[Search] Memory Start Cutoff: {self.memory_start_cutoff}")
        print(f"[Search] Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"[Search] Potential Based On Binary Success: {self.potential_based_on_binary_success}")
        print(f"[Search] Re-rank: {self.enable_rerank}")
        print(f"[Search] Rerank Train Top-1: {self.rerank_train_top1}")
        print(f"[Search] Retriever Type: {self.retriever_type}")

        self._is_eval = False

        self.skill_library = SkillLibrary(
            filepath=filepath,
            relevance_weight=relevance_weight,
            alpha=alpha,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale,
            use_description_head=self.enable_description_head,
            max_size=mem_config.get('max_size', 5000),
            retriever_type=self.retriever_type,
        )
        self.task_trajectory_history = {}
        self.task_potential_history = {}
        self.batch_previous_potentials = []
        self.current_skills = []
        self.retrieved_raw_skills = []
        self.current_retrieval_types = []
        self.batch_retrieved_types = []
        self.last_trajectories = []
        super().__init__(envs, projection_f, config)

    # --- Training progress ---
    def update_training_progress(self, current_step: int, total_steps: int):
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            self.skill_library._current_training_step = current_step

    # ==================== reset ====================
    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        if kwargs is None:
            kwargs = {}
        print("****** search environment resetting ******")
        is_eval = not kwargs.get('is_train', True)
        self._is_eval = is_eval

        per_sample = kwargs.get('per_sample', kwargs)
        obs, infos = self.envs.reset(kwargs=per_sample)
        self.tasks = obs
        self.memory.reset(batch_size=len(obs))
        self.batch_size = len(obs)
        assert self.batch_size % self.group_n == 0, \
            f"Batch size {self.batch_size} must be divisible by group size {self.group_n}"
        self.num_unique_tasks = self.batch_size // self.group_n

        # --- Retrieval with Group-based Split ---
        self.current_skills = []
        self.retrieved_raw_skills = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = []
        self.batch_retrieved_types = []
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)

        if in_warmup_period:
            print(f"[Search] Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")

        for i, task in enumerate(self.tasks):
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_skills = ""
            raw_list_of_dicts = []
            current_types_list = []
            should_retrieve = False
            retrieval_type_str = "control"
            if self.enable_memory:
                if in_warmup_period:
                    should_retrieve = False
                elif is_eval:
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False

            if should_retrieve:
                k = self.top_k
                raw_list_of_dicts = self.skill_library.retrieve(
                    current_scenario_description=task,
                    top_k=k,
                    filter_type=self.retrieve_mode
                )
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        current_types_list.append(r_type)
                        formatted_lines.append(r_text)
                    formatted_skills = "Relevant skills from the skill library:\n" + "\n".join(formatted_lines)
                    formatted_skills += "\nWarning: These lessons may be outdated. Use them only if they align with your current situation."

            self.current_skills.append(formatted_skills)
            self.retrieved_raw_skills.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            infos[i]['distill_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str

        assert len(self.current_skills) == len(self.tasks)

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        return observations, infos

    # ==================== step ====================
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    # ==================== build_text_obs ====================
    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    skills=self.current_skills[i],
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    skills=self.current_skills[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    # ==================== BiGen: Query Generation ====================
    def build_query_generation_obs(self) -> Dict[str, Any]:
        query_obs_texts = []
        for i, task in enumerate(self.tasks):
            obs_text = SEARCH_QUERY_GENERATION_TEMPLATE.format(
                task_description=task,
            )
            query_obs_texts.append(obs_text)
        return {'text': query_obs_texts, 'image': None, 'anchor': query_obs_texts}

    def apply_generated_queries(self, query_texts: List[str]):
        for i, query_text in enumerate(query_texts):
            if self.current_retrieval_types[i] not in ("experiment", "eval_retrieval"):
                continue
            match = re.search(r'<query>(.*?)</query>', query_text, re.DOTALL)
            parsed = match.group(1).strip() if match else query_text.strip()
            if not parsed:
                continue
            raw_list_of_dicts = self.skill_library.retrieve(
                current_scenario_description=parsed,
                top_k=self.top_k,
                filter_type=self.retrieve_mode
            )
            if raw_list_of_dicts:
                formatted_lines = [item.get('text', '') for item in raw_list_of_dicts]
                self.current_skills[i] = (
                    "Relevant skills from the skill library:\n" + "\n".join(formatted_lines)
                    + "\nWarning: These lessons may be outdated. Use them only if they align with your current situation."
                )
                self.retrieved_raw_skills[i] = raw_list_of_dicts

    def rebuild_initial_obs(self):
        full_text_obs = self.build_text_obs(self.tasks, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': self.tasks}

    # ==================== Re-rank ====================
    def build_rerank_obs(self) -> Dict[str, Any]:
        rerank_obs_texts = []
        self.rerank_candidates = []
        for i, task in enumerate(self.tasks):
            raw_items = self.retrieved_raw_skills[i]
            candidates_with_scores = self._get_candidate_scores(raw_items)
            if len(candidates_with_scores) >= 2:
                candidate_lines = []
                for idx, cand in enumerate(candidates_with_scores):
                    candidate_lines.append(f"[Experience {idx + 1}]: {cand['text']}")
                candidate_str = "\n\n".join(candidate_lines)
                obs_text = SEARCH_RERANK_TEMPLATE.format(
                    task_description=task,
                    n_candidates=len(candidates_with_scores),
                    candidate_experiences=candidate_str,
                )
            else:
                obs_text = SEARCH_RERANK_DUMMY_TEMPLATE.format(task_description=task)
            rerank_obs_texts.append(obs_text)
            self.rerank_candidates.append(candidates_with_scores)
        return {'text': rerank_obs_texts, 'image': None, 'anchor': rerank_obs_texts}

    def _get_candidate_scores(self, raw_items: List[Dict]) -> List[Dict]:
        result = []
        for item in raw_items:
            r_text = item.get('text', '')
            utility = 0.5
            for mem_entry in self.skill_library.data:
                if mem_entry['strategy'] == r_text:
                    utility = mem_entry.get('utility_score', 0.5)
                    break
            result.append({'text': r_text, 'type': item.get('type', 'unknown'), 'utility': utility})
        return result

    def apply_rerank_results(self, text_actions: List[str]):
        self._rerank_parse_successes = 0
        self._rerank_total = 0
        self._rerank_predicted_orders = []
        for i, action_text in enumerate(text_actions):
            candidates = self.rerank_candidates[i]
            n = len(candidates)
            self._rerank_total += 1
            if n < 2:
                self._rerank_predicted_orders.append(None)
                continue
            match = re.search(r'<rank>(.*?)</rank>', action_text, re.DOTALL)
            if not match:
                self._rerank_predicted_orders.append(None)
                continue
            rank_str = match.group(1).strip()
            if rank_str.lower() == 'none':
                self._rerank_predicted_orders.append(None)
                continue
            parsed_ids = []
            for token in rank_str.split(','):
                token = token.strip()
                if token.isdigit():
                    idx = int(token) - 1
                    if 0 <= idx < n and idx not in parsed_ids:
                        parsed_ids.append(idx)
            if len(parsed_ids) == 0:
                self._rerank_predicted_orders.append(None)
                continue
            self._rerank_parse_successes += 1
            self._rerank_predicted_orders.append(parsed_ids)
            remaining = [j for j in range(n) if j not in parsed_ids]
            full_order = parsed_ids + remaining
            reordered_texts = [candidates[j]['text'] for j in full_order]
            if self.rerank_train_top1 and not self._is_eval:
                reordered_texts = reordered_texts[:1]
            formatted = "Relevant skills from the skill library:\n" + "\n".join(reordered_texts)
            formatted += "\nWarning: These lessons may be outdated. Use them only if they align with your current situation."
            self.current_skills[i] = formatted

    def compute_rerank_rewards(self) -> np.ndarray:
        rewards = np.zeros(len(self.tasks), dtype=np.float32)
        for i in range(len(self.tasks)):
            candidates = self.rerank_candidates[i]
            predicted_order = self._rerank_predicted_orders[i]
            if predicted_order is None or len(candidates) < 2:
                rewards[i] = 0.0
                continue
            utilities = np.array([c['utility'] for c in candidates])
            ideal_order = np.argsort(-utilities)
            ideal_gains = utilities[ideal_order]
            remaining = [j for j in range(len(candidates)) if j not in predicted_order]
            full_predicted = predicted_order + remaining
            predicted_gains = utilities[full_predicted]
            def _dcg(gains):
                positions = np.arange(len(gains), dtype=np.float64) + 2.0
                return np.sum(gains / np.log2(positions))
            ideal_dcg = _dcg(ideal_gains)
            if ideal_dcg < 1e-8:
                rewards[i] = 0.0
                continue
            predicted_dcg = _dcg(predicted_gains)
            ndcg = predicted_dcg / ideal_dcg
            rewards[i] = float(ndcg)
        self._rerank_rewards = rewards
        return rewards

    # ==================== distill ====================
    def distill(self, infos: List[Dict]):
        distill_obs_text = self.build_distill_text_obs(infos)
        observations = {
            'text': distill_obs_text,
            'image': None,
            'anchor': distill_obs_text
        }
        for info in infos:
            info['is_action_valid'] = to_numpy(True)

        self.query_contrastive_rewards = np.zeros(len(self.tasks), dtype=np.float32)
        self._bigen_ctrl_success_rates = []
        self._bigen_exp_success_rates = []
        self._bigen_selection_hits = 0
        self._bigen_selection_total = 0

        batch_size = len(self.tasks)
        num_groups = batch_size // self.group_n
        group_split_index = 0 if self.full_group_memory else self.group_n // 2

        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index

            control_wins = sum(1 for i in range(start_idx, mid_idx) if infos[i].get("won", False))
            experiment_wins = sum(1 for i in range(mid_idx, end_idx) if infos[i].get("won", False))
            group_outperformed = experiment_wins > control_wins

            n_ctrl = max(group_split_index, 1)
            n_exp = max(self.group_n - group_split_index, 1)
            self._bigen_ctrl_success_rates.append(control_wins / n_ctrl)
            self._bigen_exp_success_rates.append(experiment_wins / n_exp)

            if self.enable_query_generation:
                contrastive = (experiment_wins / n_exp) - (control_wins / n_ctrl)
                for i in range(mid_idx, end_idx):
                    self.query_contrastive_rewards[i] = contrastive

            for i in range(mid_idx, end_idx):
                if self.retrieved_raw_skills[i]:
                    self._bigen_selection_total += 1
                    if infos[i].get("won", False) and group_outperformed:
                        self._bigen_selection_hits += 1

            for i in range(mid_idx, end_idx):
                task_desc = self.tasks[i]
                raw_skill_items = self.retrieved_raw_skills[i]
                is_success = infos[i].get("won", False)
                if self.group_outperformance:
                    utility_score = 1.0 if (is_success and group_outperformed) else 0.0
                else:
                    utility_score = 1.0 if is_success else 0.0
                if raw_skill_items:
                    for item in raw_skill_items:
                        strategy_text = item.get('text', '')
                        if strategy_text:
                            self.skill_library.update_utility(
                                scenario_description=task_desc,
                                strategy_text=strategy_text,
                                score=utility_score
                            )

        self._bigen_memory_buffer_size = len(self.skill_library.data)
        return observations, infos

    # ==================== build_distill_text_obs ====================
    def build_distill_text_obs(self, infos: List[Dict]) -> List[str]:
        postprocess_text_obs = []
        memory_contexts, valid_lens = self.memory.fetch(
            50,
            obs_key="information",
            action_key="search"
        )
        for i in range(len(infos)):
            task = self.tasks[i]
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])

        self.last_trajectories = memory_contexts
        for i in range(len(infos)):
            task = self.tasks[i]
            is_won = infos[i].get("won", False)
            reference_traj_str = "No reference history available yet."
            if is_won:
                success_str = "successfully"
                success_detail = "correct"
                failed_hist = self.task_trajectory_history[task]["failed"]
                if failed_hist:
                    reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                else:
                    reference_traj_str = "No failed attempts available for comparison."
            else:
                success_str = "unsuccessfully"
                success_detail = "incorrect"
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."

            distill_tmpl = SEARCH_DISTILL_TEMPLATE_WITH_DESC_HEAD if self.enable_description_head else SEARCH_DISTILL_TEMPLATE
            obs = distill_tmpl.format(
                task_description=self.tasks[i],
                success=success_str,
                success_detail=success_detail,
                reference_trajectory=reference_traj_str,
                current_trajectory=memory_contexts[i],
            )
            postprocess_text_obs.append(obs)

        if postprocess_text_obs:
            print("[Search] processed_distill_text [0]: ", postprocess_text_obs[0][:500])
        return postprocess_text_obs

    # ==================== step_distill ====================
    def step_distill(self, text_actions: List[str], infos: List[Dict]):
        import json

        print("[Search] text_actions for distillation:", text_actions[:2])

        distill_rewards = []
        current_scores = np.zeros(self.batch_size)
        raw_improvements = np.zeros(self.batch_size)
        is_won_array = np.zeros(self.batch_size, dtype=bool)

        self._bigen_desc_head_total = 0
        self._bigen_desc_head_parsed = 0
        self._bigen_desc_head_saved = 0
        self._distill_correct_count = 0
        self._distill_total_count = 0
        self._distill_ops = []
        self._distill_u_hat_values = []
        self._distill_r_values = []

        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        for i, strategy_text in enumerate(text_actions):
            task_desc = self.tasks[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""
            prev_phi = self.batch_previous_potentials[i]
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0

            distill_op = "none"
            try:
                json_str = ""
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', strategy_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = strategy_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]
                if not json_str:
                    raise ValueError("No JSON found")

                distill_data = json.loads(json_str)

                subtasks = distill_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for t in subtasks
                    if isinstance(t, dict) and t.get('status', '').strip().lower() == 'completed'
                )
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0

                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                predicted_success = distill_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']

                # --- Distillation reward (mutually exclusive modes) ---
                if self.distill_reward_type == 'first_order_diff':
                    retrieved_items = self.retrieved_raw_skills[i] if i < len(self.retrieved_raw_skills) else []
                    u_scores = [item.get('utility_score', 0.5) for item in retrieved_items] if retrieved_items else []
                    u_hat = (max(u_scores) if self.u_hat_aggregation == 'max' else sum(u_scores) / len(u_scores)) if u_scores else 0.0
                    r_tau = 1.0 if actual_success else 0.0
                    current_reward = r_tau - u_hat
                    self._distill_u_hat_values.append(u_hat)
                    self._distill_r_values.append(current_reward)
                else:
                    current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                distill_rewards.append(current_reward)
                self._distill_total_count += 1
                if predicted_success == actual_success:
                    self._distill_correct_count += 1

                # Memory saving
                should_save = (actual_success and json_str) if self.distill_reward_type == 'first_order_diff' else (predicted_success == actual_success and json_str)
                if should_save:
                    search_lesson = distill_data.get('search_lesson')
                    reasoning_lesson = distill_data.get('reasoning_lesson')
                    description_head = (distill_data.get('description_head') or '') if self.enable_description_head else ''
                    if self.enable_description_head:
                        self._bigen_desc_head_total += 1
                        if description_head and len(str(description_head).strip()) > 5:
                            self._bigen_desc_head_parsed += 1
                    lessons_to_save = []
                    if search_lesson and len(str(search_lesson)) > 5:
                        lessons_to_save.append(f"Search Insight: {search_lesson}")
                    if reasoning_lesson and len(str(reasoning_lesson)) > 5:
                        lessons_to_save.append(f"Reasoning Insight: {reasoning_lesson}")
                    if lessons_to_save:
                        distill_op = "add"
                        final_lesson = " | ".join(lessons_to_save)
                        self.skill_library.admit(
                            scenario_description=task_desc,
                            strategy_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio,
                            description_head=str(description_head) if description_head else ''
                        )
                        if self.enable_description_head and description_head and len(str(description_head).strip()) > 5:
                            self._bigen_desc_head_saved += 1
                self._distill_ops.append(distill_op)

            except Exception as e:
                print(f"[Search] Error task {i}: {e}")
                distill_rewards.append(0.0)
                self._distill_ops.append("none")
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0

            current_scores[i] = current_phi
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement

        # Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            task_desc = self.tasks[start_idx]
            group_improvements = raw_improvements[start_idx:end_idx]

            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)
            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate

        print("[Search] raw_improvements: ", raw_improvements)
        print("[Search] final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        return None, to_numpy(distill_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    # ==================== get_bigen_metrics ====================
    def get_bigen_metrics(self, prefix: str = "bigen") -> Dict[str, float]:
        p = prefix
        metrics = {}

        mem_data = self.skill_library.data
        if mem_data:
            util_scores = [float(entry.get('utility_score', 0.5)) for entry in mem_data]
            n = len(util_scores)
            for bin_lo in range(10):
                lo = bin_lo / 10.0
                hi = lo + 0.1
                if bin_lo == 9:
                    cnt = sum(1 for u in util_scores if lo <= u <= hi)
                else:
                    cnt = sum(1 for u in util_scores if lo <= u < hi)
                metrics[f'{p}/utility_dist/{bin_lo*10:02d}'] = float(cnt) / n

        s = 'skill_statics'
        buf_size = getattr(self, '_bigen_memory_buffer_size', len(mem_data) if mem_data else 0)
        metrics[f'{s}/memory_buffer_size'] = float(buf_size)
        if mem_data:
            util_scores = [float(entry.get('utility_score', 0.5)) for entry in mem_data]
            counts = [float(entry.get('count', 1)) for entry in mem_data]
            n = len(util_scores)
            metrics[f'{s}/library_quality'] = sum(util_scores) / n
            metrics[f'{s}/memory_avg_count'] = sum(counts) / n
            metrics[f'{s}/memory_low_utility_frac'] = sum(1 for u in util_scores if u < 0.3) / n
            total_chars = sum(len(entry.get('strategy', '')) + len(entry.get('trajectory', '')) for entry in mem_data)
            metrics[f'{s}/memory_total_chars'] = float(total_chars)
        metrics[f'{s}/memory_evicted_count'] = float(getattr(self.skill_library, '_last_evicted_count', 0))

        ctrl_rates = getattr(self, '_bigen_ctrl_success_rates', [])
        exp_rates = getattr(self, '_bigen_exp_success_rates', [])
        if ctrl_rates:
            metrics[f'{s}/ctrl_win_rate/mean'] = float(np.mean(ctrl_rates))
            metrics[f'{s}/exp_win_rate/mean'] = float(np.mean(exp_rates))
            metrics[f'{s}/exp_minus_ctrl/mean'] = float(np.mean(exp_rates)) - float(np.mean(ctrl_rates))

        sel_total = getattr(self, '_bigen_selection_total', 0)
        sel_hits = getattr(self, '_bigen_selection_hits', 0)
        if sel_total > 0:
            metrics[f'{s}/selection_precision'] = float(sel_hits) / float(sel_total)
        ref_total = getattr(self, '_distill_total_count', 0)
        ref_correct = getattr(self, '_distill_correct_count', 0)
        if ref_total > 0:
            metrics[f'{s}/self_assess_accuracy'] = float(ref_correct) / float(ref_total)

        if self.enable_query_generation:
            q_rewards = getattr(self, 'query_contrastive_rewards', None)
            if q_rewards is not None and len(q_rewards) > 0:
                metrics[f'{p}/query_contrastive_reward/mean'] = float(np.mean(q_rewards))
                exp_mask = q_rewards != 0
                if exp_mask.any():
                    metrics[f'{p}/query_contrastive_reward/exp_mean'] = float(np.mean(q_rewards[exp_mask]))

        if self.enable_description_head:
            total = getattr(self, '_bigen_desc_head_total', 0)
            parsed = getattr(self, '_bigen_desc_head_parsed', 0)
            saved = getattr(self, '_bigen_desc_head_saved', 0)
            metrics[f'{p}/desc_head/total_attempts'] = float(total)
            metrics[f'{p}/desc_head/parsed_count'] = float(parsed)
            if total > 0:
                metrics[f'{p}/desc_head/parse_rate'] = float(parsed / total)

        if self.enable_rerank:
            rr_rewards = getattr(self, '_rerank_rewards', None)
            if rr_rewards is not None and len(rr_rewards) > 0:
                metrics[f'{p}/rerank_ndcg/mean'] = float(np.mean(rr_rewards))
                nonzero_mask = rr_rewards > 0
                if nonzero_mask.any():
                    metrics[f'{p}/rerank_ndcg/active_mean'] = float(np.mean(rr_rewards[nonzero_mask]))
            rr_total = getattr(self, '_rerank_total', 0)
            rr_success = getattr(self, '_rerank_parse_successes', 0)
            metrics[f'{p}/rerank_parse/total'] = float(rr_total)
            if rr_total > 0:
                metrics[f'{p}/rerank_parse/parse_rate'] = float(rr_success / rr_total)

        # Selection evolution metrics
        retrieved_refs = getattr(self, 'retrieved_raw_skills', [])
        if retrieved_refs:
            from collections import Counter
            import math as _math
            all_skill_ids = []
            per_task_skill_ids = {}
            for idx, ref_list in enumerate(retrieved_refs):
                task_group_idx = idx // self.group_n if self.group_n > 0 else idx
                if task_group_idx not in per_task_skill_ids:
                    per_task_skill_ids[task_group_idx] = []
                for item in ref_list:
                    sid = item.get('skill_id', '') or item.get('text', '')[:50]
                    if sid:
                        all_skill_ids.append(sid)
                        per_task_skill_ids[task_group_idx].append(sid)
            if all_skill_ids:
                counter = Counter(all_skill_ids)
                total_count = len(all_skill_ids)
                probs = [c / total_count for c in counter.values()]
                entropy = -sum(p_val * _math.log2(p_val) for p_val in probs if p_val > 0)
                metrics[f'{s}/retrieval_entropy'] = float(entropy)
                lib_size = max(len(mem_data) if mem_data else 1, 1)
                max_entropy = _math.log2(lib_size) if lib_size > 1 else 1.0
                metrics[f'{s}/retrieval_entropy_normalized'] = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            consistency_values = []
            for task_idx, sids in per_task_skill_ids.items():
                if sids:
                    counter = Counter(sids)
                    mode_count = counter.most_common(1)[0][1]
                    consistency_values.append(mode_count / len(sids))
            if consistency_values:
                metrics[f'{s}/retrieval_consistency'] = float(np.mean(consistency_values))
            all_sims = []
            for ref_list in retrieved_refs:
                for item in ref_list:
                    sim = item.get('similarity_score')
                    if sim is not None:
                        all_sims.append(float(sim))
            if all_sims:
                metrics[f'{s}/retrieval_similarity/mean'] = float(np.mean(all_sims))
                metrics[f'{s}/retrieval_similarity/min'] = float(np.min(all_sims))
                metrics[f'{s}/retrieval_similarity/max'] = float(np.max(all_sims))

        # Distillation ops distribution
        distill_ops = getattr(self, '_distill_ops', [])
        if distill_ops:
            from collections import Counter
            ops_counter = Counter(distill_ops)
            total_ops = len(distill_ops)
            metrics[f'{s}/distill_ops/add_rate'] = float(ops_counter.get('add', 0)) / total_ops
            metrics[f'{s}/distill_ops/none_rate'] = float(ops_counter.get('none', 0)) / total_ops
            metrics[f'{s}/distill_ops/total'] = float(total_ops)

        # Skill lifetime distribution
        if mem_data:
            current_step = getattr(self.skill_library, '_current_training_step', 0)
            lifetimes = [current_step - entry.get('created_at_step', 0) for entry in mem_data]
            if lifetimes:
                lifetimes_arr = np.array(lifetimes, dtype=np.float64)
                metrics[f'{s}/skill_lifetime/mean'] = float(np.mean(lifetimes_arr))
                metrics[f'{s}/skill_lifetime/q1'] = float(np.percentile(lifetimes_arr, 25))
                metrics[f'{s}/skill_lifetime/q3'] = float(np.percentile(lifetimes_arr, 75))

        # First-order difference reward metrics (skill1_v2)
        u_hat_vals = getattr(self, '_distill_u_hat_values', [])
        r_vals = getattr(self, '_distill_r_values', [])
        if u_hat_vals:
            metrics[f'{s}/distill_u_hat/mean'] = float(np.mean(u_hat_vals))
            metrics[f'{s}/distill_u_hat/max'] = float(np.max(u_hat_vals))
            metrics[f'{s}/distill_u_hat/min'] = float(np.min(u_hat_vals))
            metrics[f'{s}/distill_r_diff/mean'] = float(np.mean(r_vals))
            metrics[f'{s}/distill_r_diff/positive_rate'] = float(sum(1 for r in r_vals if r > 0) / len(r_vals))
            metrics[f'{s}/distill_r_diff/negative_rate'] = float(sum(1 for r in r_vals if r < 0) / len(r_vals))

        return metrics

    # ==================== success_evaluator ====================
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        distill_rewards = kwargs.get('distill_rewards', None)
        batch_size = len(total_batch_list)
        success = defaultdict(list)

        for bs in range(batch_size):
            r_reward = None
            if distill_rewards is not None:
                try:
                    r_reward = distill_rewards[bs]
                except IndexError:
                    r_reward = 0.0
            self._process_batch(bs, total_batch_list, total_infos, success, distill_reward=r_reward)

        assert len(success['success_rate']) == batch_size
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, distill_reward=None):
        # Reflect reward
        if distill_reward is not None:
            val = float(distill_reward.item()) if hasattr(distill_reward, 'item') else float(distill_reward)
            success['distill_success_rate'].append(val)
        else:
            success['distill_success_rate'].append(0.0)

        # Play phase
        trajectory = total_batch_list[batch_idx]
        n_prepended = sum(1 for s in trajectory if s.get('phase') in ('query', 'rerank'))
        for i in reversed(range(len(trajectory))):
            batch_item = trajectory[i]
            if batch_item.get('phase') in ('query', 'rerank'):
                continue
            if batch_item.get('active_masks', True):
                phase = batch_item.get('phase', 'play')
                if phase == 'play':
                    info = total_infos[batch_idx][i - n_prepended]
                    won_value = float(info.get('won', 0.0))
                    success['play_success_rate'].append(won_value)
                    success['success_rate'].append(won_value)
                    # Per-benchmark logging
                    data_source = info.get("data_source")
                    if data_source:
                        success[f"{data_source}_success_rate"].append(won_value)
                    return
        success['play_success_rate'].append(0.0)
        success['success_rate'].append(0.0)

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, retrieve_type):
        self.memory = SimpleMemory()
        # --- NEW: Group and Eval Configuration ---
        self.group_n = config.env.rollout.n  # e.g., 8
        
        # --- Extract Hyperparameters from Config ---
        mem_config = config.env.get('skill_library', {})
        filepath = mem_config.get('filepath', "skill_library.json")
        import os
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        relevance_weight = mem_config.get('relevance_weight', 0.7)
        alpha = mem_config.get('alpha', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)

        # --- NEW: Memory Start Cutoff Configuration ---
        # Memory retrieval starts only when progress > memory_start_cutoff
        # Default is 0.0 (start immediately)
        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0) 
        self.current_progress_ratio = 0.0 # Track progress internally
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        assert not (self.full_group_memory and self.group_outperformance), \
            "group_outperformance requires split group (full_group_memory must be False)"
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)
        self.distill_reward_type = mem_config.get('distill_reward_type', 'self_assess')
        self.u_hat_aggregation = mem_config.get('u_hat_aggregation', 'max')

        # --- BiGen-Retrieval config switches ---
        self.enable_query_generation = mem_config.get('enable_query_generation', False)
        self.enable_description_head = mem_config.get('enable_description_head', False)
        self.enable_rerank = mem_config.get('enable_rerank', False)

        # --- NEW: Config to only give memory to 1 agent per group ---
        # If True, 1 agent retrieves, (group_n - 1) agents are control.
        # If False, (group_n / 2) agents retrieve, (group_n / 2) are control.
        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        self.single_distill_per_group = mem_config.get('single_distill_per_group', False)
        # EMA Decay rate for the baseline (matches LaTeX gamma)
        self.ema_gamma = 0.9
        print("memory retrieve_type: ", retrieve_type)
        print("memory retrieve_mode: ", self.retrieve_mode)
        print("top_k_retrieved_memory: ", self.top_k)
        print(f"Memory Start Cutoff: {self.memory_start_cutoff}") 
        print(f"Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"Single Distill Per Group: {self.single_distill_per_group}")
        print(f"Potential Based On Binary Success Only: {self.potential_based_on_binary_success}")
        print(f"BiGen-Retrieval Query Generation: {self.enable_query_generation}")
        print(f"BiGen-Retrieval Description Head: {self.enable_description_head}")
        self.rerank_train_top1 = mem_config.get('rerank_train_top1', False)
        self.retriever_type = mem_config.get('retriever_type', 'dense')
        print(f"Skill1 Re-rank: {self.enable_rerank}")
        print(f"Rerank Train Top-1 Selection: {self.rerank_train_top1}")
        print(f"Retriever Type: {self.retriever_type}")

        self._is_eval = False

        # Initialize the persistent skill library
        self.skill_library = SkillLibrary(
            filepath=filepath,
            relevance_weight=relevance_weight,
            alpha=alpha,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale,
            use_description_head=self.enable_description_head,
            max_size=mem_config.get('max_size', 5000),
            retriever_type=self.retriever_type,
        )
        # Initialize containers for retrieval tracking
        self.task_trajectory_history = {} # Added initialization
        self.task_potential_history = {} 
        self.batch_previous_potentials = [] 
        self.current_skills = []      # Formatted strings for the prompt
        self.retrieved_raw_skills = [] # List of lists of raw strings for utility updates
        self.current_retrieval_types = []
        self.batch_retrieved_types = []
        # Store the trajectories generated during the distillation phase
        # so they can be saved to memory in step_distill
        self.last_trajectories = []
        super().__init__(envs, projection_f, config)

    # --- NEW: Method to update training progress ---
    def update_training_progress(self, current_step: int, total_steps: int):
        """
        Updates the environment with the current training progress.
        This triggers memory pruning if a 20% milestone is reached.
        """
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            self.skill_library._current_training_step = current_step
            # Pass the ratio to memory to check for pruning triggers
            # We keep top-3 as requested
            # self.skill_library.check_and_prune(progress_ratio=ratio, top_k=3)

    def reset(self, kwargs) -> Dict[str, Any]:
        if kwargs is None:
            kwargs = {}
        print("****** environment resetting ******")
        # Determine mode based on kwargs
        is_eval = not kwargs.get('is_train', True)
        self._is_eval = is_eval

        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)

        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)
        self.batch_size = len(text_obs)          # Expected: 128
        assert self.batch_size % self.group_n == 0, "Batch size must be divisible by group size"
        self.num_unique_tasks = self.batch_size // self.group_n

        # --- NEW: Retrieval Logic with Group-based Split ---
        self.current_skills = []
        self.retrieved_raw_skills = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = [] 
        self.batch_retrieved_types = [] # Reset the type tracker
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        # If we are training AND progress <= cutoff, we are in warmup -> Force memory OFF.
        # If progress > cutoff, we allow memory logic to proceed.
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)
        
        if in_warmup_period:
            # Optional: Log occasionally if needed
            print(f"Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")
            # pass 
        for i, task in enumerate(self.tasks):
            # Retrieve Phi(s) - the historical best FAILED completion for this task
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_skills = ""
            raw_list_of_dicts = [] # This will hold [{'text':..., 'type':...}]
            current_types_list = [] # List to hold types for this specific agent
            should_retrieve = False
            retrieval_type_str = "control"
            if self.enable_memory:
                if in_warmup_period:
                    # Explicitly disable retrieval during warmup
                    should_retrieve = False
                elif is_eval:
                    # During Eval: Everyone retrieves (or based on config)
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False
            else:
                should_retrieve = False

            if should_retrieve:
                # When rerank is enabled, always retrieve top_k to provide candidates for re-ranking
                k = self.top_k
                raw_list_of_dicts = self.skill_library.retrieve(
                    current_scenario_description=task,
                    top_k=k,
                    filter_type=self.retrieve_mode
                )
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        
                        # Store the type for logging
                        current_types_list.append(r_type)
                        
                        formatted_lines.append(r_text)
                    
                    formatted_skills = "Relevant skills from the skill library:\n" + "\n".join(formatted_lines)
                    formatted_skills += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            
            
            self.current_skills.append(formatted_skills)
            self.retrieved_raw_skills.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            print("retrieved_raw_skills: ", self.retrieved_raw_skills)
            # print("current_skills: ", self.current_skills)
            # --- NEW: Inject types into infos immediately ---
            infos[i]['distill_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str
            print("infos[i]['retrieval_group']: ", infos[i]['retrieval_group'])
            
        assert len(self.current_skills) == len(self.tasks)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def distill(self, infos: List[Dict]):
        """
        Called at the end of the 'play' phase.
        Updates utility based on Group B (Retrieved) vs Group A (Not Retrieved) performance.
        """
        # Build observation creates self.last_trajectories side-effect
        distill_obs_text = self.build_distill_text_obs(infos)
        
        observations = {
            'text': distill_obs_text,
            'image': None,
            'anchor': distill_obs_text
        }

        # Mark actions as valid for the distillation phase
        for info in infos:
            info['is_action_valid'] = to_numpy(True)


        # BiGen: query contrastive reward = exp_success_rate - ctrl_success_rate
        self.query_contrastive_rewards = np.zeros(len(self.tasks), dtype=np.float32)
        # BiGen metrics: per-group ctrl/exp success rates
        self._bigen_ctrl_success_rates = []
        self._bigen_exp_success_rates = []
        # BiGen metrics: selection precision (exp agent with retrieval won AND group outperformed)
        self._bigen_selection_hits = 0
        self._bigen_selection_total = 0

        batch_size = len(self.tasks)
        if batch_size % self.group_n != 0:
            print(f"WARNING: Batch size {batch_size} not divisible by group_n {self.group_n}")

        num_groups = batch_size // self.group_n
        group_split_index = 0 if self.full_group_memory else self.group_n // 2

        # Iterate over each group independently
        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index

            # 1. Calculate Wins for Control (First half)
            control_wins = 0
            for i in range(start_idx, mid_idx):
                if infos[i].get("won", False):
                    control_wins += 1

            # 2. Calculate Wins for Experiment (Second half)
            experiment_wins = 0
            for i in range(mid_idx, end_idx):
                if infos[i].get("won", False):
                    experiment_wins += 1

            # 3. Determine Utility Score for THIS group
            group_outperformed = experiment_wins > control_wins

            n_ctrl = max(group_split_index, 1)
            n_exp = max(self.group_n - group_split_index, 1)
            self._bigen_ctrl_success_rates.append(control_wins / n_ctrl)
            self._bigen_exp_success_rates.append(experiment_wins / n_exp)

            # 3b. BiGen: contrastive reward for experiment agents' query steps
            if self.enable_query_generation:
                contrastive = (experiment_wins / n_exp) - (control_wins / n_ctrl)
                for i in range(mid_idx, end_idx):
                    self.query_contrastive_rewards[i] = contrastive
                print(f"[BiGen] Group {g}: ctrl={control_wins/n_ctrl:.2f}, exp={experiment_wins/n_exp:.2f}, q_reward={contrastive:.2f}")

            # 3c. Selection precision: exp agents with retrieval that won when group outperformed
            for i in range(mid_idx, end_idx):
                if self.retrieved_raw_skills[i]:
                    self._bigen_selection_total += 1
                    if infos[i].get("won", False) and group_outperformed:
                        self._bigen_selection_hits += 1

            # 4. Update Memory for experiment agents (all agents when full_group_memory)
            for i in range(mid_idx, end_idx):
                task_desc = self.tasks[i]
                raw_skill_items = self.retrieved_raw_skills[i] # List[Dict]
                is_success = infos[i].get("won", False)

                # 1.0 if the agent won AND the retrieval group beat the control group.
                if self.group_outperformance:
                    if is_success and group_outperformed:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
                else:
                    if is_success:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0


                if raw_skill_items:
                    for item in raw_skill_items:
                        # --- UPDATED: Extract text from dict for utility update ---
                        strategy_text = item.get('text', '')
                        if strategy_text:
                            self.skill_library.update_utility(
                                scenario_description=task_desc, 
                                strategy_text=strategy_text, 
                                score=utility_score
                            )

        # BiGen metrics: memory buffer size after distill utility updates
        self._bigen_memory_buffer_size = len(self.skill_library.data)

        return observations, infos

    def step_distill(self, text_actions: List[str], infos: List[Dict]):
        """
        Calculates intrinsic rewards based on improvement over an EMA baseline,
        normalizes them group-wise, and updates the baseline.
        """
        import json
        import re
        import copy
        import numpy as np
        
        def to_numpy(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        print("text_actions for distillation:", text_actions)
        
        # 1. Initialize Containers
        distill_rewards = [] # This is the immediate reward for the distillation step itself (e.g. self-consistency)
        current_scores = np.zeros(self.batch_size) # The raw potential (phi)
        raw_improvements = np.zeros(self.batch_size) # The raw I (improvement)
        is_won_array = np.zeros(self.batch_size, dtype=bool)

        # BiGen metrics: description_head parsing stats
        self._bigen_desc_head_total = 0
        self._bigen_desc_head_parsed = 0
        self._bigen_desc_head_saved = 0
        self._distill_correct_count = 0
        self._distill_total_count = 0
        self._distill_ops = []  # Track distillation operation per rollout: "add" or "none"
        self._distill_u_hat_values = []
        self._distill_r_values = []

        # Ensure batch_previous_potentials is synced
        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        # 2. Calculate Raw Scores (Phi) and Raw Improvements (I)
        for i, strategy_text in enumerate(text_actions):
            task_desc = self.tasks[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""

            # Get the baseline (Phi_{t-1})
            prev_phi = self.batch_previous_potentials[i]
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0
            # ... (JSON Parsing Logic - same as before) ...
            try:
                # --- JSON Extraction ---
                json_str = ""
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', strategy_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = strategy_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]

                if not json_str: raise ValueError("No JSON found")

                distill_data = json.loads(json_str)

                # --- Subtask Scoring (Phi calculation) ---
                subtasks = distill_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for task in subtasks
                    if isinstance(task, dict) and task.get('status', '').strip().lower() == 'completed'
                )

                # Calculate subtask-based potential
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0

                # --- DETERMINE CURRENT PHI ---
                if self.potential_based_on_binary_success:
                    # STRICT MODE: Only actual success matters for potential
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    # DEFAULT MODE: Use subtasks, but override if actual success
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                # --- Reflection Consistency Reward (Auxiliary) ---
                predicted_success = distill_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']

                if self.distill_reward_type == 'first_order_diff':
                    retrieved_items = self.retrieved_raw_skills[i] if i < len(self.retrieved_raw_skills) else []
                    u_scores = [item.get('utility_score', 0.5) for item in retrieved_items] if retrieved_items else []
                    u_hat = (max(u_scores) if self.u_hat_aggregation == 'max' else sum(u_scores) / len(u_scores)) if u_scores else 0.0
                    r_tau = 1.0 if actual_success else 0.0
                    current_reward = r_tau - u_hat
                    self._distill_u_hat_values.append(u_hat)
                    self._distill_r_values.append(current_reward)
                else:
                    current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                distill_rewards.append(current_reward)
                self._distill_total_count += 1
                if predicted_success == actual_success:
                    self._distill_correct_count += 1

                # --- Memory Saving Logic ---
                distill_op = "none"
                should_save = (actual_success and json_str) if self.distill_reward_type == 'first_order_diff' else (predicted_success == actual_success and json_str)
                if should_save:
                    action_lesson = distill_data.get('action_lesson')
                    nav_lesson = distill_data.get('navigation_lesson')
                    description_head = (distill_data.get('description_head') or '') if self.enable_description_head else ''
                    if self.enable_description_head:
                        self._bigen_desc_head_total += 1
                        if description_head and len(str(description_head).strip()) > 5:
                            self._bigen_desc_head_parsed += 1
                    lessons_to_save = []
                    if action_lesson and len(str(action_lesson)) > 5: lessons_to_save.append(f"Action Insight: {action_lesson}")
                    if nav_lesson and len(str(nav_lesson)) > 5: lessons_to_save.append(f"Navigation Insight: {nav_lesson}")

                    if lessons_to_save:
                        distill_op = "add"
                        final_lesson = " | ".join(lessons_to_save)
                        self.skill_library.admit(
                            scenario_description=task_desc,
                            strategy_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio,
                            description_head=str(description_head) if description_head else ''
                        )
                        if self.enable_description_head and description_head and len(str(description_head).strip()) > 5:
                            self._bigen_desc_head_saved += 1
                self._distill_ops.append(distill_op)

            except Exception as e:
                print(f"Error task {i}: {e}")
                distill_rewards.append(0.0)
                self._distill_ops.append("none")
                # Fallback logic for Phi on error
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0

            # --- Calculate Raw Improvement (I) ---
            current_scores[i] = current_phi
            # Improvement is strictly positive gain over history
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement

        # 3. Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            
            task_desc = self.tasks[start_idx]
            
            # Extract group data
            group_improvements = raw_improvements[start_idx:end_idx]
            # group_scores = current_scores[start_idx:end_idx]
            
            # A. Normalization (Centering)
            # As per LaTeX Eq (8): R_int = I - Mean(I)
            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                # Note: We do NOT divide by std here, just centering is sufficient 
                # to maintain the zero-sum property for the intrinsic component.
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)

            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate
            # B. Update Historical Baseline (EMA)
            # As per LaTeX Eq (9): Phi_t = gamma * Phi_{t-1} + (1-gamma) * Mean(Phi_t)
            # if len(group_scores) > 0:
            #     current_group_mean_score = np.mean(group_scores)
            #     old_baseline = self.task_potential_history.get(task_desc, 0.0)

            #     if current_group_mean_score > old_baseline:
            #         self.task_potential_history[task_desc] = current_group_mean_score
                # # EMA Update
                # new_baseline = (self.ema_gamma * old_baseline) + ((1 - self.ema_gamma) * current_group_mean_score)
                # self.task_potential_history[task_desc] = new_baseline

        print("raw_improvements: ", raw_improvements)
        print("final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        # Convert to numpy for compatibility
        return None, to_numpy(distill_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'reward': rewards, 'dones': dones, 'won': [info['won'] for info in infos]})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')

            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    # --- BiGen-Retrieval: Query Generation support ---
    def build_query_generation_obs(self) -> Dict[str, Any]:
        """Build observations for query generation phase (Phase 0)."""
        query_obs_texts = []
        for i, task in enumerate(self.tasks):
            obs_text = ALFWORLD_QUERY_GENERATION_TEMPLATE.format(
                task_description=task,
                initial_observation=self.pre_text_obs[i]
            )
            query_obs_texts.append(obs_text)
        return {
            'text': query_obs_texts,
            'image': None,
            'anchor': query_obs_texts
        }

    def apply_generated_queries(self, query_texts: List[str]):
        """Use actor-generated queries to re-retrieve from memory, only for experiment group."""
        for i, query_text in enumerate(query_texts):
            # Only re-retrieve for experiment / eval agents, skip control
            if self.current_retrieval_types[i] not in ("experiment", "eval_retrieval"):
                continue

            # Parse <query>...</query> tag
            match = re.search(r'<query>(.*?)</query>', query_text, re.DOTALL)
            parsed = match.group(1).strip() if match else query_text.strip()
            if not parsed:
                continue

            raw_list_of_dicts = self.skill_library.retrieve(
                current_scenario_description=parsed,
                top_k=self.top_k,
                filter_type=self.retrieve_mode
            )
            if raw_list_of_dicts:
                formatted_lines = [item.get('text', '') for item in raw_list_of_dicts]
                self.current_skills[i] = (
                    "Relevant skills from the skill library:\n" + "\n".join(formatted_lines)
                    + "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
                )
                self.retrieved_raw_skills[i] = raw_list_of_dicts

    def rebuild_initial_obs(self):
        """Rebuild initial play observation after query generation / rerank updated skills."""
        full_text_obs = self.build_text_obs(self.pre_text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': self.pre_text_obs}

    # ---- Re-rank Phase Methods ----

    def build_rerank_obs(self) -> Dict[str, Any]:
        """Build observations for the re-rank phase.
        For agents with >=2 retrieved candidates, present them for ranking.
        For others, present a dummy prompt.
        """
        rerank_obs_texts = []
        self.rerank_candidates = []  # store per-agent candidate info for reward computation

        for i, task in enumerate(self.tasks):
            raw_items = self.retrieved_raw_skills[i]  # List[Dict] with 'text', 'type', possibly 'score'
            # Also fetch utility scores from memory for NDCG computation
            candidates_with_scores = self._get_candidate_scores(raw_items)

            if len(candidates_with_scores) >= 2:
                candidate_lines = []
                for idx, cand in enumerate(candidates_with_scores):
                    candidate_lines.append(f"[Experience {idx + 1}]: {cand['text']}")
                candidate_str = "\n\n".join(candidate_lines)

                obs_text = ALFWORLD_RERANK_TEMPLATE.format(
                    task_description=task,
                    initial_observation=self.pre_text_obs[i],
                    n_candidates=len(candidates_with_scores),
                    candidate_experiences=candidate_str,
                )
            else:
                obs_text = ALFWORLD_RERANK_DUMMY_TEMPLATE.format(
                    task_description=task,
                    initial_observation=self.pre_text_obs[i],
                )

            rerank_obs_texts.append(obs_text)
            self.rerank_candidates.append(candidates_with_scores)

        return {
            'text': rerank_obs_texts,
            'image': None,
            'anchor': rerank_obs_texts,
        }

    def _get_candidate_scores(self, raw_items: List[Dict]) -> List[Dict]:
        """Look up utility_score from skill_library for each retrieved item."""
        result = []
        for item in raw_items:
            r_text = item.get('text', '')
            utility = 0.5  # default
            for mem_entry in self.skill_library.data:
                if mem_entry['strategy'] == r_text:
                    utility = mem_entry.get('utility_score', 0.5)
                    break
            result.append({'text': r_text, 'type': item.get('type', 'unknown'), 'utility': utility})
        return result

    def apply_rerank_results(self, text_actions: List[str]):
        """Parse model's ranking output and reorder skills accordingly."""
        import re as _re

        self._rerank_parse_successes = 0
        self._rerank_total = 0
        self._rerank_predicted_orders = []

        for i, action_text in enumerate(text_actions):
            candidates = self.rerank_candidates[i]
            n = len(candidates)
            self._rerank_total += 1

            if n < 2:
                self._rerank_predicted_orders.append(None)
                continue

            # Parse <rank>...</rank>
            match = _re.search(r'<rank>(.*?)</rank>', action_text, _re.DOTALL)
            if not match:
                self._rerank_predicted_orders.append(None)
                continue

            rank_str = match.group(1).strip()
            if rank_str.lower() == 'none':
                self._rerank_predicted_orders.append(None)
                continue

            # Parse comma-separated IDs (1-indexed)
            parsed_ids = []
            for token in rank_str.split(','):
                token = token.strip()
                if token.isdigit():
                    idx = int(token) - 1  # convert to 0-indexed
                    if 0 <= idx < n and idx not in parsed_ids:
                        parsed_ids.append(idx)

            if len(parsed_ids) == 0:
                self._rerank_predicted_orders.append(None)
                continue

            self._rerank_parse_successes += 1
            self._rerank_predicted_orders.append(parsed_ids)

            # Fill in any missing indices at the end (preserve original order for unranked)
            remaining = [j for j in range(n) if j not in parsed_ids]
            full_order = parsed_ids + remaining

            # Reorder skills text
            reordered_texts = [candidates[j]['text'] for j in full_order]

            # During training, only keep the top-1 ranked experience so the
            # model learns to surface the most useful skill.
            if self.rerank_train_top1 and not self._is_eval:
                reordered_texts = reordered_texts[:1]

            formatted = "Relevant skills from the skill library:\n" + "\n".join(reordered_texts)
            formatted += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            self.current_skills[i] = formatted

    def compute_rerank_rewards(self) -> np.ndarray:
        """Compute NDCG-based rewards for the re-rank phase.
        Ground truth order = sorted by utility_score descending.
        Predicted order = model's output ranking.
        """
        import numpy as np
        rewards = np.zeros(len(self.tasks), dtype=np.float32)

        for i in range(len(self.tasks)):
            candidates = self.rerank_candidates[i]
            predicted_order = self._rerank_predicted_orders[i]

            if predicted_order is None or len(candidates) < 2:
                rewards[i] = 0.0
                continue

            utilities = np.array([c['utility'] for c in candidates])
            # Ground truth: indices sorted by utility descending
            ideal_order = np.argsort(-utilities)
            ideal_gains = utilities[ideal_order]

            # Predicted gains in predicted order
            # Fill unranked at the end
            remaining = [j for j in range(len(candidates)) if j not in predicted_order]
            full_predicted = predicted_order + remaining
            predicted_gains = utilities[full_predicted]

            # DCG computation: sum(gain / log2(rank + 2)) — rank is 0-indexed
            def _dcg(gains):
                positions = np.arange(len(gains), dtype=np.float64) + 2.0  # log2(2), log2(3), ...
                return np.sum(gains / np.log2(positions))

            ideal_dcg = _dcg(ideal_gains)
            if ideal_dcg < 1e-8:
                rewards[i] = 0.0
                continue

            predicted_dcg = _dcg(predicted_gains)
            ndcg = predicted_dcg / ideal_dcg
            rewards[i] = float(ndcg)

        self._rerank_rewards = rewards
        return rewards

    def get_bigen_metrics(self, prefix: str = "bigen") -> Dict[str, float]:
        """Collect all BiGen-Retrieval metrics for wandb logging.

        Args:
            prefix: Key prefix, e.g. "bigen_train" or "bigen_val".

        Safe to call even during validation when distill() has not run.
        Key names never contain 'success_rate' (reserved by metric_utils).
        """
        p = prefix
        metrics = {}

        # --- Utility distribution (own wandb section) ---
        mem_data = self.skill_library.data
        if mem_data:
            util_scores = [float(entry.get('utility_score', 0.5)) for entry in mem_data]
            n = len(util_scores)
            for bin_lo in range(10):
                lo = bin_lo / 10.0
                hi = lo + 0.1
                if bin_lo == 9:
                    cnt = sum(1 for u in util_scores if lo <= u <= hi)
                else:
                    cnt = sum(1 for u in util_scores if lo <= u < hi)
                metrics[f'{p}/utility_dist/{bin_lo*10:02d}'] = float(cnt) / n

        # --- skill_statics: memory health & co-evolution metrics ---
        s = 'skill_statics'
        buf_size = getattr(self, '_bigen_memory_buffer_size', len(mem_data))
        metrics[f'{s}/memory_buffer_size'] = float(buf_size)
        if mem_data:
            counts = [float(entry.get('count', 1)) for entry in mem_data]
            metrics[f'{s}/library_quality'] = sum(util_scores) / n
            metrics[f'{s}/memory_avg_count'] = sum(counts) / n
            metrics[f'{s}/memory_low_utility_frac'] = sum(1 for u in util_scores if u < 0.3) / n
            total_chars = sum(len(entry.get('strategy', '')) + len(entry.get('trajectory', '')) for entry in mem_data)
            metrics[f'{s}/memory_total_chars'] = float(total_chars)
        metrics[f'{s}/memory_evicted_count'] = float(getattr(self.skill_library, '_last_evicted_count', 0))

        ctrl_rates = getattr(self, '_bigen_ctrl_success_rates', [])
        exp_rates = getattr(self, '_bigen_exp_success_rates', [])
        if ctrl_rates:
            metrics[f'{s}/ctrl_win_rate/mean'] = float(np.mean(ctrl_rates))
            metrics[f'{s}/exp_win_rate/mean'] = float(np.mean(exp_rates))
            metrics[f'{s}/exp_minus_ctrl/mean'] = float(np.mean(exp_rates)) - float(np.mean(ctrl_rates))

        sel_total = getattr(self, '_bigen_selection_total', 0)
        sel_hits = getattr(self, '_bigen_selection_hits', 0)
        if sel_total > 0:
            metrics[f'{s}/selection_precision'] = float(sel_hits) / float(sel_total)
        ref_total = getattr(self, '_distill_total_count', 0)
        ref_correct = getattr(self, '_distill_correct_count', 0)
        if ref_total > 0:
            metrics[f'{s}/self_assess_accuracy'] = float(ref_correct) / float(ref_total)

        # Query generation metrics
        if self.enable_query_generation:
            q_rewards = getattr(self, 'query_contrastive_rewards', None)
            if q_rewards is not None and len(q_rewards) > 0:
                metrics[f'{p}/query_contrastive_reward/mean'] = float(np.mean(q_rewards))
                metrics[f'{p}/query_contrastive_reward/min'] = float(np.min(q_rewards))
                metrics[f'{p}/query_contrastive_reward/max'] = float(np.max(q_rewards))
                exp_mask = q_rewards != 0
                if exp_mask.any():
                    metrics[f'{p}/query_contrastive_reward/exp_mean'] = float(np.mean(q_rewards[exp_mask]))

        # Description head metrics
        if self.enable_description_head:
            total = getattr(self, '_bigen_desc_head_total', 0)
            parsed = getattr(self, '_bigen_desc_head_parsed', 0)
            saved = getattr(self, '_bigen_desc_head_saved', 0)
            metrics[f'{p}/desc_head/total_attempts'] = float(total)
            metrics[f'{p}/desc_head/parsed_count'] = float(parsed)
            metrics[f'{p}/desc_head/saved_count'] = float(saved)
            if total > 0:
                metrics[f'{p}/desc_head/parse_rate'] = float(parsed / total)

        # Re-rank metrics
        if self.enable_rerank:
            rr_rewards = getattr(self, '_rerank_rewards', None)
            if rr_rewards is not None and len(rr_rewards) > 0:
                metrics[f'{p}/rerank_ndcg/mean'] = float(np.mean(rr_rewards))
                metrics[f'{p}/rerank_ndcg/min'] = float(np.min(rr_rewards))
                metrics[f'{p}/rerank_ndcg/max'] = float(np.max(rr_rewards))
                nonzero_mask = rr_rewards > 0
                if nonzero_mask.any():
                    metrics[f'{p}/rerank_ndcg/active_mean'] = float(np.mean(rr_rewards[nonzero_mask]))
            rr_total = getattr(self, '_rerank_total', 0)
            rr_success = getattr(self, '_rerank_parse_successes', 0)
            metrics[f'{p}/rerank_parse/total'] = float(rr_total)
            metrics[f'{p}/rerank_parse/success'] = float(rr_success)
            if rr_total > 0:
                metrics[f'{p}/rerank_parse/parse_rate'] = float(rr_success / rr_total)

        # --- Selection evolution metrics ---
        retrieved_refs = getattr(self, 'retrieved_raw_skills', [])
        if retrieved_refs:
            from collections import Counter
            import math as _math
            # Collect all retrieved skill_ids (skip empty retrievals)
            all_skill_ids = []
            per_task_skill_ids = {}  # task_index -> list of skill_ids
            for idx, ref_list in enumerate(retrieved_refs):
                task_group_idx = idx // self.group_n if self.group_n > 0 else idx
                if task_group_idx not in per_task_skill_ids:
                    per_task_skill_ids[task_group_idx] = []
                for item in ref_list:
                    sid = item.get('skill_id', '') or item.get('text', '')[:50]
                    if sid:
                        all_skill_ids.append(sid)
                        per_task_skill_ids[task_group_idx].append(sid)

            # 2.1 Retrieval Entropy
            if all_skill_ids:
                counter = Counter(all_skill_ids)
                total_count = len(all_skill_ids)
                probs = [c / total_count for c in counter.values()]
                entropy = -sum(p * _math.log2(p) for p in probs if p > 0)
                metrics[f'{s}/retrieval_entropy'] = float(entropy)
                lib_size = max(len(mem_data), 1)
                max_entropy = _math.log2(lib_size) if lib_size > 1 else 1.0
                metrics[f'{s}/retrieval_entropy_normalized'] = float(entropy / max_entropy) if max_entropy > 0 else 0.0

            # 2.2 Retrieval Consistency (per-task mode fraction)
            consistency_values = []
            for task_idx, sids in per_task_skill_ids.items():
                if sids:
                    counter = Counter(sids)
                    mode_count = counter.most_common(1)[0][1]
                    consistency_values.append(mode_count / len(sids))
            if consistency_values:
                metrics[f'{s}/retrieval_consistency'] = float(np.mean(consistency_values))

            # Retrieval similarity stats
            all_sims = []
            for ref_list in retrieved_refs:
                for item in ref_list:
                    sim = item.get('similarity_score')
                    if sim is not None:
                        all_sims.append(float(sim))
            if all_sims:
                metrics[f'{s}/retrieval_similarity/mean'] = float(np.mean(all_sims))
                metrics[f'{s}/retrieval_similarity/min'] = float(np.min(all_sims))
                metrics[f'{s}/retrieval_similarity/max'] = float(np.max(all_sims))

        # --- Distillation ops distribution ---
        distill_ops = getattr(self, '_distill_ops', [])
        if distill_ops:
            from collections import Counter
            ops_counter = Counter(distill_ops)
            total_ops = len(distill_ops)
            metrics[f'{s}/distill_ops/add_rate'] = float(ops_counter.get('add', 0)) / total_ops
            metrics[f'{s}/distill_ops/none_rate'] = float(ops_counter.get('none', 0)) / total_ops
            metrics[f'{s}/distill_ops/total'] = float(total_ops)

        # --- Skill lifetime distribution ---
        if mem_data:
            current_step = getattr(self.skill_library, '_current_training_step', 0)
            lifetimes = [current_step - entry.get('created_at_step', 0) for entry in mem_data]
            if lifetimes:
                lifetimes_arr = np.array(lifetimes, dtype=np.float64)
                metrics[f'{s}/skill_lifetime/mean'] = float(np.mean(lifetimes_arr))
                metrics[f'{s}/skill_lifetime/q1'] = float(np.percentile(lifetimes_arr, 25))
                metrics[f'{s}/skill_lifetime/q3'] = float(np.percentile(lifetimes_arr, 75))

        # First-order difference reward metrics (skill1_v2)
        u_hat_vals = getattr(self, '_distill_u_hat_values', [])
        r_vals = getattr(self, '_distill_r_values', [])
        if u_hat_vals:
            metrics[f'{s}/distill_u_hat/mean'] = float(np.mean(u_hat_vals))
            metrics[f'{s}/distill_u_hat/max'] = float(np.max(u_hat_vals))
            metrics[f'{s}/distill_u_hat/min'] = float(np.min(u_hat_vals))
            metrics[f'{s}/distill_r_diff/mean'] = float(np.mean(r_vals))
            metrics[f'{s}/distill_r_diff/positive_rate'] = float(sum(1 for r in r_vals if r > 0) / len(r_vals))
            metrics[f'{s}/distill_r_diff/negative_rate'] = float(sum(1 for r in r_vals if r < 0) / len(r_vals))

        return metrics

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    skills=self.current_skills[i], # Add this if template supports it
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    skills=self.current_skills[i], # <--- INJECTED HERE
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def build_distill_text_obs(self, infos: List[str]) -> List[str]:
        """
        This function builds the text observation for the agent during distillation.
        It uses ALFWORLD_DISTILL_TEMPLATE which requires task_description and current_trajectory.
        """
        postprocess_text_obs = []
        memory_contexts, valid_lens = self.memory.fetch(
                50,
                obs_key="text_obs",
                action_key="action")
        # self.task_trajectory_history[task] = {"successful": [], "failed": []}
        for i in range(len(infos)):
            task = self.tasks[i]
            # Ensure key exists (it should from reset, but safety first)
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
                
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])

        # --- CRITICAL: Store these so step_distill can access them ---
        self.last_trajectories = memory_contexts
        for i in range(len(infos)):
            task = self.tasks[i]
            is_won = infos[i].get("won", False)
            
            # Determine success string and select Contrastive Reference
            # If we WON, we want to see a FAIL to understand what to avoid (or just compare)
            # If we LOST, we want to see a SUCCESS to understand what to do
            
            reference_traj_str = "No reference history available yet."
            
            if is_won:
                SUCCESS = "successfully"
                # Try to get a failed example
                failed_hist = self.task_trajectory_history[task]["failed"]
                if failed_hist:
                    # Use the most recent failure
                    reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                else:
                    reference_traj_str = "No failed attempts available for comparison."
            else:
                SUCCESS = "unsuccessfully" # Changed from "NOT successfully" for better grammar
                # Try to get a successful example
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    # Use the most recent success
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."

            distill_tmpl = ALFWORLD_DISTILL_TEMPLATE_WITH_DESC_HEAD if self.enable_description_head else ALFWORLD_DISTILL_TEMPLATE
            obs = distill_tmpl.format(
                task_description=self.tasks[i],
                success=SUCCESS,
                reference_trajectory=reference_traj_str,
                current_trajectory=memory_contexts[i],
            )
            postprocess_text_obs.append(obs)
        
        # Debug print
        if len(postprocess_text_obs) > 0:
            print("processed_distill_text [0]: ", postprocess_text_obs[0])
            
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        
        Args:
            kwargs: Must contain:
                - total_infos (List[List[Dict]]): Info dicts for every step.
                - total_batch_list (List[List[Dict]]): Trajectory data.
                - distill_rewards (np.ndarray or List, optional): Rewards specifically for the distillation phase.
        
        Returns:
            - success (Dict[str, np.ndarray]): Dictionary of success metrics.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        
        # Extract distill_rewards. It might be None if not in training mode.
        distill_rewards = kwargs.get('distill_rewards', None)

        batch_size = len(total_batch_list)
        success = defaultdict(list)
        
        for bs in range(batch_size):
            # Extract the specific distill reward for this batch index
            r_reward = None
            if distill_rewards is not None:
                # Handle case where distill_rewards is a list, numpy array, or tensor
                try:
                    r_reward = distill_rewards[bs]
                except IndexError:
                    # Fallback if sizes mismatch (though they shouldn't)
                    r_reward = 0.0
            
            self._process_batch(bs, total_batch_list, total_infos, success, distill_reward=r_reward)
        
        # Ensure consistency in list lengths
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, distill_reward=None):
        """
        Process a single batch trajectory to extract success metrics.
        
        Args:
            batch_idx: Index of the current batch.
            total_batch_list: The full list of trajectories.
            total_infos: The full list of info dicts.
            success: The dictionary to append results to.
            distill_reward: The specific reward value for the distillation phase (float, Tensor, or None).
        """
        # --- 1. Process Reflection Phase ---
        # Since distill_rewards are passed explicitly, we process them directly here.
        if distill_reward is not None:
            # Convert Tensor or numpy scalar to python float
            if hasattr(distill_reward, 'item'):
                val = float(distill_reward.item())
            else:
                val = float(distill_reward)
            success['distill_success_rate'].append(val)
        else:
            # If no distill rewards provided (e.g. eval mode), log 0.0
            success['distill_success_rate'].append(0.0)

        # --- 2. Process Play Phase ---
        play_success_found = False
        
        # Iterate backwards to find the last active step of the 'play' phase.
        # We do this because the 'won' flag is usually at the end of the trajectory.
        trajectory = total_batch_list[batch_idx]
        # Count non-play steps prepended (query/rerank) to offset into total_infos
        n_prepended = sum(1 for s in trajectory if s.get('phase') in ('query', 'rerank'))

        for i in reversed(range(len(trajectory))):
            batch_item = trajectory[i]

            # Skip inactive steps (padding)
            if not batch_item.get('active_masks', True):
                continue

            # Check phase. Based on your logs, these are likely 'play'.
            phase = batch_item.get('phase', 'play')

            if phase == 'play':
                info_idx = i - n_prepended
                info = total_infos[batch_idx][info_idx]
                won_value = float(info.get('won', 0.0))
                
                # General Play Success
                success['play_success_rate'].append(won_value)
                success['success_rate'].append(won_value) # Main metric usually tracks play
                
                # Task-Specific Success (e.g., pick_and_place, alfworld)
                # Check 'extra.gamefile' (ALFWorld specific)
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                
                # Fallback: Check 'data_source'
                elif "data_source" in info:
                    data_source = info.get("data_source")
                    success[f"{data_source}_success_rate"].append(won_value)
                
                play_success_found = True
                
                # Once we find the valid end of the play phase, we stop searching this batch
                break
        
        # --- 3. Handle Missing Play Phase ---
        # If for some reason a trajectory has no 'play' phase or is entirely padding
        if not play_success_found:
             success['play_success_rate'].append(0.0)
             success['success_rate'].append(0.0)

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break

class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    def __init__(self, envs, projection_f, config, retrieve_type=None):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        self.num_actions_per_turn = config.env.get('num_actions_per_turn', 3)
        self.max_turns = config.env.get('max_turns', 7)
        # --- Reflection Configuration Start ---
        self.group_n = config.env.rollout.n 
        
        mem_config = config.env.get('skill_library', {})
        filepath = mem_config.get('filepath', "sokoban_skill_library.json")
        import os
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        relevance_weight = mem_config.get('relevance_weight', 0.7)
        alpha = mem_config.get('alpha', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)
        
        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0) 
        self.current_progress_ratio = 0.0 # Track progress internally
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        assert not (self.full_group_memory and self.group_outperformance), \
            "group_outperformance requires split group (full_group_memory must be False)"
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)
        self.distill_reward_type = mem_config.get('distill_reward_type', 'self_assess')
        self.u_hat_aggregation = mem_config.get('u_hat_aggregation', 'max')

        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        self.single_distill_per_group = mem_config.get('single_distill_per_group', False)
        # EMA Decay rate for the baseline (matches LaTeX gamma)
        self.ema_gamma = 0.9
        print("memory retrieve_type: ", retrieve_type)
        print("memory retrieve_mode: ", self.retrieve_mode)
        print("top_k_retrieved_memory: ", self.top_k)
        print(f"Memory Start Cutoff: {self.memory_start_cutoff}") 
        print(f"Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"Single Distill Per Group: {self.single_distill_per_group}")
        print(f"Potential Based On Binary Success Only: {self.potential_based_on_binary_success}")
        self.retriever_type = mem_config.get('retriever_type', 'dense')
        print(f"Retriever Type: {self.retriever_type}")
        # Initialize persistent skill library
        self.skill_library = SkillLibrary(
            filepath=filepath,
            relevance_weight=relevance_weight,
            alpha=alpha,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale,
            max_size=5000,
            retriever_type=self.retriever_type,
        )
        self.task_trajectory_history = {}
        self.task_potential_history = {} 
        self.batch_previous_potentials = [] 
        
        
        # Initialize containers for retrieval tracking
        self.current_skills = []       # Formatted strings for the prompt
        self.retrieved_raw_skills = []  # List of lists of raw strings for utility updates
        self.init_states = []
        self.current_retrieval_types = []
        # Store the trajectories generated during the distillation phase
        # so they can be saved to memory in step_distill
        self.last_trajectories = []        
        super().__init__(envs, projection_f, config)

    def update_training_progress(self, current_step: int, total_steps: int):
        """Updates the environment with the current training progress."""
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            # self.skill_library.check_and_prune(progress_ratio=ratio, top_k=3)

    def reset(self, kwargs):
        if kwargs is None:
            kwargs = {}
            
        # Determine mode based on kwargs
        is_eval = not kwargs.get('is_train', True)
        # print("is_eval:", is_eval)

        obs, infos = self.envs.reset()
        self.init_states = obs # Store initial state for retrieval key
        
        if self.is_multi_modal:
            obs_array = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            # Note: For visual sokoban, we might lack a text description for retrieval.
            # We assume 'obs' or 'infos' contains a level string/ID for retrieval keys.
        else:
            self.pre_text_obs = obs
        
        self.batch_size = len(obs)
        assert self.batch_size % self.group_n == 0, "Batch size must be divisible by group size"
        self.num_unique_tasks = self.batch_size // self.group_n

        self.current_skills = []
        self.retrieved_raw_skills = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = [] 
        self.batch_retrieved_types = [] # Reset the type tracker
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)
        
        if in_warmup_period:
            # Optional: Log occasionally if needed
            print(f"Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")
            pass 
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        
        for i, task in enumerate(self.init_states):
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_skills = ""
            raw_list_of_dicts = [] # This will hold [{'text':..., 'type':...}]
            current_types_list = [] # List to hold types for this specific agent

            should_retrieve = False
            retrieval_type_str = "control"
            if self.enable_memory:
                if in_warmup_period:
                    # Explicitly disable retrieval during warmup
                    should_retrieve = False
                elif is_eval:
                    # During Eval: Everyone retrieves (or based on config)
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False
            else:
                should_retrieve = False

            if should_retrieve:
                # Retrieve top_k items
                k = self.top_k
                raw_list_of_dicts = self.skill_library.retrieve(
                    current_scenario_description=task, 
                    top_k=k, 
                    filter_type=self.retrieve_mode
                )
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        
                        # Store the type for logging
                        current_types_list.append(r_type)
                        
                        formatted_lines.append(r_text)
                    
                    formatted_skills = "Relevant skills from the skill library:\n" + "\n".join(formatted_lines)
                    formatted_skills += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            
            self.current_skills.append(formatted_skills)
            self.retrieved_raw_skills.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            print("retrieved_raw_skills: ", self.retrieved_raw_skills)
            # print("current_skills: ", self.current_skills)
            # --- NEW: Inject types into infos immediately ---
            infos[i]['distill_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str
        
        # -----------------------------------------------------------
        assert len(self.current_skills) == len(self.init_states)
        # Build observations
        if self.is_multi_modal:
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs_array,   
                'anchor': obs_array
            }
        else:
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
            
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        # Store in memory. Note: If is_multi_modal, pre_text_obs is an image array.
        # For distillation purposes, we might prefer storing the text action and maybe a simplified state if available.
        self.memory.store({
            'text_obs': self.pre_text_obs, 
            'action': [self.ACTION_LOOKUP[act] for act in actions],
            'reward': rewards,
            'dones': dones,
            'won': [info.get('won', False) for info in infos] # Ensure 'won' is tracked
        })

        if self.is_multi_modal:
            next_obs_array = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs_array,
                'anchor': next_obs_array 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def distill(self, infos: List[Dict]):
        """
        Called at the end of the 'play' phase.
        Updates utility based on Group B (Retrieved) vs Group A (Not Retrieved) performance.
        """
        observations = {
            'text': self.build_distill_text_obs(infos),
            'image': None,
            'anchor': self.build_distill_text_obs(infos) # Anchor needed for some pipelines
        }
        
        # Ensure action validity is set for all
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        
        batch_size = len(self.init_states)
        
        # Ensure batch size is divisible by group_n
        if batch_size % self.group_n != 0:
            print(f"WARNING: Batch size {batch_size} not divisible by group_n {self.group_n}")
        
        num_groups = batch_size // self.group_n
        group_split_index = 0 if self.full_group_memory else self.group_n // 2
        
        # Iterate over each group independently
        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index
            
            # Calculate wins for control group (first half)
            control_wins = 0
            for i in range(start_idx, mid_idx):
                if infos[i].get("won", False):
                    control_wins += 1
            
            # Calculate wins for experiment group (second half)
            experiment_wins = 0
            for i in range(mid_idx, end_idx):
                if infos[i].get("won", False):
                    experiment_wins += 1
            
            # 3. Determine Utility Score for THIS group
            group_outperformed = experiment_wins > control_wins
            
            # 4. Update Memory ONLY for the Experiment agents
            for i in range(mid_idx, end_idx):
                task_desc = self.init_states[i]
                raw_skill_items = self.retrieved_raw_skills[i] # List[Dict]
                is_success = infos[i].get("won", False)
                
                # 1.0 if the agent won AND the retrieval group beat the control group.
                if self.group_outperformance:
                    if is_success and group_outperformed:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
                else:
                    if is_success:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0

                if raw_skill_items:
                    for item in raw_skill_items:
                        # --- UPDATED: Extract text from dict for utility update ---
                        strategy_text = item.get('text', '')
                        if strategy_text:
                            self.skill_library.update_utility(
                                scenario_description=task_desc, 
                                strategy_text=strategy_text, 
                                score=utility_score
                            )
        
        return observations, infos

    def step_distill(self, text_actions: List[str], infos: List[Dict]):
        """
        Stores the distilled skill into the persistent memory.
        Parses the JSON output containing specific subtasks and the critical lesson.
        """
        import json
        import re
        import copy
        import numpy as np
        
        def to_numpy(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        print("text_actions for distillation:", text_actions)
            
        
        # 1. Initialize Containers
        distill_rewards = [] # This is the immediate reward for the distillation step itself (e.g. self-consistency)
        current_scores = np.zeros(self.batch_size) # The raw potential (phi)
        raw_improvements = np.zeros(self.batch_size) # The raw I (improvement)
        is_won_array = np.zeros(self.batch_size, dtype=bool)
        self._distill_u_hat_values = []
        self._distill_r_values = []

        # Ensure batch_previous_potentials is synced
        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        # 2. Calculate Raw Scores (Phi) and Raw Improvements (I)
        for i, strategy_text in enumerate(text_actions):
            task_desc = self.init_states[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""
            
            # Get the baseline (Phi_{t-1})
            prev_phi = self.batch_previous_potentials[i]
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0
            
            try:
                # --- JSON Extraction ---
                json_str = ""       
                # 1. Isolate the content AFTER the <think> block to avoid parsing errors
                # (e.g., preventing the parser from grabbing a '{' inside the reasoning text)
                content_to_parse = strategy_text
                if "</think>" in strategy_text:
                    content_to_parse = strategy_text.split("</think>")[-1]
                
                # 2. Try to find a Markdown JSON code block
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', strategy_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = strategy_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]
                
                if not json_str: raise ValueError("No JSON found")
                
                distill_data = json.loads(json_str)

                # --- Subtask Scoring (Phi calculation) ---
                subtasks = distill_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for task in subtasks 
                    if isinstance(task, dict) and task.get('status', '').strip().lower() == 'completed'
                )
                
                # Calculate subtask-based potential
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
                
                # --- DETERMINE CURRENT PHI ---
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                # --- Reflection Consistency Reward (Auxiliary) ---
                predicted_success = distill_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']

                # --- Distillation reward (mutually exclusive modes) ---
                if self.distill_reward_type == 'first_order_diff':
                    retrieved_items = self.retrieved_raw_skills[i] if i < len(self.retrieved_raw_skills) else []
                    u_scores = [item.get('utility_score', 0.5) for item in retrieved_items] if retrieved_items else []
                    u_hat = (max(u_scores) if self.u_hat_aggregation == 'max' else sum(u_scores) / len(u_scores)) if u_scores else 0.0
                    r_tau = 1.0 if actual_success else 0.0
                    current_reward = r_tau - u_hat
                    self._distill_u_hat_values.append(u_hat)
                    self._distill_r_values.append(current_reward)
                else:
                    current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                distill_rewards.append(current_reward)

                # --- Memory Saving Logic ---
                should_save = (actual_success and json_str) if self.distill_reward_type == 'first_order_diff' else (predicted_success == actual_success and json_str)
                if should_save:
                    next_priority = distill_data.get('next_priority')
                    lessons_to_save = []
                    if next_priority and len(str(next_priority)) > 5:
                        lessons_to_save.append(f"New Plan: {next_priority}")

                    if lessons_to_save:
                        final_lesson = " | ".join(lessons_to_save)
                        self.skill_library.admit(
                            scenario_description=task_desc,
                            strategy_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio
                        )
            except Exception as e:
                print(f"Error task {i}: {e}")
                distill_rewards.append(0.0)
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0

            # --- Calculate Raw Improvement (I) ---
            current_scores[i] = current_phi
            # Improvement is strictly positive gain over history
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement

        # 3. Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n

            task_desc = self.init_states[start_idx]

            # Extract group data
            group_improvements = raw_improvements[start_idx:end_idx]
            # group_scores = current_scores[start_idx:end_idx]

            # A. Normalization (Centering)
            # As per LaTeX Eq (8): R_int = I - Mean(I)
            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                # Note: We do NOT divide by std here, just centering is sufficient
                # to maintain the zero-sum property for the intrinsic component.
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)

            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate
            # B. Update Historical Baseline (EMA)
            # As per LaTeX Eq (9): Phi_t = gamma * Phi_{t-1} + (1-gamma) * Mean(Phi_t)
            # if len(group_scores) > 0:
            #     current_group_mean_score = np.mean(group_scores)
            #     old_baseline = self.task_potential_history.get(task_desc, 0.0)

            #     if current_group_mean_score > old_baseline:
            #         self.task_potential_history[task_desc] = current_group_mean_score
                # # EMA Update
                # new_baseline = (self.ema_gamma * old_baseline) + ((1 - self.ema_gamma) * current_group_mean_score)
                # self.task_potential_history[task_desc] = new_baseline

        print("raw_improvements: ", raw_improvements)
        print("final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)

        return None, to_numpy(distill_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            # Inject skills into the prompt
            skills_str = self.current_skills[i] if hasattr(self, 'current_skills') else ""
            
            if init or self.config.env.history_length <= 0:
                if self.is_multi_modal:
                    # For visual, we might append skills to the system prompt elsewhere, 
                    # or assume the model handles visual + text context.
                    obs = SOKOBAN_VISUAL_TEMPLATE 
                else:
                    obs = SOKOBAN_TEMPLATE_NO_HIS.format(
                        skills=skills_str, 
                        current_observation=text_obs[i]
                    )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        skills=skills_str,
                        current_observation=text_obs[i]
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def build_distill_text_obs(self, infos: List[Dict]) -> List[str]:
        """
        Builds the text observation for the distillation phase.
        """
        postprocess_text_obs = []
        memory_contexts, valid_lens = self.memory.fetch(
            15, # Sokoban games can be long
            obs_key="text_obs",
            action_key="action"
        )

        # self.task_trajectory_history[task] = {"successful": [], "failed": []}
        for i in range(len(infos)):
            task = self.init_states[i]
            # Ensure key exists (it should from reset, but safety first)
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
                
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])
        # --- CRITICAL: Store these so step_distill can access them ---
        self.last_trajectories = memory_contexts

        for i in range(len(infos)):
            task = self.init_states[i]
            is_won = infos[i].get("won", False)
            reference_traj_str = "No reference history available yet."
            if is_won:
                SUCCESS = "successfully"
                # Try to get a failed example
                failed_hist = self.task_trajectory_history[task]["failed"]
                if failed_hist:
                    # Use the most recent failure
                    reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                else:
                    reference_traj_str = "No failed attempts available for comparison."
            else:
                SUCCESS = "unsuccessfully" # Changed from "NOT successfully" for better grammar
                # Try to get a successful example
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    # Use the most recent success
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."
            # If multi-modal, memory_contexts[i] might contain image arrays which we can't print to text.
            # We need to sanitize the history for the distillation prompt.
            history_str = ""
            if self.is_multi_modal:
                # If visual, we rely on the action history primarily
                # We assume memory_contexts returns a list of (obs, action) or similar.
                # Since SimpleMemory.fetch usually returns formatted strings if configured, 
                # we might need to manually reconstruct the action sequence here.
                
                # Fallback: just list actions
                actions = self.memory.get_all_actions(i) # Hypothetical helper or manual access
                # If get_all_actions doesn't exist, we rely on what fetch returned.
                # If fetch returned images, we skip them.
                
                # Assuming memory_contexts[i] is a string representation of history:
                if isinstance(memory_contexts[i], str):
                    history_str = memory_contexts[i]
                else:
                    # If it's not string (e.g. list of images), we construct a simple action log
                    raw_actions = self.memory[i]['action'] # Access raw storage
                    history_str = " -> ".join([str(a) for a in raw_actions])
            else:
                history_str = memory_contexts[i]
            obs = SOKOBAN_DISTILL_TEMPLATE.format(
                success=SUCCESS,
                reference_trajectory=reference_traj_str,
                current_trajectory=history_str
            )
            postprocess_text_obs.append(obs)
        
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        from collections import defaultdict
        
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        distill_rewards = kwargs.get('distill_rewards', None)
        
        batch_size = len(total_batch_list)
        success = defaultdict(list)
        
        for bs in range(batch_size):
            r_reward = 0.0
            if distill_rewards is not None and bs < len(distill_rewards):
                try:
                    r_reward = float(distill_rewards[bs])
                except:
                    r_reward = 0.0
            
            success['distill_success_rate'].append(r_reward)
            
            # Process play phase
            play_success_found = False
            trajectory = total_batch_list[bs]
            n_prepended = sum(1 for s in trajectory if s.get('phase') in ('query', 'rerank'))

            for i in reversed(range(len(trajectory))):
                batch_item = trajectory[i]
                if not batch_item.get('active_masks', True): continue

                phase = batch_item.get('phase', 'play')
                if phase == 'play':
                    info = total_infos[bs][i - n_prepended]
                    won_value = float(info.get('won', 0.0))

                    success['play_success_rate'].append(won_value)
                    success['success_rate'].append(won_value)
                    play_success_found = True
                    break

            if not play_success_found:
                success['play_success_rate'].append(0.0)
                success['success_rate'].append(0.0)

        return {key: np.array(value) for key, value in success.items()}

class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs

class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, retrieve_type):
        # print("config: ", config)
        self.memory = WebshopSimpleMemory()
        self.group_n = config.env.rollout.n  # e.g., 8
        
        # --- Extract Hyperparameters from Config ---
        mem_config = config.env.get('skill_library', {})
        filepath = mem_config.get('filepath', "skill_library.json")
        import os
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        relevance_weight = mem_config.get('relevance_weight', 0.7)
        alpha = mem_config.get('alpha', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)
        
        # --- NEW: Memory Start Cutoff Configuration ---
        # Memory retrieval starts only when progress > memory_start_cutoff
        # Default is 0.0 (start immediately)
        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0) 
        self.current_progress_ratio = 0.0 # Track progress internally
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        assert not (self.full_group_memory and self.group_outperformance), \
            "group_outperformance requires split group (full_group_memory must be False)"
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)
        self.distill_reward_type = mem_config.get('distill_reward_type', 'self_assess')
        self.u_hat_aggregation = mem_config.get('u_hat_aggregation', 'max')
        self.success_reference_induction = mem_config.get('success_reference_induction', False)

        # --- NEW: Potential Calculation Config ---
        # If True, current_phi is strictly 1.0 (won) or 0.0 (lost).
        # If False, current_phi is calculated via subtask completion % from JSON, overridden by 1.0 if won.
        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        # --- NEW: Config to only give memory to 1 agent per group ---
        # If True, 1 agent retrieves, (group_n - 1) agents are control.
        # If False, (group_n / 2) agents retrieve, (group_n / 2) are control.
        self.single_distill_per_group = mem_config.get('single_distill_per_group', False)
        # --- NEW: Reflection Decay Configuration ---
        # If True, gradually reduce the number of skill-receiving agents
        # from half-group (at progress=0) to 0 (at progress=1).
        # This overrides full_group_memory and single_distill_per_group during training.
        self.skill_decay = mem_config.get('skill_decay', False)
        # Optional: define the progress point at which skills fully vanish.
        # Default is 1.0 (skills reach 0 at the end of training).
        self.skill_decay_end = mem_config.get('skill_decay_end', 1.0)
        # EMA Decay rate for the baseline (matches LaTeX gamma)
        self.ema_gamma = 0.9
        # --- Skill1 Config Flags ---
        self.enable_query_generation = mem_config.get('enable_query_generation', False)
        self.enable_description_head = mem_config.get('enable_description_head', False)
        self.enable_rerank = mem_config.get('enable_rerank', False)
        print("memory retrieve_type: ", retrieve_type)
        print("memory retrieve_mode: ", self.retrieve_mode)
        print("top_k_retrieved_memory: ", self.top_k)
        print(f"Memory Start Cutoff: {self.memory_start_cutoff}")
        print(f"Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"Single Distill Per Group: {self.single_distill_per_group}")
        print(f"Reflection Decay Enabled: {self.skill_decay}")
        print(f"Reflection Decay End Point: {self.skill_decay_end}")
        print(f"Potential Based On Binary Success Only: {self.potential_based_on_binary_success}")
        print(f"Skill1 Query Generation: {self.enable_query_generation}")
        print(f"Skill1 Description Head: {self.enable_description_head}")
        self.rerank_train_top1 = mem_config.get('rerank_train_top1', False)
        self.retriever_type = mem_config.get('retriever_type', 'dense')
        print(f"Skill1 Re-rank: {self.enable_rerank}")
        print(f"Rerank Train Top-1 Selection: {self.rerank_train_top1}")
        print(f"Retriever Type: {self.retriever_type}")

        self._is_eval = False

        # Initialize the persistent skill library
        self.skill_library = SkillLibrary(
            filepath=filepath,
            relevance_weight=relevance_weight,
            alpha=alpha,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale,
            use_description_head=self.enable_description_head,
            max_size=mem_config.get('max_size', 5000),
            retriever_type=self.retriever_type,
        )
        self.task_trajectory_history = {}
        self.task_potential_history = {} 
        self.batch_previous_potentials = [] 
        
        # Initialize containers for retrieval tracking
        self.current_skills = []      # Formatted strings for the prompt
        self.retrieved_raw_skills = [] # List of LISTS OF DICTS (updated structure)  
        # --- NEW: Track the type of retrieval for the current batch ---
        self.current_retrieval_types = []
        # Store the trajectories generated during the distillation phase
        # so they can be saved to memory in step_distill
        self.last_trajectories = []        
        super().__init__(envs, projection_f, config)

    def update_training_progress(self, current_step: int, total_steps: int):
        """
        Updates the environment with the current training progress.
        This triggers memory pruning if a 20% milestone is reached.
        """
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            self.skill_library._current_training_step = current_step
            # Pass the ratio to memory to check for pruning triggers
            # self.skill_library.check_and_prune(progress_ratio=self.current_progress_ratio, top_k=3)

    def _compute_group_split_index(self, is_eval: bool) -> int:
        """
        Computes the group_split_index determining how many agents per group
        receive skills. Agents at positions >= group_split_index get skills.

        - Default (no decay): half the group receives skills (split_index = group_n // 2).
        - full_group_memory: all agents receive skills (split_index = 0).
        - skill_decay (training only): linearly reduces the number of skills
          agents from half-group to 0 as progress goes from 0 to skill_decay_end.
        """
        half_group = self.group_n // 2
        # --- Reflection Decay Logic (training only) ---
        if self.skill_decay:
            # Clamp the decay progress ratio to [0, 1]
            if self.skill_decay_end > 0:
                decay_ratio = min(self.current_progress_ratio / self.skill_decay_end, 1.0)
            else:
                decay_ratio = 1.0  # If end is 0, immediately decay to 0

            # Number of agents that should receive skills:
            # Linearly from half_group (at decay_ratio=0) to 0 (at decay_ratio=1)
            num_skill_agents = max(0, round(half_group * (1.0 - decay_ratio)))

            # group_split_index = group_n - num_skill_agents
            # e.g., group_n=8, half=4, decay_ratio=0.5 -> agents=2 -> split=6
            group_split_index = self.group_n - num_skill_agents

            print(
                f"[Reflection Decay] progress={self.current_progress_ratio:.3f}, "
                f"decay_ratio={decay_ratio:.3f}, "
                f"skill_agents_per_group={num_skill_agents}/{self.group_n}, "
                f"group_split_index={group_split_index}"
            )
            return group_split_index

        # --- Standard (non-decay) Logic ---
        if self.full_group_memory:
            return 0
        return half_group

    def reset(self, kwargs) -> Dict[str, Any]:
        # 1. Check the flag. Default to False if not provided.
        if kwargs is None:
            kwargs = {}
        print("****** environment resetting ******")
        # Determine mode based on kwargs
        is_eval = not kwargs.get('is_train', True)
        self._is_eval = is_eval

        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)

        obs = self.format_obs(obs)
        self.initial_infos = infos  # store for rebuild_initial_obs
        observations = {
            'text': self.build_text_obs(obs, infos, init=True),
            'image': None,
            'anchor': obs.copy()
        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size=len(infos))
        self.batch_size = len(obs)
        assert self.batch_size % self.group_n == 0, "Batch size must be divisible by group size"
        self.num_unique_tasks = self.batch_size // self.group_n

        self.current_skills = []
        self.retrieved_raw_skills = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = [] 
        self.batch_retrieved_types = [] # Reset the type tracker
        # # Example: group_n = 8. split_index = 4.
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        # --- NEW: Check if we have passed the cutoff ---
        # If we are training AND progress <= cutoff, we are in warmup -> Force memory OFF.
        # If progress > cutoff, we allow memory logic to proceed.
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)
        
        if in_warmup_period:
            # Optional: Log occasionally if needed
            print(f"Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")
            pass 

        # --- Use the new helper to compute the split index ---
        # group_split_index = self._compute_group_split_index(is_eval)
        for i, task in enumerate(self.tasks):
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_skills = ""
            raw_list_of_dicts = [] # This will hold [{'text':..., 'type':...}]
            current_types_list = [] # List to hold types for this specific agent
            
            should_retrieve = False
            retrieval_type_str = "control"
            
            if self.enable_memory:
                if in_warmup_period:
                    # Explicitly disable retrieval during warmup
                    should_retrieve = False
                elif is_eval:
                    # During Eval: Everyone retrieves (or based on config)
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False
            else:
                should_retrieve = False

            if should_retrieve:
                # Retrieve top_k items (use full top_k when rerank is enabled)
                k = self.top_k

                raw_list_of_dicts = self.skill_library.retrieve(
                    current_scenario_description=task,
                    top_k=k,
                    filter_type=self.retrieve_mode
                )

                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        current_types_list.append(r_type)
                        formatted_lines.append(r_text)

                    formatted_skills = "Relevant skills from the skill library:\n" + "\n".join(formatted_lines)
                    formatted_skills += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."

            self.current_skills.append(formatted_skills)
            self.retrieved_raw_skills.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            infos[i]['distill_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str
            print("infos[i]['retrieval_group']: ", infos[i]['retrieval_group'])
        # Debug prints
        # print("retrieved_raw_skills: ", self.retrieved_raw_skills)
        # exit(0)
        assert len(self.current_skills) == len(self.tasks)
        return observations, infos

    def distill(self, infos: List[Dict]):
        """
        Called at the end of the 'play' phase.
        Updates utility based on Group B (Retrieved) vs Group A (Not Retrieved) performance.
        """
        # Build observation creates self.last_trajectories side-effect
        distill_obs_text = self.build_distill_text_obs(infos)
        
        observations = {
            'text': distill_obs_text,
            'image': None,
            'anchor': distill_obs_text
        }

        # Mark actions as valid for the distillation phase
        for info in infos:
            info['is_action_valid'] = to_numpy(True)

        # BiGen: query contrastive reward = exp_success_rate - ctrl_success_rate
        self.query_contrastive_rewards = np.zeros(len(self.tasks), dtype=np.float32)
        self._bigen_ctrl_success_rates = []
        self._bigen_exp_success_rates = []
        self._bigen_selection_hits = 0
        self._bigen_selection_total = 0

        batch_size = len(self.tasks)
        if batch_size % self.group_n != 0:
            print(f"WARNING: Batch size {batch_size} not divisible by group_n {self.group_n}")

        num_groups = batch_size // self.group_n
        group_split_index = 0 if self.full_group_memory else self.group_n // 2

        # Iterate over each group independently
        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index

            # 1. Calculate Wins for Control (First half)
            control_wins = 0
            for i in range(start_idx, mid_idx):
                if infos[i].get("won", False):
                    control_wins += 1

            # 2. Calculate Wins for Experiment (Second half)
            experiment_wins = 0
            for i in range(mid_idx, end_idx):
                if infos[i].get("won", False):
                    experiment_wins += 1

            # 3. Determine Utility Score for THIS group
            group_outperformed = experiment_wins > control_wins

            n_ctrl = max(group_split_index, 1)
            n_exp = max(self.group_n - group_split_index, 1)
            self._bigen_ctrl_success_rates.append(control_wins / n_ctrl)
            self._bigen_exp_success_rates.append(experiment_wins / n_exp)

            # 3b. BiGen: contrastive reward for experiment agents' query steps
            if self.enable_query_generation:
                contrastive = (experiment_wins / n_exp) - (control_wins / n_ctrl)
                for i in range(mid_idx, end_idx):
                    self.query_contrastive_rewards[i] = contrastive
                print(f"[BiGen] Group {g}: ctrl={control_wins/n_ctrl:.2f}, exp={experiment_wins/n_exp:.2f}, q_reward={contrastive:.2f}")

            # 3c. Selection precision
            for i in range(mid_idx, end_idx):
                if self.retrieved_raw_skills[i]:
                    self._bigen_selection_total += 1
                    if infos[i].get("won", False) and group_outperformed:
                        self._bigen_selection_hits += 1

            # 4. Update Memory ONLY for the Experiment agents
            for i in range(mid_idx, end_idx):
                task_desc = self.tasks[i]
                raw_skill_items = self.retrieved_raw_skills[i] # List[Dict]
                is_success = infos[i].get("won", False)

                # 1.0 if the agent won AND the retrieval group beat the control group.
                if self.group_outperformance:
                    if is_success and group_outperformed:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
                else:
                    if is_success:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0


                if raw_skill_items:
                    for item in raw_skill_items:
                        # --- UPDATED: Extract text from dict for utility update ---
                        strategy_text = item.get('text', '')
                        if strategy_text:
                            self.skill_library.update_utility(
                                scenario_description=task_desc, 
                                strategy_text=strategy_text, 
                                score=utility_score
                            )

        return observations, infos

    def step_distill(self, text_actions: List[str], infos: List[Dict]):
        """
        Calculates intrinsic rewards based on improvement over an EMA baseline,
        normalizes them group-wise, and updates the baseline.
        """
        import json
        import re
        import copy
        import numpy as np
        
        def to_numpy(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        print("text_actions for distillation:", text_actions)
        
        # 1. Initialize Containers
        distill_rewards = [] # This is the immediate reward for the distillation step itself (e.g. self-consistency)
        current_scores = np.zeros(self.batch_size) # The raw potential (phi)
        raw_improvements = np.zeros(self.batch_size) # The raw I (improvement)
        is_won_array = np.zeros(self.batch_size, dtype=bool)
        self._distill_correct_count = 0
        self._distill_total_count = 0
        self._distill_ops = []  # Track distillation operation per rollout: "add" or "none"
        self._distill_u_hat_values = []
        self._distill_r_values = []

        # Ensure batch_previous_potentials is synced
        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        # 2. Calculate Raw Scores (Phi) and Raw Improvements (I)
        for i, strategy_text in enumerate(text_actions):
            task_desc = self.tasks[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""

            # Get the baseline (Phi_{t-1})
            prev_phi = self.batch_previous_potentials[i]
            # --- Extract Actual Success First ---
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0

            # ... (JSON Parsing Logic - same as before) ...
            try:
                # --- JSON Extraction ---
                json_str = ""
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', strategy_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = strategy_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]

                if not json_str: raise ValueError("No JSON found")

                distill_data = json.loads(json_str)

                # --- Subtask Scoring (Phi calculation) ---
                subtasks = distill_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for task in subtasks
                    if isinstance(task, dict) and task.get('status', '').strip().lower() == 'completed'
                )

                # Calculate subtask-based potential
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0

                # --- DETERMINE CURRENT PHI ---
                if self.potential_based_on_binary_success:
                    # STRICT MODE: Only actual success matters for potential
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    # DEFAULT MODE: Use subtasks, but override if actual success
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                # --- Reflection Consistency Reward (Auxiliary) ---
                predicted_success = distill_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']

                # --- Distillation reward (mutually exclusive modes) ---
                if self.distill_reward_type == 'first_order_diff':
                    retrieved_items = self.retrieved_raw_skills[i] if i < len(self.retrieved_raw_skills) else []
                    u_scores = [item.get('utility_score', 0.5) for item in retrieved_items] if retrieved_items else []
                    u_hat = (max(u_scores) if self.u_hat_aggregation == 'max' else sum(u_scores) / len(u_scores)) if u_scores else 0.0
                    r_tau = 1.0 if actual_success else 0.0
                    current_reward = r_tau - u_hat
                    self._distill_u_hat_values.append(u_hat)
                    self._distill_r_values.append(current_reward)
                else:
                    current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                distill_rewards.append(current_reward)
                self._distill_total_count += 1
                if predicted_success == actual_success:
                    self._distill_correct_count += 1

                # --- Memory Saving Logic (Same as before) ---
                distill_op = "none"
                should_save = (actual_success and json_str) if self.distill_reward_type == 'first_order_diff' else (predicted_success == actual_success and json_str)
                if should_save:
                    action_lesson = distill_data.get('action_lesson')
                    nav_lesson = distill_data.get('navigation_lesson')
                    description_head = (distill_data.get('description_head') or '') if self.enable_description_head else ''
                    if self.enable_description_head:
                        if not hasattr(self, '_bigen_desc_head_total'):
                            self._bigen_desc_head_total = 0
                            self._bigen_desc_head_parsed = 0
                            self._bigen_desc_head_saved = 0
                        self._bigen_desc_head_total += 1
                        if description_head and len(str(description_head).strip()) > 5:
                            self._bigen_desc_head_parsed += 1
                    lessons_to_save = []
                    if action_lesson and len(str(action_lesson)) > 5: lessons_to_save.append(f"Action Insight: {action_lesson}")
                    if nav_lesson and len(str(nav_lesson)) > 5: lessons_to_save.append(f"Navigation Insight: {nav_lesson}")

                    if lessons_to_save:
                        distill_op = "add"
                        final_lesson = " | ".join(lessons_to_save)
                        self.skill_library.admit(
                            scenario_description=task_desc,
                            strategy_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio,
                            description_head=str(description_head) if description_head else ''
                        )
                        if self.enable_description_head and description_head and len(str(description_head).strip()) > 5:
                            self._bigen_desc_head_saved += 1
                self._distill_ops.append(distill_op)

            except Exception as e:
                print(f"Error task {i}: {e}")
                distill_rewards.append(0.0)
                self._distill_ops.append("none")
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0

            # --- Calculate Raw Improvement (I) ---
            current_scores[i] = current_phi
            # Improvement is strictly positive gain over history
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement

        # 3. Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            
            task_desc = self.tasks[start_idx]
            
            # Extract group data
            group_improvements = raw_improvements[start_idx:end_idx]
            # group_scores = current_scores[start_idx:end_idx]
            
            # A. Normalization (Centering)
            # As per LaTeX Eq (8): R_int = I - Mean(I)
            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                # Note: We do NOT divide by std here, just centering is sufficient 
                # to maintain the zero-sum property for the intrinsic component.
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)

            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate
            # B. Update Historical Baseline (EMA)
            # As per LaTeX Eq (9): Phi_t = gamma * Phi_{t-1} + (1-gamma) * Mean(Phi_t)
            # if len(group_scores) > 0:
            #     current_group_mean_score = np.mean(group_scores)
            #     old_baseline = self.task_potential_history.get(task_desc, 0.0)

            #     if current_group_mean_score > old_baseline:
            #         self.task_potential_history[task_desc] = current_group_mean_score
                # # EMA Update
                # new_baseline = (self.ema_gamma * old_baseline) + ((1 - self.ema_gamma) * current_group_mean_score)
                # self.task_potential_history[task_desc] = new_baseline

        print("raw_improvements: ", raw_improvements)
        print("final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        # Convert to numpy for compatibility
        return None, to_numpy(distill_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions, 'reward': rewards, 'dones': dones, 'won': [info['won'] for info in infos]})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            if len(parts) > 2 and parts[1] == 'Instruction:':
                tasks.append(parts[2])
            else:
                tasks.append(obs)
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except (ValueError, IndexError):
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []
        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        postprocess_text_obs = []
        
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
        else:
            memory_contexts = [""] * len(text_obs)
            valid_lens = [0] * len(text_obs)
            
        for i in range(len(text_obs)):
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)
            
            if i < len(self.current_skills):
                skill_context = self.current_skills[i]
            else:
                skill_context = ""

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    skills=skill_context, 
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    skills=skill_context, 
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                
                if len(obs) > 13000:
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        skills=skill_context, 
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def build_distill_text_obs(self, infos: List[str]) -> List[str]:
        postprocess_text_obs = []
        # memory_contexts, valid_lens = self.memory.fetch(
        #         15,
        #         obs_key="text_obs",
        #         action_key="action")
        memory_contexts, valid_lens = self.memory.fetch(
                15,
                obs_key="text_obs",
                action_key="action",
                max_to_show=6)
        # self.task_trajectory_history[task] = {"successful": [], "failed": []}
        for i in range(len(infos)):
            task = self.tasks[i]
            # Ensure key exists (it should from reset, but safety first)
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
                
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])

        # --- CRITICAL: Store these so step_distill can access them ---
        self.last_trajectories = memory_contexts
        for i in range(len(infos)):
            task = self.tasks[i]
            is_won = infos[i].get("won", False)
            
            # Determine success string and select Contrastive Reference
            # If we WON, we want to see a FAIL to understand what to avoid (or just compare)
            # If we LOST, we want to see a SUCCESS to understand what to do
            
            reference_traj_str = "No reference history available yet."
            if self.success_reference_induction:
                if is_won:
                    SUCCESS = "successfully"
                else: 
                    SUCCESS = "unsuccessfully"
                # Try to get a successful example
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    # Use the most recent success
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."
            else:
                if is_won:
                    SUCCESS = "successfully"
                    # Try to get a failed example
                    failed_hist = self.task_trajectory_history[task]["failed"]
                    if failed_hist:
                        # Use the most recent failure
                        reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                    else:
                        reference_traj_str = "No failed attempts available for comparison."
                else:
                    SUCCESS = "unsuccessfully" # Changed from "NOT successfully" for better grammar
                    # Try to get a successful example
                    success_hist = self.task_trajectory_history[task]["successful"]
                    if success_hist:
                        # Use the most recent success
                        reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                    else:
                        reference_traj_str = "No successful attempts available for reference."

            distill_tmpl = WEBSHOP_DISTILL_TEMPLATE_WITH_DESC_HEAD if self.enable_description_head else WEBSHOP_DISTILL_TEMPLATE
            obs = distill_tmpl.format(
                task_description=task,
                success=SUCCESS,
                reference_trajectory=reference_traj_str,
                current_trajectory=memory_contexts[i]
            )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    # ---- Skill1: Query Generation, Re-rank, Description Head Methods ----

    def build_query_generation_obs(self) -> Dict[str, Any]:
        """Build observations for query generation phase (Phase 0)."""
        from agent_system.environments.prompts.webshop import WEBSHOP_QUERY_GENERATION_TEMPLATE
        query_obs_texts = []
        for i, task in enumerate(self.tasks):
            obs_text = WEBSHOP_QUERY_GENERATION_TEMPLATE.format(
                task_description=task,
                initial_observation=self.pre_text_obs[i]
            )
            query_obs_texts.append(obs_text)
        return {'text': query_obs_texts, 'image': None, 'anchor': query_obs_texts}

    def apply_generated_queries(self, query_texts: List[str]):
        """Use actor-generated queries to re-retrieve from memory, only for experiment group."""
        import re as _re
        for i, query_text in enumerate(query_texts):
            if self.current_retrieval_types[i] not in ("experiment", "eval_retrieval"):
                continue
            match = _re.search(r'<query>(.*?)</query>', query_text, _re.DOTALL)
            parsed = match.group(1).strip() if match else query_text.strip()
            if not parsed:
                continue
            raw_list_of_dicts = self.skill_library.retrieve(
                current_scenario_description=parsed,
                top_k=self.top_k,
                filter_type=self.retrieve_mode
            )
            if raw_list_of_dicts:
                formatted_lines = [item.get('text', '') for item in raw_list_of_dicts]
                self.current_skills[i] = (
                    "Relevant skills from the skill library:\n" + "\n".join(formatted_lines)
                    + "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
                )
                self.retrieved_raw_skills[i] = raw_list_of_dicts

    def rebuild_initial_obs(self):
        """Rebuild initial play observation after query generation / rerank updated skills."""
        full_text_obs = self.build_text_obs(self.pre_text_obs, self.initial_infos, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': self.pre_text_obs.copy()}

    def build_rerank_obs(self) -> Dict[str, Any]:
        """Build observations for the re-rank phase."""
        from agent_system.environments.prompts.webshop import WEBSHOP_RERANK_TEMPLATE, WEBSHOP_RERANK_DUMMY_TEMPLATE
        rerank_obs_texts = []
        self.rerank_candidates = []
        for i, task in enumerate(self.tasks):
            raw_items = self.retrieved_raw_skills[i]
            candidates_with_scores = self._get_candidate_scores(raw_items)
            if len(candidates_with_scores) >= 2:
                candidate_lines = []
                for idx, cand in enumerate(candidates_with_scores):
                    candidate_lines.append(f"[Experience {idx + 1}]: {cand['text']}")
                candidate_str = "\n\n".join(candidate_lines)
                obs_text = WEBSHOP_RERANK_TEMPLATE.format(
                    task_description=task,
                    initial_observation=self.pre_text_obs[i],
                    n_candidates=len(candidates_with_scores),
                    candidate_experiences=candidate_str,
                )
            else:
                obs_text = WEBSHOP_RERANK_DUMMY_TEMPLATE.format(
                    task_description=task,
                    initial_observation=self.pre_text_obs[i],
                )
            rerank_obs_texts.append(obs_text)
            self.rerank_candidates.append(candidates_with_scores)
        return {'text': rerank_obs_texts, 'image': None, 'anchor': rerank_obs_texts}

    def _get_candidate_scores(self, raw_items: List[Dict]) -> List[Dict]:
        """Look up utility_score from skill_library for each retrieved item."""
        result = []
        for item in raw_items:
            r_text = item.get('text', '')
            utility = 0.5
            for mem_entry in self.skill_library.data:
                if mem_entry['strategy'] == r_text:
                    utility = mem_entry.get('utility_score', 0.5)
                    break
            result.append({'text': r_text, 'type': item.get('type', 'unknown'), 'utility': utility})
        return result

    def apply_rerank_results(self, text_actions: List[str]):
        """Parse model's ranking output and reorder skills accordingly."""
        import re as _re
        self._rerank_parse_successes = 0
        self._rerank_total = 0
        self._rerank_predicted_orders = []
        for i, action_text in enumerate(text_actions):
            candidates = self.rerank_candidates[i]
            n = len(candidates)
            self._rerank_total += 1
            if n < 2:
                self._rerank_predicted_orders.append(None)
                continue
            match = _re.search(r'<rank>(.*?)</rank>', action_text, _re.DOTALL)
            if not match:
                self._rerank_predicted_orders.append(None)
                continue
            rank_str = match.group(1).strip()
            if rank_str.lower() == 'none':
                self._rerank_predicted_orders.append(None)
                continue
            parsed_ids = []
            for token in rank_str.split(','):
                token = token.strip()
                if token.isdigit():
                    idx = int(token) - 1
                    if 0 <= idx < n and idx not in parsed_ids:
                        parsed_ids.append(idx)
            if len(parsed_ids) == 0:
                self._rerank_predicted_orders.append(None)
                continue
            self._rerank_parse_successes += 1
            self._rerank_predicted_orders.append(parsed_ids)
            remaining = [j for j in range(n) if j not in parsed_ids]
            full_order = parsed_ids + remaining
            reordered_texts = [candidates[j]['text'] for j in full_order]

            if self.rerank_train_top1 and not self._is_eval:
                reordered_texts = reordered_texts[:1]

            formatted = "Relevant skills from the skill library:\n" + "\n".join(reordered_texts)
            formatted += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            self.current_skills[i] = formatted

    def compute_rerank_rewards(self) -> np.ndarray:
        """Compute NDCG-based rewards for the re-rank phase."""
        rewards = np.zeros(len(self.tasks), dtype=np.float32)
        for i in range(len(self.tasks)):
            candidates = self.rerank_candidates[i]
            predicted_order = self._rerank_predicted_orders[i]
            if predicted_order is None or len(candidates) < 2:
                rewards[i] = 0.0
                continue
            utilities = np.array([c['utility'] for c in candidates])
            ideal_order = np.argsort(-utilities)
            ideal_gains = utilities[ideal_order]
            remaining = [j for j in range(len(candidates)) if j not in predicted_order]
            full_predicted = predicted_order + remaining
            predicted_gains = utilities[full_predicted]
            def _dcg(gains):
                positions = np.arange(len(gains), dtype=np.float64) + 2.0
                return np.sum(gains / np.log2(positions))
            ideal_dcg = _dcg(ideal_gains)
            if ideal_dcg < 1e-8:
                rewards[i] = 0.0
                continue
            predicted_dcg = _dcg(predicted_gains)
            ndcg = predicted_dcg / ideal_dcg
            rewards[i] = float(ndcg)
        self._rerank_rewards = rewards
        return rewards

    def get_bigen_metrics(self, prefix: str = "bigen") -> Dict[str, float]:
        """Collect all Skill1 metrics for wandb logging.
        Key names never contain 'success_rate' (reserved by metric_utils).
        """
        p = prefix
        metrics = {}
        # --- Utility distribution (own wandb section) ---
        mem_data = self.skill_library.data
        if mem_data:
            util_scores = [float(entry.get('utility_score', 0.5)) for entry in mem_data]
            n = len(util_scores)
            for bin_lo in range(10):
                lo = bin_lo / 10.0
                hi = lo + 0.1
                if bin_lo == 9:
                    cnt = sum(1 for u in util_scores if lo <= u <= hi)
                else:
                    cnt = sum(1 for u in util_scores if lo <= u < hi)
                metrics[f'{p}/utility_dist/{bin_lo*10:02d}'] = float(cnt) / n

        # --- skill_statics: memory health & co-evolution metrics ---
        s = 'skill_statics'
        buf_size = getattr(self, '_bigen_memory_buffer_size', len(mem_data))
        metrics[f'{s}/memory_buffer_size'] = float(buf_size)
        if mem_data:
            counts = [float(entry.get('count', 1)) for entry in mem_data]
            metrics[f'{s}/library_quality'] = sum(util_scores) / n
            metrics[f'{s}/memory_avg_count'] = sum(counts) / n
            metrics[f'{s}/memory_low_utility_frac'] = sum(1 for u in util_scores if u < 0.3) / n
            total_chars = sum(len(entry.get('strategy', '')) + len(entry.get('trajectory', '')) for entry in mem_data)
            metrics[f'{s}/memory_total_chars'] = float(total_chars)
        metrics[f'{s}/memory_evicted_count'] = float(getattr(self.skill_library, '_last_evicted_count', 0))
        ctrl_rates = getattr(self, '_bigen_ctrl_success_rates', [])
        exp_rates = getattr(self, '_bigen_exp_success_rates', [])
        if ctrl_rates:
            metrics[f'{s}/ctrl_win_rate/mean'] = float(np.mean(ctrl_rates))
            metrics[f'{s}/exp_win_rate/mean'] = float(np.mean(exp_rates))
            metrics[f'{s}/exp_minus_ctrl/mean'] = float(np.mean(exp_rates)) - float(np.mean(ctrl_rates))
        sel_total = getattr(self, '_bigen_selection_total', 0)
        sel_hits = getattr(self, '_bigen_selection_hits', 0)
        if sel_total > 0:
            metrics[f'{s}/selection_precision'] = float(sel_hits) / float(sel_total)
        ref_total = getattr(self, '_distill_total_count', 0)
        ref_correct = getattr(self, '_distill_correct_count', 0)
        if ref_total > 0:
            metrics[f'{s}/self_assess_accuracy'] = float(ref_correct) / float(ref_total)
        if self.enable_query_generation:
            q_rewards = getattr(self, 'query_contrastive_rewards', None)
            if q_rewards is not None and len(q_rewards) > 0:
                metrics[f'{p}/query_contrastive_reward/mean'] = float(np.mean(q_rewards))
                metrics[f'{p}/query_contrastive_reward/min'] = float(np.min(q_rewards))
                metrics[f'{p}/query_contrastive_reward/max'] = float(np.max(q_rewards))
                exp_mask = q_rewards != 0
                if exp_mask.any():
                    metrics[f'{p}/query_contrastive_reward/exp_mean'] = float(np.mean(q_rewards[exp_mask]))

        if self.enable_description_head:
            total = getattr(self, '_bigen_desc_head_total', 0)
            parsed = getattr(self, '_bigen_desc_head_parsed', 0)
            saved = getattr(self, '_bigen_desc_head_saved', 0)
            metrics[f'{p}/desc_head/total_attempts'] = float(total)
            metrics[f'{p}/desc_head/parsed_count'] = float(parsed)
            metrics[f'{p}/desc_head/saved_count'] = float(saved)
            if total > 0:
                metrics[f'{p}/desc_head/parse_rate'] = float(parsed / total)

        if self.enable_rerank:
            rr_rewards = getattr(self, '_rerank_rewards', None)
            if rr_rewards is not None and len(rr_rewards) > 0:
                metrics[f'{p}/rerank_ndcg/mean'] = float(np.mean(rr_rewards))
                metrics[f'{p}/rerank_ndcg/min'] = float(np.min(rr_rewards))
                metrics[f'{p}/rerank_ndcg/max'] = float(np.max(rr_rewards))
                nonzero_mask = rr_rewards > 0
                if nonzero_mask.any():
                    metrics[f'{p}/rerank_ndcg/active_mean'] = float(np.mean(rr_rewards[nonzero_mask]))
            rr_total = getattr(self, '_rerank_total', 0)
            rr_success = getattr(self, '_rerank_parse_successes', 0)
            metrics[f'{p}/rerank_parse/total'] = float(rr_total)
            metrics[f'{p}/rerank_parse/success'] = float(rr_success)
            if rr_total > 0:
                metrics[f'{p}/rerank_parse/parse_rate'] = float(rr_success / rr_total)

        # --- Selection evolution metrics ---
        retrieved_refs = getattr(self, 'retrieved_raw_skills', [])
        if retrieved_refs:
            from collections import Counter
            import math as _math
            all_skill_ids = []
            per_task_skill_ids = {}
            for idx, ref_list in enumerate(retrieved_refs):
                task_group_idx = idx // self.group_n if self.group_n > 0 else idx
                if task_group_idx not in per_task_skill_ids:
                    per_task_skill_ids[task_group_idx] = []
                for item in ref_list:
                    sid = item.get('skill_id', '') or item.get('text', '')[:50]
                    if sid:
                        all_skill_ids.append(sid)
                        per_task_skill_ids[task_group_idx].append(sid)

            if all_skill_ids:
                counter = Counter(all_skill_ids)
                total_count = len(all_skill_ids)
                probs = [c / total_count for c in counter.values()]
                entropy = -sum(p * _math.log2(p) for p in probs if p > 0)
                metrics[f'{s}/retrieval_entropy'] = float(entropy)
                lib_size = max(len(mem_data), 1)
                max_entropy = _math.log2(lib_size) if lib_size > 1 else 1.0
                metrics[f'{s}/retrieval_entropy_normalized'] = float(entropy / max_entropy) if max_entropy > 0 else 0.0

            consistency_values = []
            for task_idx, sids in per_task_skill_ids.items():
                if sids:
                    counter = Counter(sids)
                    mode_count = counter.most_common(1)[0][1]
                    consistency_values.append(mode_count / len(sids))
            if consistency_values:
                metrics[f'{s}/retrieval_consistency'] = float(np.mean(consistency_values))

            all_sims = []
            for ref_list in retrieved_refs:
                for item in ref_list:
                    sim = item.get('similarity_score')
                    if sim is not None:
                        all_sims.append(float(sim))
            if all_sims:
                metrics[f'{s}/retrieval_similarity/mean'] = float(np.mean(all_sims))
                metrics[f'{s}/retrieval_similarity/min'] = float(np.min(all_sims))
                metrics[f'{s}/retrieval_similarity/max'] = float(np.max(all_sims))

        # --- Distillation ops distribution ---
        distill_ops = getattr(self, '_distill_ops', [])
        if distill_ops:
            from collections import Counter
            ops_counter = Counter(distill_ops)
            total_ops = len(distill_ops)
            metrics[f'{s}/distill_ops/add_rate'] = float(ops_counter.get('add', 0)) / total_ops
            metrics[f'{s}/distill_ops/none_rate'] = float(ops_counter.get('none', 0)) / total_ops
            metrics[f'{s}/distill_ops/total'] = float(total_ops)

        # --- Skill lifetime distribution ---
        if mem_data:
            current_step = getattr(self.skill_library, '_current_training_step', 0)
            lifetimes = [current_step - entry.get('created_at_step', 0) for entry in mem_data]
            if lifetimes:
                lifetimes_arr = np.array(lifetimes, dtype=np.float64)
                metrics[f'{s}/skill_lifetime/mean'] = float(np.mean(lifetimes_arr))
                metrics[f'{s}/skill_lifetime/q1'] = float(np.percentile(lifetimes_arr, 25))
                metrics[f'{s}/skill_lifetime/q3'] = float(np.percentile(lifetimes_arr, 75))

        # First-order difference reward metrics (skill1_v2)
        u_hat_vals = getattr(self, '_distill_u_hat_values', [])
        r_vals = getattr(self, '_distill_r_values', [])
        if u_hat_vals:
            metrics[f'{s}/distill_u_hat/mean'] = float(np.mean(u_hat_vals))
            metrics[f'{s}/distill_u_hat/max'] = float(np.max(u_hat_vals))
            metrics[f'{s}/distill_u_hat/min'] = float(np.min(u_hat_vals))
            metrics[f'{s}/distill_r_diff/mean'] = float(np.mean(r_vals))
            metrics[f'{s}/distill_r_diff/positive_rate'] = float(sum(1 for r in r_vals if r > 0) / len(r_vals))
            metrics[f'{s}/distill_r_diff/negative_rate'] = float(sum(1 for r in r_vals if r < 0) / len(r_vals))

        return metrics

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        distill_rewards = kwargs.get('distill_rewards', None)

        batch_size = len(total_batch_list)
        success = defaultdict(list)

        for bs in range(batch_size):
            r_reward = None
            if distill_rewards is not None:
                try:
                    r_reward = distill_rewards[bs]
                except IndexError:
                    r_reward = 0.0

            self._process_batch(bs, total_batch_list, total_infos, success, distill_reward=r_reward)

        assert len(success['success_rate']) == batch_size
        return {key: np.array(value) for key, value in success.items()}

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, distill_reward=None):
        if distill_reward is not None:
            val = float(distill_reward.item()) if hasattr(distill_reward, 'item') else float(distill_reward)
            success['distill_success_rate'].append(val)
        else:
            success['distill_success_rate'].append(0.0)

        trajectory = total_batch_list[batch_idx]
        n_prepended = sum(1 for s in trajectory if s.get('phase') in ('query', 'rerank'))
        found_active_step = False
        for i in reversed(range(len(trajectory))):
            batch_item = trajectory[i]
            if batch_item.get('phase') in ('query', 'rerank'):
                continue
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i - n_prepended]
                won_value = float(info.get('won', 0.0))
                score_value = float(info.get('task_score', 0.0))

                success['play_success_rate'].append(won_value)
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                found_active_step = True
                return

        if not found_active_step:
            success['play_success_rate'].append(0.0)
            success['success_rate'].append(0.0)
            success['webshop_task_score (not success_rate)'].append(0.0)

class MineSweeperEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config, retrieve_type=None):
        self.n_mines = config.env.minesweeper.n_mines
        self.board_size = config.env.minesweeper.board_size
        self.memory = SimpleMemory()
        
        # Group and evaluation configuration for distillation
        self.group_n = config.env.rollout.n  # e.g., 8
        # Extract skill library hyperparameters from config
        mem_config = config.env.get('skill_library', {})
        filepath = mem_config.get('filepath', "minesweeper_skill_library.json")
        import os
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        relevance_weight = mem_config.get('relevance_weight', 0.7)
        alpha = mem_config.get('alpha', 0.05)
        temp = mem_config.get('temperature', 0.5)
        ucb_scale = mem_config.get('ucb_scale', 1.0)
        self.top_k = mem_config.get('top_k', 1)
        
        self.memory_start_cutoff = mem_config.get('memory_start_cutoff', 0.0) 
        self.current_progress_ratio = 0.0 # Track progress internally
        self.retrieve_mode = mem_config.get('retrieve_mode', 'both')
        self.enable_memory = mem_config.get('enable_memory', True)
        self.group_outperformance = mem_config.get('group_outperformance', False)
        self.full_group_memory = mem_config.get('full_group_memory', False)
        assert not (self.full_group_memory and self.group_outperformance), \
            "group_outperformance requires split group (full_group_memory must be False)"
        self.group_relative_intrinsic_rewards = mem_config.get('group_relative_intrinsic_rewards', False)
        self.distill_reward_type = mem_config.get('distill_reward_type', 'self_assess')
        self.u_hat_aggregation = mem_config.get('u_hat_aggregation', 'max')

        self.potential_based_on_binary_success = mem_config.get('potential_based_on_binary_success', False)
        self.single_distill_per_group = mem_config.get('single_distill_per_group', False)
        # EMA Decay rate for the baseline (matches LaTeX gamma)
        self.ema_gamma = 0.9
        print("memory retrieve_type: ", retrieve_type)
        print("memory retrieve_mode: ", self.retrieve_mode)
        print("top_k_retrieved_memory: ", self.top_k)
        print(f"Memory Start Cutoff: {self.memory_start_cutoff}") 
        print(f"Global Memory Retrieval Enabled: {self.enable_memory}")
        print(f"Single Distill Per Group: {self.single_distill_per_group}")
        print(f"Potential Based On Binary Success Only: {self.potential_based_on_binary_success}")
        self.retriever_type = mem_config.get('retriever_type', 'dense')
        print(f"Retriever Type: {self.retriever_type}")

        # Initialize persistent skill library
        self.skill_library = SkillLibrary(
            filepath=filepath,
            relevance_weight=relevance_weight,
            alpha=alpha,
            temperature=temp,
            retrieve_type=retrieve_type,
            ucb_scale=ucb_scale,
            max_size=5000,
            retriever_type=self.retriever_type,
        )
        self.task_trajectory_history = {}
        self.task_potential_history = {} 
        self.batch_previous_potentials = [] 
        
        
        # Initialize containers for retrieval tracking
        self.current_skills = []       # Formatted strings for the prompt
        self.retrieved_raw_skills = []  # List of lists of raw strings for utility updates
        self.init_states = []
        self.current_retrieval_types = []
        # Store the trajectories generated during the distillation phase
        # so they can be saved to memory in step_distill
        self.last_trajectories = []        
        super().__init__(envs, projection_f, config)

    def update_training_progress(self, current_step: int, total_steps: int):
        """
        Updates the environment with the current training progress.
        This triggers memory pruning if a 20% milestone is reached.
        """
        if total_steps > 0:
            self.current_progress_ratio = current_step / total_steps
            
            # self.skill_library.check_and_prune(progress_ratio=ratio, top_k=3)

    def reset(self, kwargs):
        if kwargs is None:
            kwargs = {}
        
        # Determine mode based on kwargs
        is_eval = not kwargs.get('is_train', True)
        # print("is_eval:", is_eval)

        obs, infos = self.envs.reset()
        self.init_states = obs
        assert len(self.init_states) == len(infos)
        
        # print("obs[0]: ", obs[0])
        # print("infos[0]: ", infos[0])
        self.pre_text_obs = obs
        
        self.memory.reset(batch_size = len(infos))
        self.batch_size = len(obs)
        assert self.batch_size % self.group_n == 0, "Batch size must be divisible by group size"
        self.num_unique_tasks = self.batch_size // self.group_n

        self.current_skills = []
        self.retrieved_raw_skills = []
        self.batch_previous_potentials = []
        self.current_retrieval_types = [] 
        self.batch_retrieved_types = [] # Reset the type tracker
        in_warmup_period = (not is_eval) and (self.current_progress_ratio <= self.memory_start_cutoff)
        
        if in_warmup_period:
            # Optional: Log occasionally if needed
            print(f"Warmup Phase: Progress {self.current_progress_ratio:.2f} <= Cutoff {self.memory_start_cutoff}. Memory Disabled.")
            pass 
        group_split_index = self.group_n // 2
        if self.full_group_memory:
            group_split_index = 0
        for i, task in enumerate(self.init_states):
            prev_potential = self.task_potential_history.get(task, 0.0)
            self.batch_previous_potentials.append(prev_potential)
            formatted_skills = ""
            raw_list_of_dicts = [] # This will hold [{'text':..., 'type':...}]
            current_types_list = [] # List to hold types for this specific agent

            should_retrieve = False
            retrieval_type_str = "control"
            if self.enable_memory:
                if in_warmup_period:
                    # Explicitly disable retrieval during warmup
                    should_retrieve = False
                elif is_eval:
                    # During Eval: Everyone retrieves (or based on config)
                    should_retrieve = True
                    retrieval_type_str = "eval_retrieval"
                else:
                    position_in_group = i % self.group_n
                    if position_in_group >= group_split_index:
                        should_retrieve = True
                        retrieval_type_str = "experiment"
                    else:
                        should_retrieve = False
            else:
                should_retrieve = False
            
            if should_retrieve:
                # Retrieve top_k items
                k = self.top_k
                raw_list_of_dicts = self.skill_library.retrieve(
                    current_scenario_description=task, 
                    top_k=k, 
                    filter_type=self.retrieve_mode
                )
                if raw_list_of_dicts:
                    formatted_lines = []
                    for item in raw_list_of_dicts:
                        r_text = item.get('text', '')
                        r_type = item.get('type', 'unknown')
                        
                        # Store the type for logging
                        current_types_list.append(r_type)
                        
                        formatted_lines.append(r_text)
                    
                    formatted_skills = "Relevant skills from the skill library:\n" + "\n".join(formatted_lines)
                    formatted_skills += "\nWarning: These lessons may be outdated. Use them only if they align with your current observation."
            
            
            self.current_skills.append(formatted_skills)
            self.retrieved_raw_skills.append(raw_list_of_dicts)
            self.current_retrieval_types.append(retrieval_type_str)
            self.batch_retrieved_types.append(current_types_list)
            print("retrieved_raw_skills: ", self.retrieved_raw_skills)
            print("current_skills: ", self.current_skills)
            # --- NEW: Inject types into infos immediately ---
            infos[i]['distill_types'] = current_types_list
            infos[i]['retrieval_group'] = retrieval_type_str
        
        # -----------------------------------------------------------
        assert len(self.current_skills) == len(self.init_states)
        # Now it is safe to build observations
        observations = {
            'text': self.build_text_obs(infos, obs, init=True),
            'image': None, 
            'anchor': obs
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        # print("text_actions: ", text_actions)
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({
                            'text_obs': self.pre_text_obs,
                            'action': actions, 
                            'reward': rewards,
                            'dones': dones,
                            'won': [info['won'] for info in infos]
                        })
        
        self.pre_text_obs = next_obs
        next_observations = {
            'text': self.build_text_obs(infos, next_obs), 
            'image': None, 
            'anchor': next_obs
        }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def distill(self, infos: List[Dict]):
        """
        Called at the end of the 'play' phase.
        Updates utility based on Group B (Retrieved) vs Group A (Not Retrieved) performance.
        """
        observations = {
            'text': self.build_distill_text_obs(infos),
            'image': None,
            'anchor': self.build_distill_text_obs(infos)
        }
        
        # Ensure action validity is set for all
        for info in infos:
            info['is_action_valid'] = to_numpy(True)
        
        batch_size = len(self.init_states)
        assert batch_size == len(infos)
        
        # Ensure batch size is divisible by group_n
        if batch_size % self.group_n != 0:
            print(f"WARNING: Batch size {batch_size} not divisible by group_n {self.group_n}")
        
        num_groups = batch_size // self.group_n
        group_split_index = 0 if self.full_group_memory else self.group_n // 2
        
        # Iterate over each group independently
        for g in range(num_groups):
            start_idx = g * self.group_n
            end_idx = start_idx + self.group_n
            mid_idx = start_idx + group_split_index
            
            # Calculate wins for control group (first half)
            control_wins = 0
            for i in range(start_idx, mid_idx):
                if infos[i].get("won", False):
                    control_wins += 1
            
            # Calculate wins for experiment group (second half)
            experiment_wins = 0
            for i in range(mid_idx, end_idx):
                if infos[i].get("won", False):
                    experiment_wins += 1
            
            # 3. Determine Utility Score for THIS group
            group_outperformed = experiment_wins > control_wins
            
            # 4. Update Memory ONLY for the Experiment agents
            for i in range(mid_idx, end_idx):
                task_desc = self.init_states[i]
                raw_skill_items = self.retrieved_raw_skills[i] # List[Dict]
                is_success = infos[i].get("won", False)
                
                # 1.0 if the agent won AND the retrieval group beat the control group.
                if self.group_outperformance:
                    if is_success and group_outperformed:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
                else:
                    if is_success:
                        utility_score = 1.0
                    else:
                        utility_score = 0.0
            
                if raw_skill_items:
                    for item in raw_skill_items:
                        # --- UPDATED: Extract text from dict for utility update ---
                        strategy_text = item.get('text', '')
                        if strategy_text:
                            self.skill_library.update_utility(
                                scenario_description=task_desc, 
                                strategy_text=strategy_text, 
                                score=utility_score
                            )
        
        return observations, infos

    def step_distill(self, text_actions: List[str], infos: List[Dict]):
        """
        Calculates intrinsic rewards based on improvement over an EMA baseline,
        normalizes them group-wise, and updates the baseline.
        Adapted for MINESWEEPER_DISTILL_TEMPLATE.
        """
        import json
        import re
        import copy
        import numpy as np
        
        def to_numpy(x):
            return np.array(x) if not isinstance(x, np.ndarray) else x

        print("text_actions for distillation:", text_actions)
        
        # 1. Initialize Containers
        distill_rewards = [] # This is the immediate reward for the distillation step itself (e.g. self-consistency)
        current_scores = np.zeros(self.batch_size) # The raw potential (phi)
        raw_improvements = np.zeros(self.batch_size) # The raw I (improvement)
        is_won_array = np.zeros(self.batch_size, dtype=bool)
        self._distill_u_hat_values = []
        self._distill_r_values = []

        # Ensure batch_previous_potentials is synced
        if len(self.batch_previous_potentials) != self.batch_size:
            self.batch_previous_potentials = [0.0] * self.batch_size

        # 2. Calculate Raw Scores (Phi) and Raw Improvements (I)
        for i, strategy_text in enumerate(text_actions):
            task_desc = self.init_states[i]
            current_trajectory = self.last_trajectories[i] if i < len(self.last_trajectories) else ""
            
            # Get the baseline (Phi_{t-1})
            prev_phi = self.batch_previous_potentials[i]
            # --- Extract Actual Success First ---
            actual_success = bool(infos[i].get('won', False))
            is_won_array[i] = actual_success
            current_phi = 0.0
            try:
                # --- JSON Extraction ---
                json_str = ""
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', strategy_text, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    clean_text = strategy_text.strip()
                    start_idx = clean_text.find('{')
                    end_idx = clean_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = clean_text[start_idx:end_idx+1]
                
                if not json_str: raise ValueError("No JSON found")
                
                distill_data = json.loads(json_str)
                
                # 1. Trust but verify the trajectory_value against the subtasks list
                subtasks = distill_data.get('subtasks', [])
                total_subtasks = len(subtasks)
                completed_subtasks = sum(
                    1 for task in subtasks 
                    if isinstance(task, dict) and task.get('status', '').strip().lower() == 'completed'
                )
                
                # Calculate subtask-based potential
                subtask_phi = completed_subtasks / total_subtasks if total_subtasks > 0 else 0.0
                
                # --- DETERMINE CURRENT PHI ---
                if self.potential_based_on_binary_success:
                    # STRICT MODE: Only actual success matters for potential
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    # DEFAULT MODE: Use subtasks, but override if actual success
                    current_phi = subtask_phi
                    if actual_success:
                        current_phi = 1.0

                # --- Reflection Consistency Reward (Auxiliary) ---
                predicted_success = distill_data.get('task_success', False)
                if isinstance(predicted_success, str):
                    predicted_success = predicted_success.lower() in ['true', '1', 'yes']

                # --- Distillation reward (mutually exclusive modes) ---
                if self.distill_reward_type == 'first_order_diff':
                    retrieved_items = self.retrieved_raw_skills[i] if i < len(self.retrieved_raw_skills) else []
                    u_scores = [item.get('utility_score', 0.5) for item in retrieved_items] if retrieved_items else []
                    u_hat = (max(u_scores) if self.u_hat_aggregation == 'max' else sum(u_scores) / len(u_scores)) if u_scores else 0.0
                    r_tau = 1.0 if actual_success else 0.0
                    current_reward = r_tau - u_hat
                    self._distill_u_hat_values.append(u_hat)
                    self._distill_r_values.append(current_reward)
                else:
                    current_reward = 10.0 if predicted_success == actual_success and json_str else 0.0
                distill_rewards.append(current_reward)
                # --- Memory Saving Logic (Same as before) ---
                should_save = (actual_success and json_str) if self.distill_reward_type == 'first_order_diff' else (predicted_success == actual_success and json_str)
                if should_save:
                    next_priority = distill_data.get('next_priority')
                    lessons_to_save = []
                    if next_priority and len(str(next_priority)) > 5:
                        lessons_to_save.append(f"New Plan: {next_priority}")

                    if lessons_to_save:
                        final_lesson = " | ".join(lessons_to_save)
                        self.skill_library.admit(
                            scenario_description=task_desc,
                            strategy_text=final_lesson,
                            trajectory=current_trajectory,
                            initial_score=0.5,
                            attempt_type="success" if actual_success else "failure",
                            current_progress_ratio=self.current_progress_ratio
                        )
            except Exception as e:
                print(f"Error task {i}: {e}")
                distill_rewards.append(0.0)
                # Fallback logic for Phi on error
                if self.potential_based_on_binary_success:
                    current_phi = 1.0 if actual_success else 0.0
                else:
                    current_phi = 0.0

            # --- Calculate Raw Improvement (I) ---
            current_scores[i] = current_phi
            # Improvement is strictly positive gain over history
            improvement = max(0.0, current_phi - prev_phi)
            raw_improvements[i] = improvement
                    
        # 3. Group-Relative Normalization & Baseline Update
        num_unique_tasks = self.batch_size // self.group_n
        final_intrinsic_rewards = np.zeros(self.batch_size)

        for group_idx in range(num_unique_tasks):
            start_idx = group_idx * self.group_n
            end_idx = start_idx + self.group_n
            
            task_desc = self.init_states[start_idx]
            
            # Extract group data
            group_improvements = raw_improvements[start_idx:end_idx]
            # group_scores = current_scores[start_idx:end_idx]
            
            # A. Normalization (Centering)
            # As per LaTeX Eq (8): R_int = I - Mean(I)
            if self.group_relative_intrinsic_rewards:
                group_mean_imp = np.mean(group_improvements)
                # Note: We do NOT divide by std here, just centering is sufficient 
                # to maintain the zero-sum property for the intrinsic component.
                centered_improvements = group_improvements - group_mean_imp
                final_intrinsic_rewards[start_idx:end_idx] = centered_improvements
            else:
                final_intrinsic_rewards[start_idx:end_idx] = group_improvements

            group_success_rate = np.mean(is_won_array[start_idx:end_idx].astype(float))
            old_baseline = self.task_potential_history.get(task_desc, 0.0)

            if group_success_rate > old_baseline:
                self.task_potential_history[task_desc] = group_success_rate
            # B. Update Historical Baseline (EMA)
            # As per LaTeX Eq (9): Phi_t = gamma * Phi_{t-1} + (1-gamma) * Mean(Phi_t)
            # if len(group_scores) > 0:
            #     current_group_mean_score = np.mean(group_scores)
            #     old_baseline = self.task_potential_history.get(task_desc, 0.0)

            #     if current_group_mean_score > old_baseline:
            #         self.task_potential_history[task_desc] = current_group_mean_score
                # # EMA Update
                # new_baseline = (self.ema_gamma * old_baseline) + ((1 - self.ema_gamma) * current_group_mean_score)
                # self.task_potential_history[task_desc] = new_baseline

        print("raw_improvements: ", raw_improvements)
        print("final_intrinsic_rewards (centered): ", final_intrinsic_rewards)
        infos = copy.deepcopy(infos)
        for info in infos:
            info['is_action_valid'] = to_numpy(True)

        # Convert to numpy for compatibility
        return None, to_numpy(distill_rewards), to_numpy(final_intrinsic_rewards), None, copy.deepcopy(infos), to_numpy(current_scores)

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action"
            )

        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = MINESWEEPER_TEMPLATE_NO_HIS.format(
                    board_size=self.board_size,
                    n_mines=self.n_mines,
                    skills=self.current_skills[i],  # Add skills
                    current_observation=text_obs[i],
                )
            else:
                obs = MINESWEEPER_TEMPLATE.format(
                    board_size=self.board_size,
                    n_mines=self.n_mines,
                    skills=self.current_skills[i],  # Add skills
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                )
            postprocess_text_obs.append(obs)
        
        print("postprocessed_text_obs [0]:", postprocess_text_obs[0])
        return postprocess_text_obs

    def build_distill_text_obs(self, infos: List[Dict]) -> List[str]:
        """
        This function builds the text observation for the agent during distillation.
        """
        postprocess_text_obs = []
        memory_contexts, valid_lens = self.memory.fetch(
            15,  # Get full game history for distillation
            obs_key="text_obs",
            action_key="action"
        )
        # self.task_trajectory_history[task] = {"successful": [], "failed": []}
        for i in range(len(infos)):
            task = self.init_states[i]
            # Ensure key exists (it should from reset, but safety first)
            if task not in self.task_trajectory_history:
                self.task_trajectory_history[task] = {"successful": [], "failed": []}
                
            if infos[i].get("won", False):
                self.task_trajectory_history[task]["successful"].append(memory_contexts[i])
            else:
                self.task_trajectory_history[task]["failed"].append(memory_contexts[i])

        # --- CRITICAL: Store these so step_distill can access them ---
        self.last_trajectories = memory_contexts

        for i in range(len(infos)):
            task = self.init_states[i]
            is_won = infos[i].get("won", False)
            
            # Determine success string and select Contrastive Reference
            # If we WON, we want to see a FAIL to understand what to avoid (or just compare)
            # If we LOST, we want to see a SUCCESS to understand what to do
            
            reference_traj_str = "No reference history available yet."
            
            if is_won:
                SUCCESS = "successfully"
                # Try to get a failed example
                failed_hist = self.task_trajectory_history[task]["failed"]
                if failed_hist:
                    # Use the most recent failure
                    reference_traj_str = "Reference Failed Trajectory (for comparison):\n" + failed_hist[-1]
                else:
                    reference_traj_str = "No failed attempts available for comparison."
            else:
                SUCCESS = "unsuccessfully" # Changed from "NOT successfully" for better grammar
                # Try to get a successful example
                success_hist = self.task_trajectory_history[task]["successful"]
                if success_hist:
                    # Use the most recent success
                    reference_traj_str = "Reference Successful Trajectory (for comparison):\n" + success_hist[-1]
                else:
                    reference_traj_str = "No successful attempts available for reference."
            obs = MINESWEEPER_DISTILL_TEMPLATE.format(
                board_size=self.board_size,
                n_mines=self.n_mines,
                success=SUCCESS,
                reference_trajectory=reference_traj_str,
                current_trajectory=memory_contexts[i]
            )
            postprocess_text_obs.append(obs)
        
        if len(postprocess_text_obs) > 0:
            print("processed_distill_text [0]:", postprocess_text_obs[0])
        
        return postprocess_text_obs

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not.
        """
        from collections import defaultdict
        
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        distill_rewards = kwargs.get('distill_rewards', None)
        
        batch_size = len(total_batch_list)
        success = defaultdict(list)
        
        for bs in range(batch_size):
            r_reward = None
            if distill_rewards is not None:
                try:
                    r_reward = distill_rewards[bs]
                except IndexError:
                    r_reward = 0.0
            
            self._process_batch(bs, total_batch_list, total_infos, success, distill_reward=r_reward)
        
        assert len(success['success_rate']) == batch_size
        
        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, distill_reward=None):
        """
        Process a single batch trajectory to extract success metrics.
        """
        # Process distillation phase
        if distill_reward is not None:
            if hasattr(distill_reward, 'item'):
                val = float(distill_reward.item())
            else:
                val = float(distill_reward)
            success['distill_success_rate'].append(val)
        else:
            success['distill_success_rate'].append(0.0)
        
        # Process play phase
        play_success_found = False
        trajectory = total_batch_list[batch_idx]
        n_prepended = sum(1 for s in trajectory if s.get('phase') in ('query', 'rerank'))

        for i in reversed(range(len(trajectory))):
            batch_item = trajectory[i]

            if not batch_item.get('active_masks', True):
                continue

            phase = batch_item.get('phase', 'play')

            if phase == 'play':
                info = total_infos[batch_idx][i - n_prepended]
                won_value = float(info.get('won', 0.0))

                success['play_success_rate'].append(won_value)
                success['success_rate'].append(won_value)

                # Add Minesweeper-specific success metrics
                if self.board_size and self.n_mines:
                    difficulty = f"minesweeper_{self.board_size}x{self.board_size}_{self.n_mines}mines"
                    success[f"{difficulty}_success_rate"].append(won_value)

                play_success_found = True
                break
        
        if not play_success_found:
            success['play_success_rate'].append(0.0)
            success['success_rate'].append(0.0)


class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(
            _envs, projection_f, config,
            config.env.get('train_retrieve_type', 'ucb')
        )
        val_envs = SearchEnvironmentManager(
            _val_envs, projection_f, config,
            config.env.get('eval_retrieve_type', 'greedy')
        )
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, resources_per_worker=resources_per_worker)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, resources_per_worker=resources_per_worker)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        # print("config information: ", config)
        # Ensure 'config' is passed as the 3rd argument to both constructors
        envs = AlfWorldEnvironmentManager(
            _envs, 
            projection_f, 
            config, 
            config.env.train_retrieve_type
        )
        
        val_envs = AlfWorldEnvironmentManager(
            _val_envs, 
            projection_f, 
            config, 
            config.env.eval_retrieve_type
        )
        # --- FIX END ---
        return envs, val_envs

    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth,
            'min_steps': config.env.get('min_steps', 5),  # default to 3 if not specified
            'max_sol_steps': config.env.get('max_sol_steps', config.env.max_steps) 
        }
        _envs = build_sokoban_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_sokoban_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config, config.env.train_retrieve_type)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config, config.env.eval_retrieve_type)
        return envs, val_envs

    elif "minesweeper" in config.env.env_name.lower():
        from agent_system.environments.env_package.minesweeper import build_minesweeper_envs, minesweeper_projection
        env_kwargs = {
            "board_size": config.env.minesweeper.board_size,  # e.g., 8 for 8x8 board
            "n_mines": config.env.minesweeper.n_mines,
            "board_type": config.env.minesweeper.board_type
        }
        _envs = build_minesweeper_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_minesweeper_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(minesweeper_projection)
        envs = MineSweeperEnvironmentManager(_envs, projection_f, config, config.env.train_retrieve_type)
        val_envs = MineSweeperEnvironmentManager(_val_envs, projection_f, config, config.env.eval_retrieve_type)
        return envs, val_envs

    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        _webshop_data_dir = os.environ.get(
            'WEBSHOP_DATA',
            os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data'),
        )
        if config.env.webshop.use_small:
            file_path = os.path.join(_webshop_data_dir, 'items_shuffle_1000.json')
            attr_path = os.path.join(_webshop_data_dir, 'items_ins_v2_1000.json')
        else:
            file_path = os.path.join(_webshop_data_dir, 'items_shuffle.json')
            attr_path = os.path.join(_webshop_data_dir, 'items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)

        projection_f = partial(webshop_projection)
        # envs = WebshopEnvironmentManager(_envs, projection_f, config)
        # val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        envs = WebshopEnvironmentManager(
            _envs, 
            projection_f, 
            config, 
            config.env.train_retrieve_type
        )
        val_envs = WebshopEnvironmentManager(
            _val_envs, 
            projection_f, 
            config, 
            config.env.eval_retrieve_type
        )
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs

    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0, resources_per_worker=resources_per_worker)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n, resources_per_worker=resources_per_worker)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)