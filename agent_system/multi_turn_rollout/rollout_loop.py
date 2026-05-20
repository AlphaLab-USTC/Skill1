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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import json
import collections

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.step_gamma = config.algorithm.get('step_gamma', 0.95)
        self.traj_gamma = config.algorithm.get('traj_gamma', 0.6)
        self.intrinsic_reward_coefficient = config.algorithm.get('intrinsic_reward_coefficient', -1)
        self.enable_credit_assignment = config.algorithm.get('credit_assignment', False)
        mem_cfg = config.env.get('skill_library', {})
        self.selection_trainable = mem_cfg.get('selection_trainable', True)
        self.lambda_rerank = mem_cfg.get('lambda_rerank', 1.0)
        self.lambda_distill = mem_cfg.get('lambda_distill', 1.0)
        self.tokenizer = tokenizer
        self.processor = processor
        self.use_ref_policy_for_distill = config.algorithm.get('distill_reference_policy', False)

    def _calculate_distill_coefficient(self, current_step: int, total_steps: int) -> float:
        """
        Calculates the distillation coefficient.
        If hard_cutoff is enabled, the coefficient decays to ~0.001 by the cutoff point 
        and is 0.0 thereafter to ensure a smooth transition.
        """
        if total_steps <= 0:
            return 1.0
            
        progress = current_step / total_steps
        progress = max(0.0, min(1.0, progress))
        
        hard_cutoff = self.config.algorithm.get('intrinsic_hard_cutoff', False)
        
        if hard_cutoff:
            cutoff_point = 0.10
            if progress > cutoff_point:
                return 0.0
            normalized_progress = progress / cutoff_point
            alpha = 6.9
            coefficient = np.exp(-alpha * normalized_progress)
        else:
            alpha = 5.0
            coefficient = np.exp(-alpha * progress)
        
        return float(coefficient)

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        """
        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch.get('data_source', ['unknown'] * len(gen_batch.non_tensor_batch['raw_prompt']))[item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!", flush=True)

        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        row_dict = {}
        
        if is_multi_modal:
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1
                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )

        if is_multi_modal:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length:]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        for item in range(batch_size):
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        batch = collate_fn(processed_samples)
        
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            discounted_returns: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            distill_types_list: List[List[str]] = None,
            retrieval_groups_list: List[str] = None,
            subem_scores: np.ndarray = None,
            answer_found: np.ndarray = None,
            num_turns: np.ndarray = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data.

        Parameters:
            total_batch_list (List[List[Dict]]): List of trajectory data for each environment.
            episode_rewards (np.ndarray): 1-D object array of np.ndarray, one per trajectory.
                Each inner array has length == len(total_batch_list[bs]) with the per-step
                episode reward (cumulative play reward assigned to every active step).
            discounted_returns (np.ndarray): 1-D object array of np.ndarray, one per trajectory.
                Each inner array has length == len(total_batch_list[bs]) with the per-step
                discounted return.
            episode_lengths (np.ndarray): Total steps for each environment.
            success (Dict[str, np.ndarray]): Success samples for each environment.
            traj_uid (np.ndarray): Trajectory unique identifiers.
            tool_callings (np.ndarray): Number of tool callings for each environment.
            distill_types_list (List[List[str]]): Reflection types per trajectory.
            retrieval_groups_list (List[str]): Retrieval group per trajectory.

        Returns:
            DataProto: Collected and organized trajectory data.
        """
        batch_size = len(total_batch_list)

        # Compute aggregate stats from all per-step episode rewards
        all_ep_rewards = np.concatenate([episode_rewards[bs] for bs in range(batch_size)])
        episode_rewards_mean = float(np.mean(all_ep_rewards)) if len(all_ep_rewards) > 0 else 0.0
        episode_rewards_min = float(np.min(all_ep_rewards)) if len(all_ep_rewards) > 0 else 0.0
        episode_rewards_max = float(np.max(all_ep_rewards)) if len(all_ep_rewards) > 0 else 0.0

        episode_lengths_mean = float(np.mean(episode_lengths))
        episode_lengths_min = float(np.min(episode_lengths))
        episode_lengths_max = float(np.max(episode_lengths))

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)

        effective_batch = []
        for bs in range(batch_size):
            current_distill_types = []
            current_retrieval_group = "unknown"

            if distill_types_list is not None and bs < len(distill_types_list):
                current_distill_types = distill_types_list[bs]

            if retrieval_groups_list is not None and bs < len(retrieval_groups_list):
                current_retrieval_group = retrieval_groups_list[bs]

            ep_rew_arr = episode_rewards[bs]       # np.ndarray of shape (num_steps,)
            disc_ret_arr = discounted_returns[bs]   # np.ndarray of shape (num_steps,)

            for t, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # Per-step credit-assigned values
                    data['episode_rewards'] = np.array(ep_rew_arr[t], dtype=np.float32)
                    data['step_returns'] = torch.tensor(float(disc_ret_arr[t]), dtype=torch.float32)
                    # Aggregate stats
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    data['tool_callings'] = tool_callings[bs]
                    if subem_scores is not None:
                        data['subem_scores'] = subem_scores[bs]
                        data['answer_found'] = answer_found[bs]
                        data['num_turns'] = num_turns[bs]
                    data['distill_types'] = current_distill_types
                    data['retrieval_group'] = current_retrieval_group
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)

        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            current_training_steps: int,
            total_training_steps: int,
            is_train: bool = True,
            ref_rollout_wg=None,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        """
        if total_training_steps > 0:
            if hasattr(envs, 'update_training_progress'):
                envs.update_training_progress(current_training_steps, total_training_steps)
            elif hasattr(envs, 'env') and hasattr(envs.env, 'update_training_progress'):
                envs.env.update_training_progress(current_training_steps, total_training_steps)

        batch_size = len(gen_batch.batch)

        env_kwargs = gen_batch.non_tensor_batch.pop('env_kwargs', {})
        if env_kwargs is None:
            env_kwargs = {}
        if isinstance(env_kwargs, np.ndarray):
            env_kwargs = {'is_train': is_train, 'per_sample': list(env_kwargs)}
        else:
            env_kwargs['is_train'] = is_train

        obs, infos = envs.reset(kwargs=env_kwargs)

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        subem_scores = np.zeros(batch_size, dtype=np.float32)
        answer_found = np.zeros(batch_size, dtype=np.float32)
        num_turns = np.zeros(batch_size, dtype=np.float32)
        distill_rewards = np.zeros(batch_size, dtype=np.float32)
        distill_rewards_step = np.zeros(batch_size, dtype=np.float32)
        distill_rewards_before_clipping = np.zeros(batch_size, dtype=np.float32)
        extrinsic_episode_rewards = np.zeros(batch_size, dtype=np.float32)
        final_intrinsic_rewards = np.zeros(batch_size, dtype=np.float32)

        trajectory_distill_types = [[] for _ in range(batch_size)]
        trajectory_retrieval_groups = ["unknown"] * batch_size
        unknown_distills = np.zeros(batch_size, dtype=np.float32)
        failure_distills = np.zeros(batch_size, dtype=np.float32)
        success_distills = np.zeros(batch_size, dtype=np.float32)

        for i, info in enumerate(infos):
            if 'distill_types' in info:
                trajectory_distill_types[i] = info['distill_types']
                if "failure" in info['distill_types']:
                    failure_distills[i] += 1
                elif "success" in info['distill_types']:
                    success_distills[i] += 1
            if 'retrieval_group' in info:
                trajectory_retrieval_groups[i] = info['retrieval_group']

        # --- Phase 0: Query Generation (BiGen-Retrieval, optional) ---
        # Runs during BOTH training and validation: generate query → re-retrieve → rebuild obs
        if hasattr(envs, 'enable_query_generation') and envs.enable_query_generation:
            print(f"*** BiGen Phase 0: Query Generation (is_train={is_train}) ***")
            query_obs = envs.build_query_generation_obs()

            query_batch = self.preprocess_batch(gen_batch=gen_batch, obs=query_obs)
            qb_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            qb_nt_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in query_batch.non_tensor_batch:
                qb_nt_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in query_batch.non_tensor_batch:
                qb_nt_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in query_batch.non_tensor_batch:
                qb_nt_keys_to_pop.append("tools_kwargs")
            query_batch_input = query_batch.pop(
                batch_keys=qb_keys_to_pop,
                non_tensor_batch_keys=qb_nt_keys_to_pop,
            )
            query_batch_input.meta_info = gen_batch.meta_info
            qb_padded, qb_pad_size = pad_dataproto_to_divisor(query_batch_input, actor_rollout_wg.world_size)
            qb_output_padded = actor_rollout_wg.generate_sequences(qb_padded)
            qb_output = unpad_dataproto(qb_output_padded, qb_pad_size)

            query_batch.non_tensor_batch['uid'] = uid_batch
            query_batch.non_tensor_batch['traj_uid'] = traj_uid
            query_batch.non_tensor_batch['phase'] = ['query'] * batch_size
            query_batch = query_batch.union(qb_output)

            query_texts = self.tokenizer.batch_decode(query_batch.batch['responses'], skip_special_tokens=True)
            print(f"Generated queries (sample): {query_texts[0][:200] if query_texts else 'N/A'}")

            envs.apply_generated_queries(query_texts)
            obs = envs.rebuild_initial_obs()

            # Collect query step as trainable data (training only)
            if is_train:
                query_active = self.selection_trainable and self.lambda_rerank > 0
                query_batch.non_tensor_batch['rewards'] = np.zeros(batch_size, dtype=np.float32)
                query_batch.non_tensor_batch['active_masks'] = np.ones(batch_size, dtype=bool) if query_active else np.zeros(batch_size, dtype=bool)
                query_batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
                query_batch_list = to_list_of_dict(query_batch)
                for i in range(batch_size):
                    total_batch_list[i].append(query_batch_list[i])

        # --- Phase 0b: Re-rank (optional, replaces or follows query generation) ---
        if hasattr(envs, 'enable_rerank') and envs.enable_rerank:
            print(f"*** Phase 0b: Re-rank (is_train={is_train}) ***")
            rerank_obs = envs.build_rerank_obs()

            rerank_batch = self.preprocess_batch(gen_batch=gen_batch, obs=rerank_obs)
            rr_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            rr_nt_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in rerank_batch.non_tensor_batch:
                rr_nt_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in rerank_batch.non_tensor_batch:
                rr_nt_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in rerank_batch.non_tensor_batch:
                rr_nt_keys_to_pop.append("tools_kwargs")
            rerank_batch_input = rerank_batch.pop(
                batch_keys=rr_keys_to_pop,
                non_tensor_batch_keys=rr_nt_keys_to_pop,
            )
            rerank_batch_input.meta_info = gen_batch.meta_info
            rr_padded, rr_pad_size = pad_dataproto_to_divisor(rerank_batch_input, actor_rollout_wg.world_size)
            rr_output_padded = actor_rollout_wg.generate_sequences(rr_padded)
            rr_output = unpad_dataproto(rr_output_padded, rr_pad_size)

            rerank_batch.non_tensor_batch['uid'] = uid_batch
            rerank_batch.non_tensor_batch['traj_uid'] = traj_uid
            rerank_batch.non_tensor_batch['phase'] = ['rerank'] * batch_size
            rerank_batch = rerank_batch.union(rr_output)

            rerank_texts = self.tokenizer.batch_decode(rerank_batch.batch['responses'], skip_special_tokens=True)
            print(f"Re-rank outputs (sample): {rerank_texts[0][:200] if rerank_texts else 'N/A'}")

            envs.apply_rerank_results(rerank_texts)
            rerank_rewards = envs.compute_rerank_rewards()
            obs = envs.rebuild_initial_obs()
            print(f"[Re-rank] NDCG rewards: mean={np.mean(rerank_rewards):.3f}, min={np.min(rerank_rewards):.3f}, max={np.max(rerank_rewards):.3f}")

            # Collect rerank step as trainable data
            if is_train:
                rerank_active = self.selection_trainable and self.lambda_rerank > 0
                if rerank_active:
                    rerank_batch.non_tensor_batch['rewards'] = rerank_rewards * self.lambda_rerank
                    rerank_batch.non_tensor_batch['active_masks'] = np.ones(batch_size, dtype=bool)
                else:
                    rerank_batch.non_tensor_batch['rewards'] = np.zeros(batch_size, dtype=np.float32)
                    rerank_batch.non_tensor_batch['active_masks'] = np.zeros(batch_size, dtype=bool)
                rerank_batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
                rerank_batch_list = to_list_of_dict(rerank_batch)
                for i in range(batch_size):
                    total_batch_list[i].append(rerank_batch_list[i])

        phase = 'play'
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch.non_tensor_batch['phase'] = [phase] * batch_size
            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            next_obs, rewards, dones, infos = envs.step(text_actions)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]

            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            is_done = np.logical_or(is_done, dones)
            obs = next_obs

            if is_done.all():
                break

        extrinsic_episode_rewards = episode_rewards.copy()

        # Extract per-trajectory search metrics from the first done info
        # (subsequent steps after done may corrupt chat_history-based metrics)
        has_search_metrics = False
        for i in range(batch_size):
            for info in total_infos[i]:
                if 'subem_score' in info:
                    subem_scores[i] = info['subem_score']
                    answer_found[i] = info['answer_found']
                    num_turns[i] = info.get('num_turns', episode_lengths[i])
                    has_search_metrics = True
                    break

        # --- Phase 2: Reflection (Training Only) ---
        if is_train:
            phase = 'distill'
            obs, infos = envs.distill(infos)
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            if self.use_ref_policy_for_distill and ref_rollout_wg is not None:
                distill_wg = ref_rollout_wg
                distill_meta_info = dict(gen_batch.meta_info) if gen_batch.meta_info else {}
                distill_meta_info['use_ref_policy'] = True
                batch_input.meta_info = distill_meta_info
                print("utilizing reference policy for distillation .....")
            else:
                distill_wg = actor_rollout_wg
                batch_input.meta_info = gen_batch.meta_info
                print("utilizing actor policy for distillation .....")

            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, distill_wg.world_size)
            batch_output_padded = distill_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            batch.non_tensor_batch['phase'] = [phase] * batch_size

            batch = batch.union(batch_output)

            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

            _, distill_rewards_step, final_intrinsic_rewards, _, distill_infos, completion_percentages = envs.step_distill(text_actions, infos)

            if len(final_intrinsic_rewards.shape) == 2:
                final_intrinsic_rewards = final_intrinsic_rewards.squeeze(1)

            if self.intrinsic_reward_coefficient > 0:
                distill_coeff = self.intrinsic_reward_coefficient
            elif self.intrinsic_reward_coefficient == 0:
                distill_coeff = self._calculate_distill_coefficient(current_training_steps, total_training_steps)
            else:
                distill_coeff = 0.0

            distill_rewards = final_intrinsic_rewards * distill_coeff * self.lambda_distill
            distill_rewards_before_clipping = distill_rewards.copy()

            instrinsic_rewards_upper_clipping_ratio = self.config.algorithm.get('instrinsic_rewards_upper_clipping_ratio', -5)
            instrinsic_rewards_lower_clipping_ratio = self.config.algorithm.get('instrinsic_rewards_lower_clipping_ratio', -5)
            if instrinsic_rewards_upper_clipping_ratio > -1:
                distill_rewards = np.clip(distill_rewards, -instrinsic_rewards_lower_clipping_ratio, instrinsic_rewards_upper_clipping_ratio)

            self.get_distill_logs(
                distill_obs=obs["text"],
                text_actions=text_actions,
                trajectory_success=episode_rewards,
                distill_rewards=distill_rewards_step,
                completion_percentages=completion_percentages,
                current_training_steps=current_training_steps,
                total_training_steps=total_training_steps,
                distill_rewards_before_clipping=distill_rewards_before_clipping,
                distill_rewards_after_clipping=distill_rewards
            )
            for i in range(batch_size):
                episode_rewards[i] += distill_rewards[i]

            # --- BiGen: inject query contrastive reward into query step ---
            if hasattr(envs, 'enable_query_generation') and envs.enable_query_generation:
                q_rewards = envs.query_contrastive_rewards
                query_active = self.selection_trainable and self.lambda_rerank > 0
                if query_active:
                    for i in range(batch_size):
                        if total_batch_list[i] and total_batch_list[i][0].get('phase') == 'query':
                            old_r = total_batch_list[i][0]['rewards']
                            total_batch_list[i][0]['rewards'] = old_r + q_rewards[i] * self.lambda_rerank
                print(f"[BiGen] Query contrastive rewards (active={query_active}, λ={self.lambda_rerank}): mean={np.mean(q_rewards):.3f}")

        # --- Prepare for Reflection Success Tracking ---
        distill_type_stats = collections.defaultdict(list)
        retrieval_group_rewards = collections.defaultdict(list)
        for i in range(batch_size):
            current_group = trajectory_retrieval_groups[i]
            retrieval_group_rewards[current_group].append(extrinsic_episode_rewards[i])

        for i in range(batch_size):
            current_traj_r_types = trajectory_distill_types[i]

            for step_data in total_batch_list[i]:
                step_data['distill_rewards_before_clipping'] = distill_rewards_before_clipping[i]
                step_data['distill_reward'] = distill_rewards[i]
                step_data['raw_distill_reward'] = distill_rewards_step[i]
                step_data['intrinsic_reward'] = final_intrinsic_rewards[i]
                step_data['extrinsic_episode_reward'] = extrinsic_episode_rewards[i]
                step_data["failure_distill"] = failure_distills[i]
                step_data["success_distill"] = success_distills[i]
                step_data['retrieval_group'] = trajectory_retrieval_groups[i]
                for group_name, rewards_list in retrieval_group_rewards.items():
                    if len(rewards_list) > 0:
                        step_data[f'extrinsic_reward_{group_name}'] = np.mean(rewards_list)

            current_traj_success = 0.0
            # Count non-play steps prepended (query/rerank phases) to offset into total_infos
            n_prepended = sum(1 for s in total_batch_list[i] if s.get('phase') in ('query', 'rerank'))
            for step_idx in reversed(range(len(total_batch_list[i]))):
                batch_item = total_batch_list[i][step_idx]
                if batch_item.get('phase') in ('query', 'rerank'):
                    continue
                if batch_item['active_masks']:
                    info_idx = step_idx - n_prepended
                    info = total_infos[i][info_idx]
                    current_traj_success = float(info.get('won', 0.0))
                    break

            r_types = trajectory_distill_types[i]
            if not r_types:
                distill_type_stats['none'].append(current_traj_success)
            else:
                for r_type in r_types:
                    distill_type_stats[r_type].append(current_traj_success)

        # --- BiGen: inject aggregated metrics into step data for wandb ---
        if hasattr(envs, 'get_bigen_metrics'):
            bigen_prefix = "bigen_train" if is_train else "bigen_val"
            bigen_metrics = envs.get_bigen_metrics(prefix=bigen_prefix)
            if bigen_metrics:
                for i in range(batch_size):
                    for step_data in total_batch_list[i]:
                        for k, v in bigen_metrics.items():
                            step_data[k] = v

        self.get_traj_cot_logs(total_batch_list, current_training_steps, total_training_steps, trajectory_distill_types, trajectory_retrieval_groups)

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            distill_rewards=distill_rewards_step,
        )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, trajectory_distill_types, trajectory_retrieval_groups, subem_scores, answer_found, num_turns, has_search_metrics

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            current_training_steps: int,
            total_training_steps: int,
            ref_rollout_wg=None,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_distill_types = []
        total_retrieval_groups = []
        total_subem_scores = []
        total_answer_found = []
        total_num_turns = []
        dynamic_has_search_metrics = False
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, distill_types, retrieval_groups, _subem, _af, _nt, _hsm = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                current_training_steps=current_training_steps,
                total_training_steps=total_training_steps,
                is_train=True,
                ref_rollout_wg=ref_rollout_wg,
            )
            for idx, item_list in enumerate(batch_list):
                for step in item_list:
                    step['_temp_distill_types'] = distill_types[idx]
                    step['_temp_retrieval_group'] = retrieval_groups[idx]
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(
                batch_list=batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                traj_uid=traj_uid,
                tool_callings=tool_callings,
                config=self.config,
                last_try=(try_count == max_try_count),
            )
            surviving_distill_types = []
            surviving_retrieval_groups = []
            for item_list in batch_list:
                if len(item_list) > 0:
                    if '_temp_distill_types' in item_list[0]:
                        surviving_distill_types.append(item_list[0]['_temp_distill_types'])
                    else:
                        surviving_distill_types.append([])

                    if '_temp_retrieval_group' in item_list[0]:
                        surviving_retrieval_groups.append(item_list[0]['_temp_retrieval_group'])
                    else:
                        surviving_retrieval_groups.append("unknown")

                    for step in item_list:
                        if '_temp_distill_types' in step: del step['_temp_distill_types']
                        if '_temp_retrieval_group' in step: del step['_temp_retrieval_group']
                else:
                    surviving_distill_types.append([])
                    surviving_retrieval_groups.append("unknown")

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
            total_distill_types.extend(surviving_distill_types)
            total_retrieval_groups.extend(surviving_retrieval_groups)
            if _hsm:
                dynamic_has_search_metrics = True
                total_subem_scores.append(_subem)
                total_answer_found.append(_af)
                total_num_turns.append(_nt)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        if dynamic_has_search_metrics:
            total_subem_scores = np.concatenate(total_subem_scores, axis=0)
            total_answer_found = np.concatenate(total_answer_found, axis=0)
            total_num_turns = np.concatenate(total_num_turns, axis=0)
        else:
            total_subem_scores = None
            total_answer_found = None
            total_num_turns = None

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_distill_types, total_retrieval_groups, total_subem_scores, total_answer_found, total_num_turns, dynamic_has_search_metrics

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            current_training_steps: int,
            total_training_steps: int,
            is_train: bool = True,
            ref_rollout_wg=None,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        if self.config.algorithm.filter_groups.enable and is_train:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_distill_types, total_retrieval_groups, total_subem, total_af, total_nt, _hsm = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                current_training_steps=current_training_steps,
                total_training_steps=total_training_steps,
                ref_rollout_wg=ref_rollout_wg,
            )
        else:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_distill_types, total_retrieval_groups, total_subem, total_af, total_nt, _hsm = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                current_training_steps=current_training_steps,
                total_training_steps=total_training_steps,
                is_train=is_train,
                ref_rollout_wg=ref_rollout_wg,
            )

        # Validate dimensions (total_episode_rewards is flat np.ndarray from loop)
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(total_tool_callings)

        # Credit assignment: produces np.ndarray (object dtype) of per-step np.ndarray values
        # <<< CHANGE: Conditional credit assignment >>>
        if self.enable_credit_assignment:
            per_step_episode_rewards, total_discounted_returns = self.credit_assignment(
                total_batch_list, self.step_gamma
            )
        else:
            # If disabled, we assign the final cumulative reward to every step (similar to standard GRPO)
            # and set discounted returns to 0 (or simply equal to reward, depending on algorithm needs, 
            # but usually GRPO uses the full episode reward for all steps).
            per_step_episode_rewards, total_discounted_returns = self.no_credit_assignment(
                total_batch_list, total_episode_rewards
            )
        # <<< END CHANGE >>>

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=per_step_episode_rewards,
            discounted_returns=total_discounted_returns,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=total_tool_callings,
            distill_types_list=total_distill_types,
            retrieval_groups_list=total_retrieval_groups,
            subem_scores=total_subem if _hsm else None,
            answer_found=total_af if _hsm else None,
            num_turns=total_nt if _hsm else None,
        )
        print('rollout finished')

        return gen_batch_output

    def credit_assignment(self, total_batch_list, step_gamma=0.95):
        """
        Compute the 1) per-step episode reward and 2) step discounted return
        for each step in the trajectory.  Both outputs are 1-D np.ndarray of
        dtype=object, where each element is itself a np.ndarray(float64) whose
        length equals the number of steps in that trajectory.

        Parameters:
            total_batch_list (List[List[Dict]]): Per-trajectory list of step dicts.
                Each dict should contain 'rewards' (scalar) and 'active_masks' (bool).
            step_gamma (float): Discount factor for future rewards between
                consecutive steps within a trajectory.

        Returns:
            total_episode_rewards (np.ndarray[object]): 1-D object array of length
                num_trajectories.  total_episode_rewards[i] is a np.ndarray of shape
                (num_steps_i,) where every active step receives the cumulative
                (undiscounted) play-phase reward of trajectory i.
            total_discounted_returns (np.ndarray[object]): 1-D object array of length
                num_trajectories.  total_discounted_returns[i] is a np.ndarray of
                shape (num_steps_i,) with the backward-discounted return from each step.
        """
        num_trajectories = len(total_batch_list)
        total_episode_rewards = np.empty(num_trajectories, dtype=object)
        total_discounted_returns = np.empty(num_trajectories, dtype=object)

        for traj_idx, steps in enumerate(total_batch_list):
            n = len(steps)

            if n == 0:
                total_episode_rewards[traj_idx] = np.zeros(0, dtype=np.float64)
                total_discounted_returns[traj_idx] = np.zeros(0, dtype=np.float64)
                continue

            episode_rewards = np.zeros(n, dtype=np.float64)
            discounted_returns = np.zeros(n, dtype=np.float64)

            # --- 1. Compute cumulative (undiscounted) reward for the trajectory ---
            cumulative_reward = 0.0
            for t in range(n):
                step = steps[t]
                if step.get('active_masks', True):
                    reward_val = step.get('rewards', 0.0)
                    if hasattr(reward_val, 'item'):
                        reward_val = reward_val.item()
                    cumulative_reward += float(reward_val)

            # Assign cumulative reward to every active step
            for t in range(n):
                if steps[t].get('active_masks', True):
                    episode_rewards[t] = cumulative_reward

            # --- 2. Compute discounted returns (backward pass) ---
            running_return = 0.0
            for t in reversed(range(n)):
                step = steps[t]
                if not step.get('active_masks', True):
                    discounted_returns[t] = 0.0
                    continue
                reward_val = step.get('rewards', 0.0)
                if hasattr(reward_val, 'item'):
                    reward_val = reward_val.item()
                running_return = float(reward_val) + step_gamma * running_return
                discounted_returns[t] = running_return

            total_episode_rewards[traj_idx] = episode_rewards
            total_discounted_returns[traj_idx] = discounted_returns

        return total_episode_rewards, total_discounted_returns

    # <<< CHANGE: Add helper for no credit assignment >>>
    def no_credit_assignment(self, total_batch_list, final_episode_rewards):
        """
        Assigns the total episode reward to every step, effectively disabling
        per-step credit assignment. This mimics standard GRPO behavior where
        the outcome reward is applied to all tokens/steps.
        """
        num_trajectories = len(total_batch_list)
        total_step_rewards = np.empty(num_trajectories, dtype=object)
        total_discounted_returns = np.empty(num_trajectories, dtype=object)

        for traj_idx, steps in enumerate(total_batch_list):
            n = len(steps)
            if n == 0:
                total_step_rewards[traj_idx] = np.zeros(0, dtype=np.float64)
                total_discounted_returns[traj_idx] = np.zeros(0, dtype=np.float64)
                continue
            
            # Retrieve the final accumulated reward for this trajectory
            # final_episode_rewards is a 1D array of floats
            final_reward = float(final_episode_rewards[traj_idx])

            episode_rewards = np.zeros(n, dtype=np.float64)
            # For no credit assignment, we usually don't use discounted returns in the same way,
            # but to keep data structures consistent, we can just fill it with the final reward
            # or zeros depending on how the algorithm uses it. 
            # If the algorithm uses 'step_returns' for advantage, setting it to final_reward
            # makes it equivalent to outcome supervision.
            discounted_returns = np.zeros(n, dtype=np.float64)

            for t in range(n):
                if steps[t].get('active_masks', True):
                    episode_rewards[t] = final_reward
                    discounted_returns[t] = final_reward 

            total_step_rewards[traj_idx] = episode_rewards
            total_discounted_returns[traj_idx] = discounted_returns

        return total_step_rewards, total_discounted_returns
    # <<< END CHANGE >>>
    def get_distill_logs(self, distill_obs, text_actions, trajectory_success, distill_rewards, completion_percentages, current_training_steps, total_training_steps, distill_rewards_before_clipping=None, distill_rewards_after_clipping=None):
        """
        Saves distillation logs to a JSONL file for analysis.
        """
        file_path = self.config.data.get("distill_log_path", "./skill_analysis.jsonl")
        print("collecting distillation logs ....")
        if isinstance(distill_rewards, torch.Tensor):
            distill_rewards = distill_rewards.cpu().numpy()

        if completion_percentages is None:
            completion_percentages = [0.0] * len(distill_obs)
        elif isinstance(completion_percentages, (np.ndarray, torch.Tensor)):
            if isinstance(completion_percentages, torch.Tensor):
                completion_percentages = completion_percentages.cpu().numpy()

        samples_list = []
        current_step = current_training_steps

        for i, (distill_ob, generated_action, reward, completion, traj_success) in enumerate(zip(distill_obs, text_actions, distill_rewards, completion_percentages, trajectory_success)):
            if hasattr(reward, 'item'):
                reward = reward.item()
            if hasattr(completion, 'item'):
                completion = completion.item()

            simple_distill_log = {
                'trajectory_success': float(traj_success),
                'distill_observation': distill_ob,
                'generated_skill': generated_action,
                'distill_reward': float(reward),
                'task_completion_percentage': float(completion),
                'distill_reward_before_clipping': float(distill_rewards_before_clipping[i]) if distill_rewards_before_clipping is not None else None,
                'distill_reward_after_clipping': float(distill_rewards_after_clipping[i]) if distill_rewards_after_clipping is not None else None,
            }
            samples_list.append(simple_distill_log)

        step_log_entry = {
            "step": current_step,
            "total_steps": total_training_steps,
            "distill_policy": "reference" if self.use_ref_policy_for_distill else "actor",
            "batch_samples": samples_list
        }

        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(step_log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error saving distill logs: {e}")

    def get_traj_cot_logs(self, total_batch_list, current_training_steps, total_training_steps, trajectory_distill_types=None, trajectory_retrieval_groups=None):
        '''
        Collects trajectories in text, assigns accumulated rewards, and saves them to a log file.
        '''
        file_path = self.config.data.get("trajectory_log_path", "./trajectory_analysis.jsonl")
        import os
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print(f"Collecting trajectory logs for {len(total_batch_list)} trajectories...", flush=True)

        traj_cot_logs = []

        for traj_idx, traj_batch in enumerate(total_batch_list):
            cot_log = {'reward': 0.0, 'trajectory': '', 'distill_types': [], 'retrieval_group': 'unknown'}

            if trajectory_distill_types is not None and traj_idx < len(trajectory_distill_types):
                cot_log['distill_types'] = trajectory_distill_types[traj_idx]

            if trajectory_retrieval_groups is not None and traj_idx < len(trajectory_retrieval_groups):
                cot_log['retrieval_group'] = trajectory_retrieval_groups[traj_idx]

            if not cot_log['distill_types'] and len(traj_batch) > 0:
                if 'distill_types' in traj_batch[0]:
                    cot_log['distill_types'] = traj_batch[0]['distill_types']

            if cot_log['retrieval_group'] == 'unknown' and len(traj_batch) > 0:
                if 'retrieval_group' in traj_batch[0]:
                    cot_log['retrieval_group'] = traj_batch[0]['retrieval_group']

            action_idx = 0

            for i, step in enumerate(traj_batch):
                input_text = self.tokenizer.decode(step['input_ids'], skip_special_tokens=True)
                input_text = input_text.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant", "").split('assistant')[0]

                text_action = self.tokenizer.decode(step['responses'], skip_special_tokens=True)

                if step.get('active_masks', True):
                    cot_log['trajectory'] += f"\n#### step {action_idx} #### \n"
                    cot_log['trajectory'] += f"[Input]\n {input_text.strip()}\n"
                    cot_log['trajectory'] += f"[Response]\n {text_action.strip()}\n"

                    if hasattr(step['rewards'], 'item'):
                        cot_log['reward'] = float(step['rewards'].item())
                    else:
                        cot_log['reward'] = float(step['rewards'])

                    action_idx += 1

            traj_cot_logs.append(cot_log)

        step_log_entry = {
            "step": current_training_steps,
            "total_steps": total_training_steps,
            "trajectories": traj_cot_logs
        }

        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(step_log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error saving trajectory logs: {e}")

        return traj_cot_logs