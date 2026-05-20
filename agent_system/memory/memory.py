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

from typing import List, Dict, Any, Tuple
from .base import BaseMemory

class SimpleMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )
                if 'dones' in rec.keys() and rec['dones']:
                    valid_len = step_num
                    break

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths

class WebshopSimpleMemory(BaseMemory):
    """
    Memory manager: responsible for storing & fetching per‑environment history records.
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str = "text_obs",
        action_key: str = "action",
        max_to_show=15,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str, default="text_obs"):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str, default="action"):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                ## process obs to reduce tokens, modified as per environment ##
                if j != len(recent) - 1 and len(obs.split(' [SEP] ')) >= max_to_show:
                    items = obs.split(' [SEP] ')
                    obs = ' [SEP] '.join(items[:max_to_show]) + f' ... ({len(items) - max_to_show} more)'
                ## --- ##
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )
                if 'dones' in rec.keys() and rec['dones']:
                    valid_len = step_num
                    break

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths

import json
import os
import math
import uuid
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

class SkillLibrary:
    def __init__(self,
                 filepath="memory_cache/skill_library.json",
                 model_name="all-MiniLM-L6-v2",
                 relevance_weight=0.6,
                 alpha=0.1,
                 temperature=0.5,
                 retrieve_type="ucb",
                 ucb_scale=1.0,
                 use_description_head=False,
                 max_size=5000,
                 retriever_type="dense"):
        self.filepath = filepath
        self.relevance_weight = relevance_weight
        self.alpha = alpha
        self.temperature = temperature
        self.retrieve_type = retrieve_type.lower()
        self.ucb_scale = ucb_scale
        self.use_description_head = use_description_head
        self.max_size = max_size
        self._last_evicted_count = 0
        self.retriever_type = retriever_type.lower()

        if self.retriever_type == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
            print(f"Using TF-IDF retriever (no dense model loaded)")
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = None
            self._sklearn_cosine = sklearn_cosine
            self.model = None
        else:
            model_path = os.environ.get("EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
            print(f"Loading embedding model: {model_path}...")
            self.model = SentenceTransformer(model_path)
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None

        self.data = []
        self.task_embeddings = None
        self._current_training_step = 0

        self.prune_milestones = [0.25, 0.5, 0.75]
        self.processed_milestones = set()

        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    self.data = []
        self._rebuild_embeddings()

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def _rebuild_embeddings(self):
        if not self.data:
            self.task_embeddings = None
            self.tfidf_matrix = None
            return
        if self.use_description_head:
            texts = [item.get('description_head') or item['scenario_desc'] for item in self.data]
        else:
            texts = [item['scenario_desc'] for item in self.data]
        if self.retriever_type == "tfidf":
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self.task_embeddings = None
        else:
            self.task_embeddings = self.model.encode(texts, convert_to_tensor=True)
            self.tfidf_matrix = None

    def _compute_similarity_single(self, text_a: str, text_b: str) -> float:
        if self.retriever_type == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            tmp_vec = TfidfVectorizer()
            vecs = tmp_vec.fit_transform([text_a, text_b])
            return float(self._sklearn_cosine(vecs[0:1], vecs[1:2])[0][0])
        emb1 = self.model.encode(text_a, convert_to_tensor=True)
        emb2 = self.model.encode(text_b, convert_to_tensor=True)
        return util.cos_sim(emb1, emb2).item()

    def admit(self,
            scenario_description: str,
            strategy_text: str,
            trajectory: str,
            initial_score: float = 0.5,
            attempt_type: str = "unknown",
            current_progress_ratio: float = 0.0,
            description_head: str = ""):
        """
        Adds a memory entry.
        1. Exact-match on strategy_text, then check task similarity via cached embeddings.
        2. If similar (>0.85): merge (increment count, update utility).
        3. If not found, creates new entry. Evicts if over max_size.
        """
        for i, item in enumerate(self.data):
            if item['strategy'] == strategy_text:
                # Use cached embeddings for similarity check
                if self.retriever_type == "tfidf":
                    sim = self._compute_similarity_single(item['scenario_desc'], scenario_description)
                elif self.task_embeddings is not None and i < self.task_embeddings.shape[0]:
                    query_emb = self.model.encode(scenario_description, convert_to_tensor=True)
                    sim = util.cos_sim(query_emb, self.task_embeddings[i:i+1])[0][0].item()
                else:
                    sim = self._compute_similarity_single(item['scenario_desc'], scenario_description)
                if sim > 0.85:
                    self._ema_update(item, initial_score)
                    if item.get('attempt_type', 'unknown') == 'unknown':
                        item['attempt_type'] = attempt_type
                    if description_head and not item.get('description_head'):
                        item['description_head'] = description_head
                    self.save()
                    return
        new_entry = {
            "skill_id": str(uuid.uuid4()),
            "scenario_desc": scenario_description,
            "description_head": description_head,
            "strategy": strategy_text,
            "trajectory": trajectory,
            "utility_score": initial_score,
            "count": 1,
            "attempt_type": attempt_type,
            "created_at_progress": current_progress_ratio,
            "created_at_step": self._current_training_step,
        }
        self.data.append(new_entry)

        if self.retriever_type == "tfidf":
            self._rebuild_embeddings()
        else:
            embed_text = scenario_description
            if self.use_description_head and description_head:
                embed_text = description_head
            new_emb = self.model.encode(embed_text, convert_to_tensor=True)
            if self.task_embeddings is None:
                self.task_embeddings = new_emb.unsqueeze(0)
            else:
                self.task_embeddings = torch.cat((self.task_embeddings, new_emb.unsqueeze(0)), dim=0)

        self.save()
        self._retire()

    def update_utility(self, scenario_description: str, strategy_text: str, score: float):
        updated = False
        for item in self.data:
            if item['strategy'] == strategy_text:
                self._ema_update(item, score)
                updated = True
                break
        if updated:
            self.save()

    def _ema_update(self, item, new_score):
        current_avg = item['utility_score']
        new_avg = (1 - self.alpha) * current_avg + self.alpha * new_score
        item['utility_score'] = new_avg
        item['count'] += 1

    def _retire(self):
        """Evict lowest-scoring entries when memory exceeds max_size."""
        if len(self.data) <= self.max_size:
            self._last_evicted_count = 0
            return
        import math
        # Tag each entry with its original index for embedding reordering
        for i, item in enumerate(self.data):
            item['_retirement_score'] = item.get('utility_score', 0.5) * math.log2(item.get('count', 1) + 1)
            item['_orig_idx'] = i
        self.data.sort(key=lambda x: x['_retirement_score'], reverse=True)
        n_evicted = len(self.data) - self.max_size
        self.data = self.data[:self.max_size]
        # Reorder embeddings to match new sorted order instead of rebuilding
        kept_indices = [item.pop('_orig_idx') for item in self.data]
        for item in self.data:
            item.pop('_retirement_score', None)
        if self.retriever_type == "tfidf":
            if self.tfidf_matrix is not None:
                self.tfidf_matrix = self.tfidf_matrix[kept_indices]
        else:
            if self.task_embeddings is not None and self.task_embeddings.shape[0] > 0:
                self.task_embeddings = self.task_embeddings[kept_indices]
        self._last_evicted_count = n_evicted
        self.save()
        print(f"[Memory] Evicted {n_evicted} entries, {len(self.data)} remaining")

    def check_and_prune(self, progress_ratio: float, top_k: int = 3):
        for milestone in self.prune_milestones:
            if progress_ratio >= milestone and milestone not in self.processed_milestones:
                print(f"[Memory] Milestone {milestone*100}% reached. Pruning memory...")
                self._prune_memory(top_k=top_k)
                self.processed_milestones.add(milestone)

    def _prune_memory(self, top_k: int):
        if not self.data: return
        grouped_memory = defaultdict(list)
        for item in self.data:
            grouped_memory[item['scenario_desc']].append(item)
        
        pruned_data = []
        for task_desc, items in grouped_memory.items():
            items.sort(key=lambda x: x.get('utility_score', 0.0), reverse=True)
            kept_items = items[:top_k]
            for item in kept_items:
                item['utility_score'] = item['utility_score'] * 0.9
                item['count'] = 1 
            pruned_data.extend(kept_items)

        self.data = pruned_data
        self.save()
        self._rebuild_embeddings()

    def retrieve_trajectory(self, current_scenario_description: str, target_type: str = "success") -> Optional[str]:
        """
        Specific retrieval to find a trajectory of a specific type (success/failure) 
        for the exact same task (or highly similar).
        """
        if not self.data:
            return None

        # 1. Try exact match first
        for item in self.data:
            if item['scenario_desc'] == current_scenario_description and item['attempt_type'] == target_type:
                return item['trajectory']
        
        # 2. If no exact match, try semantic similarity (optional, depending on strictness)
        # For now, we return None to ensure we only pair strictly relevant trajectories.
        return None


    # --- UPDATED RETRIEVE METHOD ---
    def retrieve(self, current_scenario_description: str, top_k=3, filter_type="both") -> List[Dict[str, str]]:
        """
        Retrieves memories based on similarity and utility.
        
        Args:
            current_scenario_description: The task to find memories for.
            top_k: Number of memories to return.
            filter_type: 'success', 'failure', or 'both'. Filters memories by their source attempt type.
            
        Returns:
            A list of dictionaries, e.g.:
            [
                {"text": "Use the search bar...", "type": "success"},
                {"text": "Do not click back...", "type": "failure"}
            ]
        """
        if not self.data:
            return []

        if self.retriever_type == "tfidf":
            if self.tfidf_matrix is None:
                self._rebuild_embeddings()
            if self.tfidf_matrix is None:
                return []
            query_vec = self.tfidf_vectorizer.transform([current_scenario_description])
            cos_scores_np = self._sklearn_cosine(query_vec, self.tfidf_matrix)[0]
        else:
            if self.task_embeddings is None:
                return []
            if self.task_embeddings.shape[0] != len(self.data):
                self._rebuild_embeddings()
            query_embedding = self.model.encode(current_scenario_description, convert_to_tensor=True)
            cos_scores_np = util.cos_sim(query_embedding, self.task_embeddings)[0].cpu().numpy()

        candidates = []
        total_system_counts = sum(item.get('count', 1) for item in self.data)
        if total_system_counts < 1: total_system_counts = 1

        for idx, item in enumerate(self.data):
            # --- FILTERING LOGIC ---
            if filter_type != "both":
                item_type = item.get('attempt_type', 'unknown')
                # If we want success, skip failures/unknowns. If we want failure, skip success/unknowns.
                if item_type != filter_type:
                    continue

            relevance = float(cos_scores_np[idx])
            if relevance < 0.4:
                continue

            utility = item.get('utility_score', 0.5)
            count = item.get('count', 1)
            final_score = 0.0

            if self.retrieve_type == "relevance_only":
                final_score = relevance
            elif self.retrieve_type == "ucb":
                if count == 0: count = 1 
                exploration_bonus = self.ucb_scale * math.sqrt(math.log(total_system_counts) / count)
                ucb_utility = utility + exploration_bonus
                final_score = (self.relevance_weight * relevance) + ((1 - self.relevance_weight) * ucb_utility)
            else:
                final_score = (self.relevance_weight * relevance) + ((1 - self.relevance_weight) * utility)

            # Store the text AND the type
            candidates.append({
                "text": item['strategy'],
                "score": final_score,
                "type": item.get('attempt_type', 'unknown'),
                "similarity_score": float(relevance),
                "skill_id": item.get('skill_id', ''),
                "utility_score": item.get('utility_score', 0.5),
            })

        if not candidates:
            return []

        def _to_result(c):
            return {"text": c["text"], "type": c["type"],
                    "similarity_score": c["similarity_score"], "skill_id": c["skill_id"],
                    "utility_score": c["utility_score"]}

        if len(candidates) <= top_k:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return [_to_result(c) for c in candidates]

        if self.retrieve_type == "softmax":
            scores = np.array([c['score'] for c in candidates])
            temp = max(self.temperature, 1e-5)
            exp_scores = np.exp((scores - np.max(scores)) / temp)
            probabilities = exp_scores / np.sum(exp_scores)
            selected_indices = np.random.choice(len(candidates), size=top_k, replace=False, p=probabilities)
            return [_to_result(candidates[i]) for i in selected_indices]
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return [_to_result(c) for c in candidates[:top_k]]


class SearchMemory(BaseMemory):
    """
    Memory manager for search tasks: responsible for storing & fetching
    """
    def __init__(self):
        self._data = None
        self.keys = None
        self.batch_size = 0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.keys = None

    def store(self, record: Dict[str, List[Any]]):
        """
        Store a new record (one step of history) for each environment instance.

        Args:
            record (Dict[str, List[Any]]):
                A dictionary where each key corresponds to a type of data 
                (e.g., 'text_obs', 'action'), and each value is a list of 
                length `batch_size`, containing the data for each environment.
        """
        if self.keys is None:
            self.keys = list(record.keys())
        assert self.keys == list(record.keys())

        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in self.keys})

    def fetch(
        self,
        history_length: int,
        obs_key: str,
        action_key: str,
    ) -> Tuple[List[str], List[int]]:
        """
        Fetch and format recent interaction history for each environment instance.
        Args:
            history_length (int):
                Maximum number of past steps to retrieve per environment.
            obs_key (str):
                The key name used to access the observation in stored records.
                For example: "text_obs" or "Observation", depending on the environment.
            action_key (str):
                The key name used to access the action in stored records.
                For example: "action" or "Action".
        Returns:
            memory_contexts : List[str]
                A list of formatted action history strings for each environment.
            valid_lengths : List[int]
                A list of the actual number of valid history steps per environment.
        """
        memory_contexts, valid_lengths = [], []

        for env_idx in range(self.batch_size):
            recent = self._data[env_idx][-history_length:]
            valid_len = len(recent)
            start_idx = len(self._data[env_idx]) - valid_len

            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                act = rec[action_key]
                obs = rec[obs_key]
                lines.append(
                    f"Step {step_num}:{act} {obs}\n"
                )

            memory_contexts.append("\n".join(lines))
            valid_lengths.append(valid_len)

        return memory_contexts, valid_lengths