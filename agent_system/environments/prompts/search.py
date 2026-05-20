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

SEARCH_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}
{reflections}
Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}
{reflections}
Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

# --------------------- Reflection --------------------- #
SEARCH_REFLECT_TEMPLATE = """
You are an expert evaluating a question-answering attempt that uses an external search engine.
The question was: {task_description}

The attempt was {success} completed (i.e., the final answer was {success_detail}).

{reference_trajectory}

Trajectory of the attempt:
{current_trajectory}

<think>
If a reference trajectory exists, compare it with the current trajectory.
Given the task outcome, analyze the trajectory to understand:
1. What search queries were issued? Were they effective at retrieving relevant information?
2. Was the retrieved information correctly interpreted and synthesized?
3. What reasoning led to the final answer? Was it sound?
4. What is the most valuable lesson from this attempt?
</think>

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "query_formulation", "description": "[describe the search queries used, e.g., 'Searched for total death row inmates US']", "status": "[completed or incomplete]"}},
{{"name": "information_extraction", "description": "[describe how search results were used, e.g., 'Extracted count from Wikipedia snippet']", "status": "[completed or incomplete]"}},
{{"name": "reasoning", "description": "[describe the reasoning chain, e.g., 'Connected multiple facts to derive answer']", "status": "[completed or incomplete]"}},
{{"name": "answer_synthesis", "description": "[describe the final answer formation, e.g., 'Provided concise factual answer']", "status": "[completed or incomplete]"}}
],
"task_success": [true if the final answer was correct, false if incorrect],
"search_lesson": "[Key insight about search strategy, e.g., 'Searching for the specific entity name rather than a broad topic yields more precise results' OR 'The first search result was misleading; cross-referencing with a second query would have helped']",
"reasoning_lesson": "[Key insight about reasoning, e.g., 'The question asked for the year, not the full date — need to parse the question more carefully' OR 'Multi-hop questions require chaining facts from separate searches']"
}}

EVALUATION GUIDELINES:
- The task outcome has been provided — use it to set task_success accordingly
- Focus on WHY the attempt had this outcome:
  * If successful: What search/reasoning strategy worked well?
  * If unsuccessful: What query was ineffective or what reasoning step failed?
- Each subtask status must reflect actual trajectory events
- Lessons should be generalizable to similar future questions
- Reference specific queries, search results, or reasoning steps from the trajectory
- Use null for lessons only if truly not applicable

Output ONLY the JSON evaluation.
"""

SEARCH_REFLECT_TEMPLATE_WITH_DESC_HEAD = """
You are an expert evaluating a question-answering attempt that uses an external search engine.
The question was: {task_description}

The attempt was {success} completed (i.e., the final answer was {success_detail}).

{reference_trajectory}

Trajectory of the attempt:
{current_trajectory}

<think>
If a reference trajectory exists, compare it with the current trajectory.
Given the task outcome, analyze the trajectory to understand:
1. What search queries were issued? Were they effective at retrieving relevant information?
2. Was the retrieved information correctly interpreted and synthesized?
3. What reasoning led to the final answer? Was it sound?
4. What is the most valuable lesson from this attempt?
5. In what general scenarios would this lesson be useful to a future agent?
</think>

Output your evaluation as JSON:

{{
"subtasks": [
{{"name": "query_formulation", "description": "[describe the search queries used]", "status": "[completed or incomplete]"}},
{{"name": "information_extraction", "description": "[describe how search results were used]", "status": "[completed or incomplete]"}},
{{"name": "reasoning", "description": "[describe the reasoning chain]", "status": "[completed or incomplete]"}},
{{"name": "answer_synthesis", "description": "[describe the final answer formation]", "status": "[completed or incomplete]"}}
],
"task_success": [true if the final answer was correct, false if incorrect],
"search_lesson": "[Key insight about search strategy]",
"reasoning_lesson": "[Key insight about reasoning]",
"description_head": "[Describe in 1-2 sentences WHEN this lesson would be useful. Focus on the general question type, conditions, or challenges — not the specific question. e.g., 'Useful when the agent needs to answer multi-hop questions requiring chaining facts from separate entities.' or 'Useful when the question involves numerical comparison across multiple data sources.']"
}}

EVALUATION GUIDELINES:
- The task outcome has been provided — use it to set task_success accordingly
- Focus on WHY the attempt had this outcome
- Each subtask status must reflect actual trajectory events
- Lessons should be generalizable to similar future questions
- description_head should describe the GENERAL SCENARIO where this lesson applies, not just restate the current question

Output ONLY the JSON evaluation.
"""

# --------------------- BiGen: Query Generation --------------------- #
SEARCH_QUERY_GENERATION_TEMPLATE = """Question: {task_description}

Write a one-sentence search query to find relevant past experiences for this type of question. Do NOT answer the question.
Example: <query>tips for answering multi-hop factual questions requiring date comparisons</query>

<query>"""

# --------------------- Re-rank --------------------- #
SEARCH_RERANK_TEMPLATE = """You are about to answer a question using an external search engine.

Question: {task_description}

Below are {n_candidates} past experiences retrieved from memory. Each is labeled with an ID.

{candidate_experiences}

Rank these experiences from MOST useful to LEAST useful for the current question.
Consider which experience addresses the specific challenges you expect to face.

Output ONLY the ranked IDs as a comma-separated list within <rank> </rank> tags.
For example, if experience 3 is most useful, then 1, then 2: <rank>3,1,2</rank>
"""

SEARCH_RERANK_DUMMY_TEMPLATE = """You are about to answer a question using an external search engine.

Question: {task_description}

No past experiences are available for this question. Output <rank>none</rank> to proceed.
"""
