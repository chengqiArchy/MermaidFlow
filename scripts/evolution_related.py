from pydantic import BaseModel, Field
import random
import asyncio
from scripts.evolution_prompt import (
    GENERATE_CHILDREN_PROMPT,
    WORKFLOW_INPUT_WITH_GRAPH,
    CHANGE_MERMAID_PROMPT,
    LLM_AS_JUDGER,
    PROMPT_FEW_SHOT,
)
from scripts.mermaid_workflow import MERMAID_CODE_GUIDANCE
from scripts.formatter import XmlFormatter
from scripts.prompts.optimize_prompt import (
    WORKFLOW_OPTIMIZE_PROMPT_WITH_MERMAID,
    WORKFLOW_CUSTOM_USE_WITH_MERMAID,
)
from scripts.mermaid_workflow import (
    GraphOptimizeForMermaid,
    GraphOptimizeWithMermaid,
    GraphForEvolutionaryAlgorithm,
    FinalSelection,
    LLMAsJudge,
    LEARN_FROM_FAILED_MERMAID,
)
import weave
import numpy as np
from scripts.logs import logger


class RoundLog(BaseModel):
    graph: str = Field(default="", description="graph written by Mermaid")
    prompt: str = Field(default="", description="prompt")
    score: float = Field(default=0.0, description="score")
    code: str = Field(default="", description="code written by Python")
    round_id: int = Field(default=0, description="round_id")


def _compute_probabilities(scores, alpha=0.2, lambda_=0.3):
    scores = np.array(scores, dtype=np.float64)
    n = len(scores)

    if n == 0:
        raise ValueError("Score list is empty.")

    uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

    max_score = np.max(scores)
    shifted_scores = scores - max_score
    exp_weights = np.exp(alpha * shifted_scores)

    sum_exp_weights = np.sum(exp_weights)
    if sum_exp_weights == 0:
        raise ValueError("Sum of exponential weights is 0, cannot normalize.")

    score_prob = exp_weights / sum_exp_weights

    mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

    total_prob = np.sum(mixed_prob)
    if not np.isclose(total_prob, 1.0):
        mixed_prob = mixed_prob / total_prob

    return mixed_prob


def get_probabilities(elites):
    scores = [item["score"] * 100 for item in elites]
    mixed_prob = _compute_probabilities(scores)
    elites_with_prob = [
        {**elite, "prob": prob} for elite, prob in zip(elites, mixed_prob)
    ]
    return elites_with_prob


def get_parents_pair(elites, pair_num, is_prob=True):
    parents_pairs = []
    used_pairs = set()  # 用于跟踪已经使用过的配对

    for _ in range(pair_num):
        elite_count = len(elites)

        if elite_count < 2:
            raise ValueError(
                "Not enough elites to form a pair. Need at least 2 elites."
            )

        # 尝试找到一个未使用过的配对
        attempts = 0
        max_attempts = 100  # 防止无限循环

        while attempts < max_attempts:
            if is_prob:
                elites_with_prob = get_probabilities(elites)
                parent1_idx = random.choices(
                    range(0, elite_count),
                    weights=[item["prob"] for item in elites_with_prob],
                    k=1,
                )[0]
                parent2_idx = random.choices(
                    range(0, elite_count),
                    weights=[item["prob"] for item in elites_with_prob],
                    k=1,
                )[0]
            else:
                parent1_idx = random.randint(0, elite_count - 1)
                parent2_idx = random.randint(0, elite_count - 1)

            while parent2_idx == parent1_idx:
                parent2_idx = random.randint(0, elite_count - 1)

            # 检查这个配对或其反向配对是否已经使用过
            pair_key = (parent1_idx, parent2_idx)
            reverse_pair_key = (parent2_idx, parent1_idx)

            if pair_key not in used_pairs and reverse_pair_key not in used_pairs:
                used_pairs.add(pair_key)
                parents_pairs.append((elites[parent1_idx], elites[parent2_idx]))
                break

            attempts += 1

        # 如果尝试了最大次数仍未找到未使用的配对，则发出警告并使用最后一个生成的配对
        if attempts == max_attempts:
            print("Warning: Could not find unique parent pairs after maximum attempts.")
            parents_pairs.append((elites[parent1_idx], elites[parent2_idx]))

    return parents_pairs


def get_next_workflow_prompt(
    parents_pair: list[dict],
    elites: list[dict],
    type: str,
):
    elites_history = "\n".join(
        [
            f"\nhistory_graph_{index}:\n{item['graph']}\nhistory_prompt_{index}:\n{item['prompt']}\nhistory_score_{index}:\n{item['score']}\n\n"
            for index, item in enumerate(elites)
        ]
    )

    prompt = GENERATE_CHILDREN_PROMPT.format(
        graph_A=parents_pair[0]["graph"],
        prompt_A=parents_pair[0]["prompt"],
        score_A=parents_pair[0]["score"],
        graph_B=parents_pair[1]["graph"],
        prompt_B=parents_pair[1]["prompt"],
        score_B=parents_pair[1]["score"],
        mermiad_custom_prompt=MERMAID_CODE_GUIDANCE[type],
        elites_history=elites_history,
        prompt_few_shot=PROMPT_FEW_SHOT[type],
    )

    return prompt


def get_graph_candidates_prompt(
    parents: list[dict],
    graph_candidates: dict,
    elites: list[dict],
    type: str,
):
    elites_history = "\n".join(
        [
            f"\nhistory_graph_{index}:\n{item['graph']}\nhistory_prompt_{index}:\n{item['prompt']}\nhistory_score_{index}:\n{item['score']}\n\n"
            for index, item in enumerate(elites)
        ]
    )
    prompt = LLM_AS_JUDGER.format(
        parent_graph_A=parents[0]["graph"],
        parent_prompt_A=parents[0]["prompt"],
        parent_graph_B=parents[1]["graph"],
        parent_prompt_B=parents[1]["prompt"],
        graph_A=graph_candidates["graph_A"],
        modification_A=graph_candidates["modification_A"],
        prompt_A=graph_candidates["prompt_A"],
        graph_B=graph_candidates["graph_B"],
        modification_B=graph_candidates["modification_B"],
        prompt_B=graph_candidates["prompt_B"],
        graph_C=graph_candidates["graph_C"],
        modification_C=graph_candidates["modification_C"],
        prompt_C=graph_candidates["prompt_C"],
        graph_D=graph_candidates["graph_D"],
        modification_D=graph_candidates["modification_D"],
        prompt_D=graph_candidates["prompt_D"],
        elites_history=elites_history,
        mermaid_usage=MERMAID_CODE_GUIDANCE[type],
    )
    return prompt


def get_prompt_for_generate_code(
    parents_pair: list[dict],
    type: str,
    operator_description: str,
    new_graph: str,
    new_prompt: str,
):
    prompt = WORKFLOW_INPUT_WITH_GRAPH.format(
        graph_A=parents_pair[0]["graph"],
        code_A=parents_pair[0]["code"][0],
        prompt_A=parents_pair[0]["prompt"],
        graph_B=parents_pair[1]["graph"],
        code_B=parents_pair[1]["code"][0],
        prompt_B=parents_pair[1]["prompt"],
        new_mermaid=new_graph,
        new_prompt=new_prompt,
        operator_description=operator_description,
    )

    graph_system = WORKFLOW_OPTIMIZE_PROMPT_WITH_MERMAID.format(type=type)
    return prompt + WORKFLOW_CUSTOM_USE_WITH_MERMAID + graph_system


@weave.op()
async def get_reponse(
    parents_pair: list[dict],
    elites: list[dict],
    operator_description: str,
    type: str,
    optimize_llm,
    mermaid_checker,
):
    last_mermaid = None
    errors_for_last_mermaid = None

    async def get_graph_candidates(parents_pair, type, last_mermaid, errors_for_last_mermaid):
        mermaid_optimize_prompt = get_next_workflow_prompt(
            parents_pair=parents_pair, elites=elites, type=type
        )
        if last_mermaid and errors_for_last_mermaid:
            extra_info = LEARN_FROM_FAILED_MERMAID.format(failed_mermaid=last_mermaid, error_message=errors_for_last_mermaid)
            mermaid_optimize_prompt += extra_info
        

        mermaid_format = XmlFormatter.from_model(GraphForEvolutionaryAlgorithm)
        graph_response = await optimize_llm.call_with_format(
            mermaid_optimize_prompt, mermaid_format
        )
        # llm as judge
        graph_candidates_prompt = get_graph_candidates_prompt(
            parents=parents_pair,
            graph_candidates=graph_response,
            elites=elites,
            type=type,
        )
        graph_candidates_format = XmlFormatter.from_model(LLMAsJudge)
        graph_candidates_response = await optimize_llm.call_with_format(
            graph_candidates_prompt, graph_candidates_format
        )
        return graph_candidates_response, graph_response


    @weave.op()
    def get_graph(graph_decision, graph_candidates):
        try:
            new_graph = graph_candidates["graph_" + graph_decision["selected_graph"]]
            new_prompt = graph_candidates["prompt_" + graph_decision["selected_graph"]]
            new_modification = graph_candidates[
                "modification_" + graph_decision["selected_graph"]
            ]
        except Exception:
            new_graph = graph_candidates["graph_A"]
            new_prompt = graph_candidates["prompt_A"]
            new_modification = graph_candidates["modification_A"]
        return new_graph, new_prompt, new_modification

    try_number = 3
    for _ in range(try_number):
        graph_decision, graph_candidates = await get_graph_candidates(parents_pair, type, last_mermaid, errors_for_last_mermaid)
        new_graph, new_prompt, new_modification = get_graph(
            graph_decision, graph_candidates
        )
        if mermaid_checker:
            mermaid_path = mermaid_checker.transfer_mmd_code_string_to_temp_file(new_graph)
            is_hard_pass, hard_check_info = mermaid_checker.hard_check(mermaid_path)
            is_soft_pass, soft_check_info = mermaid_checker.soft_check(mermaid_path)
            logger.info(hard_check_info)
            logger.info(soft_check_info)
            last_mermaid = new_graph
            errors_for_last_mermaid = hard_check_info + soft_check_info
            if is_hard_pass and is_soft_pass:
                break
        else:
            break
    
    @weave.op()
    async def generate_code(new_graph, new_prompt, new_modification):
        generate_code_prompt = get_prompt_for_generate_code(
            parents_pair=parents_pair,
            type=type,
            operator_description=operator_description,
            new_graph=new_graph,
            new_prompt=new_prompt,
        )

        generate_code_format = XmlFormatter.from_model(GraphOptimizeWithMermaid)
        python_code_response = await optimize_llm.call_with_format(
            generate_code_prompt, generate_code_format
        )
        return python_code_response

    python_code_response = await generate_code(new_graph, new_prompt, new_modification)

    final_response = {
        "code": python_code_response["code"],
        "prompt": python_code_response["prompt"],
        "graph": new_graph,
        "modification": new_modification,
    }

    return final_response
