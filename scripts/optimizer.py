# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph (updated with AsyncLLM integration)

import asyncio
import time
from typing import List, Literal, Dict

from pydantic import BaseModel, Field

from scripts.evaluator import DatasetType
from scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from scripts.optimizer_utils.data_utils import DataUtils
from scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from scripts.optimizer_utils.experience_utils import ExperienceUtils
from scripts.optimizer_utils.graph_utils import GraphUtils
from scripts.async_llm import create_llm_instance
from scripts.formatter import XmlFormatter, FormatError
from scripts.evolution_related import get_parents_pair
from scripts.logs import logger
import scripts.mermaid_workflow as mermaid_workflow
import scripts.evolution_related as evolution_related
import pickle
import random
from scripts.evolution_related import RoundLog
from MermaidFlow_Linter import MermaidCheker, ConfigCls
from pathlib import Path


import weave
from scripts.extract_dataset import (
    HardProblem,
    transfer_format,
    collect_hard_problems,
    get_single_new_instruction,
    PromptOptimizationConfig,
)

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class GraphOptimizeWithMermaid(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph represent by mermaid code")
    code: str = Field(default="", description="python code")
    prompt: str = Field(default="", description="prompt")


segment_define = Path("MermaidFlow_Linter/config/sgement_config.json")
checker_config = Path("MermaidFlow_Linter/config/checker_config.json")
# checker_config= Path("MermaidFlow_Linter/config/checker_config_code.json") # for code


@weave.op()
async def get_next_workflow(
    elites,
    elites_all,
    experience,
    score,
    code,
    prompt,
    operator_description,
    graph_mermaid,
    type,
    log_data,
    optimize_llm,
    is_evolution,
    round_id,  # this is just for log
):
    config = ConfigCls(segmentor_config=segment_define, checker_config=checker_config)

    mermaid_checker = MermaidCheker(config=config)

    if is_evolution:
        parents_pair = get_parents_pair(elites, 2)
        response = await evolution_related.get_reponse(
            parents_pair=parents_pair[0],
            operator_description=operator_description,
            elites=elites_all,
            type=type,
            optimize_llm=optimize_llm,
            mermaid_checker=mermaid_checker,
        )
    else:
        response = await mermaid_workflow.get_response(
            experience=experience,
            score=score,
            code=code,
            prompt=prompt,
            operator_description=operator_description,
            graph_mermaid=graph_mermaid,
            type=type,
            log_data=log_data,
            optimize_llm=optimize_llm,
            mermaid_checker=mermaid_checker,
        )

    return response


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
        client=None,
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)

        self.client = client

    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            test_n = 1  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test())
            return None

        for opt_round in range(self.max_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            view = weave.SavedView()

            retry_count = 0
            max_retries = 1

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(
                        f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})"
                    )
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = (
                self.convergence_utils.check_convergence(top_k=3)
            )

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break

            time.sleep(5)

    async def _optimize_graph(self):
        validation_n = self.validation_rounds  # validation datasets's execution number
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)
        view = weave.SavedView()
        view = view.filter_op("Evaluator.graph_evaluate").sort_by("started_at", "desc")

        # Create a loop until the generated graph meets the check conditions
        while True:
            directory = self.graph_utils.create_round_directory(
                graph_path, self.round + 1
            )

            top_rounds = self.data_utils.get_top_rounds(self.sample)
            sample = self.data_utils.select_round(top_rounds)

            def get_code_graph_and_prompt(round_id, score):
                prompt, graph_load = self.graph_utils.read_graph_files(
                    round_id, graph_path
                )
                code = self.graph_utils.extract_solve_graph(graph_load)
                mermaid_graph = self.graph_utils.read_mermaid_files(
                    round_id, graph_path
                )
                return {
                    "graph": mermaid_graph,
                    "prompt": prompt,
                    "score": score,
                    "code": code,
                    "round_id": round_id,
                }

            elites = [
                get_code_graph_and_prompt(key["round"], key["score"])
                for key in top_rounds
            ]

            def get_elites_all():
                top_rounds = self.data_utils.get_top_rounds(20)
                return [
                    get_code_graph_and_prompt(key["round"], key["score"])
                    for key in top_rounds
                ]

            elites_all = get_elites_all()

            prompt, graph_load = self.graph_utils.read_graph_files(
                sample["round"], graph_path
            )  # graph_load is python code
            graph = self.graph_utils.extract_solve_graph(graph_load)

            processed_experience = self.experience_utils.load_experience()
            experience = self.experience_utils.format_experience(
                processed_experience, sample["round"]
            )

            operator_description = self.graph_utils.load_operators_description(
                self.operators
            )
            log_data = self.data_utils.load_log(sample["round"])
            mermaid_graph = self.graph_utils.read_mermaid_files(
                sample["round"], graph_path
            )

            is_evolution = random.random() < 0.1

            response = await get_next_workflow(
                elites=elites,
                elites_all=elites_all,
                experience=experience,
                score=sample["score"],
                code=graph[0],
                prompt=prompt,
                operator_description=operator_description,
                graph_mermaid=mermaid_graph,
                type=self.type,
                log_data=log_data,
                optimize_llm=self.optimize_llm,
                is_evolution=self.round >= 4 and is_evolution,
                round_id=self.round + 1,
            )

            check = self.experience_utils.check_modification(
                processed_experience, response["modification"], sample["round"]
            )

            if check:
                break

        self.graph_utils.write_graph_files_with_mermaid(
            directory, response, self.round + 1, self.dataset
        )

        experience = self.experience_utils.create_experience_data(
            sample, response["modification"]
        )

        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        logger.info(directory)

        avg_score = await self.evaluation_utils.evaluate_graph(
            self, directory, validation_n, data, initial=False
        )

        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score

    def _extract_fields_from_response(self, response: str) -> Dict[str, str]:
        """
        Fallback method to extract fields from raw response text using basic parsing

        Args:
            response: Raw response text from LLM

        Returns:
            Dictionary with extracted fields or None if extraction fails
        """
        try:
            # Try to extract XML tags with regex
            import re

            # Initialize result dictionary with default values
            result = {"modification": "", "graph": "", "prompt": ""}

            # Extract each field with regex
            for field in result.keys():
                pattern = rf"<{field}>(.*?)</{field}>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    result[field] = match.group(1).strip()

            # Verify we have at least some content
            if not any(result.values()):
                logger.error("No fields could be extracted from response")
                return None

            return result
        except Exception as e:
            logger.error(f"Error extracting fields from response: {str(e)}")
            return None

    async def test(self):
        rounds = range(1, 21)  # You can choose the rounds you want to test here.
        data = []

        graph_path = f"{self.root_path}/workflows_test"
        json_file_path = self.data_utils.get_results_file_path(graph_path)

        data = self.data_utils.load_results(graph_path)

        for round in rounds:
            directory = self.graph_utils.create_round_directory(graph_path, round)
            self.graph = self.graph_utils.load_graph(round, graph_path)

            (
                score,
                avg_cost,
                total_cost,
            ) = await self.evaluation_utils.evaluate_graph_test(
                self, directory, is_test=True
            )

            new_data = self.data_utils.create_result_data(
                round, score, avg_cost, total_cost
            )
            data.append(new_data)

            self.data_utils.save_results(json_file_path, data)
