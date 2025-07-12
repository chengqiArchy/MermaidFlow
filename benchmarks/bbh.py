import re
import string
from typing import Callable, List, Tuple
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from metagpt.ext.aflow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger

class BBHBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def normalize_answer(self, s: str) -> str:
        """
        Normalize answer for evaluation by:
        1. Converting to lowercase
        2. Removing parentheses, brackets around options
        3. Removing whitespace
        """
        # Remove various forms of option markers: (A), [A], A), A.
        s = re.sub(r'[\(\[\{]([A-Za-z])[\)\]\}]|([A-Za-z])[\.:\)]', r'\1\2', s)
        return s.lower().strip()

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute exact match score between prediction and ground truth answers.
        Score is 1.0 if strings match exactly after normalization, 0.0 otherwise.
        """
        return (1.0 if self.normalize_answer(prediction) == self.normalize_answer(ground_truth) else 0.0, prediction)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = problem["input"]
        expected_output = problem["target"]
        inputs = input_text

        try:
            output, cost = await self._generate_output(graph, inputs)
            score, extracted_output = self.calculate_score(expected_output, output)

            return input_text, output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]