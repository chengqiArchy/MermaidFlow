import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from metagpt.ext.aflow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger
import inspect


class AMCBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = expected_output
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        """
        Compare two mathematical expressions for equality, handling various equivalent forms.
        """
        if str(prediction) == str(reference):
            return True

        # Clean up the expressions
        prediction = str(prediction).strip()
        reference = str(reference).strip()

        # Try numeric comparison first
        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                pred_num = self.parse_digits(prediction)
                ref_num = self.parse_digits(reference)
                return isclose(pred_num, ref_num, abs_tol=1e-3)
        except:
            pass

        # Normalize expressions before comparison
        try:
            # Remove spaces around operators
            pred_normalized = re.sub(r'\s*([\+\-\*\/\^])\s*', r'\1', prediction)
            ref_normalized = re.sub(r'\s*([\+\-\*\/\^])\s*', r'\1', reference)

            if pred_normalized == ref_normalized:
                return True

            # Normalize fractions with pi
            pred_normalized = re.sub(r'\\frac{(.+)}{(.+)}\\pi', r'\\frac{\1\\pi}{\2}', prediction)
            ref_normalized = re.sub(r'\\frac{(.+)}{(.+)}\\pi', r'\\frac{\1\\pi}{\2}', reference)

            if pred_normalized == ref_normalized:
                return True

            # Sort terms in additions (e.g., "3 + 2\sqrt{3}" vs "2\sqrt{3} + 3")
            def sort_terms(expr):
                terms = expr.split('+')
                return '+'.join(sorted(term.strip() for term in terms))

            pred_sorted = sort_terms(prediction)
            ref_sorted = sort_terms(reference)

            if pred_sorted == ref_sorted:
                return True

        except:
            pass

        # Try symbolic comparison as a last resort
        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def symbolic_equal(self, a: str, b: str) -> bool:
        """
        Compare two mathematical expressions symbolically using SymPy.
        """

        def _parse(s):
            # Clean up the expression before parsing
            s = s.replace('\\', '')  # Remove LaTeX backslashes
            s = re.sub(r'\s+', '', s)  # Remove all whitespace

            for f in [parse_latex, parse_expr]:
                try:
                    expr = f(s)
                    return expr
                except:
                    continue
            return s

        try:
            a_expr = _parse(a)
            b_expr = _parse(b)

            # Try direct simplification
            if simplify(a_expr - b_expr) == 0:
                return True

            # Try numerical evaluation
            try:
                if isclose(float(N(a_expr)), float(N(b_expr)), abs_tol=1e-3):
                    return True
            except:
                pass

            # Try expanding before comparison
            from sympy import expand
            if simplify(expand(a_expr) - expand(b_expr)) == 0:
                return True

        except:
            pass

        return False

    def parse_digits(self, num):
        """Parse different number formats including LaTeX fractions."""
        num = str(num).strip()

        # Try to parse LaTeX fraction
        frac_pattern = r'\\frac{(-?\d+)}{(-?\d+)}'
        frac_match = re.match(frac_pattern, num)
        if frac_match:
            try:
                numerator = float(frac_match.group(1))
                denominator = float(frac_match.group(2))
                if denominator != 0:
                    return numerator / denominator
            except:
                pass

        # Try normal number parsing
        num = regex.sub(",", "", num)
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def is_digit(self, num):
        """Check if string can be parsed as a number."""
        # First check for LaTeX fraction
        if isinstance(num, str) and '\\frac{' in num:
            frac_pattern = r'\\frac{(-?\d+)}{(-?\d+)}'
            if re.match(frac_pattern, num.strip()):
                return True

        # Then try normal number parsing
        return self.parse_digits(num) is not None

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        input_text = problem["question"]
        expected_output = problem["answer"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            uni_score, extracted_output = self.calculate_score(expected_output, output)

            if uni_score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output, extract_answer_code=self.get_function_code(self.extract_model_answer))

            return input_text, output, expected_output, uni_score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
