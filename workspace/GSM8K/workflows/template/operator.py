import concurrent
import sys
import traceback
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.MATH.workflows.template.operator_an import *
from workspace.MATH.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio

import weave


class Operator:
    def __init__(self, llm: AsyncLLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode)
        
        try:
            # Use the formatter with AsyncLLM
            if formatter:
                response = await self.llm.call_with_format(prompt, formatter)
            else:
                # Fallback to direct call if no formatter is needed
                response = await self.llm(prompt)
                
            # Convert to expected format based on the original implementation
            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            return {"error": str(e)}
    
    def _create_formatter(self, op_class, mode=None) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            return CodeFormatter()
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None


class RoleExpansionLLM(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "role_expansion"):
        super().__init__(llm, name)
    @weave.op()
    async def __call__(self, input):
        prompt = f"""Analyze and enhance the original instruction to create a more effective and specific version for given input.

Original instruction: 

{input.get('instruction', '')}

Input: 

{input.get('question', '')}

"""
        full_response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        filter_prompt = f"""Please extract the enhanced instruction section from the following response:

Full response:

{full_response["response"]}

The extraced instruction should be rewrite in a more clear, concise style and can be directly used.
If the full response has any few-shot example, you should keep it.

Please provide the enhanced instruction content directly without adding any titles or labels like 'instruction'. The content should be ready to use as-is.
NOTE: Focus only on enhancing the original instruction based on the input context. Provide guidance that improves the instruction's clarity and effectiveness without directly answering the question. The enhancement should offer general direction while maintaining the original intent of the instruction.
NOTE: Keep the instruction concise (under 50 words). Focus on enhancing the general approach rather than being overly specific to this particular question. Maintain clarity while being brief.
"""
        response = await self._fill_node(GenerateOp, filter_prompt, mode="single_fill") 
        return response

from scripts.async_llm import LLMsConfig, AsyncLLM

models_config = LLMsConfig.default()

global _role_expansion_llm

role_expansion_llm_config = models_config.get("gpt-4o-mini")
role_expansion_llm_config.temperature = 0.7

_role_expansion_llm = RoleExpansionLLM(llm=AsyncLLM(role_expansion_llm_config))

def question_specific_custom(func):
    """
    A decorator that customizes operator behavior based on question type.
    
    This decorator allows operators to have different behaviors depending on
    the specific dataset or question type they're processing.
    
    Args:
        func: The function to be decorated
        
    Returns:
        wrapper: The wrapped function with custom behavior
    
    Example:
        @question_specific_custom
        async def __call__(self, input, instruction):
            # This will now have dataset-specific behavior
            return await self._process(input, instruction)
    """
    async def wrapper(self, input, instruction, role):
        # the instruction implicitly means a role, in this decorator the goal is to expend the role to a more specific level according to the original question and input content.

        # I will first to do some experiments about refine the role according to the question
        global _role_expansion_llm

        _input = {
            "instruction": instruction,
            "question": input
        }

        new_instruction = await _role_expansion_llm(_input)

        result = await func(self, input, new_instruction["response"] + "\n\n", role)
        return result
    
    # Return the wrapped function
    return wrapper


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    # @question_specific_custom
    @weave.op()
    async def __call__(self, input, instruction, role: str="Solver"):
        prompt = instruction + input
        with weave.attributes({"custom_role": role}):
            response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    

class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer"):
        super().__init__(llm, name)

    async def exec_code(self, code, timeout=30):
        """
        Asynchronously execute code and return an error if timeout occurs.
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            try:
                # Submit run_code task to the process pool
                future = loop.run_in_executor(executor, run_code, code)
                # Wait for the task to complete or timeout
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Timeout, attempt to shut down the process pool
                executor.shutdown(wait=False, cancel_futures=True)
                return "Error", "Code execution timed out"
            except Exception as e:
                return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem, analysis, feedback, mode):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve")
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    @weave.op()
    async def __call__(self, problem: str, analysis: str = "None"):
        """
        Call method, generate code and execute, retry up to 3 times.
        """
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill")
            code = code_response.get("response")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output}


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)
    @weave.op()
    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}