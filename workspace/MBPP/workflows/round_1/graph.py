from typing import Literal
import workspace.MBPP.workflows.template.operator as operator
import workspace.MBPP.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
import weave

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]


class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)
    @weave.op()
    async def __call__(self, problem: str, entry_point:str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # await self.custom(input=, instruction="") 
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.SIMPLE_SOLVER) # But When you want to get standard code ,you should use customcodegenerator.
        final_solution = await self.sc_ensemble(solutions=[solution], problem=problem)
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]