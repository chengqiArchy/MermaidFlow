from typing import Literal
import workspace.GSM8K.workflows.template.operator as operator
import workspace.GSM8K.workflows.round_1.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.custom = operator.Custom(self.llm)
        self.programmer= operator.Programmer(self.llm)

    @weave.op()
    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        Each operator is callable, you can call it directly.
        """
        solution = await self.custom(input=problem, instruction=prompt_custom.SIMPLE_SOLVER_1, role="simple_solver_1")
        final_solution = await self.sc_ensemble(solutions=[solution], problem=problem)
        return final_solution['response'], self.llm.get_usage_summary()["total_cost"]