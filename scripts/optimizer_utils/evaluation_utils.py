from scripts.evaluator import Evaluator
import weave
import re


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_initial_round(self, optimizer, graph_path, directory, validation_n, data):
        # Load graph with graph_utils from optimizer
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = Evaluator(eval_path=directory)

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
            )

            new_data = optimizer.data_utils.create_result_data(optimizer.round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)

        return data

    async def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False):
        evaluator = Evaluator(eval_path=directory)
        sum_score = 0

        round_name = directory.split("/")[-1]

        for i in range(validation_n):
            with weave.attributes({"round_name": round_name, "validation_round": i}):
                score, avg_cost, total_cost = await evaluator.graph_evaluate(
                    optimizer.dataset,
                    optimizer.graph,
                    {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                    directory,
                    is_test=False,
                )

                cur_round = optimizer.round + 1 if initial is False else optimizer.round

                new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
                data.append(new_data)

                result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
                optimizer.data_utils.save_results(result_path, data)

                sum_score += score

        return sum_score / validation_n
    @weave.op()
    async def evaluate_graph_test(self, optimizer, directory, is_test=True):
        evaluator = Evaluator(eval_path=directory)
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=is_test,
        )
