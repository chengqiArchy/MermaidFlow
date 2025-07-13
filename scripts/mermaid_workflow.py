from mermaid.graph import Graph
import os
from pydantic import BaseModel, Field
from scripts.prompts.optimize_prompt import (
    WORKFLOW_CUSTOM_USE_WITH_MERMAID,
    WORKFLOW_OPTIMIZE_PROMPT_WITH_MERMAID,
)
from scripts.evolution_prompt import (
    LLM_AS_JUDGER_SINGLE,
)
from scripts.formatter import XmlFormatter
import weave
import asyncio
from scripts.evolution_prompt import PROMPT_FEW_SHOT
import logging
import time
from scripts.logs import logger


class GraphOptimizeWithMermaid(BaseModel):
    code: str = Field(default="", description="code")
    prompt: str = Field(default="", description="prompt")


class GraphOptimizeForMermaid(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph represent by mermaid code")
    prompt: str = Field(default="", description="prompt")


class GraphForEvolutionaryAlgorithm(BaseModel):
    modification_A: str = Field(default="", description="modification")
    graph_A: str = Field(default="", description="graph represent by mermaid code")
    prompt_A: str = Field(default="", description="prompt")
    modification_B: str = Field(default="", description="modification")
    graph_B: str = Field(default="", description="graph represent by mermaid code")
    prompt_B: str = Field(default="", description="prompt")
    modification_C: str = Field(default="", description="modification")
    graph_C: str = Field(default="", description="graph represent by mermaid code")
    prompt_C: str = Field(default="", description="prompt")
    modification_D: str = Field(default="", description="modification")
    graph_D: str = Field(default="", description="graph represent by mermaid code")
    prompt_D: str = Field(default="", description="prompt")


class FinalSelection(BaseModel):
    selected_graph: str = Field(default="", description="selected graph")
    justification: str = Field(default="", description="justification")


class LLMAsJudge(BaseModel):
    evaluation: dict = Field(default={}, description="evaluation")
    selected_graph: str = Field(default="", description="selected graph")
    justification: str = Field(default="", description="justification")


def get_mermaid_image(text_input, folder_path):
    file_name = "graph_mermaid"
    sequence = Graph(file_name, text_input)
    file_path = os.path.join(folder_path, file_name + ".mmd")
    sequence.save(file_path)
    return file_path, sequence


def generate_image(folder_path, text_input):
    """
    Generate a mermaid diagram image using mmdc (Mermaid CLI)

    Args:
        folder_path: Path where the image will be saved
        text_input: Mermaid diagram text content

    Returns:
        str: Path to the generated image file
    """
    import subprocess

    # First generate the mermaid file
    mmd_file_path, _ = get_mermaid_image(text_input=text_input, folder_path=folder_path)

    # Define output image path
    image_file_path = os.path.splitext(mmd_file_path)[0] + ".png"

    # Run mmdc command to generate the image
    try:
        subprocess.run(
            [
                "mmdc",
                "-i",
                mmd_file_path,
                "-o",
                image_file_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"Successfully generated mermaid diagram at: {image_file_path}")
        return image_file_path
    except subprocess.CalledProcessError as e:
        print(f"Error generating mermaid diagram: {e}")
        return None
    except FileNotFoundError:
        print(
            "Error: mmdc command not found. Please install mermaid-cli using npm: npm install -g @mermaid-js/mermaid-cli"
        )
        return None

# this prompt is for generate python code
GENERATE_PYTHON_CODE = """
Below is a graph, the corresponding Python code, and the prompt.
<old_workflow>
    <old_graph>{old_mermaid}</old_graph>
    <old_code>{old_code}</old_code>
    <old_role_prompt>{old_prompt}</old_role_prompt>(only prompt_custom)
</old_workflow>

Based on this example of old graph and code, you need to generate new Python code according to the new graph and given prompt.

<information_for_new_workflow>
    <new_graph>{new_mermaid}</new_graph>
    <new_role_prompt>{new_prompt}</new_role_prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</information_for_new_workflow>

The output format should be:

<code>New generated python code</code>

Carefully analyze the new_graph to generate corresponding code. Pay attention to:
1. The connections between each node
2. The role and function of each operator
3. The input/output relationships

For each node in the graph, implement the appropriate code and use the corresponding prompts from new_prompt. If a node or operator doesn't have an explicit prompt provided, DO NOT create one - use empty strings instead. For example, Program operators may have empty analysis fields.
Every prompt referenced in your Python code must be defined in the prompt_custom module. If you find that new_role_prompt is missing some prompts, please ensure they are properly created. When adding new functionality to your Python code, make sure to import any necessary libraries or modules, except for operator, prompt_custom, create_llm_instance, and CostManage which are automatically imported.
Your implementation must be robust - ensure all methods return appropriate values and **never return None for any field**. Pay special attention to error handling and edge cases.
Use custom methods with proper formatting to ensure your output matches the expected structure. The system will extract answers based on specific rules for scoring, so maintaining the correct output format is critical.

NOTE: especially for the final output, you should ensure that the output format is correct, you can learn it from old code.
"""

MERMAID_CUSTOM_MATH = """
# Mermaid Graph Style Guide for MATH Problems
# This comprehensive guide defines styling and structure for creating consistent workflow diagrams

# Node Style Definitions
classDef CustomOp fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
classDef ProgrammerOp fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
classDef ScEnSembleOp fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;

# ===== OPERATOR USAGE GUIDE =====

# 1. Interface Nodes (Entry/Exit Points)
# Every workflow diagram must include these two standard interface nodes:
#   PROBLEM([Problem])           - The entry point providing initial input
#   RETURN([Return response & cost]) - The exit point receiving final output
#
# Example:
#   PROBLEM([Problem])
#   RETURN([Return response & cost])
#   
#   class PROBLEM Interface
#   class RETURN Interface
#
# Connection rules:
#   - PROBLEM node: Provides input to other nodes but never receives input
#   - RETURN node: Receives output from the final node but never produces output

# 2. Custom Operator Nodes
# Format: NodeName["Custom<br/>(role: role_name)"]
#
# Example:
#   K["Custom<br/>(role: validate_1)"]
#   class K CustomOp
#
# For multiple nodes with similar roles, use numbered suffixes:
#   R1["Custom<br/>(role: review_solution_1)"]
#   R2["Custom<br/>(role: review_solution_2)"]
#   class R1 CustomOp
#   class R2 CustomOp
#
# Connection example:
#   PROBLEM --> |input| R1

# 3. Programmer Operator Nodes
# Format: P["Programmer<br/>(analysis: 'your analysis text')"]
#
# The Programmer operator requires two inputs:
# - problem: The math problem to solve
# - analysis: Instructions on how to approach the problem
#
# Examples:
#   P["Programmer<br/>(analysis: 'Calculate step by step')"]
#   class P ProgrammerOp
#
# Connection rules:
# 1. For the problem input:
#   PROBLEM --> |problem|P  # The problem must come from the PROBLEM node
#
# 2. For the analysis input (two options):
#   C --> |analysis|P  # You can use another node's output as analysis content
#   # OR
#   # You can specify the analysis directly in the node definition
#
# Complete example:
#   PROBLEM --> |problem|P
#   C --> |analysis|P
#   class P ProgrammerOp

# 4. ScEnsemble Operator Nodes
# Format: ENSEMBLE["ScEnsemble<br/>"]
#
# Example:
#   ENSEMBLE["ScEnsemble<br/>"]
#   class ENSEMBLE ScEnSembleOp
#
# CRITICAL: ScEnsemble nodes MUST have multiple inputs to function correctly
# Example connections:
#   SOLUTION_1 --> ENSEMBLE
#   SOLUTION_2 --> ENSEMBLE
#   ENSEMBLE --> NEXT_NODE

# ===== WORKFLOW PATTERNS =====

# Example Workflow Pattern
# This pattern demonstrates how to effectively combine multiple solution approaches:
# 1. Problem input flows to multiple solution-generating nodes (Custom and/or Programmer)
# 2. All solutions are combined using ScEnsemble
# 3. The ensemble result is returned as the final output
#
# Connection pattern:
#   PROBLEM --> |input| SOLUTION_1
#   PROBLEM --> |input| SOLUTION_2
#   PROBLEM --> |problem| P
#   SOLUTION_1 --> ENSEMBLE
#   SOLUTION_2 --> ENSEMBLE
#   P --> ENSEMBLE
#   ENSEMBLE --> RETURN

# ===== GENERAL RULES =====

# Connection Rules
# 1. Direct connections from PROBLEM to any node that needs the initial problem as input
# 2. Direct connections between nodes where one node requires another's output
# 3. When using ProgrammerOp, the edges should have name on it
# 4. All connections must follow logical data flow and maintain workflow coherence

# Prompt Definition Requirements
# All prompts used in the graph must be defined in this format:
# <prompt>
# PROMPT_NAME_1="Your first prompt text here"
# PROMPT_NAME_2="Your second prompt text here"
# </prompt>
#
# Each prompt must be on a separate line with its own unique variable name

# IMPORTANT: Do not create new Operator types!
# Only use the predefined operators available in the system. Creating custom operators
# will cause workflow execution failures. Stick to the operators documented in this guide.

# Best Practices:
# - Use %% for comments in separate lines for clarity, don't add it to the end of the line because it is not a valide mermaid code
# - Always include the style block (classDef section) even if not all classes are used
# - Always ensure multiple inputs to ScEnsemble nodes (this is a common mistake)
# - Maintain consistent naming conventions for all nodes and classes
# - Use descriptive role names that indicate the node's purpose
# - Label connections with |input| where appropriate for clarity
"""

MERMAID_CUSTOM_CODE = """
# Operator Style Definitions
# Each operator has a specific style for consistent visualization in the Mermaid graph
classDef CustomOp fill:#d0e1f9,stroke:#4378a2,stroke-width:2px;
classDef CustomCodeGenerateOp fill:#f9c2c2,stroke:#c23737,stroke-width:2px;
classDef ScEnSembleOp fill:#f9e4b7,stroke:#b99b37,stroke-width:2px;
classDef Decision fill:#ffffff,stroke:#444444,stroke-width:1px,stroke-dasharray:2 2;
classDef TestOp fill:#d8f0d8,stroke:#2e8b57,stroke-width:2px;
classDef Interface fill:#e2e2f2,stroke:#6a6ab2,stroke-width:2px;

# ===== OPERATOR USAGE GUIDE =====

# 1. Interface Nodes
# Interface nodes represent the entry and exit points of your workflow
# Required in every graph:
#   PROBLEM([Problem])           - Starting point that provides input
#   RETURN([Return response & cost]) - Endpoint that receives final output
#
# Example:
#   class PROBLEM Interface
#   class RETURN Interface
#
# Connection rules:
#   - PROBLEM node provides input but doesn't receive any
#   - RETURN node receives output but doesn't produce any

# 2. Custom Operator
# Used for specialized reasoning strategies with defined roles
#
# Example:
#   K["Custom<br/>(role: xxx)"]
#   class K CustomOp
#
# For multiple instances with same prompt type:
#   R1["Custom<br/>(role: xxx_1)"]
#   R2["Custom<br/>(role: xxx_2)"]
#   class R1 CustomOp
#   class R2 CustomOp
#
# Connection rules:
#   - Connect directly to PROBLEM if it needs the initial problem
#   - Connect to other nodes if it needs their output

# 3. CustomCodeGenerate Operator
# Specialized for code generation tasks based on problem descriptions
#
# Example:
#   CODE_GEN["CustomCodeGenerate<br/>(instruction: xxx)"]
#   class CODE_GEN CustomCodeGenerateOp
#
# Multiple approaches:
#   CODE_GEN_1["CustomCodeGenerate<br/>(instruction: aaa)"]
#   CODE_GEN_2["CustomCodeGenerate<br/>(instruction: bbb)"]
#   class CODE_GEN_1 CustomCodeGenerateOp
#   class CODE_GEN_2 CustomCodeGenerateOp
#

# 4. ScEnsemble Operator
# Combines multiple solutions into one cohesive result
#
# Example:
#   ENSEMBLE["ScEnsemble<br/>"]
#   class ENSEMBLE ScEnsembleOp
#
# CRITICAL: Must have multiple inputs to function correctly
#   SOLUTION_1 --> ENSEMBLE
#   SOLUTION_2 --> ENSEMBLE
#   ENSEMBLE --> other_node

# 5. Test Operator
# Validates solutions against test cases
#
# Example:
#   T["Test<br/>"]
#   class T TestOp
#
# Typical connection pattern:
#   CODE_GEN --> |solution|T
#   ENTRY_POINT --> |entry_point|T
#   PROBLEM --> |problem|T
#   T --> DECISION_NODE
#
# Decision node for test results:
#   CHECK_TEST{test_result<br/>passed?}
#   class CHECK_TEST DecisionOp
#   CHECK_TEST -- Failed --> IMPROVE_SOLUTION
#   CHECK_TEST -- Passed --> RETURN

# ===== IMPORTANT NOTES =====
# - You cannot create other class operations
# - Always ensure multiple inputs to ScEnsemble nodes
# - All prompts must be defined in the prompt section
# - Format prompts as:
#   <prompt>
#   PROMPT_NAME_1="Your first prompt text here"
#   PROMPT_NAME_2="Your second prompt text here"
#   </prompt>

# IMPORTANT: Do not create new Operator types!
# Only use the predefined operators available in the system. Creating custom operators
# will cause workflow execution failures. Stick to the operators documented in this guide.
"""

UPDATE_WORKFLOW_MERMAID_MATH = """
You are building a Graph and corresponding Prompt to jointly solve {type} problems. Here is a graph written in Mermaid and the corresponding prompt that performed excellently in a previous iteration (maximum score is 1). There are also some special operators defined in the operator_description section. Referring to this given graph and prompt, which forms a basic example of a code solution approach, please reconstruct and optimize them.
You should make further optimizations based on this graph. The modified graph should differ from the provided example or have the same architecture but with prompts fine-tuned based on logs. The specific differences should be noted within the <modification>xxx</modification> section.
<sample>
    <experience>{experience}</experience>
    <modification>(such as: add/delete/modify/...)</modification>
    <score>{score}</score>
    <graph>{graph_mermaid}</graph>
    <role_prompt>{prompt}</role_prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

You can add, modify, or delete nodes, parameters, or connections in the workflow. For each change, clearly describe your single modification within XML tags using the format: <modification>your detailed modification description</modification>. 

Your optimizations should be one of (this limitation is strict! you can only pick one action!):
1. Expanding the graph by adding new single (operators). Note: adding nodes may require modifying prompts for related nodes, you should also update the corresponding prompts if needed.
2. Deleting unnecessary nodes from the graph.
3. Modifying existing nodes or their connections or their prompt.

Prompt is very important for performance here are some exmaples you can learn from it:
{prompt_few_shot}

Ensure all necessary prompts are properly prepared and defined. The generated custom node's role is defined by their prompt. Use custom methods with proper formatting to ensure your output matches the expected structure. The system will extract answers based on specific rules for scoring, so maintaining the correct output format is critical.
The output format is critical for proper evaluation. Analyze the log data carefully to identify patterns in successful solutions and common errors. Extract specific formatting requirements and incorporate them as clear guidance in your prompts.
When optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc.

NOTE: You should also try to use different operators to enhance workflow capabilities. 
NOTE: If you are trying to answer from multiple solutions, you should try to use ensemble operator then to get the final answer.
NOTE: Each output of nodes except the interface node should be connected to the next node, ensuring data flows through the entire workflow. This guarantees all nodes properly receive inputs and pass outputs.
NOTE: If you are trying to add an ensemble node, you need to guarantee that there are multiple solution inputs available for the ensemble to work effectively.
NOTE: Program is a very powerful tool for solving MATH problems, you should try it! The Program operator can execute Python code to perform calculations and solve complex mathematical problems.
NOTE: Think big! Be bold in innovation and exploration. Don't be afraid to push boundaries and discover new approaches to problem-solving. Additionally, keep prompts concise - no more than 100 words per prompt. Shorter, focused prompts often perform better than lengthy ones.
"""

UPDATE_WORKFLOW_MERMAID_CODE = """
You are building a Graph and corresponding Prompt to jointly solve {type} problems. Here is a graph written in Mermaid and the corresponding prompt that performed excellently in a previous iteration (maximum score is 1). There are also some special operators defined in the operator_description section. Referring to this given graph and prompt, which forms a basic example of a code solution approach, please reconstruct and optimize them.
You should make further optimizations based on this graph. The modified graph should differ from the provided example or have the same architecture but with prompts fine-tuned based on logs. The specific differences should be noted within the <modification>xxx</modification> section.
<sample>
    <experience>{experience}</experience>
    <modification>(such as: add/delete/modify/...)</modification>
    <score>{score}</score>
    <graph>{graph_mermaid}</graph>
    <role_prompt>{prompt}</role_prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

You can add, modify, or delete nodes, parameters, or connections in the workflow. For each change, clearly describe your single modification within XML tags using the format: <modification>your detailed modification description</modification>. 

Your optimizations should be one of (this limitation is strict! you can only pick one action!):
1. Expanding the graph by adding new single (operators). Note: adding nodes may require modifying prompts for related nodes, you should also update the corresponding prompts if needed.
2. Deleting unnecessary nodes from the graph.
3. Modifying existing nodes or their connections or their prompt.

Prompt is very important for performance here are some exmaples you can learn from it:
{prompt_few_shot}

Ensure all necessary prompts are properly prepared and defined. The generated custom node's role is defined by their prompt. Use custom methods with proper formatting to ensure your output matches the expected structure. The system will extract answers based on specific rules for scoring, so maintaining the correct output format is critical.
Review the log data carefully to understand the expected answer format and ensure your implementation produces compatible output. Proper formatting is essential for accurate evaluation of your solution, and you should emphasize this in the prompt.
When optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. 
You should also try to use different operators to enhance workflow capabilities. Each operator should have the corresponding Mermaid class defined in the graph code. 

NOTE: If you are trying to add an ensemble node, you need to guarantee that there are multiple solution inputs available for the ensemble to work effectively.
NOTE: In the code problems, ensemble different kinds of solutions is a good idea, and you should also try to use Test operator to validate the solutions.
NOTE: For code-related tasks, the final output must be either test_result["solution"] or custom_code_generate_result["response"], as these represent valid Python code. Make sure your workflow produces one of these formats as the final output to ensure proper evaluation.
NOTE: Think big! Be bold in innovation and exploration. Don't be afraid to push boundaries and discover new approaches to problem-solving. Additionally, keep prompts concise - no more than 100 words per prompt. Shorter, focused prompts often perform better than lengthy ones.
"""

LEARN_FROM_FAILED_MERMAID = """
Avoid repeating these errors from the previous round's failed workflow:

<failed_mermaid>
{failed_mermaid}
</failed_mermaid>

<error_message>
{error_message}
</error_message>
"""

# define the extended mermaid code for each dataset type.
MERMAID_CODE_GUIDANCE = {
    "math": MERMAID_CUSTOM_MATH,
    "code": MERMAID_CUSTOM_CODE,
}

UPDATE_MERMAID_WORKFLOW = {
    "math": UPDATE_WORKFLOW_MERMAID_MATH,
    "code": UPDATE_WORKFLOW_MERMAID_CODE,
}


def create_prompt_for_generate_code(
    old_mermaid: str,
    old_code: str,
    old_prompt: str,
    modification: str,
    new_mermaid: str,
    new_prompt: str,
    operator_description: str,
    dataset_type: str,
):
    # Get the mermaid graph representation from the file

    graph_input = GENERATE_PYTHON_CODE.format(
        old_mermaid=old_mermaid,
        old_code=old_code,
        old_prompt=old_prompt,
        modification=modification,
        new_mermaid=new_mermaid,
        new_prompt=new_prompt,
        operator_description=operator_description,
        type=dataset_type,
    )

    graph_system = WORKFLOW_OPTIMIZE_PROMPT_WITH_MERMAID.format(type=dataset_type)

    return graph_input + WORKFLOW_CUSTOM_USE_WITH_MERMAID + graph_system


def create_mermaid_optimization_prompt(
    experience: str,
    score: float,
    graph_mermaid: str,
    graph_code: str,
    role_prompt: str,
    operator_description: str,
    type: str,
    log_data: str,
    last_mermaid: str = None,
    errors_for_last_mermaid: str = None,
):
    graph_input = UPDATE_MERMAID_WORKFLOW[type].format(
        experience=experience,
        score=score,
        graph_mermaid=graph_mermaid,
        graph_code=graph_code,
        prompt=role_prompt,
        operator_description=operator_description,
        type=type,
        log=log_data,
        prompt_few_shot=PROMPT_FEW_SHOT[type],
    )
    if last_mermaid and errors_for_last_mermaid:
        extra_info = LEARN_FROM_FAILED_MERMAID.format(failed_mermaid=last_mermaid, error_message=errors_for_last_mermaid)
        return graph_input + MERMAID_CODE_GUIDANCE[type] + extra_info

    return graph_input + MERMAID_CODE_GUIDANCE[type]

def get_graph_candidates_prompt(
    parent: dict,
    graph_candidates: dict,
    type: str,
):
    return LLM_AS_JUDGER_SINGLE.format(
        parent_graph=parent["graph"],
        parent_prompt=parent["prompt"],
        graph_A=graph_candidates["graph_A"],
        graph_B=graph_candidates["graph_B"],
        graph_C=graph_candidates["graph_C"],
        graph_D=graph_candidates["graph_D"],
        modification_A=graph_candidates["modification_A"],
        modification_B=graph_candidates["modification_B"],
        modification_C=graph_candidates["modification_C"],
        modification_D=graph_candidates["modification_D"],
        prompt_A=graph_candidates["prompt_A"],
        prompt_B=graph_candidates["prompt_B"],
        prompt_C=graph_candidates["prompt_C"],
        prompt_D=graph_candidates["prompt_D"],
        mermaid_usage=MERMAID_CODE_GUIDANCE[type],
    )

@weave.op()
async def get_response(
    experience: str,
    score: float,
    code: str,
    prompt: str,
    operator_description: str,
    graph_mermaid: str,
    type: str,
    log_data: str,
    optimize_llm,
    mermaid_checker=None,
):
    try_number = 3
    last_mermaid = None
    errors_for_last_mermaid = None
    parent = {
        "experience": experience,
        "score": score,
        "graph": graph_mermaid,
        "code": code,
        "prompt": prompt,
    }
    for _ in range(try_number):
        @weave.op()
        async def get_graph_candidates(parent, type):
            mermaid_code_optimize_prompt = create_mermaid_optimization_prompt(
                experience=parent["experience"],
                score=parent["score"],
                graph_mermaid=parent["graph"],
                graph_code=parent["code"],
                role_prompt=parent["prompt"],
                operator_description=operator_description,
                type=type,
                log_data=log_data,
                last_mermaid=last_mermaid,
                errors_for_last_mermaid=errors_for_last_mermaid,
            )
            mermaid_format = XmlFormatter.from_model(GraphForEvolutionaryAlgorithm)
            graph_response = await optimize_llm.call_with_format(
                mermaid_code_optimize_prompt, mermaid_format
            )
            graph_candidates_prompt = get_graph_candidates_prompt(
                parent=parent,
                graph_candidates=graph_response,
                type=type,
            )
            graph_candidates_format = XmlFormatter.from_model(LLMAsJudge)
            graph_candidates_response = await optimize_llm.call_with_format(
                graph_candidates_prompt, graph_candidates_format
            )
            return graph_candidates_response, graph_response
        
        graph_decision, graph_candidates = await get_graph_candidates(parent, type)

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
        
        new_graph, new_prompt, new_modification = get_graph(graph_decision, graph_candidates)

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

    modification = new_modification

    generate_code_prompt = create_prompt_for_generate_code(
        old_mermaid=graph_mermaid,
        old_code=code,
        old_prompt=prompt,
        modification=modification,
        new_mermaid=new_graph,
        new_prompt=new_prompt,
        operator_description=operator_description,
        dataset_type=type,
    )

    generate_code_format = XmlFormatter.from_model(GraphOptimizeWithMermaid)
    python_code_response = await optimize_llm.call_with_format(
        generate_code_prompt, generate_code_format
    )

    final_response = {
        "code": python_code_response["code"],
        "prompt": python_code_response["prompt"],
        "graph": new_graph,
        "modification": modification,
    }

    return final_response
