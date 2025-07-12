GENERATE_CHILDREN_PROMPT = """
# Evolution-Based Graph Generation

You are tasked with generating a new workflow graph by combining elements from two parent graphs. This process is inspired by genetic algorithms where offspring inherit traits from both parents.
This evolutionary approach allows us to create new solutions that may be more effective than either parent graph.

<parent_graph>
    <graph_A>
        <graph>{graph_A}</graph>
        <prompt>{prompt_A}</prompt>
        <score>{score_A}</score>
    </graph_A>
    <graph_B>
        <graph>{graph_B}</graph>
        <prompt>{prompt_B}</prompt>
        <score>{score_B}</score>
    </graph_B>
</parent_graph>

## Your Task
1. Analyze both **parent graphs** carefully (structure, nodes, connections, prompts, and purpose)
2. Create a new graph that:
   - Combines strengths from both parents
   - Introduces strategic innovations for improved performance
   - Applies evolutionary operations such as:
     * Crossover: Merging effective sections(nodes) from both parents, you should separate components in a reasonable and logical way, typically at interface or ensemble nodes
     * Mutation: Making targeted modifications
     * Insertion: Adding beneficial new nodes
     * Deletion: Removing inefficient components

## Guidelines
- Focus on the strengths of both parent graphs, especially the higher-scoring parent, to inform your design decisions
- Ensure your new graph is structurally correct and follows proper workflow patterns
- If creating a significantly different structure proves challenging, you may maintain a similar structure to the parents but with improved prompts, or, you can simply add one extra node to ensemble
- Pay careful attention to the connections between nodes to ensure proper data flow - for example, ensemble nodes should have multiple inputs
- Verify that your graph has correct input/output relationships and follows the operator usage guidelines
- Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them)

## Prompt Guidence
- **Prompt engineering is crucial for performance**: Carefully analyze and learn from the prompts used in **high-scoring** historical workflows. Pay special attention to their structure, specificity, and instruction clarity. Here are some exemplary prompts you can use as reference:
    {prompt_few_shot}

Your response should include:

1. A detailed explanation of your modifications in the <modification> section
2. The complete Mermaid code for the new graph
3. The updated prompts for any custom nodes

NOTE: Ensure the new graph is valid Mermaid syntax and represents a complete workflow solution. The following section contains critical rules and guidance for using operators in Mermaid that you MUST follow to create an effective workflow:
{mermiad_custom_prompt}
"""

LLM_AS_JUDGER = """

Your task is to select the most promising graph from the candidates. Here is the content of each graph, written in Mermaid code, along with explanations of how each new graph was generated from parent graphs:
Each new graph is derived from two parent graphs, with parent graph information as follows:

<parent_graph>
    <parent_graph_A>{parent_graph_A}</parent_graph_A>
    <parent_prompt_A>{parent_prompt_A}</parent_prompt_A>

    <parent_graph_B>{parent_graph_B}</parent_graph_B>
    <parent_prompt_B>{parent_prompt_B}</parent_prompt_B>
</parent_graph>



<graph_candidates>
    <graph_A>{graph_A}</graph_A>
    <modification_A>{modification_A}</modification_A>
    <prompt_A>{prompt_A}</prompt_A>

    <graph_B>{graph_B}</graph_B>
    <modification_B>{modification_B}</modification_B>
    <prompt_B>{prompt_B}</prompt_B>

    <graph_C>{graph_C}</graph_C>
    <modification_C>{modification_C}</modification_C>
    <prompt_C>{prompt_C}</prompt_C>

    <graph_D>{graph_D}</graph_D>
    <modification_D>{modification_D}</modification_D>
    <prompt_D>{prompt_D}</prompt_D>
</graph_candidates>

Please evaluate the graph candidates and select the most promising one based on these criteria:

1. **Workflow Coherence**: Assess how well the nodes connect and form a logical workflow
2. **Innovation**: Evaluate how the new graph improves upon the parent graphs
3. **Complexity Balance**: Check if the graph has appropriate complexity (neither too simple nor unnecessarily complex)
4. **Prompt Quality**: Examine the quality and specificity of the node prompts
5. **Modification Rationale**: Consider the thoughtfulness of the explanation provided for the changes

Here are some history of previous graphs and their corresponding score:
<history>
    {elites_history}
</history>

For each candidate graph, provide a score from 1-10 for each criterion and explain your reasoning, you can learn from the history graphs. 
You should avoid selecting graphs with structural defects, such as:
1. Incorrectly connecting nodes (e.g., CustomOp should not directly feed into ProgrammerOp)
2. Not properly using the ensemble node (all solution-generating nodes should feed into it)
3. Missing critical connections between nodes
4. Creating circular dependencies

Here are the specific structural rules for this type of workflow:
{mermaid_usage}

Additionally, consider how well the graph follows established patterns from successful historical examples.

Then select the graph with the highest total score as the most promising candidate.

<evaluation>
    <graph_A_score>
        <workflow_coherence>Score (1-10)</workflow_coherence>
        <innovation>Score (1-10)</innovation>
        <complexity_balance>Score (1-10)</complexity_balance>
        <prompt_quality>Score (1-10)</prompt_quality>
        <modification_rationale>Score (1-10)</modification_rationale>
        <total_score>Sum of all scores</total_score>
        <explanation>Detailed explanation of your evaluation</explanation>
    </graph_A_score>
    
    <graph_B_score>
        <workflow_coherence>Score (1-10)</workflow_coherence>
        <innovation>Score (1-10)</innovation>
        <complexity_balance>Score (1-10)</complexity_balance>
        <prompt_quality>Score (1-10)</prompt_quality>
        <modification_rationale>Score (1-10)</modification_rationale>
        <total_score>Sum of all scores</total_score>
        <explanation>Detailed explanation of your evaluation</explanation>
    </graph_B_score>
    
    <graph_C_score>
        <workflow_coherence>Score (1-10)</workflow_coherence>
        <innovation>Score (1-10)</innovation>
        <complexity_balance>Score (1-10)</complexity_balance>
        <prompt_quality>Score (1-10)</prompt_quality>
        <modification_rationale>Score (1-10)</modification_rationale>
        <total_score>Sum of all scores</total_score>
        <explanation>Detailed explanation of your evaluation</explanation>
    </graph_C_score>
    
    <graph_D_score>
        <workflow_coherence>Score (1-10)</workflow_coherence>
        <innovation>Score (1-10)</innovation>
        <complexity_balance>Score (1-10)</complexity_balance>
        <prompt_quality>Score (1-10)</prompt_quality>
        <modification_rationale>Score (1-10)</modification_rationale>
        <total_score>Sum of all scores</total_score>
        <explanation>Detailed explanation of your evaluation</explanation>
    </graph_D_score>
</evaluation>

<selected_graph>[A/B/C/D]</selected_graph>
<justification>
    Please provide a comprehensive justification for your selection, highlighting the key strengths of the chosen graph and how it represents the most effective approach to solving the problem.
</justification>
"""

LLM_AS_JUDGER_SINGLE = """

Your task is to select the most promising graph from the candidates. Here is the content of each graph, written in Mermaid code, along with explanations of how each new graph was generated:

<parent_graph>
    <parent_graph>{parent_graph}</parent_graph>
    <parent_prompt>{parent_prompt}</parent_prompt>
</parent_graph>

<graph_candidates>
    <graph_A>{graph_A}</graph_A>
    <modification_A>{modification_A}</modification_A>
    <prompt_A>{prompt_A}</prompt_A>

    <graph_B>{graph_B}</graph_B>
    <modification_B>{modification_B}</modification_B>
    <prompt_B>{prompt_B}</prompt_B>

    <graph_C>{graph_C}</graph_C>
    <modification_C>{modification_C}</modification_C>
    <prompt_C>{prompt_C}</prompt_C>

    <graph_D>{graph_D}</graph_D>
    <modification_D>{modification_D}</modification_D>
    <prompt_D>{prompt_D}</prompt_D>
</graph_candidates>

Please evaluate the graph candidates and select the most promising one based on these criteria:

1. **Workflow Coherence**: Assess how well the nodes connect and form a logical workflow
2. **Innovation**: Evaluate how the new graph improves upon the parent graph
3. **Complexity Balance**: Check if the graph has appropriate complexity (neither too simple nor unnecessarily complex)
4. **Prompt Quality**: Examine the quality and specificity of the node prompts
5. **Modification Rationale**: Consider the thoughtfulness of the explanation provided for the changes

For each candidate graph, provide a score from 1-10 for each criterion and explain your reasoning.
You should avoid selecting graphs with structural defects, such as:
1. Incorrectly connecting nodes (e.g., CustomOp should not directly feed into ProgrammerOp)
2. Not properly using the ensemble node (all solution-generating nodes should feed into it)
3. Missing critical connections between nodes
4. Creating circular dependencies

Here are the specific structural rules for this type of workflow:
{mermaid_usage}

Then select the graph with the highest total score as the most promising candidate.

<evaluation>
    <graph_A_score>
        <workflow_coherence>Score (1-10)</workflow_coherence>
        <innovation>Score (1-10)</innovation>
        <complexity_balance>Score (1-10)</complexity_balance>
        <prompt_quality>Score (1-10)</prompt_quality>
        <modification_rationale>Score (1-10)</modification_rationale>
        <total_score>Sum of all scores</total_score>
        <explanation>Detailed explanation of your evaluation</explanation>
    </graph_A_score>
    
    <graph_B_score>
        <workflow_coherence>Score (1-10)</workflow_coherence>
        <innovation>Score (1-10)</innovation>
        <complexity_balance>Score (1-10)</complexity_balance>
        <prompt_quality>Score (1-10)</prompt_quality>
        <modification_rationale>Score (1-10)</modification_rationale>
        <total_score>Sum of all scores</total_score>
        <explanation>Detailed explanation of your evaluation</explanation>
    </graph_B_score>
    
    <graph_C_score>
        <workflow_coherence>Score (1-10)</workflow_coherence>
        <innovation>Score (1-10)</innovation>
        <complexity_balance>Score (1-10)</complexity_balance>
        <prompt_quality>Score (1-10)</prompt_quality>
        <modification_rationale>Score (1-10)</modification_rationale>
        <total_score>Sum of all scores</total_score>
        <explanation>Detailed explanation of your evaluation</explanation>
    </graph_C_score>
    
    <graph_D_score>
        <workflow_coherence>Score (1-10)</workflow_coherence>
        <innovation>Score (1-10)</innovation>
        <complexity_balance>Score (1-10)</complexity_balance>
        <prompt_quality>Score (1-10)</prompt_quality>
        <modification_rationale>Score (1-10)</modification_rationale>
        <total_score>Sum of all scores</total_score>
        <explanation>Detailed explanation of your evaluation</explanation>
    </graph_D_score>
</evaluation>

<selected_graph>[A/B/C/D]</selected_graph>
<justification>
    Please provide a comprehensive justification for your selection, highlighting the key strengths of the chosen graph and how it represents the most effective approach to solving the problem.
</justification>
"""

WORKFLOW_INPUT_WITH_GRAPH = """
Below are two parent graphs, corresponding Python code, and the related prompts used in the graph.
<parent_workflow_A>
    <graph_A>{graph_A}</graph_A>
    <code_A>{code_A}</code_A>
    <role_prompt_A>{prompt_A}</role_prompt_A>
</parent_workflow_A>

<parent_workflow_B>
    <graph_B>{graph_B}</graph_B>
    <code_B>{code_B}</code_B>
    <role_prompt_B>{prompt_B}</role_prompt_B>
</parent_workflow_B>

Based on these parent graphs, you need to generate new Python code according to the new graph and given prompt.

<information_for_new_workflow>
    <new_graph>{new_mermaid}</new_graph>
    <new_role_prompt>{new_prompt}</new_role_prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</information_for_new_workflow>

Carefully analyze the new_graph to generate corresponding code. Pay attention to:
1. The connections between each node
2. The role and function of each operator
3. The usage of operators, you can learn it from Operator Description
4. Ensure all operators used are properly defined in the __init__ method
5. Follow the graph structure, but you should first to guarantee the executable of the python code

For each node in the graph, implement the appropriate code and use the corresponding prompts from new_prompt. If a node or operator doesn't have an explicit prompt provided, DO NOT create one - use empty strings instead. For example, Program operators may have empty analysis fields.
Every prompt referenced in your Python code must be defined in the prompt_custom module. When adding new functionality to your Python code, make sure to import any necessary libraries or modules, except for operator, prompt_custom, create_llm_instance, and CostManage which are automatically imported.
Your implementation must be robust - ensure all methods return appropriate values and **never return None for any field**. Pay special attention to error handling and edge cases.
Use custom methods with proper formatting to ensure your output matches the expected structure. The system will extract answers based on specific rules for scoring, so maintaining the correct output format is critical.

NOTE: You should directly output the code don't add it as a list, make the output code very formal like python code!
"""

CHANGE_MERMAID_PROMPT = """
You are given a mermaid graph and a prompt.
<mermaid_graph>
    <mermaid>{mermaid}</mermaid>
    <prompt>{prompt}</prompt>
    <modification>{modification}</modification>
</mermaid_graph>

You should try to {action} to generate a new improved version. Your modification should be strategic and focused on enhancing the workflow's effectiveness.

Your changes should be:
1. Reasonable and logical
2. Maintain the core functionality of the workflow
3. Improve the efficiency or effectiveness of the process
4. Ensure all connections between nodes remain valid
5. Follow mermaid syntax rules correctly

Please provide a detailed explanation of your changes and why they would improve the workflow.
You should simply change the modification information according to your changes.
"""

PROMPT_FEW_SHOT_CODE = """
IMPROVE_CODE_PROMPT = \"\"\"
The previous solution failed some test cases in the HumanEval benchmark. Please conduct a thorough analysis of the problem statement, identifying all edge cases and potential pitfalls. Then, provide an improved solution that not only fixes the issues but also optimizes performance and adheres to industry-standard coding practices. Ensure your revised code includes clear, concise comments that explain your logic and design choices, and that it robustly handles all specified requirements.
\"\"\"

GENERATE_CODE_PROMPT = \"\"\"
You are an expert programmer with deep knowledge in algorithm design and code optimization. Your task is to generate a clear, efficient, and well-documented solution to the problem described above, as presented in the HumanEval dataset. Please include all relevant code snippets along with detailed explanations of your reasoning, the algorithms used, and how you handle edge cases. Ensure that your solution is easy to understand, follows best practices, and is structured to facilitate future maintenance and enhancements.
\"\"\"

CODE_GENERATE_PROMPT = \"\"\"
Generate a Python function to solve the given problem. Ensure the function name matches the one specified in the problem. Include necessary imports. Use clear variable names and add comments for clarity.

Problem:
{problem}

Function signature:
{entry_point}

Generate the complete function below:
\"\"\"

FIX_CODE_PROMPT = \"\"\"
The provided solution failed to pass the tests. Please analyze the error and fix the code. Ensure the function name and signature remain unchanged. If necessary, add or modify imports, correct logical errors, and improve the implementation.

Problem:
{input}

Provide the corrected function below:
\"\"\"

"""

PROMPT_FEW_SHOT_MATH = """
IMPROVE_CODE_PROMPT = \"\"\"The previous solution failed some test cases. Please analyze the problem carefully and provide an improved solution that addresses all edge cases and requirements. Ensure your code is efficient and follows best practices.\"\"\"

GENERATE_SOLUTION_PROMPT = \"\"\"
Please solve the given mathematical problem step by step. Follow these guidelines:

1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your solution should be thorough, mathematically sound, and easy to understand.
\"\"\"

REFINE_ANSWER_PROMPT = \"\"\"
Given the mathematical problem and the output from the code execution, please provide a well-formatted and detailed solution. Follow these guidelines:

1. Begin with a clear statement of the problem.
2. Explain the approach and any formulas or concepts used.
3. Show step-by-step calculations, using LaTeX notation for mathematical expressions.
4. Interpret the code output and incorporate it into your explanation.
5. Provide a final answer, enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your response should be comprehensive, mathematically rigorous, and easy to follow.
\"\"\"

SOLUTION_PROMPT = \"\"\"
Provide a comprehensive, step-by-step solution to the given mathematical problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \boxed{} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.
\"\"\"

MATH_SOLUTION_PROMPT = \"\"\"
Please solve the given mathematical problem step by step. Follow these guidelines:

1. State the problem clearly.
2. Outline the approach and any relevant formulas or concepts.
3. Provide detailed calculations, using LaTeX notation for mathematical expressions.
4. Explain each step of your reasoning.
5. Present the final answer enclosed in \boxed{} LaTeX notation.
6. Ensure all mathematical notation is in LaTeX format.

Your solution should be thorough, mathematically sound, and easy to understand.
\"\"\"

MATH_SOLVE_PROMPT = \"\"\"
You are a highly skilled mathematician tasked with solving a math problem. Follow these steps carefully:

1. Read and understand the problem thoroughly.
2. Identify all key information, variables, and relationships.
3. Determine the appropriate mathematical concepts, formulas, or equations to use.
4. Solve the problem step-by-step, showing all your work clearly.
5. Double-check your calculations and reasoning at each step.
6. Provide a clear and concise final answer.
7. Verify your solution by plugging it back into the original problem or using an alternative method if possible.

Format your answer as follows:
- Use LaTeX notation for mathematical expressions where appropriate.
- Show each step of your solution process clearly.
- Clearly state your final answer at the end of your solution.
- Express numerical answers as precise values (avoid rounding unless specified).
- Ensure that your final answer is a single numerical value without any units or additional text.
- Do not include any explanatory text with your final answer, just the number itself.

For example, if the final answer is 42.5, your response should end with just:
42.5

Here's the problem to solve:

\"\"\"

DETAILED_SOLUTION_PROMPT = \"\"\"
Provide a comprehensive, step-by-step solution to the given mathematical problem. Your response should include:

1. A clear restatement of the problem.
2. An explanation of the mathematical concepts and theorems involved.
3. A detailed, logical progression of steps leading to the solution.
4. Clear explanations for each step, including the reasoning behind it.
5. All mathematical expressions and equations in LaTeX format.
6. Visual aids or diagrams if applicable (described in text).
7. A final answer clearly marked and enclosed in \boxed{} LaTeX notation.
8. A brief explanation of the significance of the result, if relevant.

Ensure your solution is rigorous, easy to follow, and educational for someone learning the concept.
\"\"\"
"""

PROMPT_FEW_SHOT = {
    "code": PROMPT_FEW_SHOT_CODE,
    "math": PROMPT_FEW_SHOT_MATH,
}