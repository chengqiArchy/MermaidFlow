"""
@Time    : 2025-03-31
@Author  : didi & Zhaoyang
@Acknowledgement https://github.com/evalplus/evalplus/blob/master/evalplus/sanitize.py
"""

import ast
import traceback
from enum import Enum
from typing import Dict, Generator, List, Optional, Set, Tuple

from tree_sitter import Node


class NodeType(Enum):
    CLASS = "class_definition"
    FUNCTION = "function_definition"
    IMPORT = ["import_statement", "import_from_statement"]
    IDENTIFIER = "identifier"
    ATTRIBUTE = "attribute"
    RETURN = "return_statement"
    EXPRESSION = "expression_statement"
    ASSIGNMENT = "assignment"


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    """
    Traverse the tree structure starting from the given node.

    :param node: The root node to start the traversal from.
    :return: A generator object that yields nodes in the tree.
    """
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str) -> str:
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_definition_name(node: Node) -> str:
    for child in node.children:
        if child.type == NodeType.IDENTIFIER.value:
            return child.text.decode("utf8")


def has_return_statement(node: Node) -> bool:
    traverse_nodes = traverse_tree(node)
    for node in traverse_nodes:
        if node.type == NodeType.RETURN.value:
            return True
    return False


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == NodeType.IDENTIFIER.value:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, str]) -> Set[str]:
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    """
    Sanitize and extract relevant parts of the given Python code.
    This function parses the input code, extracts import statements, class and function definitions,
    and variable assignments. If an entrypoint is provided, it only includes definitions that are
    reachable from the entrypoint in the call graph.

    :param code: The input Python code as a string.
    :param entrypoint: Optional name of a function to use as the entrypoint for dependency analysis.
    :return: A sanitized version of the input code, containing only relevant parts.
    """
    code = code_extract(code)
    
    try:
        # Use the more reliable fallback method directly to avoid the warnings/errors
        return fallback_sanitize_with_ast(code, entrypoint)
    except Exception as e:
        print(f"ERROR in sanitize: {str(e)}")
        # If even the fallback fails, return the original code
        return code

def fallback_sanitize_with_ast(code: str, entrypoint: Optional[str] = None) -> str:
    """A function that uses Python's built-in ast module instead of tree-sitter."""
    try:
        tree = ast.parse(code)
        imports = []
        definitions = []
        function_names = set()
        class_names = set()
        variable_names = set()
        
        # First collect all top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
                definitions.append((node.name, ast.unparse(node)))
            elif isinstance(node, ast.ClassDef):
                class_names.add(node.name)
                definitions.append((node.name, ast.unparse(node)))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variable_names.add(target.id)
                        definitions.append((target.id, ast.unparse(node)))
        
        # If entrypoint is specified, find reachable definitions
        if entrypoint:
            # Build a dependency graph
            dependencies = {}
            for name, _ in definitions:
                dependencies[name] = set()
            
            # Add edges to the dependency graph
            for name, code_str in definitions:
                # Parse the code to find references to other definitions
                node = ast.parse(code_str)
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Name) and subnode.id in dependencies:
                        dependencies[name].add(subnode.id)
            
            # Find all definitions reachable from the entrypoint
            reachable = set()
            def dfs(name):
                if name in reachable:
                    return
                reachable.add(name)
                for dep in dependencies.get(name, []):
                    dfs(dep)
            
            # Start DFS from the entrypoint
            if entrypoint in dependencies:
                dfs(entrypoint)
            
            # Filter definitions to only include reachable ones
            filtered_defs = []
            for name, code_str in definitions:
                if name in reachable:
                    filtered_defs.append(code_str)
            definitions = filtered_defs
        else:
            # If no entrypoint, include all definitions
            definitions = [code_str for _, code_str in definitions]
        
        # Combine imports and definitions
        return "\n".join(imports + definitions)
    except Exception as e:
        print(f"AST fallback failed: {str(e)}")
        return code  # Return original code if all else fails