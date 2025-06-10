import ast
import os
from pathlib import Path
from collections import defaultdict


def extract_imports(file_path: Path) -> list[str]:
    """
    Extracts all import statements from a single Python file using AST.

    Args:
        file_path: The path to the Python file.

    Returns:
        A list of full import strings (e.g., 'os', 'pathlib.Path').
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module_prefix = node.module or ''
                for alias in node.names:
                    if alias.name == '*':
                        imports.append(f"{module_prefix}.*")
                    else:
                        imports.append(f"{module_prefix}.{alias.name}")
        return imports
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Could not parse {file_path}: {e}")
        return []


def resolve_import_to_module(import_path: str, internal_modules: set[str]) -> str | None:
    """
    Resolves an import path to its originating internal module file.
    For an import like 'my_app.utils.helpers.some_func', this function
    will return 'my_app.utils.helpers' if that exists as a module.

    Args:
        import_path: The full import string.
        internal_modules: A set of all module names in the project.

    Returns:
        The resolved module name as a string, or None if it's not internal.
    """
    parts = import_path.split('.')
    for i in range(len(parts), 0, -1):
        potential_module = ".".join(parts[:i])
        if potential_module in internal_modules:
            return potential_module
    return None


def find_all_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """
    Finds all elementary cycles in a directed graph using DFS.

    Args:
        graph: An adjacency list representation of the dependency graph.

    Returns:
        A list of lists, where each inner list is a circular path.
    """
    all_cycles = []
    unique_cycles_found = set()

    def dfs(node, path, visiting):
        path.append(node)
        visiting.add(node)

        for neighbor in sorted(list(graph.get(node, set()))):
            if neighbor in visiting:  # Cycle detected
                try:
                    cycle_start_index = path.index(neighbor)
                    cycle = path[cycle_start_index:]

                    # Normalize to prevent duplicates (e.g., A->B->A vs B->A->B)
                    # We store the tuple of the sorted cycle to check for uniqueness
                    # but will report the actual path.
                    normalized_representation = tuple(sorted(cycle))
                    if normalized_representation not in unique_cycles_found:
                        all_cycles.append(cycle + [neighbor])  # Add closing node for display
                        unique_cycles_found.add(normalized_representation)
                except ValueError:
                    continue  # Should not happen
            else:
                dfs(neighbor, path, visiting)

        path.pop()
        visiting.remove(node)

    # Iterate over all nodes to find cycles in disconnected parts of the graph
    all_nodes = sorted(list(graph.keys()))
    for node in all_nodes:
        dfs(node, path=[], visiting=set())

    return all_cycles


def analyze_and_print_project_dependencies(project_dir: Path):
    """
    Analyzes project dependencies, prints internal imports for each file,
    and lists any circular dependencies found.
    """
    excluded_dirs = {".venv", "__pycache__", ".git", "node_modules", ".pytest_cache", "build", "dist"}

    print(f"Analyzing project at: {project_dir}\n")

    # 1. Find all Python files
    py_files = [
        p for p in project_dir.rglob("*.py")
        if not any(excluded in p.parts for excluded in excluded_dirs)
    ]

    # 2. Create a set of all internal module names
    internal_modules = set()
    for py_file in py_files:
        rel_path = py_file.relative_to(project_dir)
        module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
        internal_modules.add(module_path)
        if module_path.endswith('.__init__'):
            internal_modules.add(module_path.removesuffix('.__init__'))

    # 3. Build dependency graph and collect import data for printing
    dependency_graph = defaultdict(set)
    file_to_internal_imports = defaultdict(list)

    for py_file in py_files:
        current_module = str(py_file.relative_to(project_dir).with_suffix('')).replace(os.sep, '.')
        imports = extract_imports(py_file)

        for imp in imports:
            resolved_module = resolve_import_to_module(imp, internal_modules)
            # Add to graph if it's an internal import to another module
            if resolved_module and resolved_module != current_module:
                dependency_graph[current_module].add(resolved_module)
                file_to_internal_imports[current_module].append(imp)

    # 4. Print the per-file import analysis
    print("--- PROJECT INTERNAL IMPORTS ---\n")
    all_modules = {str(pf.relative_to(project_dir).with_suffix('')).replace(os.sep, '.') for pf in py_files}
    for module in sorted(list(all_modules)):
        imports = sorted(file_to_internal_imports.get(module, []))
        print(f"• {module}")
        if imports:
            for imp in imports:
                print(f"    -> {imp}")
        else:
            print("    (No internal imports)")
        print()

    # 5. Find and print circular dependencies
    print("\n--- CIRCULAR DEPENDENCIES ---\n")
    cycles = find_all_cycles(dependency_graph)

    if not cycles:
        print("No circular dependencies found. ✅")
    else:
        print(f"Found {len(cycles)} circular dependency path(s): 🚨\n")
        for i, cycle in enumerate(cycles, 1):
            print(f"{i}: {' -> '.join(cycle)}")
    print()


if __name__ == "__main__":
    debug_dir = Path(__file__).resolve().parent
    common_dir = debug_dir.parent
    project_dir = common_dir.parent
    analyze_and_print_project_dependencies(project_dir)