import ast
import os
from pathlib import Path
from collections import defaultdict


def extract_imports(file_path):
    """Extract imports from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if alias.name == '*':
                        imports.append(f"{module}.*")
                    else:
                        imports.append(f"{module}.{alias.name}")

        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def analyze_project_dependencies():
    """Analyze dependencies in the project."""
    project_dir = Path(__file__).resolve().parent.parent.parent
    excluded_dirs = {".venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}
    py_files = [p for p in project_dir.rglob("*.py")
                if not any(excluded in str(p) for excluded in excluded_dirs)]

    file_imports = {}
    internal_deps = defaultdict(set)
    external_deps = set()

    # Get all internal module names
    internal_modules = set()
    for py_file in py_files:
        rel_path = py_file.relative_to(project_dir)
        module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
        internal_modules.add(module_path)
        if module_path.endswith('.__init__'):
            internal_modules.add(module_path[:-9])  # Remove .__init__

    # Analyze each file
    for py_file in py_files:
        rel_path = py_file.relative_to(project_dir)
        module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')

        imports = extract_imports(py_file)
        file_imports[module_name] = imports

        for imp in imports:
            # Check if it's an internal dependency
            is_internal = False
            for internal_mod in internal_modules:
                if imp.startswith(internal_mod):
                    internal_deps[module_name].add(internal_mod)
                    is_internal = True
                    break

            if not is_internal:
                # It's an external dependency
                root_module = imp.split('.')[0]
                external_deps.add(root_module)

    return file_imports, internal_deps, external_deps


def print_analysis():
    """Print the dependency analysis."""
    file_imports, internal_deps, external_deps = analyze_project_dependencies()

    print("=== PROJECT DEPENDENCY ANALYSIS ===\n")

    print("EXTERNAL DEPENDENCIES:")
    for dep in sorted(external_deps):
        print(f"  - {dep}")
    print()

    print("INTERNAL DEPENDENCIES:")
    for module, deps in sorted(internal_deps.items()):
        if deps:
            print(f"  {module}:")
            for dep in sorted(deps):
                print(f"    -> {dep}")
    print()

    print("FILES WITH NO INTERNAL DEPENDENCIES:")
    for module, deps in sorted(internal_deps.items()):
        if not deps:
            print(f"  - {module}")
    print()


def generate_dot_file():
    """Generate a DOT file for visualization grouped by first directory."""
    file_imports, internal_deps, external_deps = analyze_project_dependencies()
    project_dir = Path(__file__).resolve().parent.parent

    # Group modules by first directory
    groups = defaultdict(set)
    for module in internal_deps.keys():
        if '.' in module:
            first_dir = module.split('.')[0]
            groups[first_dir].add(module)
        else:
            groups['root'].add(module)

    dot_content = ["digraph dependencies {"]
    dot_content.append("  rankdir=TB;")
    dot_content.append("  node [shape=box];")
    dot_content.append("  edge [color=blue];")
    dot_content.append("  compound=true;")
    dot_content.append("")

    # Create subgraphs for each directory
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray', 'lightcyan']
    color_idx = 0

    for group_name, modules in sorted(groups.items()):
        if modules:
            color = colors[color_idx % len(colors)]
            color_idx += 1

            dot_content.append(f'  subgraph "cluster_{group_name}" {{')
            dot_content.append(f'    label="{group_name}";')
            dot_content.append(f'    style=filled;')
            dot_content.append(f'    fillcolor={color};')
            dot_content.append(f'    fontsize=14;')
            dot_content.append(f'    fontweight=bold;')

            # Add nodes in this group
            for module in sorted(modules):
                # Use short name for display (remove group prefix)
                if module.startswith(group_name + '.'):
                    display_name = module[len(group_name) + 1:]
                else:
                    display_name = module
                dot_content.append(f'    "{module}" [label="{display_name}"];')

            dot_content.append("  }")
            dot_content.append("")

    # Add dependencies between modules
    dot_content.append("  // Dependencies")
    for module, deps in internal_deps.items():
        for dep in deps:
            if module != dep:  # Avoid self-references
                dot_content.append(f'  "{module}" -> "{dep}";')

    dot_content.append("}")

    dot_file = project_dir / "dependencies.dot"
    with open(dot_file, 'w') as f:
        f.write('\n'.join(dot_content))

    print(f"DOT file generated: {dot_file}")
    print("Groups found:")
    for group_name, modules in sorted(groups.items()):
        if modules:
            print(f"  {group_name}: {len(modules)} modules")
    print()
    print("To visualize: dot -Tpng dependencies.dot -o dependencies.png")
    print("Or use an online viewer like: http://magjac.com/graphviz-visual-editor/")


if __name__ == "__main__":
    print_analysis()