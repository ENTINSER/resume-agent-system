"""Code graph / static-analysis features.

Builds AST-level metrics over a list of code artifacts:
- cyclomatic complexity per function
- function / class / import counts
- unused import detection
- import cycle detection
- test function classification
"""

import ast
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.logger import logger


# AST node types that increase decision points for simple cyclomatic complexity.
_COMPLEXITY_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.Assert,
    ast.BoolOp,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.GeneratorExp,
    ast.DictComp,
)


def _is_test_file(file_path: str) -> bool:
    """Return True if the file lives under tests/ or starts with test_."""
    parts = file_path.replace("\\", "/").split("/")
    return "tests" in parts or any(p.startswith("test_") for p in parts)


def _is_test_function(name: str) -> bool:
    return name.startswith("test_")


def _module_name(file_path: str) -> str:
    """Normalize a file path to a dotted module name, e.g. src/core/E.py -> src.core.E."""
    path = file_path.replace("\\", "/")
    if path.endswith(".py"):
        path = path[:-3]
    return path.replace("/", ".")


def _resolve_relative(level: int, module: Optional[str], current_module: str) -> Optional[str]:
    """Resolve a relative import to an absolute module name."""
    parts = current_module.split(".")
    if level > len(parts):
        return None
    base = parts[:-level] if level > 0 else parts
    if module:
        base = base + module.split(".")
    return ".".join(base) if base else None


def _extract_import_targets(node: ast.AST, current_module: str) -> List[str]:
    """Extract the imported module names from an import statement.

    For ``from x import y`` we conservatively record both ``x`` and ``x.y``
    because ``y`` may be a submodule.  Non-existent targets are filtered out
    during cycle detection.
    """
    targets = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            targets.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        module = node.module
        if node.level:
            resolved = _resolve_relative(node.level, module, current_module)
            if resolved:
                targets.append(resolved)
        elif module:
            targets.append(module)
        # If names are imported from a package, they may be submodules.
        base = module or ""
        if node.level:
            base = _resolve_relative(node.level, base, current_module) or ""
        if base:
            for alias in node.names:
                if alias.name != "*":
                    targets.append(f"{base}.{alias.name}")
    return targets


def _extract_imported_aliases(node: ast.AST) -> List[str]:
    """Extract locally-bound names for unused-import detection.

    For `import os` -> ["os"].
    For `from x import y as z` -> ["z"].
    Star imports are skipped (cannot be statically checked).
    """
    aliases = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            aliases.append(alias.asname if alias.asname else alias.name)
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                continue
            aliases.append(alias.asname if alias.asname else alias.name)
    return aliases


class _ImportVisitor(ast.NodeVisitor):
    """Collect imports and usage information for a module."""

    def __init__(self, current_module: str):
        self.current_module = current_module
        self.module_imports: List[str] = []
        self.imported_aliases: List[str] = []
        self.used_names: Set[str] = set()
        self.functions: int = 0
        self.classes: int = 0
        self.test_functions: int = 0
        self.complexities: List[int] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.functions += 1
        if _is_test_function(node.name):
            self.test_functions += 1
        self.complexities.append(_function_complexity(node))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.functions += 1
        if _is_test_function(node.name):
            self.test_functions += 1
        self.complexities.append(_function_complexity(node))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.classes += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.module_imports.extend(_extract_import_targets(node, self.current_module))
        self.imported_aliases.extend(_extract_imported_aliases(node))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.module_imports.extend(_extract_import_targets(node, self.current_module))
        self.imported_aliases.extend(_extract_imported_aliases(node))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # For `os.path.exists`, `os` is used via attribute access.
        value = node.value
        while isinstance(value, ast.Attribute):
            value = value.value
        if isinstance(value, ast.Name):
            self.used_names.add(value.id)
        self.generic_visit(node)


def _function_complexity(node: ast.FunctionDef) -> int:
    """Compute simple cyclomatic complexity for a function."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, _COMPLEXITY_NODES):
            complexity += 1
    return complexity


def _count_cycles(graph: Dict[str, Set[str]], max_cycles: int = 1000) -> int:
    """Count distinct simple cycles in a directed graph.

    Uses bounded DFS from each node, only extending to nodes with an index
    greater than or equal to the start node to avoid duplicate cycles.
    """
    nodes = sorted(graph.keys())
    index = {node: i for i, node in enumerate(nodes)}
    cycle_count = 0

    for start in nodes:
        stack = [(start, {start})]
        while stack and cycle_count < max_cycles:
            current, visited = stack.pop()
            for neighbor in graph.get(current, set()):
                if neighbor not in index:
                    continue
                if neighbor == start:
                    cycle_count += 1
                    if cycle_count >= max_cycles:
                        break
                elif index[neighbor] >= index[start] and neighbor not in visited:
                    stack.append((neighbor, visited | {neighbor}))
    return cycle_count


class CodeGraphAnalyzer:
    """AST-based static analyzer for code artifacts."""

    @staticmethod
    def analyze(artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze artifacts and return graph metrics.

        Non-Python files are skipped. Syntax errors log a warning and skip
        the file.
        """
        total_functions = 0
        total_classes = 0
        total_imports = 0
        test_function_count = 0
        code_function_count = 0
        complexities: List[int] = []
        unused_import_count = 0

        import_graph: Dict[str, Set[str]] = defaultdict(set)
        analyzed_modules: Set[str] = set()

        for artifact in artifacts:
            file_path = artifact.get("file_path", "")
            content = artifact.get("content", "")

            if not file_path.endswith(".py"):
                continue

            current_module = _module_name(file_path)
            analyzed_modules.add(current_module)

            try:
                tree = ast.parse(content)
            except SyntaxError as exc:
                logger.warning(f"[CodeGraph] Syntax error in {file_path}, skipping: {exc}")
                continue

            visitor = _ImportVisitor(current_module)
            visitor.visit(tree)

            total_functions += visitor.functions
            total_classes += visitor.classes
            total_imports += len(visitor.module_imports)

            if _is_test_file(file_path):
                test_function_count += visitor.test_functions
            else:
                code_function_count += visitor.functions

            # Track imports for cycle detection.
            for target in visitor.module_imports:
                import_graph[current_module].add(target)

            # Unused imports: module-level only.
            unused = [
                alias for alias in visitor.imported_aliases
                if alias not in visitor.used_names
            ]
            unused_import_count += len(unused)

            complexities.extend(visitor.complexities)

        # Count import cycles restricted to the set of analyzed modules.
        filtered_graph = {
            mod: {dep for dep in deps if dep in analyzed_modules}
            for mod, deps in import_graph.items()
            if mod in analyzed_modules
        }
        import_cycle_count = _count_cycles(filtered_graph)

        if complexities:
            avg_complexity = round(sum(complexities) / len(complexities), 2)
            max_complexity = max(complexities)
        else:
            avg_complexity = 0.0
            max_complexity = 0

        test_to_code_ratio = (
            round(test_function_count / max(code_function_count, 1), 4)
        )

        return {
            "avg_cyclomatic_complexity": avg_complexity,
            "max_cyclomatic_complexity": max_complexity,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_imports": total_imports,
            "unused_import_count": unused_import_count,
            "import_cycle_count": import_cycle_count,
            "test_function_count": test_function_count,
            "code_function_count": code_function_count,
            "test_to_code_function_ratio": test_to_code_ratio,
        }
