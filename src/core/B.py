"""B - Developer（开发工程师）P1优化版

优化点：
- 多策略代码解析（标准格式、Markdown、JSON、逐行分析）
- 智能文件名推断（代码块缺少文件名时）
- 代码质量预检查（语法有效性、导入检测）
- 结构化的错误处理
"""

import os
import re
import ast
import json
import textwrap
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Set

from src.core.shared_state import SharedState
from src.core.logger import logger
from src.core.llm import get_llm
from src.core.docker_sandbox import DockerSandbox, check_docker_available
from src.vector_store.rag_retriever import RAGRetriever
from src.core.config import settings


STDLIB_MODULES = {
    "__future__", "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib", "copy",
    "csv", "dataclasses", "datetime", "decimal", "enum", "functools", "hashlib", "inspect",
    "io", "itertools", "json", "logging", "math", "os", "pathlib", "pickle", "platform",
    "random", "re", "shutil", "socket", "statistics", "string", "subprocess", "sys",
    "tempfile", "textwrap", "threading", "time", "traceback", "typing", "unittest", "uuid",
    "warnings", "weakref", "xml",
}


class ChunkIncompleteError(Exception):
    """chunk 生成的代码不完整，无法通过 ast.parse 校验。"""
    pass


class ArtifactStatus(str, Enum):
    """代码产物状态"""
    PENDING = "pending"
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


def _max_tokens_for_complexity(complexity: str, is_doc: bool = False) -> int:
    from src.core.config import settings

    if is_doc:
        return {"simple": 2048, "normal": 4096, "complex": 4096}.get(complexity, 4096)
    return {
        "simple": settings.max_completion_tokens_simple,
        "normal": settings.max_completion_tokens_normal,
        "complex": settings.max_completion_tokens_complex,
    }.get(complexity, settings.max_completion_tokens_normal)


def _estimate_file_tokens(plan: dict) -> int:
    """预估文件需要的输出 token 数。按 class/function 等叶子元素计数。"""
    element_count = 0
    for elem in plan.get("required_elements", []):
        if elem.get("type") == "class":
            element_count += 1 + len(elem.get("methods", []))
        else:
            element_count += 1

    avg_tokens_per_element = {"simple": 250, "normal": 600, "complex": 900}.get(
        plan.get("complexity", "normal"), 600
    )
    total_estimate = element_count * avg_tokens_per_element + 300
    if "test" in plan.get("path", ""):
        total_estimate = element_count * 900 + 400
    return total_estimate


def _split_required_elements(required_elements: list, max_per_chunk: int = 3) -> list:
    """将 required_elements 拆成多个 chunk，每个 chunk 2-3 个元素。"""
    chunks = []
    for i in range(0, len(required_elements), max_per_chunk):
        chunks.append(required_elements[i : i + max_per_chunk])
    return chunks


def _build_chunk_prompt(
    plan: dict,
    chunk_elements: list,
    already_generated_parts: list,
    dependency_summaries: str,
) -> str:
    """为拆分后的一个 chunk 构建生成 prompt。强制传入详细字段名和类名约束。"""

    def _format_element_spec(elem):
        if isinstance(elem, dict):
            spec = elem.get("spec") or elem.get("logic_requirements") or ""
            name = elem.get("name", "")
            etype = elem.get("type", "")
            methods = elem.get("methods", [])
            if etype == "class":
                parts = [f"class {name}"]
                if methods:
                    parts.append(f"methods: {', '.join(methods)}")
                if spec:
                    parts.append(f"spec: {spec}")
                return " | ".join(parts)
            elif etype == "function":
                parts = [f"def {name}()"]
                if spec:
                    parts.append(f"spec: {spec}")
                return " | ".join(parts)
            return spec or str(elem)
        return str(elem)

    element_specs = [_format_element_spec(e) for e in chunk_elements]
    all_specs = [
        _format_element_spec(e) for e in plan.get("required_elements", [])
    ]

    is_test = plan.get("path", "").startswith("tests/")
    has_class = any(e.get("type") == "class" for e in plan.get("required_elements", []))

    if already_generated_parts:
        previous = "\n".join(already_generated_parts[-2:])
        part_instruction = "本次只补充实现下面列出的方法/函数，把它们追加到已有文件末尾，不要重复 imports 和类定义。"
    else:
        previous = "（当前文件尚未生成）"
        if has_class:
            part_instruction = "本次生成该文件的完整基础部分，包括所有 imports、类定义以及下面列出的方法。"
        else:
            part_instruction = "本次生成该文件的完整基础部分，包括所有 imports、fixtures 以及下面列出的函数。"
    indent_note = (
        "\n- 注意：本文件包含类定义，方法需要 4 空格缩进放在类内部"
        if has_class
        else "\n- 注意：本文件只有顶层函数/类，每个函数/类都从第 0 列开始，不要额外缩进"
    )
    test_constraints = (
        "\n- 每个被测方法只写 1-2 个核心测试用例\n"
        "- 不要穷举所有边界条件\n"
        "- 每个测试方法不超过 15 行\n"
        "- 使用已定义的 fixtures，不要重复创建\n"
        if is_test
        else ""
    )

    return f"""你是 Python 代码生成器。这是对 {plan['path']} 的第 {len(already_generated_parts)} 次补充生成。

## 文件角色
{plan['role']}

## 本次需要实现的内容（只实现这些，不要重复已生成的部分）
{chr(10).join(f'- {spec}' for spec in element_specs)}

## 本文件已生成的部分
{previous}

## 本次生成说明
{part_instruction}

## 依赖文件摘要
{dependency_summaries}

## 实现要求

### 代码规模约束：
- 本次生成不超过 150 行
- 每个方法不超过 25 行
- 不重复已生成部分的 imports 和类定义
{indent_note}
{test_constraints}
### 字段名和类名必须严格使用以下名称（不得自行发明）：
{chr(10).join(f'- {spec}' for spec in all_specs)}
- 评分维度字段必须是 accuracy、completeness、readability、structure（不得使用 novelty、methodology 等其他名称）
- 异常类名必须是 RevisionLimitExceeded
- 状态字段必须与依赖文件中的定义一致

### 质量标准：
- 参数校验（非法输入抛出明确的 ValueError）
- 核心业务逻辑（不少于 8 行有效代码）
- try/except 异常处理（捕获具体异常类型）
- 关键步骤用 logging.info 记录
- 类型注解完整

### 测试文件额外要求（仅 tests/ 文件）：
- 每个被测方法只写 1-2 个核心测试用例
- 不要穷举所有边界条件
- 每个测试方法不超过 15 行
- 使用已定义的 fixtures，不要重复创建

### 禁止（违反将导致重试）：
- 生成 pass、...、return None 或空字符串作为方法体
- 生成占位符注释如 # TODO
- 重复已生成部分的代码
- 自行发明或修改上述要求的字段名和类名

只输出代码，不要解释。"""


def _is_valid_python_snippet(code: str) -> bool:
    """检查一段代码是否是完整可解析的 Python 语句。"""
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _has_stub_methods(code: str) -> bool:
    """检测代码中是否存在空方法体或占位符实现（只检测 pass、空体、return None）。"""
    if not code or not code.strip():
        return True
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        # 去掉 docstring
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        # 空方法体
        if not body:
            return True

        # 只有 pass
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            return True

        # 只有 return None 或 return
        if len(body) == 1 and isinstance(body[0], ast.Return):
            ret = body[0].value
            if ret is None or (isinstance(ret, ast.Constant) and ret.value is None):
                return True

    return False


def _generate_init_py(generated_files: dict, project_dir: str) -> str:
    """根据实际生成的文件内容，用规则引擎自动生成 __init__.py。"""
    imports = []

    for file_path, content in generated_files.items():
        if (
            not file_path.startswith("src/")
            or not file_path.endswith(".py")
            or file_path == "src/__init__.py"
        ):
            continue

        module_name = file_path.replace("src/", "").replace(".py", "").replace("/", ".")

        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                imports.append(f"from src.{module_name} import {node.name}")
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                imports.append(f"from src.{module_name} import {node.name}")

    if not imports:
        return ""

    init_content = '"""项目包初始化。自动生成，不要手动编辑。"""\n\n'
    init_content += "\n".join(sorted(set(imports)))
    init_content += "\n"

    init_path = Path(project_dir) / "src" / "__init__.py"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text(init_content)

    logger.info(f"[B] __init__.py 自动生成：{len(imports)} 个导出")
    return init_content


def _assemble_chunks(parts: list, plan: dict) -> str:
    """将拆分生成的多个部分拼装成完整文件。拼装前校验每个 chunk 完整可解析。"""
    if not parts:
        return ""
    first_part = parts[0]
    cleaned_parts = [first_part]
    for part in parts[1:]:
        if not _is_valid_python_snippet(part):
            logger.warning("[B] chunk 代码不完整，无法 ast.parse，触发重试/回退")
            raise ChunkIncompleteError(
                f"Chunk for {plan['path']} is not valid Python: {part[:200]}"
            )
        # 去掉后续 chunk 中重复的 import/from 行
        lines = part.split("\n")
        code_lines = [l for l in lines if not l.strip().startswith(("import ", "from "))]
        # 对顶层元素做 dedent，避免 chunk 之间缩进基准不一致
        dedented = textwrap.dedent("\n".join(code_lines))
        cleaned_parts.append(dedented)
    return "\n\n".join(cleaned_parts)


def _any_artifact_incomplete(artifacts: List[dict]) -> bool:
    return any(a.get("status") == ArtifactStatus.INCOMPLETE.value for a in artifacts)


def _validate_file_plan(file_plan: List[dict]) -> List[str]:
    """第一道检查：校验 file_plan 自身的结构完整性"""
    errors: List[str] = []
    paths = {p.get("path", "") for p in file_plan}

    for p in file_plan:
        path = p.get("path", "")
        complexity = p.get("complexity", "normal")
        required = p.get("required_elements", [])
        deps = p.get("deps", [])

        if complexity == "complex" and not required and not path.startswith("tests/"):
            errors.append(f"{path}: complex 文件缺少 required_elements")

        for dep in deps:
            if dep not in paths:
                errors.append(f"{path}: 依赖 {dep} 不在 file_plan 中")

    return errors


def _dry_run_imports(artifacts: List[dict], file_plan: List[dict]) -> List[str]:
    """第三道检查：把代码产物写到临时目录并尝试 import，只检查项目内部缺失的模块/符号。"""
    import tempfile
    import importlib
    import sys

    errors: List[str] = []
    plan_paths = {p.get("path", "") for p in file_plan}
    project_packages: Set[str] = set()
    for path in plan_paths:
        if path.endswith(".py"):
            parts = path.replace(".py", "").split("/")
            for i in range(1, len(parts) + 1):
                project_packages.add(".".join(parts[:i]))

    py_files = [a for a in artifacts if a.get("file_path", "").endswith(".py")]
    if not py_files:
        return errors

    with tempfile.TemporaryDirectory() as tmpdir:
        for a in py_files:
            path = a.get("file_path", "")
            full = os.path.join(tmpdir, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(a.get("content", ""))

        if tmpdir not in sys.path:
            sys.path.insert(0, tmpdir)
        try:
            for a in py_files:
                path = a.get("file_path", "")
                if path.startswith("tests/"):
                    continue
                mod_name = path.replace(".py", "").replace("/", ".")
                try:
                    importlib.import_module(mod_name)
                except (ImportError, ModuleNotFoundError) as e:
                    missing = _extract_missing_module(str(e))
                    if missing and (missing in project_packages or any(missing.startswith(p) for p in project_packages)):
                        errors.append(f"{path}: {e}")
                    else:
                        logger.debug(f"[B-DryRun] 第三方依赖缺失，忽略: {e}")
                except Exception as e:
                    errors.append(f"{path}: 导入异常 {e}")
        finally:
            if tmpdir in sys.path:
                sys.path.remove(tmpdir)
            # 清理已导入的临时模块，避免污染
            for a in py_files:
                mod_name = a.get("file_path", "").replace(".py", "").replace("/", ".")
                if mod_name in sys.modules:
                    del sys.modules[mod_name]

    return errors


STUB_FILE_TEMPLATES = {
    "requirements.txt": "# Auto-generated requirements\npytest>=7.0\n",
    "pyproject.toml": """[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "Auto-generated project"
requires-python = ">=3.10"
dependencies = [
{deps}
]

[tool.pytest.ini_options]
testpaths = ["tests"]
""",
    ".env.example": """# Example environment variables
# Copy to .env and fill in real values
""",
    "README.md": """# {project_name}

Auto-generated project.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

## Test

```bash
pytest tests/
```
""",
}


def _generate_stub_files(file_plan: List[dict], project_name: str, tech_stack: List[str]) -> List[dict]:
    """规则引擎生成非核心文件，不走 LLM"""
    stub_paths = {p["path"] for p in file_plan if p.get("path", "").split("/")[-1] in STUB_FILE_TEMPLATES}
    artifacts: List[dict] = []

    # 核心依赖：从 tech_stack 推断
    core_deps = set()
    for t in tech_stack:
        t_lower = t.lower()
        if "langgraph" in t_lower:
            core_deps.add("langgraph>=0.0.50")
        if "langchain" in t_lower:
            core_deps.add("langchain>=0.1.0")
        if "pydantic" in t_lower:
            core_deps.add("pydantic>=2.0")
        if "fastapi" in t_lower:
            core_deps.add("fastapi>=0.100")
            core_deps.add("uvicorn>=0.23")
        if "flask" in t_lower:
            core_deps.add("flask>=2.3")
        if "requests" in t_lower:
            core_deps.add("requests>=2.31")
        if "beautifulsoup" in t_lower:
            core_deps.add("beautifulsoup4>=4.12")
        if "pandas" in t_lower:
            core_deps.add("pandas>=2.0")
        if "numpy" in t_lower:
            core_deps.add("numpy>=1.24")

    deps_str = "\n".join(f'    "{d}",' for d in sorted(core_deps)) if core_deps else "    # no third-party deps"

    for path in sorted(stub_paths):
        template = STUB_FILE_TEMPLATES.get(path.split("/")[-1], "")
        content = template.format(project_name=project_name or "project", deps=deps_str)
        artifacts.append({
            "file_path": path,
            "content": content,
            "status": ArtifactStatus.COMPLETE.value,
            "truncated": False,
            "continuation_attempts": 0,
        })

    return artifacts


def _extract_missing_module(error_msg: str) -> str:
    """从 ImportError / ModuleNotFoundError 消息中提取缺失模块名"""

    patterns = [
        r"No module named '([^']+)'",
        r"cannot import name '([^']+)' from '([^']+)'",
    ]
    for pat in patterns:
        m = re.search(pat, error_msg)
        if m:
            return m.group(1)
    return ""


def _validate_skeleton_artifacts(artifacts: List[dict], file_plan: List[dict]) -> List[str]:
    """Validate skeleton signatures and import targets against file_plan.

    Only project-internal imports are errors; third-party imports are warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []
    file_paths = {a.get("file_path", "") for a in artifacts}
    plan_paths = {p.get("path", "") for p in file_plan}
    path_to_plan = {p.get("path", ""): p for p in file_plan}

    for a in artifacts:
        path = a.get("file_path", "")
        content = a.get("content", "")
        if not content:
            errors.append(f"{path}: 空内容")
            continue
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            errors.append(f"{path}: 语法错误 {e}")
            continue

        # 收集已声明的 class / function
        declared_classes: Dict[str, ast.ClassDef] = {}
        declared_funcs: Set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                declared_classes[node.name] = node
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                declared_funcs.add(node.name)

        # 检查 required_elements
        plan = path_to_plan.get(path, {})
        for elem in plan.get("required_elements", []):
            etype = elem.get("type", "")
            ename = elem.get("name", "")
            methods = elem.get("methods", [])
            if etype == "class":
                cls = declared_classes.get(ename)
                if not cls:
                    errors.append(f"{path}: 缺少必需类 {ename}")
                    continue
                cls_methods = {m.name for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
                for m in methods:
                    if m not in cls_methods:
                        errors.append(f"{path}: 类 {ename} 缺少必需方法 {m}")
            elif etype == "function":
                if ename not in declared_funcs:
                    errors.append(f"{path}: 缺少必需函数 {ename}")

        # 检查 import 目标是否存在于 file_plan
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    if base in STDLIB_MODULES:
                        continue
                    if any(f.startswith(f"{base}/") or f == f"{base}.py" or f == base for f in plan_paths):
                        continue
                    warnings.append(f"{path}: import {base} 未解析")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                base = module.split(".")[0]
                if base in STDLIB_MODULES or base == "__future__":
                    continue
                dotted_module = module.replace(".", "/")
                candidates = [
                    f"{dotted_module}.py",
                    f"{dotted_module}/__init__.py",
                ]
                if not any(c in plan_paths for c in candidates):
                    warnings.append(f"{path}: from {module} 未解析")

    if warnings:
        logger.warning(f"[B] 骨架 import 警告: {warnings[:5]}")
    return errors


def _get_file_plan_from_state(state: SharedState) -> List[dict]:
    project = state.get("project", {})
    return project.get("file_plan") or []


def _topological_sort_plans(plans: List[dict]) -> List[dict]:
    """按 deps 拓扑排序 file_plan，无依赖的在前"""
    path_to_plan = {p["path"]: p for p in plans}
    visited: Set[str] = set()
    result: List[dict] = []

    def visit(path: str):
        if path in visited:
            return
        visited.add(path)
        plan = path_to_plan.get(path)
        if not plan:
            return
        for dep in plan.get("deps", []):
            if dep in path_to_plan:
                visit(dep)
        result.append(plan)

    for p in plans:
        visit(p["path"])
    return result


def _classify_issue_type(error: str) -> str:
    """简单分类错误类型"""
    err_lower = error.lower()
    if "syntax" in err_lower or "indent" in err_lower or "eol" in err_lower:
        return "syntax"
    if "import" in err_lower or "modulenotfound" in err_lower:
        return "import"
    if "assert" in err_lower or "test" in err_lower:
        return "test"
    if "timeout" in err_lower or "deadline" in err_lower:
        return "timeout"
    return "other"


def _test_error_signature(result: dict) -> str:
    """把测试错误信息归一化，用于去重判断"""
    file_path = result.get("file", "")
    error = result.get("error", "")[:120]
    # 去掉行号、临时路径、具体变量值
    error = re.sub(r"File \".*?\", line \d+", "", error)
    error = re.sub(r"line \d+", "", error)
    error = re.sub(r"0x[0-9a-fA-F]+", "", error)
    return f"{file_path}:{error.strip()}"


def _record_repair_history(state: SharedState, test_results: List[dict]) -> None:
    """记录本轮测试错误的签名，用于检测同一问题反复出现"""
    history = state.get("repair_history", {})
    for r in test_results:
        if not r.get("passed", False):
            sig = _test_error_signature(r)
            history[sig] = history.get(sig, 0) + 1
            from src.core import metrics
            metrics.observe_b_repair_attempt(
                file=r.get("file", "unknown"),
                issue_type=_classify_issue_type(r.get("error", "")),
            )
    state["repair_history"] = history


def _should_escalate_to_hitl(state: SharedState, test_results: List[dict]) -> bool:
    """同一错误签名出现 2 次以上，或单文件失败 3 次以上，升级到 HITL"""
    history = state.get("repair_history", {})
    for r in test_results:
        if r.get("passed", False):
            continue
        sig = _test_error_signature(r)
        if history.get(sig, 0) >= 2:
            logger.warning(f"[B] 同一问题重复出现，升级到 HITL: {sig}")
            return True
    return False


def _record_b_hitl(state: SharedState, reason: str) -> None:
    """记录 B 阶段 HITL 升级并设置状态"""
    state["human_review_required"] = True
    state["human_review_reason"] = reason
    from src.core import metrics
    metrics.observe_b_hitl(reason=reason)


# ===== 优化的系统提示词 =====
DEVELOPER_SYSTEM_PROMPT = """你是一位资深 Python 开发工程师。你的任务是根据需求编写高质量、可运行的 Python 项目。

## 必须严格遵循的输出格式（禁止任何其他格式）

每个文件必须严格按照以下格式输出。禁止输出 Markdown 代码块（如 ```python ... ```），禁止输出 JSON 格式。

格式模板（必须复制这个格式，只替换文件路径和内容）：

=== FILE: main.py ===
import typer
from pydantic import BaseModel

class Calculator(BaseModel):
    pass

def add(a: float, b: float) -> float:
    \"\"\"加法运算\"\"\"
    return a + b

if __name__ == "__main__":
    print(add(1, 2))
=== END ===

=== FILE: tests/test_main.py ===
import pytest
from main import add

def test_add():
    assert add(1, 2) == 3
=== END ===

=== FILE: README.md ===
# 项目名称

## 功能
描述功能

## 安装
pip install -r requirements.txt

## 使用
python main.py
=== END ===

=== FILE: requirements.txt ===
pytest>=7.0.0
typer>=0.9.0
pydantic>=2.0.0
=== END ===

## 绝对禁止
- 使用 ```python 或 ``` 包裹代码
- 输出 JSON 格式
- 在一个代码块内放多个文件
- 省略文件名或 === END === 标记
- 在代码块前加注释如 `# file: main.py`

## 要求
1. 代码必须是完整、可运行的 Python 文件
2. 遵循 PEP8 规范，使用类型提示
3. 包含文档字符串和注释
4. 编写 pytest 格式的单元测试（测试必须与代码匹配）
5. 生成 README.md 和 requirements.txt
6. 不要使用外部网络依赖
7. 代码应可在隔离环境中运行

## 必须包含的文件
- main.py（主程序入口）
- tests/test_main.py（测试文件）
- README.md（项目说明）
- requirements.txt（依赖列表）
"""


class CodeParser:
    """多策略代码解析器 - P1核心优化"""
    
    def __init__(self, response: str):
        self.response = response
        self.artifacts: List[dict] = []
    
    def parse(self, state: Optional[SharedState] = None) -> List[dict]:
        """主解析入口 - 尝试多种策略"""
        # 策略1: 标准格式
        self._parse_standard_format()
        if self.artifacts:
            logger.info(f"[B-Parser] 标准格式解析成功: {len(self.artifacts)} 个文件")
            return self._post_process()
        
        # 策略2: Markdown代码块
        self._parse_markdown_blocks()
        if self.artifacts:
            logger.info(f"[B-Parser] Markdown格式解析成功: {len(self.artifacts)} 个文件")
            return self._post_process()
        
        # 策略3: JSON格式
        self._parse_json_format()
        if self.artifacts:
            logger.info(f"[B-Parser] JSON格式解析成功: {len(self.artifacts)} 个文件")
            return self._post_process()
        
        # 策略4: 逐行分析
        self._parse_line_by_line()
        if self.artifacts:
            logger.info(f"[B-Parser] 逐行分析成功: {len(self.artifacts)} 个文件")
            return self._post_process()
        
        # 策略5: LLM 归一化（仅在提供 state 时启用）
        if state is not None:
            normalized = self._preprocess_with_llm(state)
            if normalized:
                original_response = self.response
                self.response = normalized
                self._parse_standard_format()
                self.response = original_response
                if self.artifacts:
                    logger.info(f"[B-Parser] LLM归一化解析成功: {len(self.artifacts)} 个文件")
                    return self._post_process()
        
        logger.warning("[B-Parser] 所有解析策略失败，使用fallback")
        return []
    
    def _parse_standard_format(self) -> None:
        """策略1: 标准标记格式 === FILE: path === ... === END ===

        优先匹配完整格式；如果找不到 END 标记（常见截断场景），
        则退而求其次，按下一个 FILE 标记或文本末尾截断。
        """
        # 1) 严格匹配完整格式
        strict_pattern = r'={3,}\s*FILE\s*[:\s]\s*(.+?)\s*={3,}(.+?)={3,}\s*END\s*={3,}'
        matches = re.findall(strict_pattern, self.response, re.DOTALL)
        if matches:
            for file_path, content in matches:
                path = file_path.strip().strip(':')
                if self._is_valid_path(path):
                    self.artifacts.append({
                        "file_path": path,
                        "content": content.strip(),
                        "language": self._detect_language(path),
                        "source": "standard_format"
                    })
            return

        # 2) 截断兼容：只有 FILE 头，没有 END 或 END 被截断
        headers = list(re.finditer(
            r'={3,}\s*FILE\s*[:\s]\s*(.+?)\s*={3,}',
            self.response,
            re.DOTALL
        ))
        if not headers:
            return

        for i, header in enumerate(headers):
            path = header.group(1).strip().strip(':')
            if not self._is_valid_path(path):
                continue
            start = header.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(self.response)
            content = self.response[start:end].strip()
            # 去掉可能存在的部分 END 标记
            content = re.sub(r'={3,}\s*END\s*={3,}\s*$', '', content, flags=re.DOTALL).strip()
            self.artifacts.append({
                "file_path": path,
                "content": content,
                "language": self._detect_language(path),
                "source": "standard_format_truncated"
            })
    
    def _parse_markdown_blocks(self) -> None:
        """策略2: Markdown代码块，支持文件名注释"""
        # 匹配 ```python ... ``` 或 ``` ... ```
        pattern = r'```(?:\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, self.response, re.DOTALL)
        
        for i, content in enumerate(matches):
            # 尝试从第一行提取文件名注释
            file_path = self._extract_filename_from_content(content)
            
            if not file_path:
                # 根据内容推断文件名
                file_path = self._infer_filename_from_content(content, i)
            
            self.artifacts.append({
                "file_path": file_path,
                "content": content,
                "language": self._detect_language(file_path),
                "source": "markdown_block"
            })
    
    def _parse_json_format(self) -> None:
        """策略3: JSON格式 {"files": [{"path": "...", "content": "..."}]}"""
        try:
            # 尝试提取JSON对象
            json_match = re.search(r'\{[\s\S]*?"files"[\s\S]*?\}', self.response)
            if json_match:
                data = json.loads(json_match.group())
                if "files" in data:
                    for f in data["files"]:
                        self.artifacts.append({
                            "file_path": f.get("path", "unknown.py"),
                            "content": f.get("content", ""),
                            "language": self._detect_language(f.get("path", "")),
                            "source": "json_format"
                        })
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"[B-Parser] JSON解析失败: {e}")
    
    def _parse_line_by_line(self) -> None:
        """策略4: 逐行分析 - 最后的fallback"""
        lines = self.response.split('\n')
        current_file = None
        current_content = []
        
        for line in lines:
            # 检测文件路径行（如 "# main.py" 或 "File: main.py"）
            file_match = re.match(r'^[\s#\-*]*(?:File|文件|file)[:\s]+(\S+\.(?:py|md|txt|json|yaml|yml))', line, re.IGNORECASE)
            if file_match:
                if current_file and current_content:
                    self.artifacts.append({
                        "file_path": current_file,
                        "content": '\n'.join(current_content).strip(),
                        "language": self._detect_language(current_file),
                        "source": "line_by_line"
                    })
                current_file = file_match.group(1).strip()
                current_content = []
            elif current_file:
                current_content.append(line)
        
        # 处理最后一个文件
        if current_file and current_content:
            self.artifacts.append({
                "file_path": current_file,
                "content": '\n'.join(current_content).strip(),
                "language": self._detect_language(current_file),
                "source": "line_by_line"
            })
    
    def _preprocess_with_llm(self, state: SharedState) -> str:
        """策略5: 调用 LLM 将混乱输出归一化为标准格式。

        仅在前四种策略均失败且调用方提供了 state 时调用。
        返回归一化后的文本；如果 LLM 调用失败或返回空内容，返回空字符串。
        """
        try:
            llm = get_llm()
            system_prompt = """你是一台代码格式归一化工具。你的唯一任务是把用户提供的混合自然语言、伪代码或代码输出，转换成严格的标准文件格式。

标准格式示例：
=== FILE: main.py ===
def hello():
    print("hello")
=== END ===

=== FILE: README.md ===
# Hello
=== END ===

规则：
- 只输出标准格式块，不要解释、不要 Markdown 代码块、不要 JSON。
- 为每段代码推断一个合理的相对文件路径。
- 如果某段内容不是可运行代码（如说明文字），放入 README.md 或跳过。
- 保持代码内容完整，不要修改逻辑。"""

            user_prompt = f"请把以下内容归一化为标准格式（=== FILE: 路径 === ... === END ===）：\n\n{self.response}"
            response = llm.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0,
                max_tokens=2048,
            )
            content = response.get("content", "") if isinstance(response, dict) else response
            normalized = content.strip() if content else ""
            if normalized:
                logger.info("[B-Parser] LLM 归一化完成")
            else:
                logger.warning("[B-Parser] LLM 归一化返回空内容")
            return normalized
        except Exception as e:
            logger.warning(f"[B-Parser] LLM 归一化失败: {e}")
            return ""
    
    def _extract_filename_from_content(self, content: str) -> Optional[str]:
        """从代码内容第一行提取文件名注释"""
        lines = content.split('\n', 1)
        if not lines:
            return None
        
        first_line = lines[0].strip()
        
        # 匹配各种注释格式：# file: main.py, # main.py, etc.
        patterns = [
            r'^#\s*(?:file|path|filename)[:\s]+(\S+)$',
            r'^#\s*(\S+\.(?:py|md|txt|json|yaml|yml))$',
            r'^#\s*-\*\*\s*(\S+\.(?:py|md|txt|json|yaml|yml))\s*-\*\*$',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, first_line, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _infer_filename_from_content(self, content: str, index: int) -> str:
        """根据内容智能推断文件名"""
        content_lower = content.lower()
        
        # 检查文件类型特征
        if content.strip().startswith('#!/usr/bin/env python') or 'def ' in content or 'import ' in content:
            if 'def test_' in content or 'import pytest' in content or 'from unittest' in content:
                return f"tests/test_{index}.py"
            elif 'if __name__' in content and index == 0:
                return "main.py"
            return f"module_{index}.py"
        
        if content.strip().startswith('# ') or content.startswith('## '):
            if 'install' in content_lower or 'usage' in content_lower or 'requirements' in content_lower:
                return "README.md" if 'usage' in content_lower else f"doc_{index}.md"
            return f"doc_{index}.md"
        
        # 检查requirements格式
        if any(line.strip().startswith(f'{pkg}==') or line.strip().startswith(f'{pkg}>=') 
               for line in content.split('\n') 
               for pkg in ['pytest', 'requests', 'numpy', 'pandas', 'langgraph', 'langchain']):
            return "requirements.txt"
        
        return f"file_{index}.txt"
    
    def _is_valid_path(self, path: str) -> bool:
        """检查路径是否有效"""
        if not path or len(path) > 200:
            return False
        # 排除纯文本或代码片段被误认为路径
        invalid_patterns = ['import ', 'def ', 'class ', 'print(', 'return ', 'if ', 'for ', 'while ']
        if any(path.strip().startswith(p) for p in invalid_patterns):
            return False
        return True
    
    def _detect_language(self, file_path: str) -> str:
        """检测文件语言"""
        ext = os.path.splitext(file_path)[1].lower()
        mapping = {
            '.py': 'python',
            '.md': 'markdown',
            '.txt': 'text',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'config',
        }
        return mapping.get(ext, 'text')
    
    def _post_process(self) -> List[dict]:
        """后处理：去重、检查必需文件"""
        # 去重：按文件路径去重，保留最新
        seen = set()
        unique_artifacts = []
        for a in reversed(self.artifacts):
            path = a.get("file_path", "")
            if path not in seen:
                seen.add(path)
                unique_artifacts.insert(0, a)
        
        self.artifacts = unique_artifacts
        
        # 检查并添加缺失的必需文件
        self._ensure_required_files()
        
        return self.artifacts
    
    def _ensure_required_files(self) -> None:
        """确保包含必需文件"""
        existing_paths = {a.get("file_path", "") for a in self.artifacts}
        
        if "main.py" not in existing_paths:
            # 尝试从已有代码中提取主程序逻辑
            main_content = self._extract_main_logic()
            self.artifacts.append({
                "file_path": "main.py",
                "content": main_content,
                "language": "python",
                "source": "auto_generated"
            })
        
        if not any(a.get("file_path", "").startswith("tests/") for a in self.artifacts):
            self.artifacts.append({
                "file_path": "tests/test_main.py",
                "content": self._generate_default_test(),
                "language": "python",
                "source": "auto_generated"
            })
        
        if "README.md" not in existing_paths:
            self.artifacts.append({
                "file_path": "README.md",
                "content": "# Project\n\nAuto-generated project.\n",
                "language": "markdown",
                "source": "auto_generated"
            })
        
        if "requirements.txt" not in existing_paths:
            self.artifacts.append({
                "file_path": "requirements.txt",
                "content": self._generate_requirements(),
                "language": "text",
                "source": "auto_generated"
            })
    
    def _extract_main_logic(self) -> str:
        """从已有代码中提取主程序逻辑"""
        for a in self.artifacts:
            if a.get("language") == "python":
                content = a.get("content", "")
                # 如果已有 if __name__ == '__main__' 的代码，使用它
                if "if __name__" in content:
                    return content
        
        return "# Auto-generated main.py\ndef main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()\n"
    
    def _generate_default_test(self) -> str:
        """生成默认测试文件"""
        return """import pytest
from main import main

def test_main():
    assert main() is None

def test_main_runs_without_error():
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised {type(e).__name__}: {e}")
"""
    
    def _generate_requirements(self) -> str:
        """根据代码内容推断依赖"""
        deps = set(['pytest>=7.0.0'])
        
        for a in self.artifacts:
            if a.get("language") == "python":
                content = a.get("content", "")
                
                # 检测常见导入
                import_patterns = [
                    (r'import\s+pytest|from\s+pytest', 'pytest>=7.0.0'),
                    (r'import\s+numpy|from\s+numpy', 'numpy>=1.21.0'),
                    (r'import\s+pandas|from\s+pandas', 'pandas>=1.3.0'),
                    (r'import\s+requests|from\s+requests', 'requests>=2.25.0'),
                    (r'from\s+langgraph|import\s+langgraph', 'langgraph>=0.2.0'),
                    (r'from\s+langchain|import\s+langchain', 'langchain>=0.2.0'),
                    (r'import\s+flask|from\s+flask', 'Flask>=2.0.0'),
                    (r'import\s+fastapi|from\s+fastapi', 'fastapi>=0.68.0'),
                    (r'import\s+uvicorn', 'uvicorn>=0.15.0'),
                    (r'import\s+pydantic|from\s+pydantic', 'pydantic>=2.0.0'),
                    (r'import\s+asyncio', None),  # 标准库
                    (r'import\s+typing', None),  # 标准库
                    (r'import\s+re|import\s+json|import\s+os|import\s+sys', None),  # 标准库
                ]
                
                for pattern, dep in import_patterns:
                    if dep and re.search(pattern, content):
                        deps.add(dep)
        
        return '\n'.join(sorted(deps))


class CodeQualityChecker:
    """代码质量预检查器"""
    
    @staticmethod
    def check_syntax(code: str, file_path: str) -> dict:
        """检查Python语法"""
        if not file_path.endswith('.py'):
            return {"valid": True, "errors": []}
        
        try:
            ast.parse(code)
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [f"SyntaxError line {e.lineno}: {e.msg}"]
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Parse error: {str(e)}"]
            }
    
    @staticmethod
    def check_imports(code: str) -> dict:
        """检查导入是否可用"""
        import_pattern = re.compile(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', re.MULTILINE)
        imports = import_pattern.findall(code)
        
        # 标准库列表（常见）
        stdlib = {'os', 'sys', 're', 'json', 'time', 'datetime', 'random', 'typing', 
                  'collections', 'itertools', 'functools', 'math', 'statistics',
                  'pathlib', 'hashlib', 'base64', 'urllib', 'http', 'socket',
                  'threading', 'multiprocessing', 'asyncio', 'contextlib',
                  'inspect', 'textwrap', 'string', 'copy', 'pickle', 'csv',
                  'logging', 'argparse', 'unittest', 'pytest'}  # pytest非标准但沙箱预装
        
        issues = []
        for module, direct in imports:
            mod = module or direct
            root = mod.split('.')[0]
            if root not in stdlib:
                issues.append(f"External dependency: {mod}")
        
        return {"issues": issues, "external_count": len(issues)}
    
    @staticmethod
    def check_structure(code: str) -> dict:
        """检查代码结构质量"""
        has_docstring = '"""' in code or "'''" in code
        has_type_hints = any(': ' in line for line in code.split('\n') if 'def ' in line)
        has_tests = 'def test_' in code
        has_error_handling = 'try:' in code or 'except' in code
        
        score = 0
        score += 20 if has_docstring else 0
        score += 20 if has_type_hints else 0
        score += 20 if has_tests else 0
        score += 20 if has_error_handling else 0
        score += 20  # 基础分
        
        return {
            "score": score,
            "has_docstring": has_docstring,
            "has_type_hints": has_type_hints,
            "has_tests": has_tests,
            "has_error_handling": has_error_handling
        }

    @staticmethod
    def _scan_dangerous_subprocess(artifacts: List[dict]) -> List[dict]:
        """静态扫描生成的 Python 文件中可能危险的子进程/命令调用。

        检测 os.system、subprocess.call/run/Popen、exec、eval 与危险命令
        （如 rm -rf /、curl、wget、mkfs、dd、chmod 777、bash -c、sh -c、nc、ncat）
        同时出现的情况。返回包含 file_path、matched_pattern 等的警告列表。
        """
        shell_pattern = re.compile(
            r'\b(?:os\.system|subprocess\.(?:call|run|Popen)|exec|eval)\s*\('
        )
        dangerous_patterns = [
            (re.compile(r'\brm\s+-rf\s+/'), 'rm -rf /'),
            (re.compile(r'\bcurl\b'), 'curl'),
            (re.compile(r'\bwget\b'), 'wget'),
            (re.compile(r'\bmkfs\b'), 'mkfs'),
            (re.compile(r'\bdd\b'), 'dd'),
            (re.compile(r'\bchmod\s+777\b'), 'chmod 777'),
            (re.compile(r'\bbash\s+-c\b'), 'bash -c'),
            (re.compile(r'\bsh\s+-c\b'), 'sh -c'),
            (re.compile(r'\bnc\b'), 'nc'),
            (re.compile(r'\bncat\b'), 'ncat'),
        ]

        warnings: List[dict] = []
        for artifact in artifacts:
            if artifact.get('language') != 'python':
                continue
            file_path = artifact.get('file_path', 'unknown')
            for line_no, line in enumerate(artifact.get('content', '').splitlines(), start=1):
                if not shell_pattern.search(line):
                    continue
                for pattern, label in dangerous_patterns:
                    if pattern.search(line):
                        warnings.append({
                            'file_path': file_path,
                            'line': line_no,
                            'matched_pattern': label,
                            'line_content': line.strip(),
                        })
                        break
        return warnings


# ===== 主函数 =====

def developer_node(state: SharedState) -> SharedState:
    """开发节点 - P1优化版 + P0修复"""
    # 增加迭代计数
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    iteration = state.get("iteration_count", 0)
    logger.info(f"[B] 开始开发，迭代: {iteration}")
    
    state["current_phase"] = "development"
    state["status"] = "coding"
    
    requirements = state.get("requirements", "")
    project = state.get("project", {})
    project_name = project.get("name", "unknown") if project else "unknown"
    
    logger.info(f"[B] 项目: {project_name}")
    
    # 检测连续格式失败，升级到 HITL
    format_failures = state.get("format_failures", 0)
    if format_failures >= 2:
        logger.error(f"[B] 连续 {format_failures} 次格式解析失败，升级到 HITL")
        _record_b_hitl(state, f"代码生成格式错误（连续 {format_failures} 次）")
        return state

    # 检测是否有文件被截断且无法修复
    if _any_artifact_incomplete(state.get("code_artifacts", [])):
        logger.error("[B] 存在无法闭合的截断文件，升级到 HITL")
        _record_b_hitl(state, "文件生成被截断，自动续写失败")
        return state

    # 第一道检查：校验 D 阶段输出的 file_plan
    file_plan = _get_file_plan_from_state(state)
    if file_plan:
        plan_errors = _validate_file_plan(file_plan)
        if plan_errors:
            logger.error(f"[B] file_plan 校验失败: {plan_errors[:5]}")
            _record_b_hitl(state, f"file_plan 校验失败: {plan_errors[0]}")
            return state

    # 骨架阶段：先生成完整骨架，确保跨文件引用可解析
    if file_plan and not state.get("skeleton_complete", False):
        logger.info("[B] 进入骨架生成阶段")
        skeletons = None
        for attempt in range(1, 3):
            skeletons = _generate_code_with_llm_chunked(requirements, project_name, state, skeleton_only=True)
            if not skeletons:
                logger.warning(f"[B] 骨架生成返回空，重试 {attempt}/2")
                continue
            errors = _validate_skeleton_artifacts(skeletons, file_plan)
            if not errors:
                break
            logger.warning(f"[B] 骨架验证失败，重试 {attempt}/2: {errors[:3]}")
            # 把错误注入 state，让下一轮生成看到
            state["test_results"] = [
                {"file": e.split(":")[0], "passed": False, "error": e}
                for e in errors
            ]
        if skeletons and not errors:
            state["skeleton_artifacts"] = skeletons
            state["code_artifacts"] = skeletons
            state["skeleton_complete"] = True
            state["test_results"] = []
            logger.info("[B] 骨架 AST 校验通过")
            logger.info(f"[B] 骨架生成完成: {len(skeletons)} 个文件")
        elif skeletons:
            logger.error(f"[B] 骨架验证失败: {errors[:5]}")
            _record_b_hitl(state, f"骨架验证失败: {errors[0]}")
        else:
            logger.warning("[B] 骨架生成失败，降级为直接生成完整代码")
            state["skeleton_complete"] = True  # 跳过骨架阶段，避免卡住
        return state

    # 进入填充阶段：基于骨架生成完整实现
    logger.info("[B] 进入填充阶段")

    # 生成代码（P1修复：使用分片生成避免单条响应截断）
    try:
        artifacts = _generate_code_with_llm_chunked(requirements, project_name, state)

        # P0修复：判断是否为完整生成
        if _is_complete_generation(artifacts):
            # 完整生成：全量替换，避免旧代码污染
            state["code_artifacts"] = artifacts
            state["format_failures"] = 0
            logger.info(f"[B] 完整生成 {len(artifacts)} 个文件，全量替换")
        elif len(artifacts) > 0:
            # 部分生成：与旧代码合并（仅当旧代码存在时）
            existing = {a.get("file_path", ""): a for a in state.get("code_artifacts", [])}
            for a in artifacts:
                existing[a.get("file_path", "")] = a
            state["code_artifacts"] = list(existing.values())
            state["format_failures"] = state.get("format_failures", 0) + 1
            logger.warning(f"[B] 部分生成 {len(artifacts)} 个文件，与旧代码合并（格式失败计数: {state['format_failures']}）")
        else:
            # 零文件生成
            state["format_failures"] = state.get("format_failures", 0) + 1
            logger.warning(f"[B] 格式解析失败，计数: {state['format_failures']}")

    except Exception as e:
        logger.error(f"[B] LLM 生成失败: {e}")
        artifacts = _generate_code_stub(requirements, project_name)
        state["code_artifacts"] = artifacts
        state["format_failures"] = state.get("format_failures", 0) + 1

    logger.info(f"[B] 生成 {len(artifacts)} 个文件，总计 {len(state['code_artifacts'])} 个")

    # 规则引擎生成非核心文件（requirements.txt、README.md 等），不走 LLM
    project = state.get("project", {})
    stub_artifacts = _generate_stub_files(
        file_plan,
        project.get("name", "project"),
        project.get("tech_stack", []),
    )
    if stub_artifacts:
        existing = {a.get("file_path", ""): a for a in state.get("code_artifacts", [])}
        for a in stub_artifacts:
            existing[a["file_path"]] = a
        state["code_artifacts"] = list(existing.values())
        logger.info(f"[B] 规则引擎生成 {len(stub_artifacts)} 个非核心文件")

    # 第三道检查：import 干跑，不进 Docker 就能发现项目内部缺失的模块/符号
    import_errors = _dry_run_imports(state.get("code_artifacts", []), file_plan)
    state["import_errors"] = import_errors
    if import_errors:
        logger.error(f"[B] import 干跑发现 {len(import_errors)} 个问题，跳过 Docker 测试进入自修复")
        for err in import_errors[:5]:
            logger.error(f"  - {err}")
        # 用 import 错误构造测试失败结果，触发 _self_heal
        state["test_results"] = [
            {"file": err.split(":")[0], "passed": False, "error": err}
            for err in import_errors
        ]
        state = _self_heal(state, docker_available=False)
        return state

    logger.info("[B] import 干跑通过")

    # 静态安全扫描（只警告，不阻断测试）
    security_warnings = CodeQualityChecker._scan_dangerous_subprocess(state.get("code_artifacts", []))
    if security_warnings:
        for warning in security_warnings:
            logger.warning(
                f"[B-Security] 潜在危险调用: {warning['file_path']}:{warning.get('line', '?')} - {warning['matched_pattern']}"
            )
        state["security_warnings"] = security_warnings
    
    # 代码质量预检查
    _run_quality_checks(state)
    
    # 测试执行
    docker_available = check_docker_available()
    if docker_available:
        test_results = _run_tests_in_docker(state)
    else:
        test_results = _run_tests_mock(state)
    
    # 替换测试结果为当前迭代（避免无限累积）
    state["test_results"] = test_results
    
    # 自修复循环
    if not all(r.get("passed", False) for r in test_results):
        logger.warning("[B] 测试失败，尝试修复")
        state = _self_heal(state, docker_available)
    
    return state


def _generate_code_with_llm(requirements: str, project_name: str, state: SharedState) -> List[dict]:
    """调用 Kimi API 生成代码 - P1优化版 + P0修复 + RAG + MCP"""
    from src.mcp.bridge import create_mcp_client_for_state, ask_with_tools

    llm = get_llm()

    # RAG：检索相似代码和项目
    rag_context = ""
    try:
        rag = RAGRetriever()
        similar_code = rag.find_similar_code(requirements, top_k=3)
        if similar_code:
            rag_context = "\n\n相似历史代码参考（仅供参考，不要直接复制）：\n" + rag.format_context(similar_code)
            logger.info(f"[B] RAG 检索到 {len(similar_code)} 条相似代码")
    except Exception as e:
        logger.warning(f"[B] RAG 检索失败: {e}")

    # 收集上一轮的失败信息（如果有），帮助 LLM 避免重复错误
    error_context = ""
    prev_results = state.get("test_results", [])
    if prev_results and not all(r.get("passed", False) for r in prev_results):
        errors = []
        for r in prev_results:
            if not r.get("passed", False):
                err = r.get("error", "")[:500]
                out = r.get("output", "")[:500]
                errors.append(f"文件: {r.get('file', '')}\n错误: {err}\n输出: {out}")
        if errors:
            error_context = "\n\n上一轮测试失败信息（请修复这些问题）：\n" + "\n---\n".join(errors)

    # 收集格式失败提示
    format_failures = state.get("format_failures", 0)
    format_reminder = ""
    if format_failures > 0:
        format_reminder = f"\n\n⚠️ 警告：之前 {format_failures} 次生成因格式错误未能解析。请务必使用 === FILE: 路径 === ... === END === 格式，禁止使用 Markdown 代码块 ```python。"

    user_prompt = f"""请开发一个项目：{project_name}

需求：
{requirements}
{rag_context}

请生成完整的项目代码，包括 main.py、测试文件、README 和 requirements.txt。

重要：使用标准输出格式（=== FILE: 路径 === ... === END ===）。
如果代码块没有明确文件名，请在代码第一行添加注释：# file: 文件名
{error_context}
{format_reminder}
"""

    tool_calls = []
    tool_results = []
    try:
        with create_mcp_client_for_state(state) as mcp_client:
            response = ask_with_tools(
                llm=llm,
                mcp_client=mcp_client,
                system_prompt=DEVELOPER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=4096,
                max_rounds=2,
            )
            tool_calls = response.get("tool_calls", [])
            tool_results = response.get("tool_results", [])
    except Exception as e:
        logger.warning(f"[B] MCP 工具调用失败，回退普通 LLM: {e}")
        response = llm.chat(
            system_prompt=DEVELOPER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=1.0,
            max_tokens=8192
        )

    state["tool_calls"] = state.get("tool_calls", []) + tool_calls
    state["tool_results"] = state.get("tool_results", []) + tool_results

    # 使用多策略解析器
    content = response.get("content", "") if isinstance(response, dict) else response
    parser = CodeParser(content)
    artifacts = parser.parse(state)

    # P1诊断：解析失败时记录原始输出片段，便于排查模型格式问题
    if not artifacts and content:
        preview = content[:1200].replace("\n", " ")
        logger.warning(f"[B-Diag] 开发阶段解析失败，原始输出前1200字符: {preview}...")

    # 记录token使用
    state["llm_usage"] = state.get("llm_usage", []) + [{
        "phase": "development",
        "input_tokens": response.get("input_tokens", 0) if isinstance(response, dict) else 0,
        "output_tokens": response.get("output_tokens", 0) if isinstance(response, dict) else 0,
        "total_tokens": response.get("total_tokens", 0) if isinstance(response, dict) else 0,
        "cost_usd": response.get("cost_usd", 0.0) if isinstance(response, dict) else 0.0,
        "elapsed_ms": response.get("elapsed_ms", 0) if isinstance(response, dict) else 0
    }]

    # 记录解析统计
    state["logs"] = state.get("logs", []) + [{
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "level": "info",
        "message": f"[B] 代码解析: {len(artifacts)} 个文件"
    }]

    return artifacts


def _module_name_from_path(file_path: str) -> str:
    """从 Python 文件路径推断模块名，例如 calculator.py -> calculator"""
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    return name


def _extract_project_definitions(artifacts: List[dict]) -> Dict[str, Set[str]]:
    """提取所有非测试 Python 文件中定义的顶层函数/类名，key 为相对项目根的 dotted 模块名"""
    defs: Dict[str, Set[str]] = {}
    for a in artifacts:
        path = a.get("file_path", "")
        if not path.endswith(".py") or path.startswith("tests/"):
            continue
        content = a.get("content", "")
        # src/agents.py -> src.agents
        module = path[:-3].replace(os.sep, ".").replace("/", ".")
        if module.startswith("."):
            module = module[1:]
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        names: Set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
        defs[module] = names
    return defs


def _find_test_mismatches(artifacts: List[dict], defs: Dict[str, Set[str]]) -> List[dict]:
    """扫描测试文件，找出引用了但未在项目中定义的名称"""
    import builtins

    mismatches: List[dict] = []
    builtin_names = set(dir(builtins))
    test_globals = {"pytest", "unittest", "mock", "MagicMock", "patch", "call", "ANY", "sys", "io", "__file__", "__name__"}
    stdlib_modules = {
        "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib", "copy",
        "csv", "dataclasses", "datetime", "decimal", "enum", "functools", "hashlib", "inspect",
        "io", "itertools", "json", "logging", "math", "os", "pathlib", "pickle", "platform",
        "random", "re", "shutil", "socket", "statistics", "string", "subprocess", "sys",
        "tempfile", "textwrap", "threading", "time", "traceback", "typing", "unittest", "uuid",
        "warnings", "weakref", "xml",
    }

    for a in artifacts:
        path = a.get("file_path", "")
        if not path.startswith("tests/") or not path.endswith(".py"):
            continue
        content = a.get("content", "")
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        # 收集测试文件内的所有局部定义、导入、参数
        local_names: Set[str] = set()
        module_aliases: Dict[str, str] = {}  # alias -> module name

        def _is_submodule(module: str, attr: str) -> bool:
            prefix = f"{module}.{attr}"
            return any(k == prefix or k.startswith(prefix + ".") for k in defs)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    full_module = alias.name
                    base_module = full_module.split(".")[0]
                    local_name = alias.asname if alias.asname else base_module
                    local_names.add(local_name)
                    module_aliases[local_name] = base_module
                    # 忽略标准库/常见第三方库
                    if base_module in stdlib_modules:
                        continue
                    # 如果完整模块路径或子模块前缀已在项目中定义，无需报告
                    if full_module in defs or any(k.startswith(full_module + ".") for k in defs):
                        continue
                    # 仅当该模块名可能对应项目内某个定义时才报告（保守策略）
                    if any(base_module == m.split(".")[0] or base_module in m for m in defs):
                        mismatches.append({
                            "test_file": path,
                            "type": "module_missing",
                            "module": full_module,
                            "name": "",
                            "line": getattr(node, "lineno", 0),
                        })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                base_module = module.split(".")[0]
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    local_names.add(name)
                    if name == "*":
                        continue
                    # 忽略标准库/常见第三方库
                    if base_module in stdlib_modules:
                        continue
                    # 使用完整模块路径匹配
                    if module in defs:
                        if name not in defs[module]:
                            mismatches.append({
                                "test_file": path,
                                "type": "attribute",
                                "module": module,
                                "name": name,
                                "line": getattr(node, "lineno", 0),
                            })
                    elif any(k.startswith(module + ".") for k in defs):
                        # module 是项目内某个包的父包，子模块存在，不报告
                        continue
                    else:
                        # 仅当导入的名称确实在项目中定义时，才认为模块名缺失需要重命名
                        if any(name in d for d in defs.values()):
                            mismatches.append({
                                "test_file": path,
                                "type": "module_missing",
                                "module": module,
                                "name": name,
                                "line": getattr(node, "lineno", 0),
                            })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local_names.add(node.name)
                # 函数参数（包括 self/cls）
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for arg in node.args.args + node.args.kwonlyargs:
                        local_names.add(arg.arg)
                    if node.args.vararg:
                        local_names.add(node.args.vararg.arg)
                    if node.args.kwarg:
                        local_names.add(node.args.kwarg.arg)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for t in ast.walk(target):
                        if isinstance(t, ast.Name):
                            local_names.add(t.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    local_names.add(node.target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars:
                        for var in ast.walk(item.optional_vars):
                            if isinstance(var, ast.Name):
                                local_names.add(var.id)
            elif isinstance(node, ast.For):
                for t in ast.walk(node.target):
                    if isinstance(t, ast.Name):
                        local_names.add(t.id)

        # 遍历所有 Name 和 Attribute 节点
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if name in local_names:
                    continue
                if name in builtin_names or name in test_globals:
                    continue
                # 检查是否在所有项目定义中都不存在
                if not any(name in d for d in defs.values()):
                    mismatches.append({
                        "test_file": path,
                        "type": "name",
                        "name": name,
                        "line": getattr(node, "lineno", 0),
                    })
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                # 处理 module.attr 形式，例如 calc.add
                value = node.value
                if isinstance(value, ast.Name) and value.id in module_aliases:
                    module = module_aliases[value.id]
                    attr = node.attr
                    if module in defs and attr not in defs[module]:
                        # attr 可能是子模块（如 import src; src.agents.run）
                        if _is_submodule(module, attr):
                            continue
                        mismatches.append({
                            "test_file": path,
                            "type": "attribute",
                            "module": module,
                            "name": attr,
                            "line": getattr(node, "lineno", 0),
                        })
    return mismatches


def _check_cross_chunk_consistency(artifacts: List[dict]) -> List[dict]:
    """Q5：跨分片一致性检查入口"""
    if not any(a.get("file_path", "").endswith(".py") for a in artifacts):
        return []
    defs = _extract_project_definitions(artifacts)
    if not defs:
        return []
    mismatches = _find_test_mismatches(artifacts, defs)
    return mismatches


def _fix_consistency_mismatches(
    artifacts: List[dict],
    mismatches: List[dict],
    requirements: str,
    project_name: str,
) -> List[dict]:
    """Q5：对检测出的不一致，先尝试自动重命名模块，再调用 LLM 修复剩余问题"""
    if not mismatches:
        return artifacts

    artifact_map = {a["file_path"]: a for a in artifacts}

    # ---- 第一步：自动重命名缺失模块 ----
    # 建立 名称 -> 定义该名称的实现文件 的索引
    name_to_paths: Dict[str, List[str]] = {}
    for path, a in artifact_map.items():
        if path.startswith("tests/") or not path.endswith(".py"):
            continue
        content = a.get("content", "")
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name_to_paths.setdefault(node.name, []).append(path)

    renames: Dict[str, str] = {}
    for m in mismatches:
        if m.get("type") != "module_missing":
            continue
        expected_module = m.get("module", "")
        imported_name = m.get("name", "")
        if not expected_module:
            continue
        # 不对包（dotted module 或已有子模块的目录前缀）做扁平化重命名
        if "." in expected_module or any(
            p.startswith(f"{expected_module}/") for p in artifact_map
        ):
            continue
        # 已经存在期望模块文件，无需重命名
        if f"{expected_module}.py" in artifact_map or any(
            _module_name_from_path(p) == expected_module for p in artifact_map
        ):
            continue
        # 根据导入的名称寻找候选实现文件
        candidates = name_to_paths.get(imported_name, []) if imported_name else []
        if not candidates:
            # 未指定名称时，找与期望模块名最相似的实现文件
            candidates = [
                p for p in artifact_map
                if p.endswith(".py") and not p.startswith("tests/")
            ]
            candidates.sort(key=lambda p: _module_name_similarity(expected_module, _module_name_from_path(p)), reverse=True)
            candidates = candidates[:1]
        if not candidates:
            continue
        old_path = candidates[0]
        new_path = f"{expected_module}.py"
        if old_path == new_path:
            continue
        if old_path in renames:
            continue
        renames[old_path] = new_path
        logger.info(f"[B-Chunk] 自动重命名模块: {old_path} -> {new_path}")

    # 应用重命名：更新文件路径和内容中的 import
    def _apply_renames(content: str, renames: Dict[str, str]) -> str:
        import re
        for old_path, new_path in renames.items():
            old_mod = _module_name_from_path(old_path)
            new_mod = _module_name_from_path(new_path)
            # from old import x -> from new import x
            content = re.sub(
                rf"\bfrom\s+{re.escape(old_mod)}\b",
                f"from {new_mod}",
                content,
            )
            # import old -> import new
            content = re.sub(
                rf"\bimport\s+{re.escape(old_mod)}\b",
                f"import {new_mod}",
                content,
            )
            # import old as alias -> import new as alias
            content = re.sub(
                rf"\bimport\s+{re.escape(old_mod)}\s+as\b",
                f"import {new_mod} as",
                content,
            )
        return content

    new_artifacts = []
    for a in artifacts:
        path = a["file_path"]
        if path in renames:
            a["file_path"] = renames[path]
        a["content"] = _apply_renames(a.get("content", ""), renames)
        new_artifacts.append(a)
    artifacts = new_artifacts
    artifact_map = {a["file_path"]: a for a in artifacts}

    # 重命名后重新检查，剩余问题再走 LLM 修复
    remaining_mismatches = _check_cross_chunk_consistency(artifacts)
    if not remaining_mismatches:
        logger.info("[B-Chunk] 自动重命名后一致性检查通过")
        return artifacts

    # ---- 第二步：LLM 修复剩余不一致 ----
    llm = get_llm()
    related_files = set()
    for m in remaining_mismatches:
        related_files.add(m["test_file"])
        if m.get("module"):
            for path in artifact_map:
                if _module_name_from_path(path) == m["module"]:
                    related_files.add(path)

    file_blocks = []
    for path in sorted(related_files):
        content = artifact_map[path].get("content", "")
        file_blocks.append(f"=== FILE: {path} ===\n{content}\n=== END ===")

    mismatch_desc = "\n".join(
        f"- {m['test_file']}:{m.get('line', '?')} 引用了未定义的 {m.get('type', '')} `{m.get('name', '')}`"
        + (f" (模块 {m.get('module', '')})" if m.get("module") else "")
        for m in remaining_mismatches
    )

    sys_prompt = """你是一位资深 Python 开发工程师。以下代码存在跨文件引用不一致的问题。
请仅修改相关文件，使测试文件中的引用与实现文件中的定义保持一致。
不要改变功能，不要输出解释，只输出修复后的完整文件。

输出格式：
=== FILE: 路径 ===
文件内容
=== END ==="""

    user_prompt = f"""项目名：{project_name}
需求：{requirements}

不一致问题：
{mismatch_desc}

相关文件：
{chr(10).join(file_blocks)}

请修复上述不一致，保持原有功能。只输出修改后的文件。"""

    response = llm.chat(system_prompt=sys_prompt, user_prompt=user_prompt, temperature=0.2, max_tokens=4096)
    _record_llm_usage({"llm_usage": []}, response, "development")

    content = response.get("content", "") if isinstance(response, dict) else response
    if not content:
        logger.warning("[B-Chunk] 一致性修复 LLM 返回空内容，跳过修复")
        return artifacts

    parser = CodeParser(content)
    fixed = parser.parse({})
    if not fixed:
        logger.warning("[B-Chunk] 一致性修复解析失败，跳过修复")
        return artifacts

    for a in fixed:
        path = a.get("file_path", "")
        if path in artifact_map:
            artifact_map[path]["content"] = a.get("content", artifact_map[path]["content"])
            logger.info(f"[B-Chunk] 一致性修复更新: {path}")

    return list(artifact_map.values())


def _module_name_similarity(a: str, b: str) -> float:
    """简单模块名相似度，用于重命名候选选择"""
    if a == b:
        return 1.0
    a_set, b_set = set(a.lower()), set(b.lower())
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    return inter / max(len(a_set), len(b_set))


def _plan_project_files(requirements: str, project_name: str, state: SharedState) -> List[str]:
    """根据需求让 LLM 规划项目文件列表，返回文件路径列表。"""
    llm = get_llm()

    system_prompt = """你是一位资深软件架构师。请根据用户需求，输出该项目的最小文件清单。

输出必须是严格的 JSON，格式如下：
{
  "project_name": "简短英文项目名",
  "files": [
    {"path": "main.py", "description": "项目入口，实现核心功能"},
    {"path": "tests/test_main.py", "description": "pytest 测试"},
    {"path": "README.md", "description": "项目说明"},
    {"path": "requirements.txt", "description": "依赖列表"}
  ]
}

规则：
- 只输出 JSON，不要 Markdown 代码块，不要解释
- 文件路径必须是相对路径
- 必须包含可运行的入口文件、pytest 测试文件、README.md 和 requirements.txt
- 先列源代码文件，再列测试文件，最后列文档/依赖
- 如果需求简单，文件数量尽量精简；如果复杂，可适当拆分模块"""

    user_prompt = f"""项目名：{project_name}

需求：
{requirements}

请按上面的 JSON 格式输出项目文件清单。"""

    response = llm.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=2048,
    )
    _record_llm_usage(state, response, "planning")

    content = response.get("content", "") if isinstance(response, dict) else response
    if not content:
        logger.warning("[B-Plan] 项目规划返回空内容，使用默认文件列表")
        return ["main.py", "tests/test_main.py", "README.md", "requirements.txt"]

    # 尝试解析 JSON
    try:
        # 去掉可能的 Markdown 代码块包裹
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        plan = json.loads(cleaned)
        files = [item["path"] for item in plan.get("files", []) if "path" in item]
        if files:
            logger.info(f"[B-Plan] 规划文件列表: {files}")
            return files
    except Exception as e:
        logger.warning(f"[B-Plan] JSON 解析失败: {e}，尝试行解析")

    # 退而行解析：找所有看起来像文件路径的行
    fallback = []
    for line in content.splitlines():
        line = line.strip().strip('",')
        if "/" in line and (line.endswith(".py") or line.endswith(".md") or line.endswith(".txt") or line.endswith(".yaml") or line.endswith(".yml")):
            fallback.append(line)
    if fallback:
        logger.info(f"[B-Plan] 行解析文件列表: {fallback}")
        return fallback

    logger.warning("[B-Plan] 无法解析项目规划，使用默认文件列表")
    return ["main.py", "tests/test_main.py", "README.md", "requirements.txt"]


def _is_complete_generation(artifacts: List[dict]) -> bool:
    """判断本次生成是否为完整项目：至少包含一个源码文件、一个测试文件、README、requirements"""
    files = {a.get("file_path", "") for a in artifacts}
    has_source = any(
        f.endswith(".py") and not f.startswith("tests/") for f in files
    )
    has_test = any(
        f.startswith("tests/") and f.endswith(".py") for f in files
    )
    has_readme = "README.md" in files
    has_requirements = "requirements.txt" in files
    return (
        has_source
        and has_test
        and has_readme
        and has_requirements
        and len(artifacts) >= 4
    )


def _generate_full_file_artifact(
    llm,
    plan: dict,
    base_user: str,
    existing_artifacts: List[dict],
    state: SharedState,
    skeleton_only: bool = False,
) -> Optional[dict]:
    """回退/非拆分场景：用标准格式一次性生成整个文件。"""
    path = plan["path"]
    is_doc = bool(
        path.endswith(("README.md", "requirements.txt", ".md", ".txt", ".yaml", ".yml", ".json", ".toml"))
    )
    complexity = plan.get("complexity", "normal")
    max_tokens = _max_tokens_for_complexity(complexity, is_doc=is_doc)
    deps = set(plan.get("deps", []))

    elems = plan.get("required_elements", [])
    elem_lines = []
    for e in elems:
        if e.get("type") == "class":
            elem_lines.append(
                f"    - class {e.get('name', '')}: methods({', '.join(e.get('methods', []))})"
            )
        elif e.get("type") == "function":
            elem_lines.append(f"    - def {e.get('name', '')}()")
    elem_text = "\n".join(elem_lines) if elem_lines else "    - (无强制签名)"
    spec = (
        f"文件: {path}\n"
        f"  复杂度: {complexity}\n"
        f"  职责: {plan.get('role', '')}\n"
        f"  必须包含的签名:\n{elem_text}\n"
        f"  依赖: {', '.join(plan.get('deps', [])) or '无'}"
    )

    if skeleton_only:
        skeleton_rule = (
            "\n- 当前是骨架生成阶段：只输出 import 语句、函数/类签名、类型标注和 pass 占位，"
            "禁止写 docstring 或多行字符串，禁止写具体实现逻辑，确保文件可被 ast.parse 解析"
        )
    elif path.startswith("tests/"):
        skeleton_rule = (
            "\n- 这是测试文件：每个被测方法只写 1-2 个核心测试用例，不要穷举边界条件"
            "\n- 每个测试方法不超过 15 行，使用 fixtures 复用对象，不要重复创建"
            "\n- 测试代码必须是完整、可直接 pytest 运行的 Python 文件"
        )
    else:
        skeleton_rule = (
            "\n- 代码必须是完整、可运行的 Python 文件；每个 public 方法至少 15 行业务逻辑、带错误处理和日志记录"
        )

    sys_prompt = f"""你是一位资深 Python 开发工程师。请严格按以下规格生成文件，然后使用标准格式输出。

需要生成的文件规格：
{spec}

输出格式：
=== FILE: {path} ===
（文件内容）
=== END ===

规则：
- 只输出上面列出的文件，禁止输出其他文件
- 不要输出 Markdown 代码块 ```
- 不要输出解释、总结、规格说明或额外说明
{skeleton_rule}
- 严格按 file_plan 生成签名，不要引用 file_plan 中不存在的模块或类
- 测试文件必须 mock 外部依赖（LLM、文件系统、网络等），不能依赖真实 API"""

    code_summary = "\n\n".join(
        f"=== {a['file_path']} ===\n{a['content'][:2000]}"
        for a in existing_artifacts
        if a["file_path"] in deps
    )
    user_prompt = (
        base_user
        + f"\n\n当前步骤：生成文件：{path}\n\n"
        + (f"依赖文件摘要：\n{code_summary}\n\n" if code_summary else "")
        + "请输出上述文件，使用标准格式（=== FILE: 路径 === ... === END ===）。"
    )
    return _generate_single_file(
        llm, sys_prompt, user_prompt, path, state, max_tokens=max_tokens
    )


def _generate_one_file(
    llm,
    plan: dict,
    base_user: str,
    existing_artifacts: List[dict],
    state: SharedState,
    skeleton_only: bool = False,
) -> Optional[dict]:
    """生成单个文件，长文件自动拆分为多次 chunk 生成。"""
    path = plan["path"]
    is_doc = bool(
        path.endswith(("README.md", "requirements.txt", ".md", ".txt", ".yaml", ".yml", ".json", ".toml"))
    )
    complexity = plan.get("complexity", "normal")
    estimated = _estimate_file_tokens(plan)

    deps = set(plan.get("deps", []))
    dep_summaries = "\n\n".join(
        f"=== {a['file_path']} ===\n{a['content'][:2000]}"
        for a in existing_artifacts
        if a["file_path"] in deps
    )
    if not dep_summaries:
        dep_summaries = "（无直接依赖）"

    should_split = (
        not skeleton_only
        and not is_doc
        and estimated >= settings.split_threshold_tokens
        and plan.get("required_elements")
        and complexity != "simple"
    )

    if not should_split:
        artifact = _generate_full_file_artifact(
            llm, plan, base_user, existing_artifacts, state, skeleton_only=skeleton_only
        )
        if artifact:
            artifact["status"] = ArtifactStatus.COMPLETE.value
            artifact["truncated"] = False
            artifact["continuation_attempts"] = 0
        return artifact

    logger.info(f"[B] 文件 {path} 预估 {estimated} token，拆分为多次生成")
    chunk_size = 2 if path.startswith("tests/") else 3
    chunks = _split_required_elements(plan["required_elements"], max_per_chunk=chunk_size)
    generated_parts = []

    for i, chunk in enumerate(chunks):
        chunk_success = False
        for retry in range(2):
            sys_prompt = "你是一位资深 Python 开发工程师。请严格按要求只输出代码，不要解释。"
            user_prompt = _build_chunk_prompt(plan, chunk, generated_parts, dep_summaries)
            response = llm.chat(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                temperature=1.0,
                max_tokens=settings.chunk_max_tokens,
            )
            _record_llm_usage(state, response, "development")
            content = response.get("content", "") if isinstance(response, dict) else response
            content = _clean_continuation(content)

            if not _is_valid_python_snippet(content):
                logger.warning(
                    f"[B] chunk {i+1}/{len(chunks)} 语法校验失败，重试 {retry+1}/2"
                )
                continue

            if _has_stub_methods(content):
                logger.warning(
                    f"[B] chunk {i+1}/{len(chunks)} 存在空方法体，重试 {retry+1}/2"
                )
                continue

            generated_parts.append(content)
            logger.info(
                f"[B] chunk {i+1}/{len(chunks)} 完成且校验通过，{len(content)} 字符"
            )
            chunk_success = True
            break

        if not chunk_success:
            logger.error(
                f"[B] chunk {i+1}/{len(chunks)} 重试 2 次仍失败，回退到单文件生成"
            )
            artifact = _generate_full_file_artifact(
                llm, plan, base_user, existing_artifacts, state, skeleton_only=skeleton_only
            )
            if artifact:
                artifact["status"] = ArtifactStatus.COMPLETE.value
                artifact["truncated"] = False
                artifact["continuation_attempts"] = 0
            return artifact

    try:
        raw = _assemble_chunks(generated_parts, plan)
    except ChunkIncompleteError:
        logger.error("[B] 拼装校验失败，回退到单文件生成")
        artifact = _generate_full_file_artifact(
            llm, plan, base_user, existing_artifacts, state, skeleton_only=skeleton_only
        )
        if artifact:
            artifact["status"] = ArtifactStatus.COMPLETE.value
            artifact["truncated"] = False
            artifact["continuation_attempts"] = 0
        return artifact

    wrapped = f"=== FILE: {path} ===\n{raw}\n=== END ==="
    parser = CodeParser(wrapped)
    parsed = parser.parse(state)
    artifact = None
    if parsed:
        for a in parsed:
            if a.get("file_path") == path:
                artifact = a
                break
    if not artifact:
        artifact = {"file_path": path, "content": raw}

    artifact["status"] = ArtifactStatus.COMPLETE.value
    artifact["truncated"] = False
    artifact["continuation_attempts"] = 0
    return artifact


def _generate_code_with_llm_chunked(
    requirements: str,
    project_name: str,
    state: SharedState,
    skeleton_only: bool = False,
) -> List[dict]:
    """分片生成代码 - P3修复：先规划文件列表，再按文件列表分片生成。

    这样既能避免单条响应过长被截断，又能适配任意项目需求。
    当 ``skeleton_only=True`` 时，只生成文件骨架（签名 + docstring + pass）。
    """
    llm = get_llm()

    # RAG 上下文
    rag_context = ""
    try:
        rag = RAGRetriever()
        similar_code = rag.find_similar_code(requirements, top_k=2)
        if similar_code:
            rag_context = "\n\n相似历史代码参考（仅供参考，不要直接复制）：\n" + rag.format_context(similar_code)
            logger.info(f"[B-Chunk] RAG 检索到 {len(similar_code)} 条相似代码")
    except Exception as e:
        logger.warning(f"[B-Chunk] RAG 检索失败: {e}")

    # 上一轮错误上下文
    error_context = ""
    prev_results = state.get("test_results", [])
    if prev_results and not all(r.get("passed", False) for r in prev_results):
        errors = []
        for r in prev_results:
            if not r.get("passed", False):
                err = r.get("error", "")[:400]
                out = r.get("output", "")[:400]
                errors.append(f"文件: {r.get('file', '')}\n错误: {err}\n输出: {out}")
        if errors:
            error_context = "\n\n上一轮测试失败信息（请修复这些问题）：\n" + "\n---\n".join(errors)

    format_failures = state.get("format_failures", 0)
    format_reminder = ""
    if format_failures > 0:
        format_reminder = f"\n\n警告：之前 {format_failures} 次生成因格式错误未能解析。请务必使用 === FILE: 路径 === ... === END === 格式。"

    base_user = f"""项目名：{project_name}

需求：
{requirements}
{rag_context}
{error_context}
{format_reminder}""".strip()

    # 1) 使用 D 阶段输出的 file_plan；没有时再用 LLM 规划兜底
    file_plan = _get_file_plan_from_state(state)
    if not file_plan:
        planned_files = _plan_project_files(requirements, project_name, state)
        file_plan = [{"path": f, "role": "", "complexity": "normal", "deps": []} for f in planned_files]

    # 2) 按依赖拓扑排序并分组，实现阶段每 group 1 个复杂文件，避免截断
    # __init__.py 由规则引擎统一生成，不走 LLM
    ordered_plan = [p for p in _topological_sort_plans(file_plan) if p.get("path") != "src/__init__.py"]

    def _group_files(plans: List[dict], max_per_group: int = 1) -> List[List[dict]]:
        groups: List[List[dict]] = []
        current: List[dict] = []
        for p in plans:
            current.append(p)
            if len(current) >= max_per_group:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    file_groups = _group_files(ordered_plan, max_per_group=1)

    artifacts: List[dict] = []

    for group in file_groups:
        for plan in group:
            artifact = _generate_one_file(
                llm, plan, base_user, artifacts, state, skeleton_only=skeleton_only
            )
            if artifact:
                artifacts.append(artifact)
                logger.info(f"[B-Chunk] 生成 1 个文件: {artifact['file_path']}")
            else:
                logger.warning(f"[B-Chunk] 文件生成失败: {plan['path']}")

    # Q5：跨分片一致性检查
    consistency_mismatches = _check_cross_chunk_consistency(artifacts)
    if consistency_mismatches:
        logger.warning(
            f"[B-Chunk] 发现 {len(consistency_mismatches)} 处跨分片不一致，将尝试自动修复"
        )
        artifacts = _fix_consistency_mismatches(artifacts, consistency_mismatches, requirements, project_name)
    else:
        logger.info("[B-Chunk] 跨分片一致性检查通过")

    # 规则引擎自动生成 __init__.py，避免 LLM 导入不存在的符号
    project_dir = state.get("project", {}).get("path", f"projects/{project_name}")
    generated_files = {a["file_path"]: a["content"] for a in artifacts}
    init_content = _generate_init_py(generated_files, project_dir)
    if init_content:
        artifacts.append(
            {
                "file_path": "src/__init__.py",
                "content": init_content,
                "status": ArtifactStatus.COMPLETE.value,
                "truncated": False,
                "continuation_attempts": 0,
            }
        )

    return artifacts


def _close_open_blocks(content: str) -> str:
    """紧急闭合未完成的 class/def/if/for 块，使代码可解析"""
    open_count = 0
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("#"):
            open_count += 1
        elif stripped and not stripped.startswith("#") and not stripped.startswith(" ") and not stripped.startswith("\t"):
            # 顶层语句，减少未闭合计数
            open_count = max(0, open_count - 1)

    if open_count <= 0:
        return content

    logger.warning(f"[B-Chunk] 检测到 {open_count} 个未闭合 block，自动补全 ...")
    # 统一用 ... 占位闭合
    return content.rstrip() + "\n" + "\n".join(["    ..."] * open_count)


def _continue_truncated_content(
    llm,
    original: str,
    system_prompt: str,
    user_prompt: str,
    state: SharedState,
    max_tokens: int,
    max_attempts: int = 3,
) -> str:
    """截断续写：保留原始内容，让 LLM 从末尾继续"""
    merged = original
    for attempt in range(max_attempts):
        tail = merged[-4000:] if len(merged) > 4000 else merged
        continue_prompt = (
            "请继续完成以下代码。不要重复已有内容，从最后的地方接着写。\n\n"
            f"{tail}\n\n"
            "=== CONTINUE ==="
        )
        response = llm.chat(
            system_prompt=system_prompt,
            user_prompt=continue_prompt,
            temperature=1.0,
            max_tokens=max_tokens,
        )
        _record_llm_usage(state, response, "development")
        continuation = response.get("content", "") if isinstance(response, dict) else response
        if not continuation:
            break
        merged = _merge_without_overlap(merged, continuation)
        finish_reason = response.get("finish_reason", "")
        if finish_reason != "length":
            break
    return merged


def _merge_without_overlap(original: str, continuation: str) -> str:
    """安全合并截断内容和续写内容。

    处理三种截断场景：
    1. 截断在标识符中间：import jso + son... -> 识别前缀重叠，正确拼接
    2. 截断在字符串中间：logger.info("开始 + 处理") -> 先闭合引号再合并
    3. 截断在完整语句后：def foo():\n +     return 42 -> 在语义边界拼接
    """
    # Step 0: 清理续写内容（去掉 markdown 包装和解释文本）
    continuation = _clean_continuation(continuation)

    # Step 1: 处理未闭合的字符串
    original = _close_unclosed_string(original)

    # Step 2: 提取 original 末尾可能不完整的标识符
    safe_cut = len(original)
    for i in range(len(original) - 1, -1, -1):
        if not (original[i].isalnum() or original[i] == "_"):
            safe_cut = i + 1
            break

    dangling = original[safe_cut:]  # 不完整的标识符片段，如 "jso"
    safe_original = original[:safe_cut]  # 安全部分，如 "import "

    # Step 3: 如果 dangling 是标识符片段，在 continuation 中找完整形式
    if dangling:
        # 尝试前缀匹配：continuation 开头是否以 dangling 开头
        if continuation.startswith(dangling):
            continuation = continuation[len(dangling):]
        else:
            # 部分匹配：找 dangling 的后缀和 continuation 的前缀的最长重叠
            overlap_len = 0
            for i in range(1, len(dangling) + 1):
                suffix = dangling[-i:]
                if continuation.startswith(suffix):
                    overlap_len = i
            if overlap_len > 0:
                continuation = continuation[overlap_len:]

    # Step 4: 在语义边界拼接
    if safe_original.endswith("\n") or continuation.startswith("\n"):
        merged = safe_original + continuation
    else:
        merged = safe_original + "\n" + continuation

    return merged


def _close_unclosed_string(text: str) -> str:
    """检测并闭合未闭合的字符串，避免续写时把新内容拼进字符串内部。"""
    in_string = False
    quote_char = None
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char in ('"', "'") and not in_string:
            in_string = True
            quote_char = char
        elif char == quote_char and in_string:
            in_string = False

    if in_string and quote_char:
        return text + quote_char + "\n"
    return text


def _clean_continuation(text: str) -> str:
    """清理 LLM 续写输出中的非代码内容。"""
    # 去掉 markdown 代码块包装
    text = re.sub(r"^```\w*\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text)

    # 去掉开头的解释性文字
    lines = text.split("\n")
    skip_patterns = [
        r"^继续",
        r"^以下是",
        r"^Here",
        r"^Now ",
        r"^The ",
        r"^Continuing",
        r"^I will",
        r"^Let me",
    ]

    first_code_line = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(re.match(p, stripped) for p in skip_patterns):
            first_code_line = i + 1
        else:
            break

    return "\n".join(lines[first_code_line:])


def _find_semantic_boundary(text: str) -> int:
    """找到文本中最后一个语义边界位置（保留用于后续更精细的边界处理）。"""
    boundaries = ["\n\n", "\n", ":\n", " ", ""]
    for boundary in boundaries:
        if not boundary:
            return len(text)
        pos = text.rfind(boundary)
        if pos > 0:
            return pos + len(boundary)
    return len(text)


def _generate_single_file(llm, system_prompt: str, user_prompt: str, expected_path: str, state: SharedState, max_tokens: int = 4096) -> Optional[dict]:
    """调用 LLM 生成单个文件，支持截断续写"""
    response = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=1.0, max_tokens=max_tokens)
    _record_llm_usage(state, response, "development")

    content = response.get("content", "") if isinstance(response, dict) else response
    if not content:
        logger.warning(f"[B-Chunk] {expected_path} LLM 返回空内容")
        return None

    truncated = response.get("finish_reason", "") == "length"
    continuation_attempts = 0
    if truncated:
        logger.warning(f"[B-Chunk] {expected_path} 输出被截断，尝试续写")
        content = _continue_truncated_content(llm, content, system_prompt, user_prompt, state, max_tokens)
        continuation_attempts += 1

    parser = CodeParser(content)
    parsed = parser.parse(state)

    if not parsed:
        # 尝试紧急闭合后再解析
        closed = _close_open_blocks(content)
        parser = CodeParser(closed)
        parsed = parser.parse(state)

    if not parsed:
        preview = content[:800].replace("\n", " ")
        logger.warning(f"[B-Chunk] {expected_path} 解析失败，原始输出: {preview}...")
        return None

    # 优先找目标路径，否则取第一个非空 artifact
    for a in parsed:
        if a.get("file_path") == expected_path and a.get("content"):
            a["status"] = ArtifactStatus.TRUNCATED.value if truncated else ArtifactStatus.COMPLETE.value
            a["truncated"] = truncated
            a["continuation_attempts"] = continuation_attempts
            from src.core import metrics
            metrics.observe_b_truncation(a["status"])
            return a
    for a in parsed:
        if a.get("content"):
            a["file_path"] = expected_path
            a["status"] = ArtifactStatus.TRUNCATED.value if truncated else ArtifactStatus.COMPLETE.value
            a["truncated"] = truncated
            a["continuation_attempts"] = continuation_attempts
            from src.core import metrics
            metrics.observe_b_truncation(a["status"])
            return a
    return None


def _generate_multiple_files(llm, system_prompt: str, user_prompt: str, state: SharedState, expected: set, max_tokens: int = 2048) -> List[dict]:
    """调用 LLM 生成多个文件，支持截断续写"""
    response = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=1.0, max_tokens=max_tokens)
    _record_llm_usage(state, response, "development")

    content = response.get("content", "") if isinstance(response, dict) else response
    if not content:
        logger.warning("[B-Chunk] 文档/依赖 LLM 返回空内容")
        return []

    truncated = response.get("finish_reason", "") == "length"
    continuation_attempts = 0
    if truncated:
        logger.warning("[B-Chunk] 输出被截断，尝试续写")
        content = _continue_truncated_content(llm, content, system_prompt, user_prompt, state, max_tokens)
        continuation_attempts += 1

    parser = CodeParser(content)
    parsed = parser.parse(state)

    if not parsed:
        closed = _close_open_blocks(content)
        parser = CodeParser(closed)
        parsed = parser.parse(state)

    if not parsed:
        preview = content[:800].replace("\n", " ")
        logger.warning(f"[B-Chunk] 文档/依赖解析失败，原始输出: {preview}...")
        return []

    # 只保留期望文件，按路径去重
    result = []
    seen = set()
    from src.core import metrics
    for a in parsed:
        path = a.get("file_path", "")
        if path in expected and path not in seen and a.get("content"):
            a["status"] = ArtifactStatus.TRUNCATED.value if truncated else ArtifactStatus.COMPLETE.value
            a["truncated"] = truncated
            a["continuation_attempts"] = continuation_attempts
            metrics.observe_b_truncation(a["status"])
            result.append(a)
            seen.add(path)
    return result


def _record_llm_usage(state: SharedState, response: dict, phase: str) -> None:
    """记录 LLM token 使用"""
    state["llm_usage"] = state.get("llm_usage", []) + [{
        "phase": phase,
        "input_tokens": response.get("input_tokens", 0) if isinstance(response, dict) else 0,
        "output_tokens": response.get("output_tokens", 0) if isinstance(response, dict) else 0,
        "total_tokens": response.get("total_tokens", 0) if isinstance(response, dict) else 0,
        "cost_usd": response.get("cost_usd", 0.0) if isinstance(response, dict) else 0.0,
        "elapsed_ms": response.get("elapsed_ms", 0) if isinstance(response, dict) else 0,
    }]


def _run_quality_checks(state: SharedState) -> None:
    """运行代码质量预检查"""
    artifacts = state.get("code_artifacts", [])
    quality_results = []
    
    for artifact in artifacts:
        file_path = artifact.get("file_path", "")
        content = artifact.get("content", "")
        
        if not content:
            continue
        
        # 语法检查
        syntax_result = CodeQualityChecker.check_syntax(content, file_path)
        
        # 导入检查
        import_result = CodeQualityChecker.check_imports(content)
        
        # 结构检查
        structure_result = CodeQualityChecker.check_structure(content)
        
        quality_results.append({
            "file": file_path,
            "syntax_valid": syntax_result["valid"],
            "syntax_errors": syntax_result["errors"],
            "external_deps": import_result["issues"],
            "structure_score": structure_result["score"],
            "has_docstring": structure_result["has_docstring"],
            "has_type_hints": structure_result["has_type_hints"]
        })
    
    state["quality_results"] = quality_results
    
    # 记录检查结果
    invalid_files = [r for r in quality_results if not r["syntax_valid"]]
    if invalid_files:
        logger.warning(f"[B] 语法检查: {len(invalid_files)} 个文件有语法错误")
    else:
        logger.info("[B] 语法检查: 全部通过")


def _run_tests_in_docker(state: SharedState) -> List[dict]:
    """在 Docker 沙箱内运行测试 - P1优化版"""
    results = []

    # 允许通过环境变量调整沙箱行为（M9 大项目需要更多资源/联网安装依赖）
    network_disabled = os.getenv("AGENT_SANDBOX_NETWORK_DISABLED", "true").lower() in ("true", "1", "yes")
    timeout = int(os.getenv("AGENT_SANDBOX_TIMEOUT", "60"))
    memory_limit = os.getenv("AGENT_SANDBOX_MEMORY", "512m")

    try:
        with DockerSandbox(
            network_disabled=network_disabled,
            timeout=timeout,
            memory_limit=memory_limit,
        ) as sandbox:
            # 写入代码
            sandbox.write_files(state.get("code_artifacts", []))
            
            # 运行测试
            result = sandbox.run_tests()
            
            results.append({
                "file": "docker_tests",
                "passed": result.get("success", False),
                "duration_ms": result.get("duration_ms", 0),
                "output": result.get("stdout", ""),
                "error": result.get("stderr", ""),
                "sandbox": "docker",
                "exit_code": result.get("exit_code", 0)
            })
            
            if not result.get("success", False):
                combined = (result.get("stdout", "") + "\n" + result.get("stderr", "")).strip()
                logger.warning(f"[B] Docker 测试失败: {combined[:500]}")
    
    except Exception as e:
        logger.error(f"[B] Docker 沙箱错误: {e}")
        results.append({
            "file": "docker_tests",
            "passed": False,
            "duration_ms": 0,
            "output": "",
            "error": str(e),
            "sandbox": "docker",
            "exit_code": -1
        })
    
    return results


def _run_tests_mock(state: SharedState) -> List[dict]:
    """本地模拟测试（Docker 不可用时）"""
    results = []
    
    for artifact in state.get("code_artifacts", []):
        if artifact.get("file_path", "").startswith("tests/"):
            result = {
                "file": artifact["file_path"],
                "passed": True,
                "duration_ms": 100,
                "output": "1 passed (mock)",
                "error": "",
                "sandbox": "mock"
            }
            results.append(result)
    
    if not results:
        results.append({
            "file": "default",
            "passed": True,
            "duration_ms": 0,
            "output": "No tests found (mock)",
            "error": "",
            "sandbox": "mock"
        })
    
    return results


def _self_heal(state: SharedState, docker_available: bool, max_attempts: int = 2) -> SharedState:
    """自修复循环 - P2修复：改用分片生成，避免单条 8192 tokens 截断和 300s 超时。

    新增：记录错误签名，同一问题重复出现时升级到 HITL。
    """
    requirements = state.get("requirements", "")
    project = state.get("project", {})
    project_name = project.get("name", "unknown") if project else "unknown"

    # 先把当前失败信息记录到历史
    _record_repair_history(state, state.get("test_results", []))

    for attempt in range(1, max_attempts + 1):
        logger.info(f"[B] 自修复尝试 {attempt}/{max_attempts}")

        # 同一问题反复出现时，直接升级 HITL
        if _should_escalate_to_hitl(state, state.get("test_results", [])):
            _record_b_hitl(state, "同一测试错误反复出现，自动修复无效")
            return state

        # 调用 LLM 分片重新生成（error_context 会自动从 state['test_results'] 读取）
        try:
            fix_artifacts = _generate_code_with_llm_chunked(requirements, project_name, state)

            if _is_complete_generation(fix_artifacts):
                state["code_artifacts"] = fix_artifacts
                state["format_failures"] = 0
                logger.info(f"[B] 自修复完整生成 {len(fix_artifacts)} 个文件，全量替换")
            elif len(fix_artifacts) > 0:
                existing = {
                    a.get("file_path", ""): a
                    for a in state.get("code_artifacts", [])
                }
                for a in fix_artifacts:
                    existing[a.get("file_path", "")] = a
                state["code_artifacts"] = list(existing.values())
                state["format_failures"] = state.get("format_failures", 0) + 1
                logger.warning(
                    f"[B] 自修复部分生成 {len(fix_artifacts)} 个文件，合并到现有代码"
                )
            else:
                state["format_failures"] = state.get("format_failures", 0) + 1
                logger.warning(f"[B] 自修复未生成任何文件，格式失败计数 +1")
        except Exception as e:
            logger.error(f"[B] LLM 修复失败: {e}")

        # 重新运行质量检查
        _run_quality_checks(state)

        # 重新测试
        if docker_available:
            test_results = _run_tests_in_docker(state)
        else:
            test_results = _run_tests_mock(state)

        state["test_results"] = test_results
        _record_repair_history(state, test_results)

        if all(r.get("passed", False) for r in test_results):
            logger.info("[B] 自修复成功")
            return state

    logger.error("[B] 自修复失败，已达最大尝试次数，升级到 HITL")
    _record_b_hitl(state, "自修复达到最大次数仍无法通过测试")

    return state


def _generate_fix_with_llm(state: SharedState, error_info: List[dict], quality_issues: List[str]) -> List[dict]:
    """调用 LLM 修复代码 - P1优化版"""
    llm = get_llm()
    
    # 获取当前代码（最近5个文件）
    code_context = ""
    for artifact in state.get("code_artifacts", [])[-5:]:
        code_context += f"\n=== {artifact['file_path']} ===\n{artifact['content'][:1000]}\n"
    
    # 构建错误报告
    error_report = ""
    for err in error_info:
        error_report += f"\n文件: {err['file']}\n错误: {err['error']}\n输出: {err['output'][:500]}\n"
    
    if quality_issues:
        error_report += "\n语法错误:\n" + "\n".join(quality_issues)
    
    user_prompt = f"""请根据以下原始需求和错误信息修复代码。

原始需求：
{state.get('requirements', '')}

错误信息：
{error_report}

当前代码：
{code_context}

请生成修复后的代码，使用标准格式（=== FILE: 路径 === ... === END ===）。
注意：
1. 必须满足原始需求，不要生成与需求无关的代码
2. 只修复错误，不要改变原有功能
3. 确保所有导入的依赖都在requirements.txt中声明
4. 测试用例需要与实际代码匹配
"""
    
    response = llm.chat(
        system_prompt=DEVELOPER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=1.0,
        max_tokens=8192
    )
    
    content = response.get("content", "") if isinstance(response, dict) else response
    parser = CodeParser(content)
    artifacts = parser.parse(state)

    # P1诊断：修复阶段解析失败时记录原始输出片段
    if not artifacts and content:
        preview = content[:1200].replace("\n", " ")
        logger.warning(f"[B-Diag] 修复阶段解析失败，原始输出前1200字符: {preview}...")

    # 记录token使用（修复阶段）
    state["llm_usage"] = state.get("llm_usage", []) + [{
        "phase": "fix",
        "input_tokens": response.get("input_tokens", 0) if isinstance(response, dict) else 0,
        "output_tokens": response.get("output_tokens", 0) if isinstance(response, dict) else 0,
        "total_tokens": response.get("total_tokens", 0) if isinstance(response, dict) else 0,
        "cost_usd": response.get("cost_usd", 0.0) if isinstance(response, dict) else 0.0,
        "elapsed_ms": response.get("elapsed_ms", 0) if isinstance(response, dict) else 0
    }]
    
    return artifacts


def _generate_code_stub(requirements: str, project_name: str) -> List[dict]:
    """生成代码骨架（降级用）"""
    return [
        {
            "file_path": "main.py",
            "content": f"# {project_name}\n\ndef main():\n    print('Hello from {project_name}')\n\nif __name__ == '__main__':\n    main()\n",
            "language": "python",
            "source": "stub"
        },
        {
            "file_path": "tests/test_main.py",
            "content": "import pytest\nfrom main import main\n\ndef test_main():\n    assert main() is None\n",
            "language": "python",
            "source": "stub"
        },
        {
            "file_path": "README.md",
            "content": f"# {project_name}\n\n{requirements}\n\n## 安装\n\n```bash\npip install -r requirements.txt\n```\n\n## 使用\n\n```bash\npython main.py\n```\n",
            "language": "markdown",
            "source": "stub"
        },
        {
            "file_path": "requirements.txt",
            "content": "pytest>=7.0.0\n",
            "language": "text",
            "source": "stub"
        }
    ]


def save_project(
    state: SharedState,
    output_dir: str,
    project_dir_hint: Optional[str] = None,
) -> str:
    """保存项目到目录

    优先使用 ``project_dir_hint`` 作为目录名；否则根据 LLM 分析的项目名
    生成 ``{safe_name}-{suffix}`` 目录。
    """
    project = state.get("project", {})
    project_id = project.get("id", "project") if project else "project"

    if project_dir_hint:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", project_dir_hint).strip("_").lower()
        if not safe_name:
            safe_name = "project"
        project_dir = os.path.join(output_dir, safe_name)
    else:
        project_name = project.get("name", project_id) if project else project_id
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", project_name).strip("_").lower()
        if not safe_name:
            safe_name = "project"
        unique_suffix = project_id.split("-")[-1][-4:]
        project_dir = os.path.join(output_dir, f"{safe_name}-{unique_suffix}")

    os.makedirs(project_dir, exist_ok=True)

    for artifact in state.get("code_artifacts", []):
        file_path = os.path.join(project_dir, artifact.get("file_path", ""))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(artifact.get("content", ""))

    logger.info(f"[B] 项目已保存到: {project_dir}")
    return project_dir
