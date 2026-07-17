"""编排器分析节点"""

import json
import os
import re
from typing import TYPE_CHECKING

from src.core.shared_state import SharedState
from src.core.logger import logger
from src.core.config import settings
from src.core.orchestrator.constants import PENDING_REVIEW_FILE, HUMAN_RESPONSE_FILE

if TYPE_CHECKING:
    from src.core.orchestrator.core import Orchestrator


# 分析节点的系统提示词
_ANALYZE_SYSTEM_PROMPT = """你是一位资深技术顾问。你的任务是根据用户需求，推断最合适的项目架构。

输出格式（必须严格JSON）：
{
  "name": "项目英文名称（简短，如 'WebCrawler', 'DataPipeline', 'ChatBotAPI'）",
  "tech_stack": ["框架1", "框架2", "数据库", "测试工具"],
  "milestones": [
    "setup - 环境搭建与依赖安装",
    "core - 核心功能实现",
    "eval - 测试与评估",
    "polish - 文档与优化"
  ],
  "file_plan": [
    {
      "path": "src/models.py",
      "role": "数据模型",
      "complexity": "normal",
      "deps": [],
      "required_elements": [
        {"type": "class", "name": "ReviewResult", "methods": ["to_dict", "from_dict"]},
        {"type": "class", "name": "ArticleState", "methods": ["can_revise"]}
      ]
    },
    {
      "path": "src/agents.py",
      "role": "智能体逻辑",
      "complexity": "complex",
      "deps": ["src/models.py"],
      "required_elements": [
        {"type": "class", "name": "ResearcherAgent", "methods": ["run", "_research"]}
      ]
    }
  ]
}

规则：
1. 项目名应简洁、专业，使用英文
2. 技术栈应基于需求推断（如爬虫用 requests + bs4，Web用 FastAPI/Flask）
3. 里程碑应反映项目特点，至少4个
4. file_plan 必须覆盖所有核心文件、测试文件、README、requirements.txt
5. file_plan 中的 path 必须是相对于项目根目录的路径，**严禁**以 "projects/" 或项目目录名开头
6. complexity 只能是 simple / normal / complex 之一
7. deps 列出该文件依赖的其他 file_plan 路径
8. required_elements 必须具体：每个 complex/normal 文件列出必须包含的 class/function，以及每个方法至少 15 行业务逻辑、错误处理、日志记录的要求
9. 如果需求里提到 LangGraph / 多智能体，必须包含 src/state.py 或 src/models.py 来承载状态定义
10. file_plan 文件总数不得超过 10 个；禁止出现以下文件：pyproject.toml、.env.example、conftest.py、src/__init__.py、src/config.py、以及任何 nodes/ 目录下的拆分文件
11. 每个文件不得超过 200 行；每个类不得超过 3 个方法；不要把三个 agent 类塞进同一个文件
12. LangGraph 工作流类（如 ArticleGraph）必须放在 src/graph.py
13. 入口函数/公共 API（如 create_article、main）必须放在 src/main.py 或 src/graph.py，不要放在 src/__init__.py
14. 非核心文件（requirements.txt、README.md）由规则引擎生成，不要在 file_plan 里写复杂实现要求
15. 如果有相似历史案例，可参考其技术栈和里程碑，但不要直接复制
16. 如需查询技术文档或历史案例，可调用可用工具
"""


def analyze_node(self, state: SharedState):
    """分析节点 - 解析需求，创建项目定义（使用LLM动态分析 + RAG）"""
    requirements = state.get("requirements", "")
    logger.info(f"[D] 分析需求: {requirements[:100]}...")

    # RAG：检索相似历史项目
    rag_context = ""
    try:
        similar_projects = self.rag.find_similar_projects(requirements, top_k=3)
        if similar_projects:
            rag_context = self.rag.format_context(similar_projects)
            logger.info(f"[D] 检索到 {len(similar_projects)} 个相似历史项目")
    except Exception as e:
        logger.warning(f"[D] RAG 检索失败: {e}")

    # 调用 LLM 动态分析需求（支持 MCP 工具）
    tool_calls = []
    tool_results = []
    try:
        project_info, tool_calls, tool_results = self._analyze_with_llm(state, requirements, rag_context)
    except Exception as e:
        logger.error(f"[D] LLM分析失败: {e}，使用降级方案")
        project_info = self._fallback_analysis(requirements)

    file_plan = project_info.get("file_plan") or _build_default_file_plan(project_info, requirements)
    file_plan = _normalize_file_plan_paths(file_plan)

    state["project"] = {
        "id": f"proj-{state['session_id'][:8]}",
        "name": project_info["name"],
        "description": requirements,
        "tech_stack": project_info["tech_stack"],
        "milestones": project_info["milestones"],
        "file_plan": file_plan,
    }
    state["tool_calls"] = state.get("tool_calls", []) + tool_calls
    state["tool_results"] = state.get("tool_results", []) + tool_results

    state["current_phase"] = "analysis"
    state["status"] = "analyzing"

    self._save_state(state, "analysis")
    return state


def analyze_with_llm(
    self,
    state: SharedState,
    requirements: str,
    rag_context: str = "",
) -> tuple:
    """调用 LLM 分析需求，推断项目结构（支持 MCP 工具）

    返回: (project_info, tool_calls, tool_results)
    """
    from src.core.llm import get_llm
    from src.mcp.bridge import create_mcp_client_for_state, ask_with_tools

    llm = get_llm()

    user_prompt = f"""请分析以下需求，给出项目架构建议：

需求：
{requirements[:2000]}

{rag_context}

请只输出JSON格式。"""

    tool_calls = []
    tool_results = []
    try:
        with create_mcp_client_for_state(state) as mcp_client:
            response = ask_with_tools(
                llm=llm,
                mcp_client=mcp_client,
                system_prompt=_ANALYZE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=1.0,
                max_tokens=4096,
                max_rounds=2,
            )
            tool_calls = response.get("tool_calls", [])
            tool_results = response.get("tool_results", [])
    except Exception as e:
        logger.warning(f"[D] MCP 工具调用失败，回退普通 LLM: {e}")
        response = llm.chat(
            system_prompt=_ANALYZE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=1.0,
            max_tokens=4096,
        )

    content = response.get("content", "") if isinstance(response, dict) else response

    # 尝试解析 JSON
    json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', content)
    if json_match:
        data = json.loads(json_match.group(1))
    else:
        data = json.loads(content.strip())

    required = ["name", "tech_stack", "milestones"]
    for field in required:
        if field not in data:
            raise ValueError(f"缺少字段: {field}")

    logger.info(f"[D] LLM分析结果: {data['name']}, 技术栈: {data['tech_stack']}")
    return data, tool_calls, tool_results


def _normalize_file_plan_paths(file_plan: list) -> list:
    """确保 file_plan 中的 path 是相对于项目根目录的干净路径，并移除非 LLM 生成的 src/__init__.py。"""
    normalized = []
    init_elements = []

    for p in file_plan:
        path = p.get("path", "")
        # 去掉可能的 projects/xxx/ 前缀
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 3 and parts[0] == "projects":
            # projects/<project-name>/... -> ...
            path = "/".join(parts[2:])
        elif len(parts) >= 2 and parts[0] == "projects":
            path = "/".join(parts[1:])
        # 再处理项目目录名单独作为前缀的情况
        if len(parts) >= 2 and parts[-2].startswith(("01-", "test-", "calc")):
            # 如果以项目目录名开头，也去掉
            pass
        p["path"] = path
        p["deps"] = [_normalize_single_path(d) for d in p.get("deps", [])]

        if path == "src/__init__.py":
            # __init__.py 由规则引擎生成，其中的函数定义迁移到 src/main.py
            init_elements.extend(p.get("required_elements", []))
            continue

        normalized.append(p)

    if init_elements:
        main_plan = next((p for p in normalized if p.get("path") == "src/main.py"), None)
        if main_plan:
            existing = main_plan.get("required_elements", [])
            existing_names = {e.get("name") for e in existing}
            for e in init_elements:
                if e.get("name") not in existing_names:
                    existing.append(e)
            main_plan["required_elements"] = existing
        else:
            normalized.append(
                {
                    "path": "src/main.py",
                    "role": "程序入口与公共 API",
                    "complexity": "normal",
                    "deps": ["src/graph.py", "src/agents.py", "src/state.py"],
                    "required_elements": init_elements,
                }
            )

    return normalized


def _normalize_single_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "projects":
        return "/".join(parts[2:])
    if len(parts) >= 2 and parts[0] == "projects":
        return "/".join(parts[1:])
    return path


def _infer_required_elements(path: str) -> list:
    """根据文件路径推断默认 required_elements"""
    basename = os.path.basename(path)
    name, _ = os.path.splitext(basename)
    if path.endswith("models.py") or name == "state":
        return [
            {"type": "class", "name": "ArticleState", "methods": ["to_dict", "can_revise"]},
            {"type": "class", "name": "ReviewResult", "methods": ["to_dict", "from_dict"]},
        ]
    if "researcher" in path:
        return [{"type": "class", "name": "ResearcherNode", "methods": ["run", "research"]}]
    if "writer" in path:
        return [{"type": "class", "name": "WriterNode", "methods": ["run", "write", "revise"]}]
    if "reviewer" in path:
        return [{"type": "class", "name": "ReviewerNode", "methods": ["run", "score", "can_pass"]}]
    if "graph" in path:
        return [{"type": "function", "name": "create_graph"}, {"type": "function", "name": "run_workflow"}]
    if "main" in path:
        return [{"type": "function", "name": "main"}]
    if basename.startswith("test_") and name != "tests":
        return [{"type": "function", "name": f"test_{name.replace('test_', '')}"}]
    return []


def _build_default_file_plan(project_info: dict, requirements: str) -> list:
    """当 LLM 未返回 file_plan 时，基于规则生成默认文件清单"""
    req_lower = requirements.lower()
    tech = [t.lower() for t in project_info.get("tech_stack", [])]
    has_agents = any(kw in req_lower for kw in ["agent", "langgraph", "multi-agent", "智能体"])
    has_web = any(kw in req_lower for kw in ["web", "api", "flask", "fastapi", "接口", "网站"])
    has_cli = any(kw in req_lower for kw in ["cli", "命令行", "命令", "calculator"])

    plan = [
        {"path": "src/__init__.py", "role": "包初始化", "complexity": "simple", "deps": [], "required_elements": []},
        {"path": "src/main.py", "role": "程序入口", "complexity": "normal", "deps": [], "required_elements": _infer_required_elements("src/main.py")},
        {"path": "README.md", "role": "项目文档", "complexity": "simple", "deps": [], "required_elements": []},
        {"path": "requirements.txt", "role": "依赖声明", "complexity": "simple", "deps": [], "required_elements": []},
        {"path": "tests/test_main.py", "role": "主流程测试", "complexity": "normal", "deps": ["src/main.py"], "required_elements": _infer_required_elements("tests/test_main.py")},
    ]

    if has_agents or "langgraph" in tech:
        plan.insert(0, {"path": "src/models.py", "role": "数据模型", "complexity": "normal", "deps": [], "required_elements": _infer_required_elements("src/models.py")})
        plan.insert(1, {"path": "src/agents.py", "role": "智能体逻辑", "complexity": "complex", "deps": ["src/models.py"], "required_elements": _infer_required_elements("src/agents.py")})
        plan.insert(2, {"path": "src/graph.py", "role": "LangGraph 工作流", "complexity": "complex", "deps": ["src/models.py", "src/agents.py"], "required_elements": _infer_required_elements("src/graph.py")})
        plan.append({"path": "tests/test_agents.py", "role": "智能体测试", "complexity": "normal", "deps": ["src/agents.py"], "required_elements": _infer_required_elements("tests/test_agents.py")})
        plan.append({"path": "tests/test_graph.py", "role": "工作流测试", "complexity": "normal", "deps": ["src/graph.py"], "required_elements": _infer_required_elements("tests/test_graph.py")})
    elif has_web:
        plan.insert(0, {"path": "src/models.py", "role": "数据模型", "complexity": "normal", "deps": [], "required_elements": _infer_required_elements("src/models.py")})
        plan.insert(1, {"path": "src/routers.py", "role": "路由处理", "complexity": "normal", "deps": ["src/models.py"], "required_elements": _infer_required_elements("src/routers.py")})
        plan.append({"path": "tests/test_api.py", "role": "API 测试", "complexity": "normal", "deps": ["src/routers.py"], "required_elements": _infer_required_elements("tests/test_api.py")})
    elif has_cli:
        plan.insert(0, {"path": "src/cli.py", "role": "命令行交互", "complexity": "normal", "deps": [], "required_elements": _infer_required_elements("src/cli.py")})
        plan.insert(1, {"path": "src/core.py", "role": "核心逻辑", "complexity": "normal", "deps": [], "required_elements": _infer_required_elements("src/core.py")})
        plan.append({"path": "tests/test_cli.py", "role": "CLI 测试", "complexity": "normal", "deps": ["src/cli.py"], "required_elements": _infer_required_elements("tests/test_cli.py")})
        plan.append({"path": "tests/test_core.py", "role": "核心逻辑测试", "complexity": "normal", "deps": ["src/core.py"], "required_elements": _infer_required_elements("tests/test_core.py")})

    return plan


def fallback_analysis(requirements: str) -> dict:
    """LLM分析失败时的降级方案 - 关键词匹配"""
    req_lower = requirements.lower()

    if any(kw in req_lower for kw in ["爬虫", "crawl", "spider", "抓取"]):
        info = {
            "name": "WebCrawler",
            "tech_stack": ["requests", "beautifulsoup4", "pytest", "pyyaml"],
            "milestones": [
                "setup - 环境搭建与目标网站分析",
                "core - 爬虫核心逻辑实现",
                "eval - 反爬测试与稳定性评估",
                "polish - 文档与使用示例",
            ],
        }

    elif any(kw in req_lower for kw in ["网站", "web", "api", "接口", "flask", "fastapi"]):
        info = {
            "name": "WebAPIService",
            "tech_stack": ["fastapi", "uvicorn", "pydantic", "pytest"],
            "milestones": [
                "setup - 环境搭建与API设计",
                "core - 路由与业务逻辑实现",
                "eval - API测试与性能评估",
                "polish - 文档与部署指南",
            ],
        }

    elif any(kw in req_lower for kw in ["数据", "分析", "pandas", "可视化"]):
        info = {
            "name": "DataAnalysisTool",
            "tech_stack": ["pandas", "matplotlib", "numpy", "pytest"],
            "milestones": [
                "setup - 环境搭建与数据集准备",
                "core - 数据处理与可视化实现",
                "eval - 准确性测试与性能评估",
                "polish - 报告生成与文档",
            ],
        }

    elif any(kw in req_lower for kw in ["agent", "langgraph", "multi-agent", "智能体"]):
        info = {
            "name": "MultiAgentSystem",
            "tech_stack": ["LangGraph", "LangChain", "Kimi API", "pytest"],
            "milestones": [
                "setup - 环境搭建与架构设计",
                "core - 智能体协作逻辑实现",
                "eval - 集成测试与性能评估",
                "polish - 文档与使用示例",
            ],
        }

    else:
        info = {
            "name": "PythonProject",
            "tech_stack": ["pytest", "pyyaml"],
            "milestones": [
                "setup - 环境搭建",
                "core - 核心功能实现",
                "eval - 测试与评估",
                "polish - 文档与优化",
            ],
        }

    info["file_plan"] = _build_default_file_plan(info, requirements)
    return info
