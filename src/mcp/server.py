"""MCP Server - 为 D/B/E 智能体提供工具能力

基于官方 MCP Python SDK 的 FastMCP 实现，工具通过内存流与 Client 通信。
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from src.core.logger import logger
from src.core.config import settings
from src.vector_store.rag_retriever import RAGRetriever


# 危险命令黑名单（用于本地 fallback 时的 shell 执行）
SHELL_BLACKLIST = [
    "rm -rf /",
    "mkfs",
    "dd if=/dev/zero",
    ":(){ :|:& };:",
    "> /dev/sda",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
]


def _is_dangerous_command(command: str) -> bool:
    """检查命令是否包含危险操作"""
    cmd_lower = command.lower()
    for dangerous in SHELL_BLACKLIST:
        if dangerous.lower() in cmd_lower:
            return True
    # 禁止直接写系统目录
    if re.search(r"\s*/(bin|sbin|usr|etc|var|lib|opt|System|Applications)/", command):
        return True
    return False


def _resolve_path(workspace: str, path: str) -> Path:
    """解析并校验路径，确保在工作空间内"""
    base = Path(workspace).resolve()
    target = (base / path).resolve()
    # 确保 target 在 base 之内
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ValueError(f"路径越界: {path}") from e
    return target


def _read_file_sync(workspace: str, path: str) -> str:
    target = _resolve_path(workspace, path)
    if not target.exists():
        return f"[错误] 文件不存在: {path}"
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[错误] 读取失败: {e}"


def _write_file_sync(workspace: str, path: str, content: str) -> str:
    target = _resolve_path(workspace, path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"[成功] 已写入 {path}"
    except Exception as e:
        return f"[错误] 写入失败: {e}"


def _list_dir_sync(workspace: str, path: str) -> str:
    target = _resolve_path(workspace, path)
    if not target.exists():
        return f"[错误] 目录不存在: {path}"
    if not target.is_dir():
        return f"[错误] 不是目录: {path}"
    try:
        items = []
        for item in sorted(target.iterdir()):
            item_type = "DIR" if item.is_dir() else "FILE"
            items.append(f"{item_type}: {item.name}")
        return "\n".join(items) if items else "(空目录)"
    except Exception as e:
        return f"[错误] 列出失败: {e}"


def _shell_exec_sync(workspace: str, command: str, timeout: int = 60) -> str:
    """执行 shell 命令，必须使用 Docker 沙箱；本地执行仅在显式开启时作为开发逃生口"""
    if _is_dangerous_command(command):
        return "[错误] 命令被安全策略拒绝"

    # 优先使用 Docker 沙箱
    try:
        from src.core.docker_sandbox import DockerSandbox, check_docker_available

        if check_docker_available():
            sandbox = DockerSandbox(workspace, timeout=timeout, network_disabled=True)
            result = sandbox.run_command(command)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            return_code = result.get("returncode", -1)
            return f"[退出码 {return_code}]\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    except Exception as e:
        logger.warning(f"[MCP] Docker 沙箱执行失败: {e}")

    # 生产环境禁止本地 fallback；只有显式开启时才允许（仅用于开发调试）
    allow_local = settings.mcp_shell_allow_local
    if not allow_local:
        return (
            "[错误] Docker 沙箱不可用，shell_exec 已拒绝执行。"
            "如需本地调试，请设置 MCP_SHELL_ALLOW_LOCAL=true（不推荐生产环境）。"
        )

    logger.warning("[MCP] MCP_SHELL_ALLOW_LOCAL=true，正在本地执行命令（生产环境请勿开启）")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (
            f"[本地执行 退出码 {result.returncode}]\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    except subprocess.TimeoutExpired:
        return f"[错误] 命令执行超时 (> {timeout}s)"
    except Exception as e:
        return f"[错误] 执行失败: {e}"


def _web_fetch_sync(url: str, max_length: int = 5000) -> str:
    """获取网页内容（只读 GET）"""
    try:
        import requests

        response = requests.get(url, timeout=20, headers={"User-Agent": "resume-agent-system/1.0"})
        response.raise_for_status()
        content = response.text
        if len(content) > max_length:
            content = content[:max_length] + "\n...[内容已截断]"
        return content
    except Exception as e:
        return f"[错误] 获取失败: {e}"


def _vector_search_sync(query: str, doc_type: Optional[str] = None, top_k: int = 3) -> str:
    """检索向量记忆"""
    supported = {"project", "code", "evaluation"}
    if doc_type is not None and doc_type not in supported:
        return f"[错误] 不支持的 doc_type: {doc_type}，请使用 {', '.join(supported)}"

    try:
        rag = RAGRetriever()
        if doc_type == "code":
            results = rag.find_similar_code(query, top_k=top_k)
        elif doc_type == "evaluation":
            results = rag.find_similar_evaluations(query, top_k=top_k)
        else:
            results = rag.find_similar_projects(query, top_k=top_k)
        return rag.format_context(results)
    except Exception as e:
        logger.warning(f"[MCP] 向量检索失败: {e}")
        return "[向量检索不可用]"


def create_mcp_server(workspace: str, name: str = "resume-agent-mcp") -> FastMCP:
    """创建一个绑定到指定工作空间的 MCP Server"""
    mcp = FastMCP(name)

    @mcp.tool()
    async def file_read(path: str) -> str:
        """读取工作空间内的文件内容

        Args:
            path: 相对工作空间的文件路径
        """
        return _read_file_sync(workspace, path)

    @mcp.tool()
    async def file_write(path: str, content: str) -> str:
        """在工作空间内写入文件

        Args:
            path: 相对工作空间的文件路径
            content: 文件内容
        """
        return _write_file_sync(workspace, path, content)

    @mcp.tool()
    async def file_list(path: str = ".") -> str:
        """列出工作空间内的目录内容

        Args:
            path: 相对工作空间的目录路径，默认为根目录
        """
        return _list_dir_sync(workspace, path)

    @mcp.tool()
    async def shell_exec(command: str, timeout: int = 60) -> str:
        """在工作空间内执行 shell 命令（受安全策略限制）

        Args:
            command: 要执行的命令
            timeout: 超时时间（秒）
        """
        return _shell_exec_sync(workspace, command, timeout)

    @mcp.tool()
    async def web_fetch(url: str) -> str:
        """获取指定 URL 的网页内容（只读）

        Args:
            url: 目标网页地址
        """
        return _web_fetch_sync(url)

    @mcp.tool()
    async def vector_search(query: str, doc_type: str = "project") -> str:
        """检索历史任务向量记忆

        Args:
            query: 检索查询
            doc_type: 文档类型（project/code/evaluation），默认 project
        """
        return _vector_search_sync(query, doc_type=doc_type)

    return mcp
