"""MCP Client - 通过 stdio 连接本地 MCP Server

对外暴露同步接口，内部在后台线程维持一个持久 event loop，
确保 session 和 stdio_client 在同一线程/loop 中创建和销毁。
"""

import os
import sys
import json
import asyncio
import threading
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.core.logger import logger
from src.core.config import settings


class MCPClient:
    """MCP 工具客户端

    使用示例：
        with MCPClient(workspace="projects/xxx") as client:
            tools = client.list_tools()
            result = client.call_tool("file_read", {"path": "main.py"})
    """

    def __init__(
        self,
        workspace: str,
        server_module: str = "src.mcp.run_server",
        enabled: Optional[bool] = None,
    ):
        self.workspace = workspace
        self.server_module = server_module
        self.enabled = enabled if enabled is not None else settings.mcp_enabled
        self.session: Optional[ClientSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stdio_cm = None

    def _start(self):
        """同步启动：在后台线程创建并运行 event loop"""
        if not self.enabled:
            logger.info("[MCP] 已禁用")
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        try:
            self._run_coro(self._setup()).result(timeout=30)
            logger.info("[MCP] Client 连接成功")
        except Exception as e:
            logger.warning(f"[MCP] 连接失败，将禁用工具: {e}")
            self.enabled = False
            self._stop_loop()

    def _run_loop(self):
        """后台线程运行的 event loop"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coro(self, coro):
        """在后台 loop 中调度协程"""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _setup(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", self.server_module, "--workspace", self.workspace],
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )
        self._stdio_cm = stdio_client(params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        await self.session.initialize()

    def _stop_loop(self):
        """停止后台 event loop"""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None

    def _close(self):
        """同步关闭"""
        if self._loop is None or self.session is None:
            self._stop_loop()
            return

        async def _teardown():
            try:
                await self.session.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass

        try:
            self._run_coro(_teardown()).result(timeout=10)
        except Exception as e:
            logger.warning(f"[MCP] 关闭 session 时出错: {e}")
        finally:
            self._stop_loop()
            self.session = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        if not self.enabled or self.session is None or self._loop is None:
            return []

        async def _list():
            response = await self.session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
                for tool in response.tools
            ]

        try:
            return self._run_coro(_list()).result(timeout=30)
        except Exception as e:
            logger.warning(f"[MCP] 列出工具失败: {e}")
            return []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """调用指定工具"""
        if not self.enabled or self.session is None or self._loop is None:
            return "[MCP 未启用]"

        async def _call():
            result = await self.session.call_tool(name, arguments=arguments)
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts)

        try:
            return self._run_coro(_call()).result(timeout=60)
        except Exception as e:
            logger.warning(f"[MCP] 调用工具 {name} 失败: {e}")
            return f"[MCP 工具调用失败: {e}]"

    def format_tools_prompt(self) -> str:
        """把工具列表格式化为 LLM prompt 片段"""
        tools = self.list_tools()
        if not tools:
            return ""

        lines = ["## 可用工具", "你可以通过以下 JSON 格式调用工具：", ""]
        lines.append('{"tool": "工具名", "arguments": {"参数名": "值"}}')
        lines.append("")
        lines.append("可用工具列表：")
        for tool in tools:
            lines.append(f"- {tool['name']}: {tool['description']}")
            lines.append(f"  参数: {json.dumps(tool['parameters'], ensure_ascii=False)}")
        lines.append("")
        lines.append("如果你需要调用工具，请在回复中单独输出一个 JSON 代码块，不要添加其他说明。")
        return "\n".join(lines)
