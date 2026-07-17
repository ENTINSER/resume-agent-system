"""MCP Client 集成测试

通过 stdio 启动 MCP Server 进程，验证 Client 能连接并调用工具。
"""

import os
import tempfile

import pytest

from src.mcp.client import MCPClient


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestMCPClient:
    def test_client_lists_tools(self, workspace):
        with MCPClient(workspace=workspace) as client:
            tools = client.list_tools()
            names = [t["name"] for t in tools]
            assert "file_read" in names
            assert "file_write" in names
            assert "shell_exec" in names

    def test_client_file_write_read(self, workspace):
        with MCPClient(workspace=workspace) as client:
            write_result = client.call_tool("file_write", {"path": "test.txt", "content": "hello mcp"})
            assert "成功" in write_result

            read_result = client.call_tool("file_read", {"path": "test.txt"})
            assert read_result == "hello mcp"

    def test_client_shell_exec(self, workspace, monkeypatch):
        # 测试环境通常没有 Docker，开启本地执行逃生口以验证 shell_exec 链路
        monkeypatch.setenv("MCP_SHELL_ALLOW_LOCAL", "true")
        with MCPClient(workspace=workspace) as client:
            result = client.call_tool("shell_exec", {"command": "echo hello"})
            assert "hello" in result

    def test_client_disabled(self, workspace):
        with MCPClient(workspace=workspace, enabled=False) as client:
            assert client.list_tools() == []
            assert client.call_tool("file_read", {"path": "x"}) == "[MCP 未启用]"
