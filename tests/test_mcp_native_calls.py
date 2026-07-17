"""MCP 原生 function calling 桥接单测"""

import pytest
from unittest.mock import MagicMock

from src.mcp.prompts import build_tool_messages, parse_native_tool_calls
from src.mcp.bridge import _format_tools_for_api
from src.mcp.schema_adapter import normalize_tool_calls


class TestBuildToolMessages:
    def test_openai_tool_messages(self):
        calls = [{"id": "c1", "name": "file_read", "arguments": {"path": "main.py"}}]
        results = [{"tool": "file_read", "result": "content"}]
        messages = build_tool_messages(calls, results, api_format="openai")
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["function"]["name"] == "file_read"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "c1"

    def test_anthropic_tool_messages(self):
        calls = [{"id": "c1", "name": "file_read", "arguments": {"path": "main.py"}}]
        results = [{"tool": "file_read", "result": "content"}]
        messages = build_tool_messages(calls, results, api_format="anthropic")
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["type"] == "tool_use"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"][0]["type"] == "tool_result"


class TestFormatToolsForApi:
    def test_openai_format(self):
        tools = [{"name": "file_read", "description": "read", "parameters": {}}]
        result = _format_tools_for_api(tools, "openai")
        assert result[0]["type"] == "function"

    def test_anthropic_format(self):
        tools = [{"name": "file_read", "description": "read", "parameters": {}}]
        result = _format_tools_for_api(tools, "anthropic")
        assert result[0]["name"] == "file_read"
        assert "input_schema" in result[0]
