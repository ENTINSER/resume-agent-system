"""MCP schema 转换单元测试"""

import pytest

from src.mcp.schema_adapter import (
    mcp_tool_to_openai,
    mcp_tools_to_openai,
    mcp_tool_to_anthropic,
    mcp_tools_to_anthropic,
    normalize_tool_calls,
)


@pytest.fixture
def mcp_tool():
    return {
        "name": "file_read",
        "description": "读取文件内容",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
            },
            "required": ["path"],
        },
    }


class TestSchemaAdapter:
    def test_mcp_tool_to_openai(self, mcp_tool):
        result = mcp_tool_to_openai(mcp_tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "file_read"
        assert "path" in result["function"]["parameters"]["properties"]

    def test_mcp_tools_to_openai(self, mcp_tool):
        result = mcp_tools_to_openai([mcp_tool])
        assert len(result) == 1
        assert result[0]["function"]["name"] == "file_read"

    def test_mcp_tool_to_anthropic(self, mcp_tool):
        result = mcp_tool_to_anthropic(mcp_tool)
        assert result["name"] == "file_read"
        assert "input_schema" in result
        assert "path" in result["input_schema"]["properties"]

    def test_mcp_tools_to_anthropic(self, mcp_tool):
        result = mcp_tools_to_anthropic([mcp_tool])
        assert len(result) == 1
        assert result[0]["name"] == "file_read"


class TestNormalizeToolCalls:
    def test_openai_tool_calls(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "file_read",
                                    "arguments": '{"path": "main.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        calls = normalize_tool_calls(response, api_format="openai")
        assert len(calls) == 1
        assert calls[0]["id"] == "call_1"
        assert calls[0]["name"] == "file_read"
        assert calls[0]["arguments"]["path"] == "main.py"

    def test_anthropic_tool_use(self):
        response = {
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "tu_1", "name": "file_read", "input": {"path": "main.py"}},
            ]
        }
        calls = normalize_tool_calls(response, api_format="anthropic")
        assert len(calls) == 1
        assert calls[0]["id"] == "tu_1"
        assert calls[0]["arguments"]["path"] == "main.py"
