"""MCP Prompt 解析单元测试"""

import pytest

from src.mcp.prompts import (
    inject_tools_prompt,
    parse_tool_calls,
    remove_tool_calls,
    build_tool_result_message,
)


class TestMCPPrompts:
    def test_inject_tools_prompt(self):
        base = "你是一个助手。"
        tools = "## 可用工具\n- file_read"
        result = inject_tools_prompt(base, tools)
        assert base in result
        assert "可用工具" in result

    def test_parse_tool_calls_codeblock(self):
        content = """
我需要读取文件。
```json
{"tool": "file_read", "arguments": {"path": "main.py"}}
```
"""
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0] == ("file_read", {"path": "main.py"})

    def test_parse_tool_calls_plain_json(self):
        content = '{"tool": "shell_exec", "arguments": {"command": "ls"}}'
        calls = parse_tool_calls(content)
        assert len(calls) == 1
        assert calls[0] == ("shell_exec", {"command": "ls"})

    def test_parse_no_tool_calls(self):
        content = "这是一个普通回复，没有工具调用。"
        calls = parse_tool_calls(content)
        assert calls == []

    def test_remove_tool_calls(self):
        content = """正常文本
```json
{"tool": "x"}
```
"""
        cleaned = remove_tool_calls(content)
        assert "```json" not in cleaned
        assert "正常文本" in cleaned

    def test_build_tool_result_message(self):
        msg = build_tool_result_message("file_read", "hello")
        assert "file_read" in msg
        assert "hello" in msg
