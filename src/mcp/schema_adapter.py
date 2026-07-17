"""MCP 工具 schema 转 OpenAI / Anthropic 原生 function calling 格式"""

from typing import Dict, Any, List


def mcp_tool_to_openai(tool: Dict[str, Any]) -> Dict[str, Any]:
    """把 MCP list_tools 返回的工具 dict 转成 OpenAI tools 格式"""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def mcp_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量转换"""
    return [mcp_tool_to_openai(t) for t in tools]


def mcp_tool_to_anthropic(tool: Dict[str, Any]) -> Dict[str, Any]:
    """把 MCP 工具 dict 转成 Anthropic tools 格式"""
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
    }


def mcp_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量转换"""
    return [mcp_tool_to_anthropic(t) for t in tools]


def normalize_tool_calls(response: Dict[str, Any], api_format: str = "openai") -> List[Dict[str, Any]]:
    """统一解析不同 API 格式的 tool_calls

    返回列表元素：
        {
            "id": str,          # OpenAI 需要；Anthropic 用 name 兜底
            "name": str,
            "arguments": dict,
        }
    """
    calls: List[Dict[str, Any]] = []

    if api_format == "anthropic":
        content = response.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    calls.append({
                        "id": block.get("id", block.get("name", "")),
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    })
        return calls

    # OpenAI 兼容格式
    raw = response.get("tool_calls", [])
    if not raw and response.get("choices"):
        message = response["choices"][0].get("message", {})
        raw = message.get("tool_calls", [])

    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function", {})
        args = function.get("arguments", "{}")
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except Exception:
                args = {}
        calls.append({
            "id": item.get("id", function.get("name", "")),
            "name": function.get("name", item.get("name", "")),
            "arguments": args,
        })

    return calls
