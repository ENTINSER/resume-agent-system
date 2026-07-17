"""MCP Prompt 工具 - 在 prompt 中注入工具定义并解析工具调用"""

import json
import re
from typing import Dict, Any, List, Tuple, Optional


def inject_tools_prompt(base_prompt: str, tools_json: str) -> str:
    """把工具说明追加到 system prompt"""
    if not tools_json:
        return base_prompt
    return f"""{base_prompt}

{tools_json}

重要规则：
1. 只有在确实需要外部信息或执行操作时才调用工具
2. 调用工具时只输出一个 JSON 代码块，格式为 ```json {{"tool": "...", "arguments": {{...}}}} ```
3. 如果不需要工具，直接输出正常回复，不要输出 JSON
"""


def parse_tool_calls(content: str) -> List[Tuple[str, Dict[str, Any]]]:
    """从 LLM 回复中解析工具调用

    返回 [(tool_name, arguments), ...]
    """
    calls = []

    # 匹配 ```json ... ``` 代码块
    json_blocks = re.findall(r"```json\s*([\s\S]*?)\s*```", content)
    for block in json_blocks:
        try:
            data = json.loads(block.strip())
            if isinstance(data, dict) and "tool" in data:
                tool_name = data["tool"]
                arguments = data.get("arguments", {})
                if isinstance(arguments, dict):
                    calls.append((tool_name, arguments))
        except json.JSONDecodeError:
            continue

    # 如果没有代码块，尝试直接解析整段 JSON
    if not calls:
        try:
            data = json.loads(content.strip())
            if isinstance(data, dict) and "tool" in data:
                calls.append((data["tool"], data.get("arguments", {})))
        except json.JSONDecodeError:
            pass

    return calls


def remove_tool_calls(content: str) -> str:
    """移除回复中的工具调用代码块，保留正常文本"""
    cleaned = re.sub(r"```json\s*[\s\S]*?\s*```", "", content)
    cleaned = cleaned.strip()
    return cleaned


def build_tool_result_message(tool_name: str, result: str) -> str:
    """构造工具结果消息，追加到对话上下文"""
    return f"""工具 {tool_name} 返回结果：
```
{result}
```
请基于以上结果继续回答。"""


def build_tool_messages(
    calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    api_format: str = "openai",
) -> List[Dict[str, Any]]:
    """构造原生 function calling 的多轮消息

    calls: [{"id": ..., "name": ..., "arguments": ...}]
    results: [{"tool": ..., "result": ...}]
    """
    messages = []
    if api_format == "anthropic":
        # Anthropic 把 tool_use 和 tool_result 都作为 assistant/user content 块
        tool_use_blocks = []
        tool_result_blocks = []
        for call, res in zip(calls, results):
            tool_use_blocks.append({
                "type": "tool_use",
                "id": call.get("id", call.get("name", "")),
                "name": call.get("name", ""),
                "input": call.get("arguments", {}),
            })
            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": call.get("id", call.get("name", "")),
                "content": str(res.get("result", "")),
            })
        messages.append({"role": "assistant", "content": tool_use_blocks})
        messages.append({"role": "user", "content": tool_result_blocks})
        return messages

    # OpenAI 兼容格式
    tool_calls = []
    for call in calls:
        tool_calls.append({
            "id": call.get("id", call.get("name", "")),
            "type": "function",
            "function": {
                "name": call.get("name", ""),
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            },
        })
    messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    for call, res in zip(calls, results):
        messages.append({
            "role": "tool",
            "tool_call_id": call.get("id", call.get("name", "")),
            "name": call.get("name", ""),
            "content": str(res.get("result", "")),
        })
    return messages


def parse_native_tool_calls(
    response: Dict[str, Any],
    api_format: str = "openai",
) -> List[Dict[str, Any]]:
    """从 LLMClient 返回的 dict 中解析原生 tool_calls

    返回 [{"id": ..., "name": ..., "arguments": ...}]
    """
    from src.mcp.schema_adapter import normalize_tool_calls

    return normalize_tool_calls(response, api_format=api_format)
