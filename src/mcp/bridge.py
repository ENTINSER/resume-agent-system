"""MCP 与 LLM 的桥接层

提供 `ask_with_tools` 通用方法，让智能体在需要时自动调用 MCP 工具。
支持两种模式：
1. prompt-based（默认兼容）：在 system prompt 中注入工具说明，解析 JSON 代码块。
2. native function calling：使用 OpenAI/Anthropic 原生 tools 接口（由配置开关）。
"""

import os
from typing import Optional

from src.core.llm import LLMClient
from src.core.config import settings
from src.mcp.client import MCPClient
from src.mcp.prompts import (
    parse_tool_calls,
    remove_tool_calls,
    build_tool_result_message,
    parse_native_tool_calls,
    build_tool_messages,
)
from src.mcp.schema_adapter import (
    mcp_tools_to_openai,
    mcp_tools_to_anthropic,
)
from src.core.logger import logger


def get_workspace(state: dict) -> str:
    """根据 state 推断项目工作空间路径"""
    project = state.get("project") or {}
    project_id = project.get("id") or f"proj-{state.get('session_id', 'unknown')[:8]}"
    return os.path.join("projects", project_id)


def create_mcp_client_for_state(state: dict, enabled: Optional[bool] = None) -> MCPClient:
    """为当前 state 创建 MCPClient"""
    workspace = get_workspace(state)
    os.makedirs(workspace, exist_ok=True)
    return MCPClient(workspace=workspace, enabled=enabled)


def _format_tools_for_api(tools: list, api_format: str) -> list:
    """把 MCP 工具列表转成对应 API 的 tools schema"""
    if api_format == "anthropic":
        return mcp_tools_to_anthropic(tools)
    return mcp_tools_to_openai(tools)


def _supports_native(llm: LLMClient) -> bool:
    """判断是否满足原生 function calling 条件"""
    return (
        settings.mcp_native_function_calling
        and llm.provider.supports_tools
    )


def ask_with_tools(
    llm: LLMClient,
    mcp_client: MCPClient,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    max_rounds: int = None,
) -> dict:
    """调用 LLM，支持自动 MCP 工具调用（多轮）

    返回值与 LLMClient.chat() 保持一致，并额外包含：
      - tool_calls: 所有调用记录
      - tool_results: 所有调用结果
    """
    max_rounds = max_rounds if max_rounds is not None else settings.mcp_max_tool_rounds
    tools = mcp_client.list_tools()

    if not tools:
        # MCP 未启用或无可用工具，走普通 chat
        return llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    native_mode = _supports_native(llm)
    accumulated_calls = []
    accumulated_results = []

    if native_mode:
        return _ask_native(
            llm=llm,
            mcp_client=mcp_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            max_rounds=max_rounds,
        )

    # prompt-based 模式
    tools_prompt = mcp_client.format_tools_prompt()
    response = llm.chat_with_tools(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools_prompt=tools_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    current_prompt = user_prompt
    for round_idx in range(max_rounds):
        content = response.get("content", "")
        calls = parse_tool_calls(content)
        if not calls:
            break

        logger.info(f"[MCP] 第 {round_idx + 1} 轮工具调用: {[n for n, _ in calls]}")
        current_prompt, response, accumulated_calls, accumulated_results = _run_prompt_round(
            llm=llm,
            mcp_client=mcp_client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            current_prompt=current_prompt,
            calls=calls,
            tools_prompt=tools_prompt,
            response=response,
            accumulated_calls=accumulated_calls,
            accumulated_results=accumulated_results,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # 清理最终内容中的工具调用块
    response["content"] = remove_tool_calls(response.get("content", ""))
    response["tool_calls"] = accumulated_calls
    response["tool_results"] = accumulated_results
    return response


def _run_prompt_round(
    llm: LLMClient,
    mcp_client: MCPClient,
    system_prompt: str,
    user_prompt: str,
    current_prompt: str,
    calls: list,
    tools_prompt: str,
    response: dict,
    accumulated_calls: list,
    accumulated_results: list,
    temperature: float,
    max_tokens: int,
):
    """执行 prompt-based 模式的一轮工具调用"""
    for name, args in calls:
        result = mcp_client.call_tool(name, args)
        accumulated_calls.append({"tool": name, "arguments": args})
        accumulated_results.append({"tool": name, "result": result})

    result_texts = [
        build_tool_result_message(item["tool"], item["result"])
        for item in accumulated_results
    ]
    current_prompt = (
        f"原始问题：{user_prompt}\n\n"
        f"你已调用工具，结果如下：\n\n"
        f"{'\n\n'.join(result_texts)}\n\n"
        f"请基于工具结果直接给出最终回答。"
    )
    response = llm.chat_with_tools(
        system_prompt=system_prompt,
        user_prompt=current_prompt,
        tools_prompt=tools_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return current_prompt, response, accumulated_calls, accumulated_results


def _ask_native(
    llm: LLMClient,
    mcp_client: MCPClient,
    system_prompt: str,
    user_prompt: str,
    tools: list,
    temperature: float,
    max_tokens: int,
    max_rounds: int,
) -> dict:
    """原生 function calling 多轮调用"""
    api_format = llm.api_format
    tools_schema = _format_tools_for_api(tools, api_format)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.chat_with_tools(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools_schema,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    accumulated_calls = []
    accumulated_results = []

    for round_idx in range(max_rounds):
        calls = parse_native_tool_calls(response, api_format=api_format)
        if not calls:
            break

        logger.info(f"[MCP] 第 {round_idx + 1} 轮原生工具调用: {[c.get('name') for c in calls]}")

        results = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            result = mcp_client.call_tool(name, args)
            results.append({"tool": name, "result": result})
            accumulated_calls.append({"tool": name, "arguments": args})
            accumulated_results.append({"tool": name, "result": result})

        # 把 tool_calls 和 tool_results 加入 messages，继续下一轮
        tool_messages = build_tool_messages(calls, results, api_format=api_format)
        messages.extend(tool_messages)

        response = llm.chat_with_tools(
            system_prompt=system_prompt,
            user_prompt=messages,
            tools=tools_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    response["tool_calls"] = accumulated_calls
    response["tool_results"] = accumulated_results
    return response
