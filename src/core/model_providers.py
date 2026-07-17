"""LLM 模型适配器抽象层

为不同模型/API 格式提供统一接口，解决：
- temperature/thinking 参数差异
- payload 字段差异（max_tokens vs max_completion_tokens）
- 响应解析差异
- 原生 function calling 支持差异
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional


class ModelProvider(ABC):
    """模型适配器基类"""

    def __init__(self, model: str):
        self.model = model

    @property
    @abstractmethod
    def api_format(self) -> Literal["openai", "anthropic"]:
        """返回底层 API 格式"""
        ...

    @property
    @abstractmethod
    def supports_temperature(self) -> bool:
        ...

    @property
    @abstractmethod
    def supports_thinking(self) -> bool:
        ...

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        ...

    @property
    @abstractmethod
    def max_tokens_limit(self) -> int:
        ...

    @property
    @abstractmethod
    def default_temperature(self) -> float:
        ...

    def normalize_temperature(self, temperature: float, enable_thinking: bool = False) -> float:
        """将外部温度参数归一化为模型可接受的值"""
        if not self.supports_temperature:
            return self.default_temperature
        return temperature

    @abstractmethod
    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """构造请求 payload"""
        ...

    @abstractmethod
    def parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解析响应，返回统一结构：content, input_tokens, output_tokens, finish_reason, reasoning_content"""
        ...


class KimiProvider(ModelProvider):
    """Kimi Code API（k2p5 等）适配器

    默认使用 OpenAI 兼容格式，thinking 可控。
    """

    @property
    def api_format(self) -> Literal["openai", "anthropic"]:
        return "openai"

    @property
    def supports_temperature(self) -> bool:
        return True

    @property
    def supports_thinking(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def max_tokens_limit(self) -> int:
        # kimi-k2.5 官方上下文 256k，输出上限受 input + output <= 256k 约束。
        # 这里保留一个安全的单次输出上限，复杂文件可在调用方申请更高值。
        return 16384

    @property
    def default_temperature(self) -> float:
        return 1.0

    def normalize_temperature(self, temperature: float, enable_thinking: bool = False) -> float:
        # kimi-k2p5 对 temperature 有严格限制：
        # - thinking enabled 时只能接受 1.0
        # - thinking disabled 时只能接受 0.6
        if enable_thinking:
            return 1.0
        return 0.6

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.normalize_temperature(temperature, enable_thinking),
            "max_completion_tokens": min(max_tokens, self.max_tokens_limit),
        }
        if self.supports_thinking:
            payload["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
        if tools and self.supports_tools:
            payload["tools"] = tools
        return payload

    def parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        content = ""
        reasoning_content = ""
        finish_reason = ""
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning_content = message.get("reasoning_content", "")
            finish_reason = choice.get("finish_reason", "")
            # 原生 function calling
            tool_calls = message.get("tool_calls")
            if tool_calls:
                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    "finish_reason": finish_reason,
                    "reasoning_content": reasoning_content,
                }
        usage = data.get("usage", {})
        return {
            "content": content,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "finish_reason": finish_reason,
            "reasoning_content": reasoning_content,
        }


class OpenAIProvider(ModelProvider):
    """通用 OpenAI 兼容格式适配器"""

    @property
    def api_format(self) -> Literal["openai", "anthropic"]:
        return "openai"

    @property
    def supports_temperature(self) -> bool:
        return True

    @property
    def supports_thinking(self) -> bool:
        return False

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def max_tokens_limit(self) -> int:
        return 4096

    @property
    def default_temperature(self) -> float:
        return 1.0

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": min(max_tokens, self.max_tokens_limit),
        }
        if tools and self.supports_tools:
            payload["tools"] = tools
        return payload

    def parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        content = ""
        finish_reason = ""
        choices = data.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            finish_reason = choice.get("finish_reason", "")
            tool_calls = message.get("tool_calls")
            if tool_calls:
                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    "finish_reason": finish_reason,
                    "reasoning_content": "",
                }
        usage = data.get("usage", {})
        return {
            "content": content,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "finish_reason": finish_reason,
            "reasoning_content": "",
        }


class AnthropicProvider(ModelProvider):
    """通用 Anthropic 兼容格式适配器"""

    @property
    def api_format(self) -> Literal["openai", "anthropic"]:
        return "anthropic"

    @property
    def supports_temperature(self) -> bool:
        return True

    @property
    def supports_thinking(self) -> bool:
        return False

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def max_tokens_limit(self) -> int:
        return 4096

    @property
    def default_temperature(self) -> float:
        return 1.0

    def normalize_temperature(self, temperature: float, enable_thinking: bool = False) -> float:
        return 1.0

    def build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if messages is None:
            messages = [{"role": "user", "content": user_prompt}]
        else:
            # Anthropic 要求 system 作为顶层字段，messages 中不能包含 system 角色
            system_msg = [m for m in messages if m.get("role") == "system"]
            if system_msg:
                system_prompt = system_msg[0].get("content", system_prompt)
            messages = [m for m in messages if m.get("role") != "system"]

        payload: Dict[str, Any] = {
            "model": self.model,
            "system": system_prompt,
            "messages": messages,
            "temperature": self.normalize_temperature(temperature),
            "max_tokens": min(max_tokens, self.max_tokens_limit),
        }
        if tools and self.supports_tools:
            payload["tools"] = tools
        return payload

    def parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text_parts = []
        reasoning_parts = []
        tool_calls = []
        if isinstance(data.get("content"), list):
            for block in data["content"]:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", block.get("name", "")),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        },
                    })
                elif block_type in ("thinking", "reasoning"):
                    reasoning_parts.append(block.get("text", ""))
                elif "text" in block:
                    text_parts.append(block.get("text", ""))
        elif "content" in data:
            text_parts.append(str(data.get("content"))[:2000])

        usage = data.get("usage", {})
        return {
            "content": "".join(text_parts),
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
            "finish_reason": data.get("stop_reason", ""),
            "reasoning_content": "\n".join(reasoning_parts),
            "tool_calls": tool_calls,
        }


def get_provider(model: str, api_format: Literal["openai", "anthropic"]) -> ModelProvider:
    """根据模型名和 API 格式选择合适的适配器

    优先按显式指定的 ``api_format`` 选择；Kimi 模型在 OpenAI 格式下使用专用适配器。
    """
    if api_format == "anthropic":
        return AnthropicProvider(model)

    model_lower = model.lower()
    if model_lower in ("k2p5", "kimi-k2.5", "kimi-k2.6", "kimi-k2p5"):
        return KimiProvider(model)
    return OpenAIProvider(model)
