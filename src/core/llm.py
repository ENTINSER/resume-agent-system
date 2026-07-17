"""LLM 客户端 - 统一封装 Kimi Code API

支持两种 API 格式：
1. OpenAI 兼容格式（默认）
   - base_url: https://api.kimi.com/coding/v1
   - endpoint: /chat/completions
   - 环境变量: KIMI_API_FORMAT=openai
2. Anthropic 兼容格式
   - base_url: https://api.kimi.com/coding/
   - endpoint: /v1/messages
   - 环境变量: KIMI_API_FORMAT=anthropic

配置来源优先级：
1. 环境变量 KIMI_API_FORMAT
2. 根据 KIMI_BASE_URL 自动推断
3. 默认使用 openai

P6 重构：引入 ModelProvider 抽象层，统一处理不同模型的参数/解析差异。
"""

import os
import random
import time
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urljoin

import requests

from src.core.logger import logger
from src.core.config import settings
from src.core.model_providers import get_provider, ModelProvider


ApiFormat = Literal["openai", "anthropic"]


def _detect_api_format(base_url: str, explicit_format: Optional[str] = None) -> ApiFormat:
    """根据配置自动推断或校验 API 格式"""
    if explicit_format:
        fmt = explicit_format.lower().strip()
        if fmt in ("openai", "anthropic"):
            return fmt
        logger.warning(f"[LLM] 未知的 KIMI_API_FORMAT={explicit_format}，将自动推断")

    # 自动推断规则
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return "openai"

    # Kimi Code API 同时支持 OpenAI 与 Anthropic 格式。
    # 历史配置中大量 .env 使用 /coding（不带 /v1），默认行为应为更常见的 OpenAI 格式。
    if "kimi" in url.lower():
        logger.info(f"[LLM] 检测到 Kimi URL，默认使用 openai 格式: {base_url}")
        return "openai"

    if url.endswith("/coding") or url.endswith("/coding/"):
        return "anthropic"

    # 兜底：OpenAI 兼容格式更常见
    logger.info(f"[LLM] 无法从 URL 推断格式，默认使用 openai: {base_url}")
    return "openai"


def _normalize_base_url(base_url: str, fmt: ApiFormat) -> str:
    """规范化 base_url，避免拼接出重复路径"""
    url = base_url.rstrip("/")
    if fmt == "openai":
        # OpenAI 格式要求 base_url 以 /v1 结尾
        if not url.endswith("/v1"):
            url = url + "/v1"
    elif fmt == "anthropic":
        # Anthropic 格式要求 base_url 以 /coding/ 结尾
        if url.endswith("/v1"):
            url = url[:-3]
    return url


class LLMClient:
    """LLM 客户端 - 统一封装 Kimi Code API"""

    ENDPOINTS = {
        "openai": "chat/completions",
        "anthropic": "v1/messages",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_format: Optional[str] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.kimi_api_key
        raw_base_url = base_url if base_url is not None else settings.kimi_base_url
        self.model = model if model is not None else settings.kimi_model
        self.api_format = _detect_api_format(
            raw_base_url, api_format if api_format is not None else settings.kimi_api_format
        )
        self.base_url = _normalize_base_url(raw_base_url, self.api_format)

        if not self.api_key:
            raise ValueError("KIMI_API_KEY 未设置")

        self.provider: ModelProvider = get_provider(self.model, self.api_format)
        self.enable_thinking = os.getenv("KIMI_ENABLE_THINKING", "false").lower() in (
            "true", "1", "yes"
        )

        logger.info(
            f"[LLM] 初始化: format={self.api_format}, "
            f"base_url={self.base_url}, model={self.model}, "
            f"provider={self.provider.__class__.__name__}"
        )

    def _endpoint_url(self) -> str:
        return urljoin(self.base_url + "/", self.ENDPOINTS[self.api_format])

    def _headers(self) -> Dict[str, str]:
        if self.api_format == "openai":
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> dict:
        """统一 chat 接口，返回结构化结果（含 tokens）

        注：``max_tokens`` 对应 Kimi API 的 ``max_completion_tokens``，
        用于限制模型单次输出的最大 token 数。返回值中包含 ``finish_reason``，
        调用方可据此判断是否因长度被截断。
        """
        start_time = time.time()

        payload = self.provider.build_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=self.enable_thinking,
        )

        result = self._do_request(payload)
        if not result.get("success", False):
            return result

        elapsed_ms = int((time.time() - start_time) * 1000)
        result["elapsed_ms"] = elapsed_ms
        return result

    def chat_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> dict:
        """支持工具提示的 chat 接口

        兼容两种模式：
        1. tools_prompt（str）：在 system prompt 中注入工具说明，让模型决定是否需要调用工具。
           这是当前 MCP 桥接使用的模式。
        2. tools（list）：原生 OpenAI/Anthropic function calling。
        3. messages（list）：传入完整消息列表（用于原生 function calling 多轮）。
        """
        if tools_prompt is not None:
            from src.mcp.prompts import inject_tools_prompt

            full_system_prompt = inject_tools_prompt(system_prompt, tools_prompt)
            return self.chat(
                system_prompt=full_system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # 原生 function calling 模式
        if tools is None:
            tools = []
        payload = self.provider.build_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=self.enable_thinking,
            tools=tools,
            messages=messages,
        )
        result = self._do_request(payload)
        return result

    def _do_request(self, payload: Dict[str, Any]) -> dict:
        """执行 HTTP 请求并解析响应（含指数退避重试）"""
        url = self._endpoint_url()
        headers = self._headers()
        max_attempts = 3
        base_delay = 1.0

        last_error = None
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=600,
                )
            except requests.exceptions.RequestException as exc:
                last_error = exc
                logger.warning(f"[LLM] 请求异常（第 {attempt}/{max_attempts} 次）: {exc}")
            else:
                if response.status_code == 200:
                    last_error = None
                    break

                last_error = response
                if self._is_retryable_status(response.status_code):
                    logger.warning(
                        f"[LLM] API 返回 {response.status_code}（第 {attempt}/{max_attempts} 次），"
                        f"{response.text[:200]}"
                    )
                else:
                    logger.error(f"[LLM] API 错误: {response.status_code} - {response.text[:300]}")
                    return self._error_result(response.status_code, response.text)

            if attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)), 10.0)
                delay += random.uniform(0, 0.5)
                logger.info(f"[LLM] 将在 {delay:.1f}s 后重试...")
                time.sleep(delay)

        if last_error is not None:
            if isinstance(last_error, requests.exceptions.RequestException):
                logger.error(f"[LLM] 请求最终失败: {last_error}")
                return self._error_result(0, str(last_error))
            logger.error(f"[LLM] API 错误: {last_error.status_code} - {last_error.text[:300]}")
            return self._error_result(last_error.status_code, last_error.text)

        try:
            data = response.json()
        except Exception as e:
            logger.error(f"[LLM] 响应 JSON 解析失败: {e}")
            return self._error_result(200, response.text[:300])

        parsed = self.provider.parse_response(data)
        content = parsed["content"]
        input_tokens = parsed["input_tokens"]
        output_tokens = parsed["output_tokens"]
        total_tokens = input_tokens + output_tokens

        # 成本计算（Kimi k2p5 约 $0.5/1M input, $1.5/1M output）
        cost_usd = (input_tokens / 1_000_000) * 0.5 + (output_tokens / 1_000_000) * 1.5

        success = True
        if not content or not content.strip():
            success = False
            content = "# LLM 返回空内容"
            reasoning = parsed.get("reasoning_content", "")
            finish = parsed.get("finish_reason", "")
            if reasoning:
                logger.warning(
                    f"[LLM] 模型返回空 content，finish_reason={finish}，"
                    f"reasoning_content 前500字符: {reasoning[:500]}..."
                )
            else:
                logger.warning(f"[LLM] 模型返回空 content，finish_reason={finish}")
        elif parsed.get("finish_reason") == "length":
            logger.warning("[LLM] 模型输出因长度限制被截断（finish_reason=length）")

        logger.info(
            f"[LLM] 生成完成，tokens: {total_tokens} (in: {input_tokens}, out: {output_tokens}), "
            f"成本: ${cost_usd:.4f}"
        )

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "elapsed_ms": 0,
            "cost_usd": cost_usd,
            "success": success,
            "finish_reason": parsed.get("finish_reason", ""),
            "reasoning_content": parsed.get("reasoning_content", ""),
            "tool_calls": parsed.get("tool_calls", []),
        }

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        if status_code == 429:
            return True
        if status_code >= 500:
            return True
        return False

    @staticmethod
    def _error_result(status_code: int, text: str) -> dict:
        return {
            "content": f"# API 错误: {status_code} - {text[:200]}",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "elapsed_ms": 0,
            "cost_usd": 0.0,
            "success": False,
        }


# 全局 LLM 实例
_llm: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    """获取 LLM 实例（单例）"""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def reset_llm() -> None:
    """重置全局 LLM 实例（主要用于测试）"""
    global _llm
    _llm = None
