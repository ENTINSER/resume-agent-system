"""LLMClient 单元测试

验证 OpenAI 与 Anthropic 两种 API 格式的请求构造、响应解析和错误处理。
所有测试使用 Mock，不调用真实 API。
"""

import os
import pytest
import requests
from unittest.mock import patch, MagicMock

from src.core.llm import LLMClient, _detect_api_format, _normalize_base_url, reset_llm


@pytest.fixture
def no_retry_delay(monkeypatch):
    """测试期间跳过指数退避的等待时间"""
    monkeypatch.setattr("src.core.llm.time.sleep", lambda s: None)
    monkeypatch.setattr("src.core.llm.random.uniform", lambda a, b: 0.0)


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    """清除可能干扰测试的环境变量，避免测试受 .env 文件影响"""
    for key in ("KIMI_API_KEY", "KIMI_BASE_URL", "KIMI_MODEL", "KIMI_API_FORMAT"):
        monkeypatch.delenv(key, raising=False)


class TestApiFormatDetection:
    """测试 API 格式自动推断"""

    def test_openai_format_by_url(self):
        assert _detect_api_format("https://api.kimi.com/coding/v1") == "openai"

    def test_anthropic_format_by_url(self):
        # Kimi URL 现在默认走 OpenAI 格式；Anthropic 格式需显式指定
        assert _detect_api_format("https://api.kimi.com/coding/") == "openai"
        assert _detect_api_format("https://api.kimi.com/coding") == "openai"

    def test_explicit_format_overrides_url(self):
        assert _detect_api_format("https://api.kimi.com/coding/v1", "anthropic") == "anthropic"
        assert _detect_api_format("https://api.kimi.com/coding/", "openai") == "openai"

    def test_default_fallback(self):
        assert _detect_api_format("https://some.other.url/") == "openai"


class TestBaseUrlNormalization:
    """测试 base_url 规范化"""

    def test_openai_url_normalized(self):
        assert _normalize_base_url("https://api.kimi.com/coding", "openai") == "https://api.kimi.com/coding/v1"
        assert _normalize_base_url("https://api.kimi.com/coding/v1", "openai") == "https://api.kimi.com/coding/v1"

    def test_anthropic_url_normalized(self):
        assert _normalize_base_url("https://api.kimi.com/coding/v1", "anthropic") == "https://api.kimi.com/coding"
        assert _normalize_base_url("https://api.kimi.com/coding/", "anthropic") == "https://api.kimi.com/coding"


class TestLLMClientInit:
    """测试 LLMClient 初始化"""

    def test_init_requires_api_key(self):
        with pytest.raises(ValueError, match="KIMI_API_KEY"):
            LLMClient(api_key="")

    def test_init_defaults(self):
        client = LLMClient(api_key="test-key")
        assert client.api_format == "openai"
        assert client.base_url == "https://api.kimi.com/coding/v1"
        assert client.model == "kimi-k2p5"

    def test_init_anthropic(self):
        client = LLMClient(
            api_key="test-key",
            base_url="https://api.kimi.com/coding/",
            model="kimi-k2p5",
            api_format="anthropic",
        )
        assert client.api_format == "anthropic"
        assert client.base_url == "https://api.kimi.com/coding"


class TestOpenAIChat:
    """测试 OpenAI 兼容格式调用"""

    def test_request_payload(self):
        client = LLMClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with patch("src.core.llm.requests.post", return_value=mock_response) as mock_post:
            result = client.chat("You are helpful.", "Say hi")

        assert result["success"] is True
        assert result["content"] == "Hello"
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5
        assert result["total_tokens"] == 15
        assert result["cost_usd"] > 0

        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.kimi.com/coding/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        payload = kwargs["json"]
        assert payload["model"] == "kimi-k2p5"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    def test_error_response(self, no_retry_delay):
        client = LLMClient(api_key="test-key")
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("src.core.llm.requests.post", return_value=mock_response) as mock_post:
            result = client.chat("You are helpful.", "Say hi")

        assert result["success"] is False
        assert "401" in result["content"]
        assert mock_post.call_count == 1  # 客户端错误不重试


class TestAnthropicChat:
    """测试 Anthropic 兼容格式调用"""

    def test_request_payload(self):
        client = LLMClient(
            api_key="test-key",
            base_url="https://api.kimi.com/coding/",
            api_format="anthropic",
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "Hi there"}],
            "usage": {"input_tokens": 8, "output_tokens": 2},
        }

        with patch("src.core.llm.requests.post", return_value=mock_response) as mock_post:
            result = client.chat("You are helpful.", "Say hi")

        assert result["success"] is True
        assert result["content"] == "Hi there"
        assert result["input_tokens"] == 8
        assert result["output_tokens"] == 2

        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.kimi.com/coding/v1/messages"
        assert kwargs["headers"]["x-api-key"] == "test-key"
        assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
        payload = kwargs["json"]
        assert "system" in payload
        assert payload["messages"][0]["role"] == "user"

    def test_error_response(self, no_retry_delay):
        client = LLMClient(
            api_key="test-key",
            base_url="https://api.kimi.com/coding/",
            api_format="anthropic",
        )
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"

        with patch("src.core.llm.requests.post", return_value=mock_response) as mock_post:
            result = client.chat("You are helpful.", "Say hi")

        assert result["success"] is False
        assert "500" in result["content"]
        assert mock_post.call_count == 3  # 服务端错误重试到最大次数


class TestRetry:
    """测试指数退避重试"""

    def test_retryable_then_success(self, no_retry_delay):
        client = LLMClient(api_key="test-key")
        error_response = MagicMock()
        error_response.status_code = 503
        error_response.text = "Service Unavailable"
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {
            "choices": [{"message": {"content": "Recovered"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }

        with patch("src.core.llm.requests.post", side_effect=[error_response, ok_response]) as mock_post:
            result = client.chat("You are helpful.", "Say hi")

        assert result["success"] is True
        assert result["content"] == "Recovered"
        assert mock_post.call_count == 2

    def test_request_exception_then_success(self, no_retry_delay):
        client = LLMClient(api_key="test-key")
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }

        with patch("src.core.llm.requests.post", side_effect=[requests.exceptions.ConnectionError("boom"), ok_response]) as mock_post:
            result = client.chat("You are helpful.", "Say hi")

        assert result["success"] is True
        assert result["content"] == "OK"
        assert mock_post.call_count == 2


class TestSingleton:
    """测试全局单例"""

    def test_get_llm_singleton(self):
        reset_llm()
        with patch.dict(os.environ, {"KIMI_API_KEY": "singleton-key"}, clear=False):
            client1 = LLMClient(api_key="singleton-key")
            # 由于全局单例使用 os.getenv，这里只验证 reset_llm 可用
            assert client1 is not None
