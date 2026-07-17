"""Go API -> Celery 消息格式集成测试

验证 Go API 投递到 Redis 的 Celery 任务消息能被 kombu 正确解析。
此测试需要 Redis 和 Go API 同时运行。
"""

import base64
import json
import os
import time
import uuid

import pytest
import redis


API_URL = os.getenv("AGENT_API_URL", "http://localhost:8080")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _api_available():
    import urllib.request
    try:
        urllib.request.urlopen(f"{API_URL}/readyz", timeout=2)
        return True
    except Exception:
        return False


def _redis_available():
    try:
        redis.from_url(REDIS_URL).ping()
        return True
    except Exception:
        return False


def _worker_running():
    import subprocess
    try:
        output = subprocess.check_output(
            ["pgrep", "-f", "celery -A src.tasks worker"],
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return False


@pytest.mark.skipif(not _api_available(), reason="Go API not available")
@pytest.mark.skipif(not _redis_available(), reason="Redis not available")
@pytest.mark.skipif(_worker_running(), reason="Celery worker is running, would consume the message")
class TestGoAPICeleryMessage:
    def test_message_envelope(self):
        """Go API 投递的消息包含 kombu 需要的 body/headers/properties"""
        client = redis.from_url(REDIS_URL)
        # 清空队列，避免历史消息干扰
        client.delete("celery")

        requirements = f"test requirements {uuid.uuid4()}"
        resp = requests.post(
            f"{API_URL}/api/v1/tasks",
            json={"requirements": requirements},
            timeout=5,
        )
        assert resp.status_code in (200, 202), resp.text
        data = resp.json()
        assert data["status"] == "pending"
        task_id = data["task_id"]

        # 从 Redis 读取消息，最多等 5 秒
        raw = client.brpop("celery", timeout=5)
        assert raw is not None, "Redis 队列中没有收到 Celery 消息"

        envelope = json.loads(raw[1])
        assert "body" in envelope
        assert "headers" in envelope
        assert "properties" in envelope
        assert envelope.get("content-type") == "application/json"
        assert envelope.get("content-encoding") == "utf-8"

        headers = envelope["headers"]
        assert headers.get("task") == "src.tasks.agent_task.run_agent_task"
        assert headers.get("lang") == "py"
        assert "id" in headers

        properties = envelope["properties"]
        assert "correlation_id" in properties
        assert properties.get("body_encoding") == "base64"

        # body base64 解码后应为 [[args], {kwargs}, {callbacks...}]
        body_bytes = base64.b64decode(envelope["body"])
        body = json.loads(body_bytes)
        assert isinstance(body, list) and len(body) == 3
        assert body[0] == [task_id, requirements]
        assert "api_key" in body[1]
        assert body[2].get("callbacks") is None
        assert body[2].get("errbacks") is None


# 保持顶层导入兼容：如果 requests 未安装则跳过整个模块
try:
    import requests
except ImportError:
    pytest.skip("requests not installed", allow_module_level=True)
