"""平台异常分级

将运行时错误划分为三类，便于编排器和 Celery 采取不同策略：
- TransientError：临时故障（网络、LLM 限流/5xx、Redis 抖动），可重试。
- DegradableError：非致命能力缺失（RAG、MCP、ML/SFT 飞轮失败），可降级继续。
- FatalError：无法恢复的错误（配置缺失、认证失败、非法输入），直接失败。
"""

from __future__ import annotations

from typing import Optional


class AgentPlatformError(Exception):
    """平台异常基类"""

    def __init__(self, message: str, *, original: Optional[Exception] = None):
        super().__init__(message)
        self.original = original


class TransientError(AgentPlatformError):
    """临时性故障，建议指数退避重试"""


class DegradableError(AgentPlatformError):
    """可降级错误，主流程应继续执行，但可能跳过某项增强能力"""


class FatalError(AgentPlatformError):
    """致命错误，任务应直接失败，不宜重试"""


class TaskCancelledError(AgentPlatformError):
    """任务被取消"""


def classify_exception(exc: Exception) -> AgentPlatformError:
    """根据异常类型分类为平台异常"""
    import requests
    import redis.exceptions as redis_exc
    import sqlalchemy.exc as db_exc

    message = str(exc)

    if isinstance(exc, AgentPlatformError):
        return exc

    # 网络 / LLM 临时错误
    if isinstance(
        exc,
        (
            requests.exceptions.RequestException,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    ):
        return TransientError(message, original=exc)

    # Redis 临时错误
    if isinstance(exc, (redis_exc.ConnectionError, redis_exc.TimeoutError)):
        return TransientError(message, original=exc)

    # 数据库临时错误（连接/死锁等）
    if isinstance(
        exc,
        (
            db_exc.OperationalError,
            db_exc.DBAPIError,
        ),
    ):
        return TransientError(message, original=exc)

    # 配置 / 认证 / 输入错误通常不可重试
    if isinstance(
        exc,
        (
            ValueError,
            KeyError,
            TypeError,
            FileNotFoundError,
            PermissionError,
        ),
    ):
        return FatalError(message, original=exc)

    # 其他未知错误视为可降级，保留日志后继续
    return DegradableError(message, original=exc)
