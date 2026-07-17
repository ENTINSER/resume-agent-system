"""异常分级单元测试"""

import pytest
import requests
import redis.exceptions as redis_exc
import sqlalchemy.exc as db_exc

from src.core.exceptions import (
    AgentPlatformError,
    TransientError,
    DegradableError,
    FatalError,
    classify_exception,
)


class TestClassifyException:
    """测试异常分类器"""

    def test_requests_exception_is_transient(self):
        exc = requests.exceptions.ConnectionError("boom")
        classified = classify_exception(exc)
        assert isinstance(classified, TransientError)
        assert classified.original is exc

    def test_redis_timeout_is_transient(self):
        exc = redis_exc.TimeoutError("redis slow")
        classified = classify_exception(exc)
        assert isinstance(classified, TransientError)

    def test_db_operational_error_is_transient(self):
        exc = db_exc.OperationalError("stmt", {}, "db down")
        classified = classify_exception(exc)
        assert isinstance(classified, TransientError)

    def test_value_error_is_fatal(self):
        exc = ValueError("bad input")
        classified = classify_exception(exc)
        assert isinstance(classified, FatalError)

    def test_permission_error_is_fatal(self):
        exc = PermissionError("denied")
        classified = classify_exception(exc)
        assert isinstance(classified, FatalError)

    def test_unknown_error_is_degradable(self):
        exc = RuntimeError("something weird")
        classified = classify_exception(exc)
        assert isinstance(classified, DegradableError)

    def test_platform_error_passes_through(self):
        exc = TransientError("timeout")
        classified = classify_exception(exc)
        assert classified is exc
