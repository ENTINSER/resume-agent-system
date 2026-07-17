"""日志模块单元测试"""

import json
import logging
import sys

import pytest

from src.core.logger import JsonFormatter, log_agent_event


class TestJsonFormatter:
    """测试 JSON 日志格式化器"""

    @pytest.fixture
    def formatter(self):
        return JsonFormatter()

    def test_basic_json_output(self, formatter):
        record = logging.LogRecord(
            name="AgentPlatform",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "info"
        assert parsed["logger"] == "AgentPlatform"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed

    def test_extra_fields_included(self, formatter):
        record = logging.LogRecord(
            name="AgentPlatform",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="alert",
            args=(),
            exc_info=None,
        )
        record.agent = "D"
        record.request_id = "req-123"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["agent"] == "D"
        assert parsed["request_id"] == "req-123"
        assert parsed["level"] == "warning"

    def test_exception_included(self, formatter):
        try:
            raise ValueError("boom")
        except ValueError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="AgentPlatform",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="failed",
                args=(),
                exc_info=exc_info,
            )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "failed"
        assert "exception" in parsed
        assert "boom" in parsed["exception"]


class TestLogAgentEvent:
    """测试智能体事件日志接口"""

    def test_log_agent_event_emits_json(self, caplog):
        caplog.set_level(logging.INFO, logger="AgentPlatform")

        log_agent_event("D", "phase_change", "analysis")

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.agent == "D"
        assert "phase_change" in record.message
        assert "analysis" in record.message
