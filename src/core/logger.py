"""日志系统 - 平台级日志

默认输出结构化 JSON 到 stderr，与 Go Gateway 的 zap 日志格式对齐。
可通过环境变量控制：
- LOG_LEVEL: DEBUG/INFO/WARNING/ERROR/CRITICAL（默认 INFO）
- JSON_LOG_FORMAT: true/false（默认 true）
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from src.core.config import settings


class JsonFormatter(logging.Formatter):
    """将日志记录格式化为 JSON。

    自动包含 record 中通过 `extra=` 传入的自定义字段（如 agent、request_id），
    方便后续接入 ELK / Loki 等日志系统。
    """

    # 标准库内部字段，不需要作为自定义字段输出
    _INTERNAL_KEYS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in self._INTERNAL_KEYS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _setup_logging() -> None:
    level_name = settings.log_level.upper()
    use_json = settings.json_log_format

    level = getattr(logging, level_name, logging.INFO)

    if use_json:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-10s | %(levelname)-8s | %(message)s"
        )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )


_setup_logging()

logger = logging.getLogger("AgentPlatform")


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的平台日志记录器"""
    return logging.getLogger(name)


def log_agent_event(agent: str, event: str, details: str = ""):
    """记录智能体事件

    agent 名称通过 extra 字段输出，便于日志系统按智能体过滤。
    """
    logger.info(f"{event}: {details}", extra={"agent": agent})
