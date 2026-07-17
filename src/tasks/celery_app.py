"""Celery 应用配置

提供任务队列能力，供 Go API 投递、Python Worker 消费。
"""

from celery import Celery

from src.core.config import settings

# Redis URL 优先从环境变量读取，默认本地 Redis
REDIS_URL = settings.redis_url

app = Celery(
    "agent_platform",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks.agent_task", "src.tasks.ml_training_task"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 单个任务最多 30 分钟
    worker_prefetch_multiplier=1,  # 公平调度，Worker 一次只取一个任务
    result_expires=3600 * 24 * 7,  # 结果保留 7 天
)
