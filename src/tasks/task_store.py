"""任务状态存储

使用 Redis 作为跨语言（Go API + Python Worker）共享的任务状态存储。
存储格式简单，便于 Go 端直接读取。
"""

import json
import time
from typing import Optional, Dict, Any

import redis

from src.db.base import SessionLocal
from src.db.task_repository import TaskRepository


class TaskStore:
    """任务状态存储

    Redis 负责跨语言快速读写 + 事件发布；
    SQL（SQLite/Postgres）负责持久化归档。
    """

    TASK_KEY_PREFIX = "agent:task:"
    TASK_EVENTS_CHANNEL = "agent:task:events"

    def __init__(self, redis_url: str = "redis://localhost:6379/0", repository: Optional[TaskRepository] = None):
        self.client = redis.from_url(redis_url, decode_responses=True)
        # 复用同一个 SQL 会话/仓库，避免每次操作都新建会话（Q14 性能优化）
        if repository is not None:
            self.repository = repository
            self._db = None
        else:
            self._db = SessionLocal()
            self.repository = TaskRepository(self._db)

    def _key(self, task_id: str) -> str:
        return f"{self.TASK_KEY_PREFIX}{task_id}"

    def create_task(self, task_id: str, requirements: str) -> None:
        """创建任务记录"""
        now = str(int(time.time()))
        self.client.hset(self._key(task_id), mapping={
            "task_id": task_id,
            "requirements": requirements,
            "status": "pending",
            "current_phase": "init",
            "created_at": now,
            "updated_at": now,
            "result": "",
            "error": "",
        })
        # 持久化到 SQL
        self.repository.create(task_id, requirements)

    def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        current_phase: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """更新任务状态"""
        mapping = {"updated_at": str(int(time.time()))}
        if status is not None:
            mapping["status"] = status
        if current_phase is not None:
            mapping["current_phase"] = current_phase
        if result is not None:
            mapping["result"] = json.dumps(result, ensure_ascii=False, default=str)
        if error is not None:
            mapping["error"] = error

        self.client.hset(self._key(task_id), mapping=mapping)
        # 发布事件，供 SSE 推送
        self.client.publish(self.TASK_EVENTS_CHANNEL, json.dumps({
            "task_id": task_id,
            "status": mapping.get("status"),
            "current_phase": mapping.get("current_phase"),
            "updated_at": mapping["updated_at"],
        }, default=str))

        # 持久化到 SQL
        self.repository.update(
            task_id,
            status=status,
            current_phase=current_phase,
            result=result,
            error=error,
        )

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        data = self.client.hgetall(self._key(task_id))
        if not data:
            return None

        result_str = data.get("result", "")
        if result_str:
            try:
                data["result"] = json.loads(result_str)
            except json.JSONDecodeError:
                pass
        return dict(data)

    def publish_phase(self, task_id: str, phase: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """发布阶段事件"""
        self.client.publish(self.TASK_EVENTS_CHANNEL, json.dumps({
            "task_id": task_id,
            "status": "running",
            "current_phase": phase,
            "payload": payload or {},
            "updated_at": str(int(time.time())),
        }, default=str))

    def increment_quota(self, api_key: str, tokens: int, cost: float) -> None:
        """按 API Key 累计 token/cost 用量"""
        key = f"agent:quota:{api_key}"
        pipe = self.client.pipeline()
        pipe.hincrby(key, "total_tokens", tokens)
        pipe.hincrby(key, "task_count", 1)
        pipe.hset(key, "total_cost", f"{cost:.6f}")
        pipe.execute()
