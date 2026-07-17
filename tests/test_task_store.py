"""TaskStore 单元测试

验证 Redis + SQL 双写、事件发布、任务状态读写。
使用 fakeredis 避免依赖真实 Redis。
"""

import pytest
from unittest.mock import patch

from src.tasks.task_store import TaskStore


@pytest.fixture
def fake_redis(monkeypatch):
    """用 fakeredis 替代真实 Redis"""
    try:
        import fakeredis
    except ImportError:
        pytest.skip("fakeredis not installed")

    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    monkeypatch.setattr("redis.from_url", lambda url, **kwargs: client)
    return client


@pytest.fixture
def store(fake_redis):
    return TaskStore()


class TestTaskStore:
    def test_create_task(self, store):
        store.create_task("task-001", "开发计算器")
        data = store.get_task("task-001")
        assert data is not None
        assert data["status"] == "pending"
        assert data["requirements"] == "开发计算器"

    def test_update_task(self, store):
        store.create_task("task-001", "开发计算器")
        store.update_task("task-001", status="running", current_phase="development")
        data = store.get_task("task-001")
        assert data["status"] == "running"
        assert data["current_phase"] == "development"

    def test_update_with_result(self, store):
        store.create_task("task-001", "开发计算器")
        store.update_task("task-001", status="completed", result={"score": 90})
        data = store.get_task("task-001")
        assert data["status"] == "completed"
        assert data["result"]["score"] == 90

    def test_publish_event(self, store, fake_redis):
        store.create_task("task-001", "开发计算器")
        pubsub = fake_redis.pubsub()
        pubsub.subscribe(store.TASK_EVENTS_CHANNEL)
        store.publish_phase("task-001", "development")
        # 读取消息
        message = None
        for msg in pubsub.listen():
            if msg["type"] == "message":
                message = msg
                break
        assert message is not None
        assert "task-001" in message["data"]
