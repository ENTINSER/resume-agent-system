"""数据库模块单元测试"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.db.task_repository import TaskRepository


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestTaskRepository:
    def test_create_and_get(self, db_session):
        repo = TaskRepository(db_session)
        task = repo.create("task-001", "开发计算器")
        assert task.id == "task-001"
        assert task.status == "pending"

        found = repo.get("task-001")
        assert found is not None
        assert found.requirements == "开发计算器"

    def test_update(self, db_session):
        repo = TaskRepository(db_session)
        repo.create("task-001", "开发计算器")
        updated = repo.update("task-001", status="running", current_phase="development")
        assert updated.status == "running"
        assert updated.current_phase == "development"

    def test_list(self, db_session):
        repo = TaskRepository(db_session)
        repo.create("task-001", "需求1")
        repo.create("task-002", "需求2")
        tasks = repo.list(limit=10)
        assert len(tasks) == 2
