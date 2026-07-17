"""任务仓库

提供任务 CRUD 操作，作为 SQL 持久化层。
"""

from typing import Optional, List

from sqlalchemy.orm import Session

from src.db.models import Task


class TaskRepository:
    """任务 SQL 仓库"""

    def __init__(self, db: Session):
        self.db = db

    def create(self, task_id: str, requirements: str) -> Task:
        existing = self.get(task_id)
        if existing:
            existing.requirements = requirements
            existing.status = "pending"
            existing.current_phase = "init"
            existing.result = None
            existing.error = None
            self.db.commit()
            self.db.refresh(existing)
            return existing

        task = Task(id=task_id, requirements=requirements, status="pending", current_phase="init")
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task

    def get(self, task_id: str) -> Optional[Task]:
        return self.db.query(Task).filter(Task.id == task_id).first()

    def list(self, limit: int = 100, offset: int = 0) -> List[Task]:
        return (
            self.db.query(Task)
            .order_by(Task.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def update(
        self,
        task_id: str,
        status: Optional[str] = None,
        current_phase: Optional[str] = None,
        session_id: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> Optional[Task]:
        task = self.get(task_id)
        if not task:
            return None

        if status is not None:
            task.status = status
        if current_phase is not None:
            task.current_phase = current_phase
        if session_id is not None:
            task.session_id = session_id
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error

        self.db.commit()
        self.db.refresh(task)
        return task
