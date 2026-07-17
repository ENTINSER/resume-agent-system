"""SQLAlchemy 数据模型"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, DateTime, JSON

from src.db.base import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Task(Base):
    """任务表：保存所有由 API 投递的 Agent 任务"""

    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    requirements = Column(Text, nullable=False)
    status = Column(String(32), default="pending", nullable=False, index=True)
    current_phase = Column(String(64), default="init", nullable=False)
    session_id = Column(String(36), nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "requirements": self.requirements,
            "status": self.status,
            "current_phase": self.current_phase,
            "session_id": self.session_id,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
