"""SQLAlchemy 数据库基座

支持 SQLite 和 Postgres，通过 DATABASE_URL 环境变量切换。
默认 SQLite，便于本地开发和测试。
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.core.config import settings

DATABASE_URL = settings.database_url

# SQLite 需要 check_same_thread=False；Postgres 等生产环境使用连接池
create_engine_kwargs = {
    "pool_pre_ping": True,
    "pool_size": 10,
    "max_overflow": 20,
    "echo": False,
}
if DATABASE_URL.startswith("sqlite"):
    create_engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **create_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """生成数据库会话，用于依赖注入"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """初始化所有表"""
    Base.metadata.create_all(bind=engine)
