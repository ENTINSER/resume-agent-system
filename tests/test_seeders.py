"""RAG 数据种子测试"""

import os
import sqlite3

import pytest

from src.core.persistence import SessionStore
from src.vector_store.seeders.external_repo import ExternalRepoSeeder
from src.vector_store.seeders.historical_indexer import HistoricalIndexer
from src.vector_store.seeders.synthetic_generator import SyntheticExampleGenerator
from src.vector_store.vector_store import VectorStore
from src.vector_store.embedding_service import EmbeddingService


@pytest.fixture
def memory_store(mock_embedding_service):
    return VectorStore(
        url=":memory:",
        collection_name="test_seeders",
        vector_size=EmbeddingService.VECTOR_SIZE,
        embedding_service=mock_embedding_service,
    )


def test_external_repo_seeder_local_dir(tmp_path, memory_store, monkeypatch):
    # 构造一个临时项目目录
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir()
    (repo_dir / "main.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo_dir / "README.md").write_text("# Sample\nA sample repo.\n", encoding="utf-8")
    (repo_dir / "node_modules").mkdir()
    (repo_dir / "node_modules" / "big.js").write_text("// ignored\n")

    seeder = ExternalRepoSeeder(source=str(repo_dir), name="sample")
    monkeypatch.setattr("src.vector_store.seeders.external_repo.get_vector_store", lambda: memory_store)
    stats = seeder.seed()

    assert stats["indexed"] > 0
    assert stats["files"] == 2  # main.py + README.md
    assert memory_store.count() == stats["indexed"]


def test_synthetic_generator_seeds(memory_store, monkeypatch):
    monkeypatch.setattr("src.vector_store.seeders.synthetic_generator.get_vector_store", lambda: memory_store)
    stats = SyntheticExampleGenerator().seed()
    assert stats["indexed"] > 0
    assert stats["examples"] >= 4
    assert memory_store.count() == stats["indexed"]


def test_historical_indexer(tmp_path, memory_store, monkeypatch):
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path=str(db_path))

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at, updated_at, requirements, status, passed, project_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s-test", "2024-01-01T00:00:00", "2024-01-01T00:00:00",
             "Build a calculator", "complete", 1, "calc"),
        )
        conn.execute(
            "INSERT INTO artifacts (session_id, iteration, file_path, content, language, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s-test", 1, "main.py", "def add(a, b): return a + b", "python", "agent", "2024-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO evaluations (session_id, iteration, overall_score, passed, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("s-test", 1, 85, 1, "2024-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO test_runs (session_id, iteration, file, passed, output, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s-test", 1, "test_main.py", 0, "", "ZeroDivisionError", "2024-01-01T00:00:00"),
        )
        conn.commit()

    indexer = HistoricalIndexer(projects_dir=str(tmp_path / "projects"))
    indexer.store = store
    monkeypatch.setattr("src.vector_store.seeders.historical_indexer.get_vector_store", lambda: memory_store)

    stats = indexer.index_all()
    assert stats["sessions"]["indexed"] > 0
    assert stats["sessions"]["sessions"] == 1
    # projects/ 目录不存在时返回 0
    assert stats["projects"]["indexed"] == 0
    assert memory_store.count() == stats["sessions"]["indexed"]


def test_historical_projects_dir(tmp_path, memory_store, monkeypatch):
    proj_dir = tmp_path / "projects" / "legacy"
    proj_dir.mkdir(parents=True)
    (proj_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")

    indexer = HistoricalIndexer(projects_dir=str(tmp_path / "projects"))
    monkeypatch.setattr("src.vector_store.seeders.historical_indexer.get_vector_store", lambda: memory_store)
    stats = indexer.index_projects_dir()
    assert stats["indexed"] > 0
    assert stats["project_dirs"] == 1
    assert memory_store.count() == stats["indexed"]
