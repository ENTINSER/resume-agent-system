"""SessionStore 持久化单元测试

验证 SQLite 写入、读取、列表查询以及修复后的“不重复写入”行为。
"""

import os
import tempfile
import pytest

from src.core.persistence import SessionStore


@pytest.fixture
def store(tmp_path):
    """每个测试使用独立的数据库文件"""
    db_path = tmp_path / "test_sessions.db"
    return SessionStore(db_path=str(db_path))


@pytest.fixture
def sample_state():
    """构造一个包含各类子记录的示例状态"""
    return {
        "session_id": "test-session-001",
        "created_at": "2024-01-01T00:00:00",
        "requirements": "开发一个计算器",
        "status": "complete",
        "current_phase": "done",
        "project": {"name": "Calculator"},
        "iteration_count": 2,
        "code_artifacts": [
            {"file_path": "main.py", "content": "print(1)", "language": "python", "source": "test"},
            {"file_path": "tests/test_main.py", "content": "pass", "language": "python", "source": "test"},
        ],
        "test_results": [
            {"file": "docker_tests", "passed": True, "duration_ms": 100, "output": "1 passed", "error": "", "sandbox": "docker"},
        ],
        "evaluation_result": {
            "overall_score": 85.0,
            "passed": True,
            "pass_threshold": 80,
            "dimensions": {
                "code_quality": {"score": 80, "weight": 0.3, "details": []},
                "functionality": {"score": 90, "weight": 0.3, "details": []},
                "performance": {"score": 85, "weight": 0.2, "details": []},
                "documentation": {"score": 85, "weight": 0.2, "details": []},
            },
            "issues": [],
            "recommendations": [],
        },
        "llm_usage": [
            {"phase": "development", "input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost_usd": 0.0001, "elapsed_ms": 1000},
            {"phase": "evaluation", "input_tokens": 80, "output_tokens": 40, "total_tokens": 120, "cost_usd": 0.00008, "elapsed_ms": 800},
        ],
    }


class TestSessionStoreInit:
    """测试存储初始化"""

    def test_db_file_created(self, store):
        assert os.path.exists(store.db_path)


class TestSaveAndRetrieve:
    """测试保存与读取"""

    def test_get_session(self, store, sample_state):
        store.save_session(sample_state)
        row = store.get_session(sample_state["session_id"])
        assert row is not None
        assert row["session_id"] == sample_state["session_id"]
        assert row["project_name"] == "Calculator"
        assert row["overall_score"] == 85.0
        assert row["passed"] == 1
        assert row["iteration_count"] == 2

    def test_list_sessions(self, store, sample_state):
        store.save_session(sample_state)
        sessions = store.list_sessions(limit=10)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == sample_state["session_id"]


class TestNoDuplicateWrites:
    """测试修复后的不重复写入"""

    def test_save_twice_no_duplicate_llm_usage(self, store, sample_state):
        store.save_session(sample_state)
        store.save_session(sample_state)

        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM llm_usage WHERE session_id = ?",
                (sample_state["session_id"],)
            ).fetchone()[0]

        # 两次保存后仍应是 2 条，而不是 4 条
        assert count == 2

    def test_save_twice_no_duplicate_artifacts(self, store, sample_state):
        store.save_session(sample_state)
        store.save_session(sample_state)

        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM artifacts WHERE session_id = ?",
                (sample_state["session_id"],)
            ).fetchone()[0]

        assert count == 2

    def test_save_twice_no_duplicate_evaluations(self, store, sample_state):
        store.save_session(sample_state)
        store.save_session(sample_state)

        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM evaluations WHERE session_id = ?",
                (sample_state["session_id"],)
            ).fetchone()[0]

        assert count == 1

    def test_save_twice_no_duplicate_test_runs(self, store, sample_state):
        store.save_session(sample_state)
        store.save_session(sample_state)

        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM test_runs WHERE session_id = ?",
                (sample_state["session_id"],)
            ).fetchone()[0]

        assert count == 1

    def test_save_updates_session(self, store, sample_state):
        store.save_session(sample_state)

        # 修改状态后再次保存
        sample_state["status"] = "human_review"
        sample_state["iteration_count"] = 3
        store.save_session(sample_state)

        row = store.get_session(sample_state["session_id"])
        assert row["status"] == "human_review"
        assert row["iteration_count"] == 3


class TestCostSummary:
    """测试成本汇总"""

    def test_cost_summary(self, store, sample_state):
        store.save_session(sample_state)
        summary = store.get_cost_summary(sample_state["session_id"])

        assert summary["session_count"] == 1
        assert summary["total_tokens"] == 270
        assert summary["total_cost_usd"] == pytest.approx(0.00018)
