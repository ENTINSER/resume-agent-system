"""向量索引器单元测试"""

import pytest

from src.vector_store.indexer import index_task_result, get_vector_store
from src.vector_store.vector_store import VectorStore
from src.vector_store.embedding_service import EmbeddingService


@pytest.fixture
def vector_store(monkeypatch, mock_embedding_service):
    store = VectorStore(
        url=":memory:",
        collection_name="test_indexer",
        vector_size=EmbeddingService.VECTOR_SIZE,
        embedding_service=mock_embedding_service,
    )
    monkeypatch.setattr("src.vector_store.indexer.get_vector_store", lambda: store)
    return store


class TestIndexer:
    def test_index_task_result(self, vector_store):
        state = {
            "session_id": "sess-001",
            "task_id": "task-001",
            "requirements": "开发一个 Python 计算器",
            "project": {
                "name": "Calculator",
                "tech_stack": ["pytest"],
                "milestones": ["setup", "core"],
            },
            "code_artifacts": [
                {"file_path": "main.py", "content": "def add(a, b): return a + b"},
                {"file_path": "README.md", "content": "# Calc"},
            ],
            "evaluation_result": {
                "overall_score": 88,
                "passed": True,
                "issues": [],
                "recommendations": [],
            },
        }

        index_task_result(state)

        # 应该有：requirement、project_summary、code、README(project_summary)、evaluation = 5 条
        assert vector_store.count() == 5

        # 再次索引应幂等（确定性 ID 不新增重复点）
        index_task_result(state)
        assert vector_store.count() == 5

        # 验证可以检索到
        results = vector_store.search("计算器", top_k=10)
        assert len(results) >= 1
